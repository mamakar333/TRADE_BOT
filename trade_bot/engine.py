"""Simulated execution engine: polls a watchlist, feeds a strategy, fills
paper trades at live book prices, and enforces risk limits.

Safety: SIMULATION_MODE below is checked in the constructor, before the
engine will do anything at all. Independently of that flag, KalshiClient
(trade_bot/client.py) only ever implements HTTP GET -- there is no POST
method anywhere in this codebase, so nothing here is structurally capable
of placing a real order even if this check were bypassed.
"""
from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Callable

from .client import KalshiAPIError, KalshiClient
from .data import get_market, get_market_orderbook
from .fees import taker_fee_dollars
from .portfolio import PaperPortfolio, Position
from .strategy import MANUAL_STRATEGY_NAME, MarketSnapshot, Signal, Strategy, StrategyDecision, force_settle_exit

SIMULATION_MODE = True

DEFAULT_LOG_PATH = Path(__file__).resolve().parent.parent / "logs" / "decisions.log"


@dataclass
class RiskLimits:
    max_position_size_dollars: float = 100.0
    max_total_exposure_dollars: float = 500.0
    daily_stop_loss_dollars: float = 50.0


@dataclass
class RiskCheckResult:
    allowed: bool
    reason: str = ""


class ExecutionEngine:
    def __init__(
        self,
        client: KalshiClient,
        portfolio: PaperPortfolio,
        strategy: Strategy,
        watchlist: list[str],
        risk_limits: RiskLimits | None = None,
        history_window: int = 200,
        log_path: Path | str = DEFAULT_LOG_PATH,
        watchlist_resolver: Callable[[], list[str]] | None = None,
        strategy_router: Callable[[str], Strategy] | None = None,
        reentry_cooldown_minutes: float = 8.0,
    ):
        if not SIMULATION_MODE:
            raise RuntimeError(
                "SIMULATION_MODE is False. This engine only ever simulates fills against a "
                "paper portfolio; it refuses to run unless that flag is True. Flipping it to "
                "False must not happen without implementing and reviewing real order-placement "
                "code first -- which does not exist in this codebase."
            )
        self.client = client
        self.portfolio = portfolio
        self.strategy = strategy
        # If given, called per-ticker each cycle to pick which Strategy handles it
        # (e.g. a crypto-specific strategy for BTC/ETH tickers, a different one for
        # sports tickers). Falls back to self.strategy for any ticker it doesn't
        # have an opinion on. The resolved strategy's own .name is what gets
        # recorded on the resulting position/trade -- callers always see which
        # actual strategy acted, never a generic "router" label.
        self.strategy_router = strategy_router
        self.watchlist = list(watchlist)
        self.risk_limits = risk_limits or RiskLimits()
        self.history_window = history_window
        # Called each cycle (if given) to refresh self.watchlist -- for rotating
        # 15min/hourly crypto series where the "current" ticker changes over time.
        self._watchlist_resolver = watchlist_resolver
        self._cycles_run = 0
        # In-memory only -- a restart rebuilds this from scratch via normal polling
        # (a few interval cycles to refill the momentum window). Not persisted because
        # the momentum strategy's lookback is short (minutes) relative to how quickly
        # polling refills it, and it keeps the persistence layer (portfolio.py) focused
        # on what actually needs durability: cash, positions, and trade history.
        self._history: dict[str, list[MarketSnapshot]] = {t: [] for t in self.watchlist}
        # Diagnosed 2026-07-13: without this, a stop-loss exit could be
        # immediately followed by a fresh entry on the exact same ticker if the
        # signal that caused the loss hadn't actually changed yet -- observed
        # as the same market stopping out 10+ times in a row. After a
        # stop-loss, that ticker is skipped for new entries for this long.
        self.reentry_cooldown_minutes = reentry_cooldown_minutes
        self._cooldown_until: dict[str, datetime] = {}

        self._logger = self._make_logger(Path(log_path))

    def _make_logger(self, log_path: Path) -> logging.Logger:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        logger = logging.getLogger(f"trade_bot.engine.{id(self)}")
        logger.setLevel(logging.INFO)
        logger.propagate = False
        if not logger.handlers:
            handler = logging.FileHandler(log_path)
            handler.setFormatter(logging.Formatter("%(message)s"))
            logger.addHandler(handler)
        return logger

    def _log(self, **fields) -> None:
        fields.setdefault("timestamp", datetime.now(timezone.utc).isoformat())
        self._logger.info(json.dumps(fields, default=str))

    def _fetch_snapshot(self, ticker: str) -> MarketSnapshot | None:
        try:
            m = get_market(self.client, ticker)
        except KalshiAPIError as e:
            self._log(event="fetch_error", ticker=ticker, error=str(e))
            return None
        return MarketSnapshot(
            ticker=ticker,
            timestamp=datetime.now(timezone.utc),
            yes_bid_pct=m.get("yes_bid_pct"),
            yes_ask_pct=m.get("yes_ask_pct"),
            no_bid_pct=m.get("no_bid_pct"),
            no_ask_pct=m.get("no_ask_pct"),
            last_price_pct=m.get("last_price_pct"),
            volume=m.get("volume") or 0,
            status=m.get("status", ""),
            orderbook_imbalance=self._fetch_orderbook_imbalance(ticker),
            close_time=self._parse_close_time(m.get("close_time")),
        )

    @staticmethod
    def _parse_close_time(raw: str | None) -> datetime | None:
        if not raw:
            return None
        try:
            return datetime.fromisoformat(raw.replace("Z", "+00:00"))
        except ValueError:
            return None

    def _fetch_orderbook_imbalance(self, ticker: str) -> float | None:
        try:
            book = get_market_orderbook(self.client, ticker)
        except KalshiAPIError:
            return None
        yes_depth = sum(level["quantity"] for level in book.get("yes", []) if level.get("quantity") is not None)
        no_depth = sum(level["quantity"] for level in book.get("no", []) if level.get("quantity") is not None)
        total = yes_depth + no_depth
        if total <= 0:
            return None
        return (yes_depth - no_depth) / total

    def _strategy_for(self, ticker: str) -> Strategy:
        if self.strategy_router is not None:
            return self.strategy_router(ticker)
        return self.strategy

    @staticmethod
    def _manual_position_decision(snapshot: MarketSnapshot, position: Position) -> StrategyDecision:
        if snapshot.status not in ("active", "open"):
            return force_settle_exit(snapshot, position)
        return StrategyDecision(
            Signal.HOLD, reason="manual position -- left to user control, bot does not auto-manage entries/exits for it"
        )

    @staticmethod
    def _today_start_iso() -> str:
        return datetime.now(timezone.utc).date().isoformat() + "T00:00:00+00:00"

    def _daily_pnl(self) -> float:
        realized = self.portfolio.get_realized_pnl_since(self._today_start_iso())
        unrealized = 0.0
        for pos in self.portfolio.get_all_open_positions():
            hist = self._history.get(pos.ticker)
            if not hist:
                continue
            mark = hist[-1].yes_bid_pct if pos.side == "YES" else hist[-1].no_bid_pct
            if mark is None:
                continue
            unrealized += pos.quantity * (mark - pos.entry_price_pct) / 100
        return realized + unrealized

    def _check_entry_risk(self, size: int, price_pct: float) -> RiskCheckResult:
        daily_pnl = self._daily_pnl()
        if daily_pnl <= -self.risk_limits.daily_stop_loss_dollars:
            return RiskCheckResult(
                False, f"daily stop-loss hit (P&L {daily_pnl:.2f} <= -{self.risk_limits.daily_stop_loss_dollars})"
            )

        cost = size * price_pct / 100
        if cost > self.risk_limits.max_position_size_dollars:
            return RiskCheckResult(
                False, f"position cost {cost:.2f} exceeds max_position_size_dollars "
                f"{self.risk_limits.max_position_size_dollars}"
            )

        exposure = self.portfolio.get_total_exposure()
        if exposure + cost > self.risk_limits.max_total_exposure_dollars:
            return RiskCheckResult(
                False, f"total exposure {exposure:.2f}+{cost:.2f} would exceed "
                f"max_total_exposure_dollars {self.risk_limits.max_total_exposure_dollars}"
            )

        return RiskCheckResult(True)

    def run_once(self) -> None:
        if self._watchlist_resolver is not None:
            try:
                self.watchlist = self._watchlist_resolver()
            except KalshiAPIError as e:
                self._log(event="watchlist_refresh_error", error=str(e))

        open_tickers = [p.ticker for p in self.portfolio.get_all_open_positions()]
        # Union, not just self.watchlist: a rotating 15min/hourly market can roll
        # off the resolved watchlist while we still hold a position in it -- that
        # position must keep getting evaluated (and become closeable) regardless.
        tickers_to_check = list(dict.fromkeys(self.watchlist + open_tickers))

        for ticker in tickers_to_check:
            snapshot = self._fetch_snapshot(ticker)
            if snapshot is None:
                continue

            self.portfolio.record_price_tick(
                ticker, snapshot.yes_bid_pct, snapshot.yes_ask_pct, snapshot.last_price_pct, snapshot.volume
            )

            history = self._history.setdefault(ticker, [])
            position = self.portfolio.get_open_position(ticker)

            cooldown_until = self._cooldown_until.get(ticker)

            if position is not None and position.strategy_name == MANUAL_STRATEGY_NAME:
                # Manual positions are user-controlled -- the bot must never enter,
                # exit, or otherwise touch them via strategy logic (that's the whole
                # point of manual mode). The only exception is forced settlement:
                # once the underlying market actually resolves, the position has to
                # be closed to keep the books accurate, regardless of who opened it.
                decision = self._manual_position_decision(snapshot, position)
                strategy_label = MANUAL_STRATEGY_NAME
                acting_strategy = None  # never referenced: manual decisions are HOLD/SELL only, never BUY
            elif position is None and cooldown_until is not None and snapshot.timestamp < cooldown_until:
                acting_strategy = self._strategy_for(ticker)
                remaining = (cooldown_until - snapshot.timestamp).total_seconds() / 60
                decision = StrategyDecision(
                    Signal.HOLD, reason=f"re-entry cooldown after stop-loss, {remaining:.1f}min remaining"
                )
                strategy_label = acting_strategy.name
            else:
                acting_strategy = self._strategy_for(ticker)
                decision = acting_strategy.evaluate(snapshot, history, position)
                strategy_label = acting_strategy.name

            self._log(
                event="decision",
                ticker=ticker,
                signal=decision.signal.value,
                size=decision.size,
                reason=decision.reason,
                yes_bid=snapshot.yes_bid_pct,
                yes_ask=snapshot.yes_ask_pct,
                no_bid=snapshot.no_bid_pct,
                no_ask=snapshot.no_ask_pct,
                orderbook_imbalance=snapshot.orderbook_imbalance,
                status=snapshot.status,
                strategy=strategy_label,
                has_position=position is not None,
                on_watchlist=ticker in self.watchlist,
            )

            self._act(ticker, snapshot, decision, position, acting_strategy)

            history.append(snapshot)
            if len(history) > self.history_window:
                del history[: len(history) - self.history_window]

        self._record_equity()

        if self.portfolio.reset_if_busted():
            self._log(event="account_reset", reason="cash balance hit $0, topped back up to starting balance")

        self._cycles_run += 1
        if self._cycles_run % 50 == 0:
            self.portfolio.prune_price_history()

    def _act(self, ticker, snapshot: MarketSnapshot, decision, position, strategy: Strategy | None) -> None:
        if decision.signal == Signal.HOLD:
            return

        if decision.signal in (Signal.BUY_YES, Signal.BUY_NO):
            if position is not None:
                self._log(event="skip", ticker=ticker, reason="already have an open position in this market")
                return
            side = "YES" if decision.signal == Signal.BUY_YES else "NO"
            fill_price = snapshot.yes_ask_pct if side == "YES" else snapshot.no_ask_pct
            if fill_price is None:
                self._log(event="skip", ticker=ticker, reason=f"no {side} ask available to fill against")
                return

            risk = self._check_entry_risk(decision.size, fill_price)
            if not risk.allowed:
                self._log(event="risk_blocked", ticker=ticker, reason=risk.reason)
                return

            fee = taker_fee_dollars(decision.size, fill_price)
            try:
                self.portfolio.open_position(ticker, side, decision.size, fill_price, fee, strategy.name)
            except ValueError as e:
                self._log(event="fill_error", ticker=ticker, reason=str(e))
                return
            self._log(
                event="fill", action="open", ticker=ticker, side=side,
                quantity=decision.size, price=fill_price, fee=fee,
            )

        elif decision.signal == Signal.SELL:
            if position is None:
                self._log(event="skip", ticker=ticker, reason="no open position to sell")
                return
            fill_price = snapshot.yes_bid_pct if position.side == "YES" else snapshot.no_bid_pct
            if fill_price is None:
                # Matches Strategy._force_settle_exit's fallback: once a market goes
                # terminal the bid can disappear before last_price does, and we still
                # need a price to close the position out at.
                fill_price = snapshot.last_price_pct
            if fill_price is None:
                self._log(event="skip", ticker=ticker, reason=f"no {position.side} bid or last price available to fill against")
                return
            fee = taker_fee_dollars(position.quantity, fill_price)
            trade = self.portfolio.close_position(ticker, fill_price, fee, close_reason=decision.reason)
            self._log(
                event="fill", action="close", ticker=ticker, side=position.side,
                quantity=position.quantity, price=fill_price, fee=fee, realized_pnl=trade.realized_pnl,
            )
            if "stop-loss" in decision.reason and position.strategy_name != MANUAL_STRATEGY_NAME:
                until = snapshot.timestamp + timedelta(minutes=self.reentry_cooldown_minutes)
                self._cooldown_until[ticker] = until
                self._log(
                    event="cooldown_start", ticker=ticker,
                    reason=f"stop-loss exit; blocking new entries on this ticker for {self.reentry_cooldown_minutes}min",
                )

    def _record_equity(self) -> None:
        positions_value = 0.0
        for pos in self.portfolio.get_all_open_positions():
            mark = pos.entry_price_pct
            hist = self._history.get(pos.ticker)
            if hist:
                side_mark = hist[-1].yes_bid_pct if pos.side == "YES" else hist[-1].no_bid_pct
                if side_mark is not None:
                    mark = side_mark
            positions_value += pos.quantity * mark / 100
        self.portfolio.record_equity_snapshot(positions_value)

    def run_forever(self, interval_seconds: float = 45.0) -> None:
        while True:
            try:
                self.run_once()
            except Exception as e:
                # Defense in depth on top of KalshiClient's own retries: a 24/7
                # loop must survive whatever it hasn't seen before rather than
                # take the whole process down over one bad cycle.
                self._log(event="cycle_error", error=str(e), error_type=type(e).__name__)
            time.sleep(interval_seconds)
