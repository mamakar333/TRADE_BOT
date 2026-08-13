"""Live (real-money) execution engine. Trades across a resolved watchlist of
fast-moving crypto markets (see trade_bot/watchlist.py's ACTIVE_CRYPTO_*) --
originally KXBTC15M only, extended 2026-08-03 per explicit user request to
cover every fast-rotating crypto series (~10-11 tickers at a time).

Safety model -- read before changing any of this:

1. The EXCHANGE is the only source of truth for cash and positions. Every
   cycle re-fetches real balance (`data.get_available_balance_dollars`) and
   real positions (`data.get_real_positions`, filtered to `allowed_series_
   prefixes`) from Kalshi directly. `LiveLedger` (live_ledger.py) is an
   audit trail only -- it is reconciled against exchange truth every cycle,
   never trusted blindly, and lives in a separate database file from paper
   trading's so the two can never cross-contaminate.

2. Every real position must match one of `allowed_series_prefixes`.
   `watchlist_resolver` and every position fetch are prefix-filtered so this
   engine can never see, reason about, or touch anything else on the user's
   real Kalshi account, no matter what else is in it -- this is what keeps
   "trade more markets" from silently becoming "trade any market."

3. Order placement is single-attempt (see `KalshiClient.post`'s docstring)
   and always `fill_or_kill` -- it either fills immediately or does nothing,
   and this code never leaves a resting order behind. On any ambiguous
   failure it logs and does nothing further; it never assumes success and
   never blindly retries a mutating call.

4. Two independent hard caps on capital, both enforced in code, not just
   config: `LiveRiskLimits` (the numbers a human actually chose for this
   run) AND `ABSOLUTE_MAX_*_DOLLARS` below (a circuit breaker that a bad
   config value literally cannot exceed without editing this file). Raising
   the absolute ceiling is a deliberate code change, same spirit as the
   paper engine's SIMULATION_MODE flag.

5. `PerformanceGovernor` (adaptive.py) can reduce size or pause new entries
   based on this strategy's own trailing REAL results -- the "learns as it
   goes" piece. It only ever makes the bot more conservative, never more
   aggressive, and only in response to trades that have actually happened.
"""
from __future__ import annotations

import json
import logging
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

from typing import Callable

from .adaptive import PerformanceGovernor
from .client import KalshiAPIError, KalshiClient
from .data import get_available_balance_dollars, get_market, get_market_orderbook, get_real_positions
from .fees import taker_fee_dollars
from .interval_tracker import IntervalTracker, tracked_asset
from .live_ledger import LiveLedger
from .notifications import send_notification
from .online_learner import OnlineLogisticLearner
from .portfolio import Position
from .push import send_push
from .strategy import MarketSnapshot, Signal, Strategy, StrategyDecision

# Circuit breaker: no LiveRiskLimits passed to this engine may exceed these,
# regardless of what run_live_trading.py configures. Raising these requires
# editing this file -- a deliberate, reviewable act.
ABSOLUTE_MAX_CAPITAL_DOLLARS = 500.0
ABSOLUTE_MAX_POSITION_DOLLARS = 100.0

# User-controlled soft cap, independent of the two hard limits above --
# toggled live from the dashboard/app (LiveLedger.set_small_bets_only) with
# no restart or redeploy needed, since _act() re-reads it fresh on every
# entry. Added 2026-08-05 per explicit user request: a quick way to force
# every new entry down to a small, low-risk size without touching config.
SMALL_BETS_ONLY_CAP_DOLLARS = 5.0

DEFAULT_LOG_PATH = Path(__file__).resolve().parent.parent / "logs" / "live_decisions.log"

# Diagnosed 2026-08-03: right after a restart (or whenever several tickers'
# history windows fill in on the same cycle), one real BTC/ETH move can make
# multiple tickers that all just track that same underlying signal at once
# -- e.g. KXBTC15M plus four different KXBTCD strikes all firing within the
# same minute. That's not diversification, it's one bet split into several
# same-direction positions that all lose together if the move reverses,
# while the risk engine (per-ticker position caps, total-capital cap) has no
# way to see that they're correlated. Maps a ticker to the real-world asset
# it's actually a bet on, so the engine can cap concurrent EXPOSURE per
# asset, not just per ticker.
UNDERLYING_ASSET_PREFIXES: dict[str, str] = {
    "KXBTC15M-": "BTC",
    "KXBTCD-": "BTC",
    "KXETH15M-": "ETH",
    "KXETHD-": "ETH",
    "KXBNB15M-": "BNB",
}


def underlying_asset(ticker: str) -> str:
    for prefix, asset in UNDERLYING_ASSET_PREFIXES.items():
        if ticker.startswith(prefix):
            return asset
    return ticker  # unknown ticker: treat as its own asset, never grouped with anything


@dataclass
class LiveRiskLimits:
    max_capital_dollars: float
    # Per-trade cap once available balance is at or above this amount --
    # below it, a trade is instead capped at `low_balance_fraction` of
    # whatever's actually left (see `_dynamic_position_cap`). This is what
    # lets the bot keep trading in smaller and smaller sizes as it draws
    # down, rather than either being unable to size a trade at all once
    # balance drops under the flat cap, or risking a large fraction of a
    # small remaining balance on one trade.
    max_position_size_dollars: float
    low_balance_fraction: float = 0.25
    daily_stop_loss_dollars: float = 35.0
    # Added 2026-08-04 per explicit user request: disable the automatic
    # daily-loss pause entirely, so the strategy keeps trading (and the
    # online learner keeps getting labeled examples) through a losing day
    # instead of going idle until UTC midnight. The user is taking over
    # this control manually via the dashboard/app Stop button instead.
    # daily_stop_loss_dollars is left in place and still SHOWN in both UIs
    # for awareness -- this flag only controls whether it actually blocks
    # anything. max_capital_dollars/max_position_size_dollars (the per-
    # trade and total sizing caps) are unrelated and still fully enforced.
    daily_kill_switch_enabled: bool = True


class LiveExecutionEngine:
    def __init__(
        self,
        client: KalshiClient,
        strategy: Strategy,
        ledger: LiveLedger,
        risk_limits: LiveRiskLimits,
        watchlist_resolver: Callable[[], list[str]],
        allowed_series_prefixes: list[str],
        history_window: int = 200,
        log_path: Path | str = DEFAULT_LOG_PATH,
        governor: PerformanceGovernor | None = None,
        fok_slippage_cents: float = 1.0,
        learner: OnlineLogisticLearner | None = None,
        min_trades_for_ml_gate: int = 20,
        ml_gate_threshold: float = 0.40,
        notify_topic: str | None = None,
        interval_tracker: IntervalTracker | None = None,
        reentry_cooldown_minutes: float = 8.0,
        extra_toggle_refresh: Callable[[], None] | None = None,
        underlying_key_fn: Callable[[str], str] | None = None,
        pause_new_entries_when: Callable[[], bool] | None = None,
    ):
        if risk_limits.max_capital_dollars > ABSOLUTE_MAX_CAPITAL_DOLLARS:
            raise RuntimeError(
                f"risk_limits.max_capital_dollars ({risk_limits.max_capital_dollars}) exceeds the "
                f"hard-coded ABSOLUTE_MAX_CAPITAL_DOLLARS ({ABSOLUTE_MAX_CAPITAL_DOLLARS}) in live_engine.py. "
                "Raise the ceiling in code, deliberately, if this is really intended."
            )
        if risk_limits.max_position_size_dollars > ABSOLUTE_MAX_POSITION_DOLLARS:
            raise RuntimeError(
                f"risk_limits.max_position_size_dollars ({risk_limits.max_position_size_dollars}) exceeds "
                f"the hard-coded ABSOLUTE_MAX_POSITION_DOLLARS ({ABSOLUTE_MAX_POSITION_DOLLARS}) in live_engine.py."
            )
        self.client = client
        self.strategy = strategy
        self.ledger = ledger
        self.risk_limits = risk_limits
        self.watchlist_resolver = watchlist_resolver
        self.allowed_series_prefixes = allowed_series_prefixes
        self.history_window = history_window
        self.learner = learner
        self.min_trades_for_ml_gate = min_trades_for_ml_gate
        self.ml_gate_threshold = ml_gate_threshold
        self.governor = governor
        self.fok_slippage_cents = fok_slippage_cents
        self.notify_topic = notify_topic
        # None by default (same optionality convention as governor/learner
        # above) -- run_live_trading.py wires up a real one; tests that
        # don't care about interval tracking pay no extra DB cost.
        self.interval_tracker = interval_tracker
        # Diagnosed live 2026-08-06: without this, the SAME ticker could hit
        # a stop-loss, re-enter, and hit another stop-loss again within
        # minutes -- observed directly, e.g. KXBTCD-26AUG0523-T64499.99 was
        # stopped/trailed out FOUR separate times within 40 minutes during
        # one choppy stretch, and a BTC/ETH scalp ticker was stopped out
        # three times in three minutes during another. The generic paper
        # ExecutionEngine (trade_bot/engine.py) already had exactly this
        # protection; it was simply never ported to the live engine when
        # this one was built. Same mechanism, same default duration.
        self.reentry_cooldown_minutes = reentry_cooldown_minutes
        self._cooldown_until: dict[str, datetime] = {}
        self._history: dict[str, list[MarketSnapshot]] = {}
        self._logger = self._make_logger(Path(log_path))
        # Generalized 2026-08-12 for the MLB strategy's own live toggle
        # (mlb_trading_enabled) -- was a hardcoded btc_dip_enabled refresh
        # inline in run_once() below. None (the default) means "nothing
        # extra to refresh"; run_live_trading.py passes its exact previous
        # lambda so crypto's behavior is unchanged.
        self.extra_toggle_refresh = extra_toggle_refresh
        # Generalized 2026-08-12 for the same-game exposure guard MLB needs
        # (mlb_watchlist.game_key groups both team-side tickers of one
        # game) -- defaults to the existing crypto underlying_asset()
        # function, so crypto's correlated-asset behavior is unchanged.
        self.underlying_key_fn = underlying_key_fn or underlying_asset
        # Added 2026-08-13 per explicit user request: crypto and MLB trade
        # the same real Kalshi account/capital pool from two independent
        # processes -- when MLB trading is turned on, new crypto entries
        # should pause rather than compete with it for the same balance.
        # None (the default) means never pause -- MLB's own engine instance
        # doesn't pass this at all, it has no reason to pause itself. Only
        # blocks NEW entries, same as the daily kill-switch/governor pause;
        # existing open positions still get managed/exited normally.
        self.pause_new_entries_when = pause_new_entries_when

    def _make_logger(self, log_path: Path) -> logging.Logger:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        logger = logging.getLogger(f"trade_bot.live_engine.{id(self)}")
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
        logging.getLogger(__name__).info(json.dumps(fields, default=str))

    @staticmethod
    def _own_side_price_pct(side: str, yes_scale_price_pct: float) -> float:
        """Convert a unified YES-scale order price (what the V2 API speaks)
        into "that side's own price" (what strategy.py's exit logic and pnl
        math expect, matching the paper engine's convention)."""
        return yes_scale_price_pct if side == "YES" else 100 - yes_scale_price_pct

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
            result=m.get("result") or None,
        )

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

    @staticmethod
    def _parse_close_time(raw: str | None) -> datetime | None:
        if not raw:
            return None
        try:
            return datetime.fromisoformat(raw.replace("Z", "+00:00"))
        except ValueError:
            return None

    @staticmethod
    def _today_start_iso() -> str:
        return datetime.now(timezone.utc).date().isoformat() + "T00:00:00+00:00"

    def _daily_realized_pnl(self) -> float:
        return self.ledger.get_realized_pnl_since(self._today_start_iso())

    def _dynamic_position_cap(self, available: float) -> float:
        """$100 flat cap while there's at least that much to trade with;
        below that, scale down to 25% of whatever's left instead of either
        refusing to size a trade or risking most of a small balance on one
        trade. Never returns more than `available` itself -- the caller
        still clamps against `available` directly too, this is just the
        per-trade fraction of it."""
        if available >= self.risk_limits.max_position_size_dollars:
            return self.risk_limits.max_position_size_dollars
        return available * self.risk_limits.low_balance_fraction

    def _update_learner(self, ticker: str, ledger_open, realized_pnl: float) -> None:
        """Feeds the just-learned outcome back into the online model and
        persists the updated weights immediately -- so learning survives a
        restart and the next entry decision (on any ticker) sees it right
        away, with no redeploy involved. Only fires for trades that carried
        an entry-time feature snapshot (our own strategy's BUY decisions);
        reconciliation-driven closes and manual trades have none and are
        silently skipped here, on purpose -- there's no clean label for
        "why" those happened."""
        if self.learner is None or not ledger_open.features_json:
            return
        try:
            features = json.loads(ledger_open.features_json)
        except json.JSONDecodeError:
            return
        label = 1 if realized_pnl > 0 else 0
        self.learner.update(features, label)
        self.ledger.save_learner_state(self.learner.to_json())
        self._log(event="ml_updated", ticker=ticker, n_updates=self.learner.n_updates, label=label, realized_pnl=realized_pnl)

    def _entry_order_terms(self, side: str, snapshot: MarketSnapshot) -> tuple[str | None, float | None]:
        if side == "YES":
            if snapshot.yes_ask_pct is None:
                return None, None
            return "bid", min(99.0, snapshot.yes_ask_pct + self.fok_slippage_cents)
        if snapshot.yes_bid_pct is None:
            return None, None
        return "ask", max(1.0, snapshot.yes_bid_pct - self.fok_slippage_cents)

    def _close_order_terms(self, side: str, snapshot: MarketSnapshot) -> tuple[str | None, float | None]:
        if side == "YES":
            if snapshot.yes_bid_pct is None:
                return None, None
            return "ask", max(1.0, snapshot.yes_bid_pct - self.fok_slippage_cents)
        if snapshot.yes_ask_pct is None:
            return None, None
        return "bid", min(99.0, snapshot.yes_ask_pct + self.fok_slippage_cents)

    def run_once(self) -> None:
        # Refreshes whatever strategy-specific live toggle this engine
        # instance was configured with (e.g. crypto's btc_dip_enabled, or
        # MLB's mlb_trading_enabled) from the shared ledger once per cycle,
        # so flipping it in the dashboard/app takes effect on the next
        # cycle with no restart. Best-effort: a plain attribute set can't
        # realistically raise, but this must never be the thing that takes
        # a cycle down regardless. See extra_toggle_refresh's constructor
        # comment.
        if self.extra_toggle_refresh is not None:
            try:
                self.extra_toggle_refresh()
            except Exception:
                self._log(event="extra_toggle_refresh_error", error="failed to refresh an extra live toggle")

        try:
            watchlist = self.watchlist_resolver()
        except KalshiAPIError as e:
            self._log(event="watchlist_resolve_error", error=str(e))
            watchlist = []

        try:
            real_positions = get_real_positions(self.client, ticker_prefixes=self.allowed_series_prefixes)
        except KalshiAPIError as e:
            self._log(event="positions_fetch_error", error=str(e))
            return

        # Union of three things, not just the watchlist:
        # 1. The resolved watchlist itself.
        # 2. Real exchange positions -- a rotating market can roll off the
        #    watchlist while a position is still open in it (e.g. it was
        #    still tradeable when opened, expired since); it must keep
        #    getting evaluated so it can still be closed.
        # 3. Tickers the LEDGER still thinks are open -- found live
        #    2026-08-04: without this, a ticker that rotates off the
        #    watchlist AND has already closed on the exchange (so it's in
        #    neither #1 nor #2) never gets visited again, so its ledger row
        #    never gets reconciled -- it just sits there marked "open"
        #    forever, silently inflating open-position counts in the
        #    dashboard/app even though there's no real exposure at all.
        tickers_to_check = (
            set(watchlist) | {p.ticker for p in real_positions} | {t.ticker for t in self.ledger.get_all_open_trades()}
        )

        # Mutable and shared across this whole cycle -- _act() adds to it the
        # moment a fill actually happens, so a second correlated ticker
        # processed later in this SAME cycle also sees the asset as already
        # committed, not just on the next poll. See underlying_asset()'s
        # docstring above for why this exists (and self.underlying_key_fn's
        # constructor comment for how MLB reuses this same mechanism keyed
        # by game instead of by crypto asset).
        open_underlyings = {self.underlying_key_fn(p.ticker) for p in real_positions}

        for ticker in tickers_to_check:
            snapshot = self._fetch_snapshot(ticker)
            if snapshot is None:
                continue

            history = self._history.setdefault(ticker, [])
            real_pos = next((p for p in real_positions if p.ticker == ticker), None)
            ledger_open = self.ledger.get_open_trade(ticker)

            # Reconciliation: the exchange is authoritative. If it shows a
            # position our ledger doesn't know about (e.g. an order whose
            # response was lost after it actually filled), record it now so
            # the audit trail and PerformanceGovernor stay accurate.
            if real_pos is not None and ledger_open is None:
                self.ledger.record_open(
                    ticker, real_pos.side, real_pos.quantity, real_pos.avg_entry_price_pct,
                    real_pos.fees_paid_dollars, self.strategy.name, None, None,
                )
                self._log(
                    event="reconciled_untracked_position", ticker=ticker, side=real_pos.side,
                    quantity=real_pos.quantity, note="exchange showed a position the ledger didn't have",
                )
                ledger_open = self.ledger.get_open_trade(ticker)
            elif real_pos is None and ledger_open is not None:
                self._log(
                    event="reconciled_missing_position", ticker=ticker,
                    note="ledger showed an open position the exchange doesn't -- closing ledger row, exchange wins",
                )
                # Diagnosed live 2026-08-05: Kalshi zeroes out yes_bid/no_bid
                # on ANY finalized market regardless of which side actually
                # won, so falling back to a live quote here silently recorded
                # a 0 mark for every reconciled close once a market settled
                # -- correct by coincidence when our side lost, but a
                # phantom near-total loss when it WON (verified against two
                # real trades: $17.34 and $20.61 in fake losses on positions
                # that had actually settled as wins). Prefer the market's own
                # settlement result when it's available: a contract is worth
                # exactly 100 if it matches the result, 0 otherwise -- never
                # a live quote, which is meaningless once terminal.
                close_reason = "reconciled: exchange no longer shows this position"
                if snapshot.result:
                    settlement_mark = 100.0 if snapshot.result.lower() == ledger_open.side.lower() else 0.0
                    close_reason = f"reconciled: market settled {snapshot.result.upper()} (exchange no longer shows this position)"
                else:
                    settlement_mark = snapshot.yes_bid_pct if ledger_open.side == "YES" else snapshot.no_bid_pct
                if settlement_mark is not None:
                    pnl_est = (
                        ledger_open.quantity * (settlement_mark - ledger_open.entry_price_pct) / 100
                        - ledger_open.entry_fee
                    )
                    self.ledger.record_close(
                        ledger_open.id, settlement_mark, 0.0, pnl_est,
                        close_reason, None, None,
                    )
                ledger_open = None

            position = None
            if real_pos is not None:
                position = Position(
                    id=ledger_open.id if ledger_open else 0,
                    ticker=real_pos.ticker,
                    side=real_pos.side,
                    quantity=real_pos.quantity,
                    entry_price_pct=real_pos.avg_entry_price_pct,
                    entry_fee=ledger_open.entry_fee if ledger_open else 0.0,
                    opened_at=ledger_open.opened_at if ledger_open else datetime.now(timezone.utc).isoformat(),
                    strategy_name=self.strategy.name,
                    entry_kind=ledger_open.entry_kind if ledger_open else None,
                )

            cooldown_until = self._cooldown_until.get(ticker)
            if position is None and cooldown_until is not None and snapshot.timestamp < cooldown_until:
                remaining = (cooldown_until - snapshot.timestamp).total_seconds() / 60
                decision = StrategyDecision(
                    Signal.HOLD, reason=f"re-entry cooldown after stop-loss, {remaining:.1f}min remaining"
                )
            else:
                decision = self.strategy.evaluate(snapshot, history, position)

            self._log(
                event="decision", ticker=ticker, signal=decision.signal.value, size=decision.size,
                reason=decision.reason, yes_bid=snapshot.yes_bid_pct, yes_ask=snapshot.yes_ask_pct,
                no_bid=snapshot.no_bid_pct, no_ask=snapshot.no_ask_pct, status=snapshot.status,
                has_position=position is not None,
            )

            self._act(ticker, snapshot, decision, real_pos, ledger_open, open_underlyings)

            if self.interval_tracker is not None and tracked_asset(ticker) is not None:
                last_trade = self.ledger.get_last_trade_for_ticker(ticker)
                self.interval_tracker.record(
                    snapshot, decision,
                    trade_side=last_trade.side if last_trade else None,
                    trade_realized_pnl=last_trade.realized_pnl if last_trade else None,
                )

            history.append(snapshot)
            if len(history) > self.history_window:
                del history[: len(history) - self.history_window]

        if self.interval_tracker is not None:
            self.interval_tracker.reconcile_pending(self.client)

    def _act(self, ticker, snapshot: MarketSnapshot, decision, real_pos, ledger_open, open_underlyings: set[str]) -> None:
        if decision.signal == Signal.HOLD:
            return

        if decision.signal in (Signal.BUY_YES, Signal.BUY_NO):
            if real_pos is not None:
                self._log(event="skip", ticker=ticker, reason="already have a real open position in this market")
                return

            if self.pause_new_entries_when is not None and self.pause_new_entries_when():
                self._log(
                    event="skip", ticker=ticker,
                    reason="new entries paused -- another live strategy sharing this account is currently trading",
                )
                return

            asset = self.underlying_key_fn(ticker)
            if asset in open_underlyings:
                self._log(
                    event="skip", ticker=ticker,
                    reason=f"already have an open {asset}-linked position elsewhere -- avoiding correlated concentration",
                )
                return

            side = "YES" if decision.signal == Signal.BUY_YES else "NO"
            own_side_ask = snapshot.yes_ask_pct if side == "YES" else snapshot.no_ask_pct
            if not own_side_ask:
                self._log(event="skip", ticker=ticker, reason=f"no {side} ask available to size against")
                return

            daily_pnl = self._daily_realized_pnl()
            if self.risk_limits.daily_kill_switch_enabled and daily_pnl <= -self.risk_limits.daily_stop_loss_dollars:
                self._log(
                    event="risk_blocked", ticker=ticker,
                    reason=f"daily stop-loss hit (realized P&L today {daily_pnl:.2f} <= "
                    f"-{self.risk_limits.daily_stop_loss_dollars}) -- no new entries until UTC midnight",
                )
                return

            governor_multiplier, governor_reason = 1.0, "no governor configured"
            if self.governor is not None:
                gd = self.governor.evaluate(self.strategy.name)
                governor_multiplier, governor_reason = gd.multiplier, gd.reason
            if governor_multiplier <= 0:
                self._log(event="governor_paused", ticker=ticker, reason=governor_reason)
                return

            # Online ML gate: a live-updating, per-candidate-trade model on
            # top of the governor's coarser strategy-wide one (see
            # online_learner.py). Only ever dampens/blocks, same as the
            # governor -- never boosts size beyond what the strategy itself
            # decided. Inert (multiplier 1.0) until there's enough real
            # labeled data that a prediction means more than noise.
            ml_multiplier, ml_note = 1.0, "no learner configured"
            if self.learner is not None:
                if self.learner.n_updates < self.min_trades_for_ml_gate:
                    ml_note = f"only {self.learner.n_updates} learned trades so far (need {self.min_trades_for_ml_gate}) -- no adjustment yet"
                elif decision.features is None:
                    ml_note = "decision carried no features -- skipping ML gate for this entry"
                else:
                    p_win = self.learner.predict_proba(decision.features)
                    if p_win < self.ml_gate_threshold:
                        self._log(
                            event="ml_gated", ticker=ticker,
                            reason=f"model predicts {p_win:.0%} win probability, below the {self.ml_gate_threshold:.0%} gate",
                        )
                        return
                    ml_multiplier = 0.5 + 0.5 * min(1.0, (p_win - self.ml_gate_threshold) / (1.0 - self.ml_gate_threshold))
                    ml_note = f"model predicts {p_win:.0%} win probability -> size x{ml_multiplier:.2f}"

            try:
                available = get_available_balance_dollars(self.client)
            except KalshiAPIError as e:
                self._log(event="balance_check_failed", ticker=ticker, error=str(e))
                return

            dynamic_cap = self._dynamic_position_cap(available)
            target_cost = decision.size * own_side_ask / 100 * governor_multiplier * ml_multiplier
            target_cost = min(target_cost, dynamic_cap, self.risk_limits.max_capital_dollars, available)

            small_bets_only = self.ledger.get_small_bets_only()
            small_bets_note = ""
            if small_bets_only and target_cost > SMALL_BETS_ONLY_CAP_DOLLARS:
                target_cost = SMALL_BETS_ONLY_CAP_DOLLARS
                small_bets_note = f", small-bets-only cap ${SMALL_BETS_ONLY_CAP_DOLLARS:.0f}"

            count = int(target_cost // (own_side_ask / 100))
            if count < 1:
                self._log(
                    event="skip", ticker=ticker,
                    reason=f"governed/capped target ${target_cost:.2f} (balance ${available:.2f}, "
                    f"cap ${dynamic_cap:.2f}{small_bets_note}) too small for 1 contract at {own_side_ask:.1f}c",
                )
                return

            book_side, submit_price = self._entry_order_terms(side, snapshot)
            if submit_price is None:
                self._log(event="skip", ticker=ticker, reason="no price available to submit order")
                return

            client_order_id = str(uuid.uuid4())
            try:
                resp = self.client.create_order(ticker, book_side, submit_price, count, client_order_id=client_order_id)
            except KalshiAPIError as e:
                self._log(event="order_error", ticker=ticker, action="open", side=side, count=count, error=str(e))
                self.ledger.log_event("order_error", f"open {ticker} {side} count={count}: {e}")
                return

            fill_count = int(round(float(resp.get("fill_count", "0") or 0)))
            if fill_count <= 0:
                self._log(event="order_not_filled", ticker=ticker, action="open", response=resp)
                return

            avg_fill_price = resp.get("average_fill_price")
            yes_scale_fill = float(avg_fill_price) * 100 if avg_fill_price is not None else submit_price
            fill_price_pct = self._own_side_price_pct(side, yes_scale_fill)
            avg_fee = resp.get("average_fee_paid")
            total_fee = float(avg_fee) * fill_count if avg_fee is not None else taker_fee_dollars(fill_count, fill_price_pct)

            features_json = json.dumps(decision.features) if decision.features is not None else None
            self.ledger.record_open(
                ticker, side, fill_count, fill_price_pct, total_fee, self.strategy.name,
                resp.get("order_id"), client_order_id, features_json=features_json,
                entry_kind=decision.entry_kind,
            )
            open_underlyings.add(asset)
            self._log(
                event="fill", action="open", ticker=ticker, side=side, quantity=fill_count,
                price=fill_price_pct, fee=total_fee, order_id=resp.get("order_id"),
                governor=governor_reason, ml=ml_note, reason=decision.reason,
            )
            dollars_risked = fill_count * fill_price_pct / 100 + total_fee
            open_body = (
                f"{fill_count} contracts @ {fill_price_pct:.1f}c (${dollars_risked:.2f} incl. fees)\n{decision.reason}"
            )
            send_notification(self.notify_topic, f"Opened {side} on {ticker}", open_body)
            send_push(f"Opened {side} on {ticker}", open_body)

        elif decision.signal == Signal.SELL:
            if real_pos is None:
                self._log(event="skip", ticker=ticker, reason="no real open position to sell")
                return

            book_side, submit_price = self._close_order_terms(real_pos.side, snapshot)
            if submit_price is None:
                self._log(event="skip", ticker=ticker, reason="no price available to close position")
                return

            count = real_pos.quantity
            if count < 1:
                # Defense in depth on top of data.get_real_positions's own
                # max(1, ...) rounding -- never call create_order with an
                # invalid count regardless of how a bad value got here.
                self._log(event="skip", ticker=ticker, reason=f"real position quantity {count} is not closeable")
                return

            client_order_id = str(uuid.uuid4())
            try:
                # Kalshi rejects reduce_only on fill_or_kill orders (400
                # invalid_order) -- confirmed 2026-08-03 against the real API
                # during a manual end-to-end test, where this exact call
                # failed and would have left every closed position stuck
                # open forever. reduce_only requires immediate_or_cancel.
                # IOC is fine here (arguably better than FOK for a close: a
                # partial fill still only ever reduces risk, never adds it).
                resp = self.client.create_order(
                    ticker, book_side, submit_price, count, client_order_id=client_order_id,
                    reduce_only=True, time_in_force="immediate_or_cancel",
                )
            except KalshiAPIError as e:
                self._log(event="order_error", ticker=ticker, action="close", side=real_pos.side, count=count, error=str(e))
                self.ledger.log_event("order_error", f"close {ticker} {real_pos.side} count={count}: {e}")
                return

            fill_count = int(round(float(resp.get("fill_count", "0") or 0)))
            if fill_count <= 0:
                self._log(event="order_not_filled", ticker=ticker, action="close", response=resp)
                return

            if "stop-loss" in decision.reason:
                until = snapshot.timestamp + timedelta(minutes=self.reentry_cooldown_minutes)
                self._cooldown_until[ticker] = until
                self._log(
                    event="cooldown_start", ticker=ticker,
                    reason=f"stop-loss exit; blocking new entries on this ticker for {self.reentry_cooldown_minutes}min",
                )

            avg_fill_price = resp.get("average_fill_price")
            yes_scale_fill = float(avg_fill_price) * 100 if avg_fill_price is not None else submit_price
            exit_price_pct = self._own_side_price_pct(real_pos.side, yes_scale_fill)
            avg_fee = resp.get("average_fee_paid")
            exit_fee = float(avg_fee) * fill_count if avg_fee is not None else taker_fee_dollars(fill_count, exit_price_pct)

            realized_pnl = None
            if ledger_open is not None:
                realized_pnl = (
                    fill_count * (exit_price_pct - ledger_open.entry_price_pct) / 100
                    - ledger_open.entry_fee - exit_fee
                )
                self.ledger.record_close(
                    ledger_open.id, exit_price_pct, exit_fee, realized_pnl, decision.reason,
                    resp.get("order_id"), client_order_id,
                )
                self._update_learner(ticker, ledger_open, realized_pnl)

            self._log(
                event="fill", action="close", ticker=ticker, side=real_pos.side, quantity=fill_count,
                price=exit_price_pct, fee=exit_fee, realized_pnl=realized_pnl, order_id=resp.get("order_id"),
                reason=decision.reason,
            )
            pnl_note = f"${realized_pnl:+.2f}" if realized_pnl is not None else "unknown pnl"
            won = (realized_pnl or 0) >= 0
            close_title = f"{'WIN' if won else 'LOSS'}: Closed {real_pos.side} on {ticker} ({pnl_note})"
            entry_note = f"entry {ledger_open.entry_price_pct:.1f}c -> " if ledger_open is not None else ""
            close_body = f"{'✅' if won else '❌'} {entry_note}exit {exit_price_pct:.1f}c\n{decision.reason}"
            send_notification(
                self.notify_topic, close_title, close_body,
                priority="default" if won else "high",
            )
            send_push(close_title, close_body)

    def run_forever(self, interval_seconds: float = 20.0) -> None:
        while True:
            try:
                self.run_once()
            except Exception as e:
                self._log(event="cycle_error", error=str(e), error_type=type(e).__name__)
            time.sleep(interval_seconds)
