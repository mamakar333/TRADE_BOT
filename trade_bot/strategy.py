"""Pluggable strategy interface: `evaluate(snapshot, history, position) -> StrategyDecision`.

The execution engine owns fetching data, filling orders, and enforcing risk
limits. A strategy only ever answers "given what I can see, what should
happen next?" -- swapping strategies means writing a new `Strategy`
subclass, nothing in engine.py changes.
"""
from __future__ import annotations

import statistics
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum

from .online_learner import build_features
from .portfolio import Position

# The strategy_name the dashboard tags manual trades with (see app.py's
# _place_manual_trade). Positions opened this way must never be entered,
# exited, or otherwise touched by a Strategy's evaluate() -- the whole point
# of manual mode is that the user controls it. The engine checks this before
# ever calling evaluate() on a ticker with an open position.
MANUAL_STRATEGY_NAME = "manual"


class Signal(str, Enum):
    BUY_YES = "BUY_YES"
    BUY_NO = "BUY_NO"
    SELL = "SELL"
    HOLD = "HOLD"


@dataclass
class MarketSnapshot:
    ticker: str
    timestamp: datetime
    yes_bid_pct: float | None
    yes_ask_pct: float | None
    no_bid_pct: float | None
    no_ask_pct: float | None
    last_price_pct: float | None
    volume: float
    status: str
    # (yes_depth - no_depth) / (yes_depth + no_depth) across the top few resting
    # order-book levels on each side, range [-1, 1]. Positive = more resting size
    # wants to buy YES than NO -- a real market-microstructure signal (this is
    # what the order book actually contains), not a fundamentals prediction.
    # None when the engine doesn't fetch the book for this snapshot.
    orderbook_imbalance: float | None = None
    # When Kalshi stops taking new orders and settles the market. Used to force
    # an exit before that happens -- diagnosed 2026-07-12 as the single biggest
    # source of realized losses: positions left open through forced settlement
    # on short-lived rotating markets (15min BTC/ETH) averaged ~-$9-10 each,
    # roughly the entire strategy's net loss, vs ~-$0.40 for normally-exited
    # trades.
    close_time: datetime | None = None

    @property
    def mid_pct(self) -> float | None:
        if self.yes_bid_pct is None or self.yes_ask_pct is None:
            return None
        return (self.yes_bid_pct + self.yes_ask_pct) / 2

    @property
    def minutes_to_close(self) -> float | None:
        if self.close_time is None:
            return None
        return (self.close_time - self.timestamp).total_seconds() / 60


@dataclass
class StrategyDecision:
    signal: Signal
    size: int = 0
    reason: str = ""
    # Entry-time feature snapshot for OnlineLogisticLearner (online_learner.py),
    # populated only on BUY decisions that support it (CryptoTechnicalStrategy).
    # None elsewhere -- exits, HOLDs, and strategies that don't build features
    # just don't participate in the live learning gate.
    features: dict[str, float] | None = None


def force_settle_exit(snapshot: MarketSnapshot, position: Position) -> StrategyDecision:
    """Close a position out at whatever price is left once its market goes
    terminal. Shared between Strategy subclasses (via
    `_TakeProfitStopLossStrategy._force_settle_exit`) and the engine's manual-
    position handling (trade_bot/engine.py), since a manual position must
    still be reconciled when its market actually resolves, even though the
    bot otherwise never touches manual positions."""
    mark = snapshot.yes_bid_pct if position.side == "YES" else snapshot.no_bid_pct
    if mark is None:
        mark = snapshot.last_price_pct
    if mark is None:
        return StrategyDecision(
            Signal.HOLD, reason=f"market status is {snapshot.status!r} but no price available to settle against yet"
        )
    return StrategyDecision(
        Signal.SELL,
        size=position.quantity,
        reason=f"market status is {snapshot.status!r}; closing out at {mark:.1f} to realize settlement",
    )


class Strategy(ABC):
    name: str = "base"

    @abstractmethod
    def evaluate(
        self,
        snapshot: MarketSnapshot,
        history: list[MarketSnapshot],
        position: Position | None,
    ) -> StrategyDecision:
        """`history` is prior snapshots for this ticker, oldest first, not
        including `snapshot` itself. `position` is the currently open paper
        position in this ticker, if any (a strategy can only be asked to
        manage one position per market at a time)."""


class _TakeProfitStopLossStrategy(Strategy):
    """Shared control flow for strategies that enter on some signal and exit
    dynamically: a hard stop-loss floor, a trailing profit lock once a
    position is meaningfully ahead (let it run while momentum favors it,
    lock in gains on reversal instead of capping at a fixed target), and a
    time-decay exit that closes out before a short-lived market's forced
    settlement. Subclasses implement `_evaluate_entry` only.

    The trailing/urgency design replaced a fixed take-profit/stop-loss band
    after diagnosing 90 closed trades on 2026-07-12: positions left open
    until forced settlement (the market expired before the fixed band ever
    triggered) averaged ~-$9 to -$10 each and accounted for nearly the
    entire realized loss of the original momentum strategy -- normally-
    exited trades were close to breakeven. The fixed band simply wasn't
    reacting fast enough on 15-minute markets.
    """

    # All of these are FRACTIONAL RETURN ON CAPITAL (e.g. 0.08 = 8% return on
    # the dollars staked), not raw percentage-point price moves. A 3pp move
    # means very different things at a 20c entry (15% return) vs an 80c entry
    # (3.75% return), so pricing exits in raw pp was inconsistent by
    # construction. Reworked 2026-07-13 after the user asked for trades that
    # actually target 5-10%+ returns and noticed exits were firing on tiny,
    # inconsistent price wiggles.
    stop_loss_pct: float
    trail_activate_pct: float
    trail_giveback_pct: float
    max_profit_pct: float
    urgency_minutes: float
    max_entry_price_pct: float
    # Diagnosed 2026-08-03: a fixed stop_loss_pct treats a -15% move the same
    # whether a market has 55 minutes left or 2 -- but with lots of time
    # left, the underlying has real room to round-trip back before the
    # market resolves; with little time, it doesn't. Every one of this
    # strategy's live losses so far closed in under 2 minutes on a fixed
    # threshold that never accounted for how much time was actually left.
    # These two widen the stop tolerance the more time remains, phasing
    # linearly from stop_loss_pct (tightest, right before close) up to
    # stop_loss_pct + stop_loss_time_bonus_pct (loosest, at/after
    # stop_loss_time_reference_minutes remaining). Default bonus is 0.0 --
    # opt-in, so existing strategies/paper trading are unaffected unless
    # explicitly configured.
    stop_loss_time_bonus_pct: float = 0.0
    stop_loss_time_reference_minutes: float = 60.0
    # Minimum time a market must have left before a NEW entry is allowed.
    # Diagnosed 2026-08-01 against 57 closed crypto_technical trades: 33 were
    # held under 3 minutes and accounted for -$281 of the strategy's -$371
    # total loss, including single trades that lost $113 and $49 in under two
    # minutes. These were entries opened in a market's final 1-2 minutes,
    # where a short-dated threshold contract's price can gap straight through
    # any stop-loss between one 45s poll and the next (worst case: entry at
    # 81c, next poll shows 0.7c -- a near-total loss the exit logic never got
    # a chance to react to). urgency_minutes already forces an EXIT before
    # this window; this is the matching guard against a fresh ENTRY into it.
    min_minutes_to_close: float
    # Diagnosed 2026-08-03 against 35 real crypto_technical trades: win rate
    # climbed steadily with entry price at every cutoff tested (e.g. below
    # 45c: 1W/15L, 6%; at/above 45c: 3W/17L, 15%) -- buying the side the
    # market already considers the underdog performed consistently worse
    # than buying the side already somewhat favored, not just at the
    # extremes max_entry_price_pct already guards against. Default 0.0 --
    # opt-in, no floor unless explicitly configured. Honest caveat: even
    # above the cutoff, real win rate is still well under breakeven -- this
    # removes the worst-performing slice, it does not make the signal
    # profitable by itself.
    min_entry_price_pct: float = 0.0

    def evaluate(
        self,
        snapshot: MarketSnapshot,
        history: list[MarketSnapshot],
        position: Position | None,
    ) -> StrategyDecision:
        tradeable = snapshot.status in ("active", "open")

        if position is not None:
            if not tradeable:
                return self._force_settle_exit(snapshot, position)
            return self._evaluate_exit(snapshot, position, history)

        if not tradeable:
            return StrategyDecision(
                Signal.HOLD, reason=f"market status is {snapshot.status!r}, not tradeable for new entries"
            )
        minutes_left = snapshot.minutes_to_close
        if minutes_left is not None and minutes_left < self.min_minutes_to_close:
            return StrategyDecision(
                Signal.HOLD,
                reason=(
                    f"only {minutes_left:.1f}min to close, below min_minutes_to_close "
                    f"({self.min_minutes_to_close}) -- too close to expiry to safely open a new position"
                ),
            )
        return self._evaluate_entry(snapshot, history)

    def _evaluate_entry(self, snapshot: MarketSnapshot, history: list[MarketSnapshot]) -> StrategyDecision:
        raise NotImplementedError

    def _entry_price_ok(self, side: str, snapshot: MarketSnapshot) -> tuple[bool, str]:
        """Reject entries priced too close to the extremes. Buying at, say, 98
        cents risks 98 to make 2 -- the payout is asymmetric enough that even a
        genuinely high win-rate signal can have negative expected value there.
        Diagnosed 2026-07-12: the strategy was chasing momentum up to 98-100 cent
        YES prices right before several 15-min crypto markets flipped to 0,
        each a ~$9-10 loss on a position that could only ever have won a few
        cents."""
        ask = snapshot.yes_ask_pct if side == "YES" else snapshot.no_ask_pct
        if ask is None:
            return False, "no ask available"
        if ask > self.max_entry_price_pct:
            return False, (
                f"{side} ask {ask:.0f} exceeds max_entry_price_pct {self.max_entry_price_pct} "
                "-- risk/reward too asymmetric this close to certain"
            )
        if ask < self.min_entry_price_pct:
            return False, (
                f"{side} ask {ask:.0f} below min_entry_price_pct {self.min_entry_price_pct} "
                "-- buying the side the market already considers the underdog, diagnosed as a "
                "consistently worse-performing entry"
            )
        return True, ""

    def _effective_stop_loss_pct(self, snapshot: MarketSnapshot) -> float:
        if self.stop_loss_time_bonus_pct <= 0 or snapshot.minutes_to_close is None:
            return self.stop_loss_pct
        time_fraction = min(1.0, max(0.0, snapshot.minutes_to_close / self.stop_loss_time_reference_minutes))
        return self.stop_loss_pct + self.stop_loss_time_bonus_pct * time_fraction

    @staticmethod
    def _size_for_dollars(target_dollars: float, price_pct: float) -> int:
        """Contracts needed to stake ~target_dollars at this price. Sizing by
        dollars (not a fixed contract count) is what makes a position's actual
        $ risk predictable and comparable across markets trading at wildly
        different prices."""
        if price_pct <= 0:
            return 0
        return max(1, round(target_dollars / (price_pct / 100)))

    @staticmethod
    def _force_settle_exit(snapshot: MarketSnapshot, position: Position) -> StrategyDecision:
        return force_settle_exit(snapshot, position)

    def _evaluate_exit(
        self, snapshot: MarketSnapshot, position: Position, history: list[MarketSnapshot]
    ) -> StrategyDecision:
        mark = snapshot.yes_bid_pct if position.side == "YES" else snapshot.no_bid_pct
        if mark is None:
            return StrategyDecision(Signal.HOLD, reason=f"no {position.side} bid available to mark position")

        pct_return = (mark - position.entry_price_pct) / position.entry_price_pct

        # Hard floor: never ride a loss past this regardless of anything else.
        # Widened when there's still real time left for the underlying to
        # round-trip back -- see stop_loss_time_bonus_pct's docstring above.
        effective_stop_loss_pct = self._effective_stop_loss_pct(snapshot)
        if pct_return <= -effective_stop_loss_pct:
            widened_note = (
                f" (widened from base {self.stop_loss_pct:.0%} -- {snapshot.minutes_to_close:.0f}min still left)"
                if effective_stop_loss_pct >= self.stop_loss_pct + 0.01 and snapshot.minutes_to_close is not None
                else ""
            )
            return StrategyDecision(
                Signal.SELL,
                size=position.quantity,
                reason=(
                    f"{position.side} mark {mark:.1f} vs entry {position.entry_price_pct:.1f} "
                    f"({pct_return:+.1%} return) hit hard stop-loss (-{effective_stop_loss_pct:.0%}){widened_note} "
                    "-- minimizing damage"
                ),
            )

        # Time-decay urgency: get out before a short-lived market forces a
        # settlement-price exit, which is the single biggest loss driver we
        # found. If already losing, cut it now rather than risk it going to 0;
        # if flat/winning, take what's on the table.
        minutes_left = snapshot.minutes_to_close
        if minutes_left is not None and minutes_left <= self.urgency_minutes:
            return StrategyDecision(
                Signal.SELL,
                size=position.quantity,
                reason=(
                    f"{minutes_left:.1f}min to close -- exiting now at {pct_return:+.1%} return instead of "
                    "risking forced settlement"
                ),
            )

        # Trailing profit lock: once meaningfully ahead (>= trail_activate_pct,
        # e.g. 5%+), track the best return seen and close if it gives back too
        # much of that peak -- lets a winner run toward 10%+ while momentum
        # still favors it, but locks in gains the moment it turns instead of
        # waiting for a fixed ceiling (or letting it round-trip back to a loss).
        peak_return = self._peak_favorable_return(history, position, mark)
        if peak_return >= self.trail_activate_pct:
            giveback = peak_return - pct_return
            if giveback >= self.trail_giveback_pct:
                return StrategyDecision(
                    Signal.SELL,
                    size=position.quantity,
                    reason=(
                        f"{position.side} peaked at +{peak_return:.1%}, now {pct_return:+.1%} "
                        f"(gave back {giveback:.1%}) -- trailing stop locking in gains"
                    ),
                )

        if pct_return >= self.max_profit_pct:
            return StrategyDecision(
                Signal.SELL,
                size=position.quantity,
                reason=f"{position.side} at {pct_return:+.1%} hit max_profit_pct ({self.max_profit_pct:.0%}) cap",
            )

        return StrategyDecision(
            Signal.HOLD,
            reason=(
                f"{position.side} at {pct_return:+.1%} return (peak {peak_return:+.1%}), "
                f"{minutes_left:.1f}min to close" if minutes_left is not None else
                f"{position.side} at {pct_return:+.1%} return (peak {peak_return:+.1%})"
            ),
        )

    @staticmethod
    def _peak_favorable_return(history: list[MarketSnapshot], position: Position, current_mark: float) -> float:
        try:
            opened_at = datetime.fromisoformat(position.opened_at)
        except ValueError:
            opened_at = None

        marks = [current_mark]
        for s in history:
            if opened_at is not None and s.timestamp < opened_at:
                continue
            m = s.yes_bid_pct if position.side == "YES" else s.no_bid_pct
            if m is not None:
                marks.append(m)
        return (max(marks) - position.entry_price_pct) / position.entry_price_pct

    @staticmethod
    def _mid_at_or_before(history: list[MarketSnapshot], cutoff: datetime) -> float | None:
        candidates = [s for s in history if s.timestamp <= cutoff and s.mid_pct is not None]
        if not candidates:
            return None
        return max(candidates, key=lambda s: s.timestamp).mid_pct


class MomentumStrategy(_TakeProfitStopLossStrategy):
    """Follow the trend: if the YES mid-price has moved at least
    `threshold_pct` percentage points over the last `window_minutes`, buy
    the side the market is moving toward. Exits via the shared dynamic
    trailing-stop/time-decay logic in `_TakeProfitStopLossStrategy`.

    Chosen as the Phase 2 starter strategy because it's the simplest thing
    that's genuinely testable end-to-end on this pipeline: it only needs a
    rolling price history (which the execution engine builds for free by
    polling the watchlist) and it produces one falsifiable bet -- "a price
    that just moved keeps moving" -- that will show up cleanly in the P&L
    either way. No fundamental/model input, no multi-market logic, nothing
    that would make a bad result ambiguous to diagnose.
    """

    name = "momentum"

    def __init__(
        self,
        threshold_pct: float = 4.0,
        window_minutes: float = 5.0,
        target_dollars: float = 120.0,
        stop_loss_pct: float = 0.15,
        trail_activate_pct: float = 0.05,
        trail_giveback_pct: float = 0.03,
        max_profit_pct: float = 0.25,
        urgency_minutes: float = 3.0,
        max_entry_price_pct: float = 85.0,
        min_minutes_to_close: float = 5.0,
    ):
        self.threshold_pct = threshold_pct
        self.window_minutes = window_minutes
        self.target_dollars = target_dollars
        self.stop_loss_pct = stop_loss_pct
        self.trail_activate_pct = trail_activate_pct
        self.trail_giveback_pct = trail_giveback_pct
        self.max_profit_pct = max_profit_pct
        self.urgency_minutes = urgency_minutes
        self.max_entry_price_pct = max_entry_price_pct
        self.min_minutes_to_close = min_minutes_to_close

    def _evaluate_entry(self, snapshot: MarketSnapshot, history: list[MarketSnapshot]) -> StrategyDecision:
        mid = snapshot.mid_pct
        if mid is None:
            return StrategyDecision(Signal.HOLD, reason="no two-sided market (missing yes bid/ask)")

        old_mid = self._mid_at_or_before(history, snapshot.timestamp - timedelta(minutes=self.window_minutes))
        if old_mid is None:
            return StrategyDecision(
                Signal.HOLD, reason=f"insufficient price history for a {self.window_minutes}min window yet"
            )

        delta = mid - old_mid
        if delta >= self.threshold_pct:
            ok, reject_reason = self._entry_price_ok("YES", snapshot)
            if not ok:
                return StrategyDecision(Signal.HOLD, reason=reject_reason)
            size = self._size_for_dollars(self.target_dollars, snapshot.yes_ask_pct)
            return StrategyDecision(
                Signal.BUY_YES,
                size=size,
                reason=(
                    f"mid {old_mid:.1f}->{mid:.1f} ({delta:+.1f}pp over {self.window_minutes}min) "
                    f">= +{self.threshold_pct}pp threshold, following momentum up (${self.target_dollars:.0f})"
                ),
            )
        if delta <= -self.threshold_pct:
            ok, reject_reason = self._entry_price_ok("NO", snapshot)
            if not ok:
                return StrategyDecision(Signal.HOLD, reason=reject_reason)
            size = self._size_for_dollars(self.target_dollars, snapshot.no_ask_pct)
            return StrategyDecision(
                Signal.BUY_NO,
                size=size,
                reason=(
                    f"mid {old_mid:.1f}->{mid:.1f} ({delta:+.1f}pp over {self.window_minutes}min) "
                    f"<= -{self.threshold_pct}pp threshold, following momentum down (${self.target_dollars:.0f})"
                ),
            )
        return StrategyDecision(
            Signal.HOLD,
            reason=f"mid {old_mid:.1f}->{mid:.1f} ({delta:+.1f}pp) within +/-{self.threshold_pct}pp threshold",
        )


class CryptoTechnicalStrategy(_TakeProfitStopLossStrategy):
    """Multi-timeframe technical strategy for continuously-traded crypto
    markets (15-min/hourly BTC/ETH threshold contracts): trades only when a
    short and a longer lookback both agree on direction, sized by how strong
    that agreement is, and dampened when recent volatility is unusually high.

    This is real technical analysis on the market's own price history -- it
    does not have, and does not pretend to have, access to any external
    crypto exchange, on-chain, or order-flow data feed. "Multi-timeframe
    momentum agreement" is a standard, honest signal: only trade a short-term
    move when it isn't contradicted by the longer trend, which is a genuine
    (if imperfect) filter against pure noise -- not a prediction of where
    the "real" price goes.
    """

    name = "crypto_technical"

    def __init__(
        self,
        short_window_minutes: float = 3.0,
        long_window_minutes: float = 12.0,
        threshold_pct: float = 2.0,
        base_dollars: float = 120.0,
        max_dollars: float = 280.0,
        volatility_dampen_stdev: float = 8.0,
        stop_loss_pct: float = 0.15,
        trail_activate_pct: float = 0.05,
        trail_giveback_pct: float = 0.03,
        max_profit_pct: float = 0.25,
        urgency_minutes: float = 3.0,
        max_entry_price_pct: float = 85.0,
        min_entry_price_pct: float = 0.0,
        min_minutes_to_close: float = 5.0,
        micro_bet_enabled: bool = False,
        micro_bet_min_dollars: float = 1.0,
        micro_bet_max_dollars: float = 4.0,
        micro_bet_threshold_pct: float = 0.5,
        micro_bet_max_move_pct: float = 6.0,
        stop_loss_time_bonus_pct: float = 0.0,
        stop_loss_time_reference_minutes: float = 60.0,
    ):
        self.short_window_minutes = short_window_minutes
        self.long_window_minutes = long_window_minutes
        self.threshold_pct = threshold_pct
        self.base_dollars = base_dollars
        self.max_dollars = max_dollars
        self.volatility_dampen_stdev = volatility_dampen_stdev
        self.stop_loss_pct = stop_loss_pct
        self.trail_activate_pct = trail_activate_pct
        self.trail_giveback_pct = trail_giveback_pct
        self.max_profit_pct = max_profit_pct
        self.urgency_minutes = urgency_minutes
        self.max_entry_price_pct = max_entry_price_pct
        self.min_entry_price_pct = min_entry_price_pct
        self.min_minutes_to_close = min_minutes_to_close
        self.stop_loss_time_bonus_pct = stop_loss_time_bonus_pct
        self.stop_loss_time_reference_minutes = stop_loss_time_reference_minutes
        # Opt-in secondary entry path, OFF by default -- only meant for a
        # small real-money account that wants visible, frequent activity and
        # has explicitly accepted smaller-edge trades in exchange for it. When
        # the main multi-timeframe agreement signal doesn't fire, falls back
        # to a much looser single-timeframe (short-window only) momentum
        # check, sized tiny ($1-4 by default) instead of the strategy's
        # normal size. Still goes through every other guard unchanged
        # (min_minutes_to_close, max_entry_price_pct, the shared exit logic,
        # the engine's daily kill-switch and PerformanceGovernor) -- this
        # only loosens the ENTRY bar, and only for a capped, small stake.
        self.micro_bet_enabled = micro_bet_enabled
        self.micro_bet_min_dollars = micro_bet_min_dollars
        self.micro_bet_max_dollars = micro_bet_max_dollars
        self.micro_bet_threshold_pct = micro_bet_threshold_pct
        # Diagnosed 2026-08-03 against the first 11 real micro-bet trades
        # (1W/10L, -$8.63): every loss hit the hard stop-loss within 0-1.8min
        # of opening, and the biggest losses (-50%, -26%) came from chasing
        # the two most extreme recent moves (-28.5pp, -48pp/3min). A move
        # that large on a threshold contract has almost certainly already
        # happened by the time this fires -- it reads as a thin-order-book
        # spike, not momentum to follow, and it reverses hard. This caps how
        # large a move the micro-bet will still chase; above it, treat the
        # move as noise rather than signal.
        self.micro_bet_max_move_pct = micro_bet_max_move_pct

    def _evaluate_entry(self, snapshot: MarketSnapshot, history: list[MarketSnapshot]) -> StrategyDecision:
        mid = snapshot.mid_pct
        if mid is None:
            return StrategyDecision(Signal.HOLD, reason="no two-sided market (missing yes bid/ask)")

        short_old = self._mid_at_or_before(history, snapshot.timestamp - timedelta(minutes=self.short_window_minutes))
        if short_old is None:
            return StrategyDecision(
                Signal.HOLD, reason=f"insufficient history for {self.short_window_minutes}min short window yet"
            )
        short_delta = mid - short_old

        # long_old (and therefore the full agreement signal) may not be
        # available yet even once short_old is -- e.g. right after a
        # restart, since in-memory history rebuilds from zero every time.
        # The micro-bet fallback only needs short_delta, so it can fire well
        # before the strict signal's full 12min warmup completes; it's
        # deliberately checked here rather than after the long-window check
        # below so it isn't gated on history it doesn't use.
        long_old = self._mid_at_or_before(history, snapshot.timestamp - timedelta(minutes=self.long_window_minutes))
        long_delta = mid - long_old if long_old is not None else None
        volatility = self._recent_volatility(history)

        agree_up = long_delta is not None and short_delta >= self.threshold_pct and long_delta > 0
        agree_down = long_delta is not None and short_delta <= -self.threshold_pct and long_delta < 0

        if not (agree_up or agree_down):
            if self.micro_bet_enabled:
                micro_decision = self._evaluate_micro_bet(snapshot, short_delta, volatility)
                if micro_decision is not None:
                    return micro_decision
            long_note = (
                f"long {long_delta:+.1f}pp/{self.long_window_minutes}min"
                if long_delta is not None
                else f"long window ({self.long_window_minutes}min) not available yet"
            )
            return StrategyDecision(
                Signal.HOLD,
                reason=f"short {short_delta:+.1f}pp/{self.short_window_minutes}min, {long_note} -- no aligned signal",
            )

        side = "YES" if agree_up else "NO"
        ok, reject_reason = self._entry_price_ok(side, snapshot)
        if not ok:
            return StrategyDecision(Signal.HOLD, reason=reject_reason)

        confidence = min(abs(short_delta) / self.threshold_pct, 3.0)
        dollars = min(self.max_dollars, self.base_dollars * confidence)
        vol_note = ""
        if volatility is not None and volatility > self.volatility_dampen_stdev:
            dollars = max(self.base_dollars / 2, dollars / 2)
            vol_note = f", size dampened for {volatility:.1f}pp recent volatility"

        ask = snapshot.yes_ask_pct if side == "YES" else snapshot.no_ask_pct
        size = self._size_for_dollars(dollars, ask)
        signal = Signal.BUY_YES if agree_up else Signal.BUY_NO
        direction = "up" if agree_up else "down"
        return StrategyDecision(
            signal,
            size=size,
            reason=(
                f"short {short_delta:+.1f}pp/{self.short_window_minutes}min & "
                f"long {long_delta:+.1f}pp/{self.long_window_minutes}min agree {direction}, "
                f"confidence x{confidence:.1f} -> ${dollars:.0f}{vol_note}"
            ),
            features=build_features(
                short_delta=short_delta, long_delta=long_delta, is_micro_bet=False,
                volatility=volatility, orderbook_imbalance=snapshot.orderbook_imbalance,
                minutes_to_close=snapshot.minutes_to_close, entry_price_pct=ask, side=side,
            ),
        )

    @staticmethod
    def _recent_volatility(history: list[MarketSnapshot]) -> float | None:
        mids = [s.mid_pct for s in history[-10:] if s.mid_pct is not None]
        if len(mids) < 3:
            return None
        diffs = [b - a for a, b in zip(mids, mids[1:])]
        return statistics.pstdev(diffs)

    def _evaluate_micro_bet(
        self, snapshot: MarketSnapshot, short_delta: float, volatility: float | None
    ) -> StrategyDecision | None:
        """Fallback entry for when the strict multi-timeframe signal doesn't
        fire: still a real signal (short-term momentum), just a much lower
        bar and a correspondingly tiny stake. Returns None (not a HOLD
        decision) when even this looser bar isn't met, so the caller falls
        through to the normal "no aligned signal" HOLD."""
        if abs(short_delta) < self.micro_bet_threshold_pct:
            return None
        if abs(short_delta) > self.micro_bet_max_move_pct:
            # Too large to be trend to follow -- more likely a thin-book
            # spike that's already happened and is about to revert. Chasing
            # it is exactly what produced this strategy's two worst losses.
            return None

        side = "YES" if short_delta > 0 else "NO"
        ok, reject_reason = self._entry_price_ok(side, snapshot)
        if not ok:
            return None

        confidence = abs(short_delta) / self.micro_bet_threshold_pct
        dollars = min(self.micro_bet_max_dollars, max(self.micro_bet_min_dollars, self.micro_bet_min_dollars * confidence))
        ask = snapshot.yes_ask_pct if side == "YES" else snapshot.no_ask_pct
        size = self._size_for_dollars(dollars, ask)
        signal = Signal.BUY_YES if side == "YES" else Signal.BUY_NO
        return StrategyDecision(
            signal,
            size=size,
            reason=(
                f"micro calculated-risk bet: short-term move {short_delta:+.1f}pp/{self.short_window_minutes}min "
                f"(below the {self.threshold_pct}pp full-agreement bar, above the {self.micro_bet_threshold_pct}pp "
                f"micro bar) -> ${dollars:.2f}"
            ),
            features=build_features(
                short_delta=short_delta, long_delta=None, is_micro_bet=True,
                volatility=volatility, orderbook_imbalance=snapshot.orderbook_imbalance,
                minutes_to_close=snapshot.minutes_to_close, entry_price_pct=ask, side=side,
            ),
        )


class SportsMicrostructureStrategy(_TakeProfitStopLossStrategy):
    """For event-driven sports/game markets: combines price momentum with two
    market-microstructure signals -- a volume surge (recent trading pace vs.
    its own recent baseline) and order-book imbalance (which side has more
    resting size wanting to trade). The market's own crowd already prices in
    whatever it knows about the teams/players/form/injuries; this strategy
    does not have, and does not pretend to have, access to any sports-
    statistics feed, roster data, or injury report. It reads what the market
    is doing, not what the game "should" do.
    """

    name = "sports_microstructure"

    def __init__(
        self,
        window_minutes: float = 5.0,
        threshold_pct: float = 3.0,
        volume_surge_ratio: float = 1.5,
        imbalance_threshold: float = 0.2,
        base_dollars: float = 120.0,
        max_dollars: float = 240.0,
        stop_loss_pct: float = 0.15,
        trail_activate_pct: float = 0.05,
        trail_giveback_pct: float = 0.03,
        max_profit_pct: float = 0.25,
        urgency_minutes: float = 5.0,
        max_entry_price_pct: float = 85.0,
        min_minutes_to_close: float = 5.0,
    ):
        self.window_minutes = window_minutes
        self.threshold_pct = threshold_pct
        self.volume_surge_ratio = volume_surge_ratio
        self.imbalance_threshold = imbalance_threshold
        self.base_dollars = base_dollars
        self.max_dollars = max_dollars
        self.stop_loss_pct = stop_loss_pct
        self.trail_activate_pct = trail_activate_pct
        self.trail_giveback_pct = trail_giveback_pct
        self.max_profit_pct = max_profit_pct
        self.urgency_minutes = urgency_minutes
        self.max_entry_price_pct = max_entry_price_pct
        self.min_minutes_to_close = min_minutes_to_close

    def _evaluate_entry(self, snapshot: MarketSnapshot, history: list[MarketSnapshot]) -> StrategyDecision:
        mid = snapshot.mid_pct
        if mid is None:
            return StrategyDecision(Signal.HOLD, reason="no two-sided market (missing yes bid/ask)")

        old_mid = self._mid_at_or_before(history, snapshot.timestamp - timedelta(minutes=self.window_minutes))
        if old_mid is None:
            return StrategyDecision(
                Signal.HOLD, reason=f"insufficient price history for a {self.window_minutes}min window yet"
            )

        delta = mid - old_mid
        surge = self._volume_surge(snapshot, history)
        imbalance = snapshot.orderbook_imbalance

        signals_up = 0
        signals_down = 0
        notes = []
        if delta >= self.threshold_pct:
            signals_up += 1
            notes.append(f"momentum {delta:+.1f}pp")
        elif delta <= -self.threshold_pct:
            signals_down += 1
            notes.append(f"momentum {delta:+.1f}pp")

        if imbalance is not None and abs(imbalance) >= self.imbalance_threshold:
            if imbalance > 0:
                signals_up += 1
            else:
                signals_down += 1
            notes.append(f"book imbalance {imbalance:+.2f}")

        if surge is not None and surge >= self.volume_surge_ratio:
            notes.append(f"volume surge x{surge:.1f}")
            # amplifies whichever direction the other signals already point;
            # not directional on its own.
            if signals_up > signals_down:
                signals_up += 1
            elif signals_down > signals_up:
                signals_down += 1

        if signals_up == 0 and signals_down == 0:
            return StrategyDecision(Signal.HOLD, reason=f"no signal (mid {old_mid:.1f}->{mid:.1f})")
        if signals_up == signals_down:
            return StrategyDecision(Signal.HOLD, reason=f"conflicting signals: {', '.join(notes)}")

        side = "YES" if signals_up > signals_down else "NO"
        ok, reject_reason = self._entry_price_ok(side, snapshot)
        if not ok:
            return StrategyDecision(Signal.HOLD, reason=reject_reason)

        dollars = min(self.max_dollars, self.base_dollars * max(signals_up, signals_down))
        ask = snapshot.yes_ask_pct if side == "YES" else snapshot.no_ask_pct
        size = self._size_for_dollars(dollars, ask)
        signal = Signal.BUY_YES if side == "YES" else Signal.BUY_NO
        return StrategyDecision(signal, size=size, reason=" + ".join(notes) + f" -> ${dollars:.0f}")

    @staticmethod
    def _volume_surge(snapshot: MarketSnapshot, history: list[MarketSnapshot]) -> float | None:
        """volume is cumulative, so a surge is about the *rate of change*
        (deltas between consecutive snapshots), not the raw total -- the raw
        total only ever goes up, which would make every reading look like a
        surge."""
        vols = [s.volume for s in history[-11:] if s.volume is not None]
        if len(vols) < 4 or snapshot.volume is None:
            return None
        deltas = [max(0.0, b - a) for a, b in zip(vols, vols[1:])]
        latest_delta = max(0.0, snapshot.volume - vols[-1])
        baseline_deltas = deltas[:-1] if len(deltas) > 1 else deltas
        if not baseline_deltas:
            return None
        baseline = statistics.mean(baseline_deltas)
        if baseline <= 0:
            return None
        return latest_delta / baseline
