"""A from-scratch entry signal for the live crypto bot, replacing the
hand-tuned momentum-agreement approach in CryptoTechnicalStrategy for LIVE
trading specifically (CryptoTechnicalStrategy itself is untouched, but as
of 2026-08-06 no longer runs anywhere in this codebase by default -- the
paper bot was repurposed per explicit user request to run THIS strategy
too, just tuned more risk-tolerant; see run_paper_trading.py's docstring).

Written 2026-08-04 after validating against all 329 real closed live
trades (crypto_technical strategy, 2026-08-03/04): 122 wins / 207 losses,
37.1% win rate, -$136.14 total. That analysis is the entire justification
for every choice below -- see docs/ALGORITHM.md for the full writeup with
numbers. Short honest summary of what the data actually showed:

- The core premise the old strategy was built on -- "a price that just
  moved keeps moving, and agreement across two timeframes confirms it" --
  is NOT supported. A logistic regression fit on all 314 trades with
  entry-time features finds short_delta's coefficient is slightly NEGATIVE
  and long_delta's is ~zero. Requiring both to agree filters on a feature
  set that barely predicts anything.
- orderbook_imbalance has the single strongest coefficient of any feature
  (+0.32, more than 5x short_delta's magnitude) and was never used to drive
  a crypto entry decision before -- it was computed every cycle and fed to
  a slow-learning online model, never used directly. Trades whose direction
  agreed with a real book imbalance won 39.6% of the time; trades against
  one won 22.4%.
- volatility_scaled is meaningfully negative (-0.34) and close to
  monotonic in bucketed win rate (42.9% at 0-2pp down to 24.6% at 6-8pp) --
  a real, usable gate, previously only a size dampener.
- entry_price_scaled is positive (+0.22), consistent with and stronger
  than the min_entry_price_pct floor already found live 2026-08-03.
- asset_btc is positive (+0.18), asset_eth is negative (-0.09) -- BTC
  trades meaningfully better than ETH or BNB (the implicit baseline).
- Of the 329 trades, stop-loss exits (79 trades, 1.3% win rate, -$82.97)
  and reconciled/forced-settlement exits (41 trades, 9.8% win rate,
  -$89.72) together account for MORE than the entire net loss -- every
  other exit reason nets positive. Stop-losses fired at a -28.0% average
  realized return (current stop_loss_pct=0.15, widened up to 0.25 with
  time remaining) -- losers were allowed to run much further than the
  ~+15-20% winners were allowed to run before locking in. Reconciled exits
  were concentrated on 15-min markets (14/15) and correlate with known bot
  downtime windows now closed by the watchdog (bot_control.py) -- kept as
  a live risk anyway via a wider urgency buffer on 15-min tickers
  specifically, see urgency_minutes_15min_bonus below.
- The "gamble" hold-to-expiry mechanic (added earlier the same day) went
  0-for-41, -$31.57 -- an unambiguous, no-caveats failure. Not carried
  forward into this strategy at all (not even as a disabled option).
- "scalp" (rapid 30-60s move, fast fixed exit) had the best win rate of
  any real signal (55.0%, 22W/18L) but was still slightly net negative
  (-$3.31) purely on fee drag against its small stake -- kept, with a
  larger stake and take-profit target to clear that drag.

Honest caveats, worth repeating from online_learner.py's own docstring
philosophy: 314 labeled examples is not a lot for 12 features. The fitted
model was checked against a chronological 20% holdout (not just the
training data) and came back reasonably calibrated (predicted ~0.3 bucket
-> actual 28.8%, predicted ~0.5 bucket -> actual 47.8%), which is real
evidence this isn't pure overfit noise, but prediction markets are close
to efficiently priced by design -- nothing here should be expected to turn
a coin-flip signal into a strongly profitable one. The realistic goal is
fewer, better-chosen entries and much smaller losses on the ones that
don't work out, not a dramatically higher win rate.

The weights below are a BATCH fit (full-dataset logistic regression via
gradient descent, L2=0.02), not updated by single-trade online SGD like
OnlineLogisticLearner -- deliberately: watching that mechanism operate
live showed a handful of one-at-a-time SGD steps on ~50 examples can push
a whole model to a near-uniform pessimistic output (see run_live_trading.py's
ML_GATE_THRESHOLD note, 2026-08-04) well before it has enough data to
actually discriminate. A periodic batch refit (see refit_strategy_model.py
at the repo root) on the full accumulated history is a more statistically
stable way to keep this "fresh" than continuous per-trade nudges, and
keeps a human in the loop for reviewing what changed and why before it
goes live again -- the same non-negotiable the online learner already
holds itself to ("never invents a trade idea, never overrides an exit").
"""
from __future__ import annotations

import math
import statistics
from dataclasses import dataclass
from datetime import datetime, timedelta

from .btc_price_history import BtcPriceHistory
from .online_learner import build_features
from .portfolio import Position
from .strategy import MarketSnapshot, Signal, StrategyDecision, _TakeProfitStopLossStrategy
from .watchlist import ACTIVE_CRYPTO_ROTATING_SINGLE

# Refit 2026-08-12 against 999 real closed trades (up from the original
# 314 on 2026-08-04) via refit_strategy_model.py, at the user's explicit
# request to have this reflect everything the ledger has seen so far, not
# a permanent snapshot. Chronological 80/20 holdout (200 trades, none seen
# during fitting) showed the CURRENT live gate (0.43) barely beats doing
# nothing -- 45.3% win rate vs. a 45.0% base rate on the same holdout -- but
# a 0.50 cutoff shows a real lift, 52.9% win rate on 70/200 taken. Worth
# reassessing min_win_probability itself, not just these coefficients (see
# run_live_trading.py's STRATEGY_PARAMS) -- left unchanged here since that's
# a separate, more speculative call (n=70 at that cutoff) than shipping
# weights the script's own holdout says still generalize.
#
# What's consistent with the 2026-08-04 fit (same sign, real signal):
# asset_btc positive / asset_eth negative (BTC still the best asset of the
# three, now confirmed independently on 3x the data), entry_price_scaled
# positive and slightly stronger.
#
# What changed, worth being honest about rather than quietly pasting over:
# orderbook_imbalance -- the strongest coefficient in the original fit
# (+0.32, the whole justification for this strategy using book imbalance as
# its primary direction signal) -- dropped to +0.058, a ~6x reduction.
# Direction selection itself is unaffected (_evaluate_entry picks the side
# from the raw imbalance sign, not this weight), but the probability model's
# own confidence in that signal is now much weaker. volatility_scaled also
# roughly halved (-0.34 -> -0.15). Several already-small features flipped
# sign entirely (hour_cos, minutes_to_close_scaled, long_delta_scaled) --
# read that as "these were never strong signals, the original 314-example
# fit was fitting some noise on them," not as a real reversal.
# btc_realized_vol_scaled reads ~0 -- expected, almost no historical trade
# has it populated yet (added 2026-08-11); revisit on the next refit once
# more trades carry it.
PROBABILITY_MODEL_BIAS = -0.3011001376144315
PROBABILITY_MODEL_WEIGHTS: dict[str, float] = {
    "short_delta_scaled": -0.041174030856234806,
    "long_delta_scaled": 0.023083739095339787,
    "is_micro_bet": -0.03955025996656771,
    "volatility_scaled": -0.1468483494037117,
    "orderbook_imbalance": 0.05813540295682912,
    "minutes_to_close_scaled": -0.07516518578192966,
    "entry_price_scaled": 0.2768824631893457,
    "side_is_yes": -0.09506025550060261,
    "hour_sin": -0.020069603545994463,
    "hour_cos": -0.1184958678400709,
    "asset_btc": 0.19241340616192065,
    "asset_eth": -0.06702906552128068,
    "btc_realized_vol_scaled": 0.014811281060798211,
}

_SHORT_DURATION_PREFIXES = tuple(f"{s}-" for s in ACTIVE_CRYPTO_ROTATING_SINGLE)


def _sigmoid(z: float) -> float:
    if z < -30:
        return 0.0
    if z > 30:
        return 1.0
    return 1.0 / (1.0 + math.exp(-z))


def predict_win_probability(features: dict[str, float]) -> float:
    z = PROBABILITY_MODEL_BIAS + sum(PROBABILITY_MODEL_WEIGHTS.get(k, 0.0) * v for k, v in features.items())
    return _sigmoid(z)


@dataclass
class DataDrivenCryptoStrategy(_TakeProfitStopLossStrategy):
    """Entry: a fitted-on-real-outcomes probability model, gated to only
    fire when it clears the price-implied breakeven probability by a real
    margin (not a fixed absolute threshold -- an 80c favorite and a 50c
    coin flip need very different predicted probabilities to be worth
    taking). Direction comes from order-book imbalance (the strongest real
    signal found), not raw price momentum (the weakest). Exit: the same
    trailing-stop/time-decay-urgency structure already validated by real
    data in earlier diagnoses, with stop-loss tightened substantially
    (10% base instead of up to 25%) -- the single biggest lever the new
    analysis found, since stop-losses were firing at -28% average, far
    wider than winners were ever allowed to run.
    """

    name = "data_driven_crypto"

    short_window_minutes: float = 3.0
    long_window_minutes: float = 12.0

    # Order-book imbalance is the primary direction signal now (see module
    # docstring) -- must be at least this strong to trade at all, and the
    # short-term price move (still computed, still a model feature) must
    # not contradict it beyond a small tolerance.
    min_imbalance_pct: float = 0.12
    momentum_conflict_tolerance_pct: float = 1.5

    # Hard gate, not just a size dampener -- calibrated against real bucketed
    # win rate, which drops from ~43% under 2pp to ~25-31% above 6pp.
    max_volatility_pct: float = 6.5

    # Absolute predicted-win-probability gate, calibrated against a
    # chronological 20% holdout of the 314 labeled trades (not just fit on
    # the training data). IMPORTANT REVERSAL from the first version of this
    # file, worth being honest about: that version gated on the model
    # beating the PRICE-IMPLIED breakeven probability (the theoretically
    # "correct" definition of edge). Backtesting that against the holdout
    # found the model's predicted probability exceeded price-implied
    # breakeven on ZERO of 63 held-out trades -- Kalshi's own pricing
    # already encodes more information than this feature set beats in
    # absolute terms, so that gate would have meant the bot almost never
    # trades again, the exact failure mode already hit and fixed once
    # today (see run_live_trading.py's ML_GATE_THRESHOLD note). An absolute
    # probability threshold, used as a relative ranking tool rather than a
    # claim of beating the market outright, is what the data actually
    # supports: at p>=0.42 the holdout took 23/63 trades at 47.8% win rate
    # (vs. 41.3% base rate); at p>=0.44, 21/63 at 47.6%. 0.43 splits the
    # difference. Fee cost is real but small enough at these prices that it
    # doesn't change which side of this threshold a trade falls on for the
    # cases that matter -- not separately modeled here.
    min_win_probability: float = 0.43

    min_entry_price_pct: float = 45.0
    max_entry_price_pct: float = 85.0
    min_minutes_to_close: float = 6.0

    # REVISED 2026-08-05, widened back up -- the initial 0.10/0.05 tightening
    # (2026-08-04, see git history) was based on the OLD strategy's -28%
    # average realized stop-loss return, but running the tightened version
    # for real for ~23 hours (201 trades) showed it overcorrected: of 87
    # stop-loss exits (standard + scalp combined), 55 (63%) would have
    # WON if held to actual settlement instead of being cut -- verified by
    # cross-referencing each exit's ticker against later decision-log price
    # observations for that same market. In aggregate, actual realized P&L
    # on those 87 trades was -$24.28; holding every one to settlement would
    # have been +$12.74, a $37 swing on this data alone. These 15-min-
    # adjacent threshold contracts see real, large, non-trend swings in
    # their final minutes as the underlying price hovers near the strike --
    # a percentage-of-contract-price stop reacts to that noise, not
    # necessarily a genuine reversal. Widened back toward (and slightly
    # past) the original pre-2026-08-04 level pending more data; still
    # meaningfully data-driven, just informed by a full day of the new
    # strategy's own real results instead of the old strategy's.
    stop_loss_pct: float = 0.20
    stop_loss_time_bonus_pct: float = 0.10
    stop_loss_time_reference_minutes: float = 60.0
    # Added 2026-08-07: full-day review of 2026-08-06's 27 real standard-
    # stop-loss trades found overshoot (actual fill % beyond the nominal
    # threshold) was wildly different by series -- median 11.2pp on the
    # three 15-min rotating series vs. 1.8pp on the hourly ones (max 41.1pp
    # vs 14.3pp). Same root cause as urgency_minutes_15min_bonus above:
    # these markets swing harder and faster as they approach their own
    # close, and detection lag (even at the current 10s poll interval)
    # costs more there than anywhere else in the scope. Flat additional
    # room on top of the existing time-based widening, 15-min series only.
    stop_loss_pct_15min_bonus: float = 0.15
    trail_activate_pct: float = 0.05
    trail_giveback_pct: float = 0.03
    max_profit_pct: float = 0.25
    urgency_minutes: float = 3.0
    # Extra buffer specifically on the three 15-min rotating series, on top
    # of urgency_minutes -- defense in depth against the reconciled/forced-
    # settlement loss pattern (concentrated 14-of-15 on these tickers),
    # alongside (not instead of) the watchdog fix for the downtime that
    # caused most of the worst cases.
    urgency_minutes_15min_bonus: float = 1.5

    base_dollars: float = 25.0
    max_dollars: float = 30.0
    # How far above min_win_probability the prediction needs to be for
    # sizing to reach max_dollars -- the fitted model's scores cluster in a
    # narrow ~0.25-0.50 band (see module docstring's calibration numbers),
    # so 0.10 above the gate is already close to the strongest predictions
    # this model ever actually produces.
    full_confidence_margin: float = 0.10

    # DISABLED 2026-08-06 after three rounds of tuning (stop-loss 0.08 ->
    # 0.20 -> 0.28, take-profit 0.08 -> 0.12) failed to make this mechanism
    # profitable. The interval_outcomes table (trade_bot/interval_tracker.py)
    # gave the decisive evidence: on 70 real traded intervals, entry
    # DIRECTION was correct 81% of the time but realized P&L was positive
    # only 40% of the time -- scalp accounted for 20 of the 30 "right call,
    # still lost money" trades (67%, -$12.26). Widening the stop threshold
    # didn't help: fresh trades under the 0.28 stop still overshot it by
    # 9-15.5 percentage points, the same magnitude as under the old 0.20
    # stop -- proof the threshold was never the actual lever, execution
    # catching a fast move late is. Checked whether raising the entry
    # trigger (scalp_threshold_pct) would filter out the bad cases instead:
    # it doesn't -- EVERY trigger-size bucket lost money (buckets by
    # trigger pp: 1.5-2pp n=9 avg -$0.20/trade, 2-3pp n=17 avg -$0.17,
    # 3-5pp n=24 avg -$0.22, 5-8pp+ n=66 avg -$0.26), just with the 8pp+
    # bucket dominating total loss by sheer volume (55% of all scalp
    # trades). No configuration found across any of this made scalp net
    # profitable. Left fully intact (not deleted) in case a faster
    # execution model someday changes the picture -- see
    # scalp_window_seconds/scalp_threshold_pct/scalp_dollars/
    # scalp_take_profit_pct/scalp_stop_loss_pct/scalp_max_hold_minutes
    # below, all now unused while this is False.
    scalp_enabled: bool = False
    scalp_window_seconds: float = 45.0
    scalp_threshold_pct: float = 1.5
    # Stake/take-profit raised from the first version's $3.00/6% -- real
    # historical data (batch analysis) showed 55% win rate but net-negative
    # purely on fee drag against a small stake.
    scalp_dollars: float = 5.0
    # REVISED 2026-08-06: was 0.08. With scalp_stop_loss_pct widened to 0.20
    # below (2026-08-05), an 8% target against a 20% stop needs a 71% win
    # rate just to break even -- real results since then were ~36-42%, a
    # combination that loses money on the math alone regardless of signal
    # quality. Widening the target (not the stop -- that's being left alone
    # this round pending the polling-interval/re-entry-cooldown fixes,
    # which should matter more for a signal this short-lived) is also
    # supported directly: of the 30 real scalp stop-loss trades since
    # 2026-08-05, 53% would have WON if held, meaning genuine follow-through
    # past 8% is common, not rare.
    scalp_take_profit_pct: float = 0.12
    # REVISED 2026-08-05: the original 0.08 stop was firing on 35 of the
    # first 47 live scalp trades (74%) at only 5.7% win rate, and 24 of
    # those 35 (68.6%) would have WON if held to settlement instead --
    # same pattern as, and worse than, the standard stop_loss_pct finding
    # above, same fix. scalp_max_hold_minutes below already forces an exit
    # regardless of price, so widening this doesn't undermine the "fast
    # in/out" premise -- it just stops the stop from being the thing that
    # decides most scalp trades' outcome instead of the actual signal.
    #
    # REVISED 2026-08-06: was 0.20. The interval_outcomes table (see
    # trade_bot/interval_tracker.py) gives an independent read on this,
    # cross-referencing every trade's ACTUAL entry side against the
    # market's real settlement, not just a log-scraped price proxy: across
    # 52 traded intervals since 2026-08-05, entry direction matched the
    # final settlement 81% of the time, but only 35% were actually
    # profitable. Of the 25 trades that were directionally CORRECT but
    # still lost money, 12 (the single largest bucket, -$9.87 of -$18.36)
    # were scalp_stop_loss exits -- the stop cutting a trade that was
    # already headed the right way. Same conclusion as the 2026-08-05
    # finding, independently reconfirmed from different data.
    scalp_stop_loss_pct: float = 0.28
    scalp_max_hold_minutes: float = 2.5

    momentum_hold_enabled: bool = True
    momentum_hold_confirm_minutes: float = 1.0
    momentum_hold_confirm_pct: float = 0.5
    momentum_hold_giveback_pct: float = 0.08
    momentum_hold_max_profit_pct: float = 0.50

    # Added 2026-08-07 per explicit user request, based on a pattern they
    # observed watching KXBTC15M markets by hand: in the first few minutes
    # of a fresh 15-min contract, one side sometimes gets beaten down to a
    # cheap 20-30c band, then bounces back to 40c+ (sometimes 60c+) shortly
    # after. Off by default -- LiveExecutionEngine refreshes this flag from
    # LiveLedger.get_btc_dip_enabled() every cycle, so it's a live
    # dashboard/app toggle, not a redeploy. Deliberately KXBTC15M- only
    # (not ETH/BNB, not hourly) and deliberately buys BELOW
    # min_entry_price_pct -- this is a different bet than the strategy's
    # normal entry logic, not a variant of it, so it's fully self-contained
    # (own entry gate, own exit, own sizing) rather than layered on top of
    # the existing imbalance/probability gates the way scalp was.
    btc_dip_enabled: bool = False
    # Only look to enter in the first N minutes of a fresh 15-min market's
    # life (open_time = close_time - 15min for this series, confirmed
    # against the real Kalshi market data).
    btc_dip_window_minutes: float = 4.0
    btc_dip_entry_min_pct: float = 20.0
    btc_dip_entry_max_pct: float = 30.0
    # "$2-3 for now" per the user's explicit request -- practice/test sizing.
    btc_dip_dollars: float = 2.5
    # Absolute PRICE LEVEL (not %-return-on-entry, unlike the rest of this
    # strategy) the user described the bounce reaching -- entry price
    # varies within the 20-30c band, so a fixed price TARGET is what
    # actually matches "reaches above 40, sometimes up to 60", not a fixed
    # % return that would mean something different at 20c vs 30c entry.
    btc_dip_take_profit_activate_pct: float = 40.0
    # Once activated, lock in if price gives back this many PERCENTAGE
    # POINTS (absolute, not % of return) from its peak.
    btc_dip_trail_giveback_pct: float = 5.0
    # Hard stop if the bounce never happens and price keeps falling instead
    # -- also in absolute points below entry, not %-of-entry-return.
    btc_dip_stop_loss_pct: float = 10.0
    # Safety timeout independent of the 4min ENTRY window above (this one
    # bounds how long a filled position can be held) -- the user described
    # a "short window" for the bounce; if it hasn't happened by here,
    # something other than the expected pattern is going on.
    btc_dip_max_hold_minutes: float = 8.0

    # Added 2026-08-11: optional handle onto the continuous BTC/USD tick feed
    # (trade_bot/btc_price_history.py, 1 tick/sec from Coinbase, independent
    # of Kalshi) so the entry model can see real realized volatility of the
    # underlying asset, not just noise in the Kalshi contract's own mid-price
    # (see _recent_volatility below). None by default -- any caller that
    # doesn't wire one in (paper bot's ExecutionEngine has no equivalent,
    # tests) gets btc_realized_vol_scaled=0.0/no-signal, same contract as
    # every other optional feature here.
    btc_price_history: BtcPriceHistory | None = None

    @staticmethod
    def _is_short_duration(ticker: str) -> bool:
        return ticker.startswith(_SHORT_DURATION_PREFIXES)

    @staticmethod
    def _recent_volatility(history: list[MarketSnapshot]) -> float | None:
        mids = [s.mid_pct for s in history[-10:] if s.mid_pct is not None]
        if len(mids) < 3:
            return None
        diffs = [b - a for a, b in zip(mids, mids[1:])]
        return statistics.pstdev(diffs)

    def _btc_realized_volatility_pct(self) -> float | None:
        if self.btc_price_history is None:
            return None
        try:
            return self.btc_price_history.get_recent_volatility_pct(window_minutes=5.0)
        except Exception:
            # Best-effort, same as fetch_btc_price_usd -- a DB hiccup here
            # must never block a real trading decision.
            return None

    def _evaluate_entry(self, snapshot: MarketSnapshot, history: list[MarketSnapshot]) -> StrategyDecision:
        mid = snapshot.mid_pct
        if mid is None:
            return StrategyDecision(Signal.HOLD, reason="no two-sided market (missing yes bid/ask)")

        imbalance = snapshot.orderbook_imbalance
        volatility = self._recent_volatility(history)
        short_duration = self._is_short_duration(snapshot.ticker)

        if short_duration and self.scalp_enabled:
            scalp_decision = self._evaluate_scalp_entry(snapshot, history, volatility)
            if scalp_decision is not None:
                return scalp_decision

        if self.btc_dip_enabled:
            btc_dip_decision = self._evaluate_btc_dip_entry(snapshot)
            if btc_dip_decision is not None:
                return btc_dip_decision

        short_old = self._mid_at_or_before(history, snapshot.timestamp - timedelta(minutes=self.short_window_minutes))
        short_delta = (mid - short_old) if short_old is not None else 0.0
        long_old = self._mid_at_or_before(history, snapshot.timestamp - timedelta(minutes=self.long_window_minutes))
        long_delta = (mid - long_old) if long_old is not None else None

        if imbalance is None:
            return StrategyDecision(Signal.HOLD, reason="no order-book imbalance available yet")
        if abs(imbalance) < self.min_imbalance_pct:
            return StrategyDecision(
                Signal.HOLD,
                reason=f"book imbalance {imbalance:+.2f} below {self.min_imbalance_pct} minimum -- no real signal",
            )

        side = "YES" if imbalance > 0 else "NO"
        # Sanity co-check: don't buy YES into a market whose short-term
        # price is actively moving hard the other way, even if the book
        # currently favors YES (a moment-old signal reversing).
        if side == "YES" and short_delta <= -self.momentum_conflict_tolerance_pct:
            return StrategyDecision(
                Signal.HOLD,
                reason=(
                    f"book imbalance {imbalance:+.2f} favors YES but price just moved "
                    f"{short_delta:+.1f}pp -- conflicting signals, skipping"
                ),
            )
        if side == "NO" and short_delta >= self.momentum_conflict_tolerance_pct:
            return StrategyDecision(
                Signal.HOLD,
                reason=(
                    f"book imbalance {imbalance:+.2f} favors NO but price just moved "
                    f"{short_delta:+.1f}pp -- conflicting signals, skipping"
                ),
            )

        if volatility is not None and volatility > self.max_volatility_pct:
            return StrategyDecision(
                Signal.HOLD,
                reason=f"recent volatility {volatility:.1f}pp exceeds {self.max_volatility_pct}pp gate -- too noisy",
            )

        ok, reject_reason = self._entry_price_ok(side, snapshot)
        if not ok:
            return StrategyDecision(Signal.HOLD, reason=reject_reason)

        ask = snapshot.yes_ask_pct if side == "YES" else snapshot.no_ask_pct
        features = build_features(
            short_delta=short_delta, long_delta=long_delta, is_micro_bet=False,
            volatility=volatility, orderbook_imbalance=imbalance,
            minutes_to_close=snapshot.minutes_to_close, entry_price_pct=ask, side=side,
            hour_utc=snapshot.timestamp.hour, ticker=snapshot.ticker,
            btc_realized_vol=self._btc_realized_volatility_pct(),
        )
        p_win = predict_win_probability(features)

        if p_win < self.min_win_probability:
            return StrategyDecision(
                Signal.HOLD,
                reason=(
                    f"model predicts {p_win:.0%} win probability, below the {self.min_win_probability:.0%} "
                    "gate"
                ),
                predicted_probability=p_win,
            )

        margin = p_win - self.min_win_probability
        confidence = min(1.0, margin / self.full_confidence_margin)
        dollars = self.base_dollars + (self.max_dollars - self.base_dollars) * confidence
        size = self._size_for_dollars(dollars, ask)
        signal = Signal.BUY_YES if side == "YES" else Signal.BUY_NO
        return StrategyDecision(
            signal,
            size=size,
            reason=(
                f"book imbalance {imbalance:+.2f} -> {side}, model predicts {p_win:.0%} win probability "
                f"(gate {self.min_win_probability:.0%}) -> ${dollars:.0f}"
            ),
            features=features,
            predicted_probability=p_win,
        )

    def _evaluate_scalp_entry(
        self, snapshot: MarketSnapshot, history: list[MarketSnapshot], volatility: float | None
    ) -> StrategyDecision | None:
        """Same rapid short-fuse signal that performed best in the real
        data (55% win rate) -- kept structurally as-is, see module
        docstring for why the stake/take-profit changed instead of the
        signal itself."""
        if snapshot.minutes_to_close is not None and snapshot.minutes_to_close < self.scalp_max_hold_minutes + 1.0:
            return None
        if snapshot.mid_pct is None:
            return None
        cutoff = snapshot.timestamp - timedelta(seconds=self.scalp_window_seconds)
        old_mid = self._mid_at_or_before(history, cutoff)
        if old_mid is None:
            return None
        delta = snapshot.mid_pct - old_mid
        if abs(delta) < self.scalp_threshold_pct:
            return None
        if volatility is not None and volatility > self.max_volatility_pct:
            return None

        side = "YES" if delta > 0 else "NO"
        imbalance = snapshot.orderbook_imbalance
        if imbalance is not None:
            if side == "YES" and imbalance <= -self.min_imbalance_pct:
                return None
            if side == "NO" and imbalance >= self.min_imbalance_pct:
                return None

        ok, _ = self._entry_price_ok(side, snapshot)
        if not ok:
            return None

        ask = snapshot.yes_ask_pct if side == "YES" else snapshot.no_ask_pct
        size = self._size_for_dollars(self.scalp_dollars, ask)
        signal = Signal.BUY_YES if side == "YES" else Signal.BUY_NO
        return StrategyDecision(
            signal,
            size=size,
            reason=(
                f"scalp: rapid {delta:+.1f}pp move in {self.scalp_window_seconds:.0f}s "
                f">= {self.scalp_threshold_pct}pp bar -- fast in/out, ${self.scalp_dollars:.2f} stake"
            ),
            entry_kind="scalp",
            features=build_features(
                short_delta=delta, long_delta=None, is_micro_bet=True,
                volatility=volatility, orderbook_imbalance=imbalance,
                minutes_to_close=snapshot.minutes_to_close, entry_price_pct=ask, side=side,
                hour_utc=snapshot.timestamp.hour, ticker=snapshot.ticker,
                btc_realized_vol=self._btc_realized_volatility_pct(),
            ),
        )

    def _evaluate_btc_dip_entry(self, snapshot: MarketSnapshot) -> StrategyDecision | None:
        """Toggle-gated (see btc_dip_enabled's field comment): buy whichever
        side has been beaten down into the btc_dip_entry_min_pct..max_pct
        band within the first btc_dip_window_minutes of a fresh KXBTC15M
        contract's life. Deliberately bypasses _entry_price_ok -- that gate
        exists to keep the NORMAL strategy from buying underdogs, which is
        exactly what this mechanism is for."""
        if not snapshot.ticker.startswith("KXBTC15M-"):
            return None
        minutes_to_close = snapshot.minutes_to_close
        if minutes_to_close is None:
            return None
        # KXBTC15M's open_time is always close_time - 15min (confirmed
        # against the real Kalshi market schema) -- no separate "market
        # start" field exists to read directly.
        minutes_since_open = 15.0 - minutes_to_close
        if not (0.0 <= minutes_since_open <= self.btc_dip_window_minutes):
            return None

        for side, price in (("YES", snapshot.yes_ask_pct), ("NO", snapshot.no_ask_pct)):
            if price is None:
                continue
            if self.btc_dip_entry_min_pct <= price <= self.btc_dip_entry_max_pct:
                size = self._size_for_dollars(self.btc_dip_dollars, price)
                signal = Signal.BUY_YES if side == "YES" else Signal.BUY_NO
                return StrategyDecision(
                    signal,
                    size=size,
                    reason=(
                        f"btc_dip: {side} ask {price:.0f}c in "
                        f"[{self.btc_dip_entry_min_pct:.0f},{self.btc_dip_entry_max_pct:.0f}] "
                        f"{minutes_since_open:.1f}min into a fresh 15min market -- betting on a bounce, "
                        f"${self.btc_dip_dollars:.2f} stake"
                    ),
                    entry_kind="btc_dip",
                )
        return None

    def _evaluate_scalp_exit(self, snapshot: MarketSnapshot, position: Position) -> StrategyDecision:
        mark = snapshot.yes_bid_pct if position.side == "YES" else snapshot.no_bid_pct
        if mark is None:
            return StrategyDecision(Signal.HOLD, reason=f"no {position.side} bid available to mark scalp position")
        pct_return = (mark - position.entry_price_pct) / position.entry_price_pct

        if pct_return <= -self.scalp_stop_loss_pct:
            return StrategyDecision(
                Signal.SELL, size=position.quantity,
                reason=f"scalp {position.side} at {pct_return:+.1%} hit tight stop-loss (-{self.scalp_stop_loss_pct:.0%})",
            )
        if pct_return >= self.scalp_take_profit_pct:
            return StrategyDecision(
                Signal.SELL, size=position.quantity,
                reason=f"scalp {position.side} at {pct_return:+.1%} hit quick take-profit ({self.scalp_take_profit_pct:.0%})",
            )

        held_minutes = 0.0
        try:
            opened_at = datetime.fromisoformat(position.opened_at)
            held_minutes = (snapshot.timestamp - opened_at).total_seconds() / 60
        except ValueError:
            pass
        if held_minutes >= self.scalp_max_hold_minutes:
            return StrategyDecision(
                Signal.SELL, size=position.quantity,
                reason=(
                    f"scalp {position.side} held {held_minutes:.1f}min >= max {self.scalp_max_hold_minutes}min "
                    f"-- forcing quick exit regardless of {pct_return:+.1%} return"
                ),
            )
        return StrategyDecision(
            Signal.HOLD, reason=f"scalp {position.side} at {pct_return:+.1%}, held {held_minutes:.1f}min",
        )

    def _evaluate_btc_dip_exit(
        self, snapshot: MarketSnapshot, position: Position, history: list[MarketSnapshot]
    ) -> StrategyDecision:
        """Fully self-contained, like _evaluate_scalp_exit -- own stop,
        own take-profit/trail, own timeout, never falls through to the
        normal %-return-based exit machinery. Works in absolute PRICE
        POINTS, not % return, per btc_dip_take_profit_activate_pct's field
        comment."""
        mark = snapshot.yes_bid_pct if position.side == "YES" else snapshot.no_bid_pct
        if mark is None:
            return StrategyDecision(Signal.HOLD, reason=f"no {position.side} bid available to mark btc_dip position")

        if mark <= position.entry_price_pct - self.btc_dip_stop_loss_pct:
            return StrategyDecision(
                Signal.SELL, size=position.quantity,
                reason=(
                    f"btc_dip {position.side} at {mark:.0f}c, {position.entry_price_pct - mark:.0f}pp below "
                    f"entry {position.entry_price_pct:.0f}c -- no bounce, cutting losses "
                    f"(-{self.btc_dip_stop_loss_pct:.0f}pp stop)"
                ),
            )

        try:
            opened_at = datetime.fromisoformat(position.opened_at)
        except ValueError:
            opened_at = None
        marks = [mark]
        for s in history:
            if opened_at is not None and s.timestamp < opened_at:
                continue
            m = s.yes_bid_pct if position.side == "YES" else s.no_bid_pct
            if m is not None:
                marks.append(m)
        peak_mark = max(marks)

        if peak_mark >= self.btc_dip_take_profit_activate_pct:
            giveback = peak_mark - mark
            if giveback >= self.btc_dip_trail_giveback_pct:
                return StrategyDecision(
                    Signal.SELL, size=position.quantity,
                    reason=(
                        f"btc_dip {position.side} peaked at {peak_mark:.0f}c, now {mark:.0f}c "
                        f"(gave back {giveback:.0f}pp >= {self.btc_dip_trail_giveback_pct:.0f}pp tolerance) "
                        "-- locking in the bounce"
                    ),
                )

        held_minutes = (snapshot.timestamp - opened_at).total_seconds() / 60 if opened_at is not None else 0.0
        if held_minutes >= self.btc_dip_max_hold_minutes:
            return StrategyDecision(
                Signal.SELL, size=position.quantity,
                reason=(
                    f"btc_dip {position.side} held {held_minutes:.1f}min >= max "
                    f"{self.btc_dip_max_hold_minutes:.0f}min -- the expected bounce didn't happen, exiting"
                ),
            )

        return StrategyDecision(
            Signal.HOLD,
            reason=(
                f"btc_dip {position.side} at {mark:.0f}c (peak {peak_mark:.0f}c), "
                f"entry {position.entry_price_pct:.0f}c, held {held_minutes:.1f}min"
            ),
        )

    def _evaluate_special_exit(
        self, snapshot: MarketSnapshot, position: Position, history: list[MarketSnapshot]
    ) -> StrategyDecision | None:
        if position.entry_kind == "scalp":
            return self._evaluate_scalp_exit(snapshot, position)
        if position.entry_kind == "btc_dip":
            return self._evaluate_btc_dip_exit(snapshot, position, history)
        return None

    def _still_trending_favorably(
        self, snapshot: MarketSnapshot, history: list[MarketSnapshot], position: Position
    ) -> bool:
        cutoff = snapshot.timestamp - timedelta(minutes=self.momentum_hold_confirm_minutes)
        old_mid = self._mid_at_or_before(history, cutoff)
        if old_mid is None or snapshot.mid_pct is None:
            return False
        delta = snapshot.mid_pct - old_mid
        if position.side == "YES":
            return delta >= self.momentum_hold_confirm_pct
        return delta <= -self.momentum_hold_confirm_pct

    def _momentum_hold_active(
        self, snapshot: MarketSnapshot, position: Position, history: list[MarketSnapshot]
    ) -> bool:
        return (
            self.momentum_hold_enabled
            and position.entry_kind != "scalp"
            and self._is_short_duration(snapshot.ticker)
            and self._still_trending_favorably(snapshot, history, position)
        )

    def _trail_giveback_tolerance(
        self, snapshot: MarketSnapshot, position: Position, history: list[MarketSnapshot], peak_return: float
    ) -> float:
        if self._momentum_hold_active(snapshot, position, history):
            return self.momentum_hold_giveback_pct
        return self.trail_giveback_pct

    def _profit_ceiling(
        self, snapshot: MarketSnapshot, position: Position, history: list[MarketSnapshot], peak_return: float
    ) -> float:
        if self._momentum_hold_active(snapshot, position, history):
            return self.momentum_hold_max_profit_pct
        return self.max_profit_pct

    def _urgency_minutes_for(self, snapshot: MarketSnapshot) -> float:
        if self._is_short_duration(snapshot.ticker):
            return self.urgency_minutes + self.urgency_minutes_15min_bonus
        return self.urgency_minutes

    def _effective_stop_loss_pct(self, snapshot: MarketSnapshot) -> float:
        base = super()._effective_stop_loss_pct(snapshot)
        if self._is_short_duration(snapshot.ticker):
            return base + self.stop_loss_pct_15min_bonus
        return base
