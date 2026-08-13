"""REAL-MONEY entry point. Places actual orders on Kalshi against your real
account balance. Scoped to a resolved watchlist of Kalshi's fast-moving
crypto markets -- KXBTC15M, KXETH15M, KXBNB15M (rotate every 15min) plus
near-the-money strikes from KXBTCD/KXETHD (rotate hourly), ~10-11 tickers at
a time (trade_bot/watchlist.py's build_crypto_watchlist). Originally
KXBTC15M only; extended 2026-08-03 per explicit user request. Long-dated
crypto ladder markets (KXBTCMAXY/KXETHMAXY/KXSOLMAXMON/KXDOGEMAX1/
KXXRPMINMON, settling months out) are deliberately excluded -- this
strategy's few-minute momentum signal isn't a meaningful basis for a
months-away outcome, and nothing here has been validated against that kind
of market. Nothing outside build_crypto_watchlist's series is ever touched.

    uv run python run_live_trading.py

Hard safety parameters below were explicitly chosen by the account owner on
2026-08-01, after being shown that the same strategy (crypto_technical) had
a 26% win rate and a net loss of ~$371 in paper trading:

    total capital this bot may ever risk .... $100
    max risked on any single open trade ..... $100 while balance >= $100,
                                                else 25% of whatever's left
                                                (updated 2026-08-03 -- see
                                                LiveRiskLimits below)
    daily loss kill-switch .................. $35, auto-pauses new entries
                                                for the rest of that UTC day,
                                                resumes automatically the
                                                next day

Two fixes shipped alongside this from the paper-trading diagnosis that
justified going live at all:
  1. min_minutes_to_close entry guard (trade_bot/strategy.py) -- stops the
     strategy from opening a fresh position in a market's final minutes,
     where 76% of the paper strategy's realized loss came from (a threshold
     contract's price can gap straight through a stop-loss between polls
     right before expiry).
  2. PerformanceGovernor (trade_bot/adaptive.py) -- the honest version of
     "learns as it goes": tracks this strategy's own trailing REAL results
     and automatically halves size or pauses new entries if they turn bad,
     using only trades that have actually happened. Not a predictive model.

Before this will run at all, KALSHI_USE_DEMO must be false (real account)
and LIVE_TRADING_CONFIRMED must be set to the exact phrase checked below --
both are deliberate, unambiguous acts, not something a stray env var flips
by accident. The dashboard's Start button (trade_bot/bot_control.py) sets
that env var itself once the operator has confirmed in the UI -- the
confirmation step lives there when launched that way, not here.

STRATEGY/GOVERNOR_CONFIG/RISK_LIMITS are module-level (not buried in main())
so app.py's Live Trading tab can import the exact same objects actually
enforced here, rather than keeping a second, driftable copy of these numbers
just for display.
"""
from __future__ import annotations

import logging
import os
import sys

from trade_bot.adaptive import AdaptiveConfig, PerformanceGovernor
from trade_bot.bot_control import CONFIRMATION_PHRASE, remove_pidfile, write_pidfile
from trade_bot.btc_price_history import BtcPriceHistory
from trade_bot.client import KalshiClient
from trade_bot.data_driven_strategy import DataDrivenCryptoStrategy
from trade_bot.interval_tracker import IntervalTracker
from trade_bot.live_engine import LiveExecutionEngine, LiveRiskLimits
from trade_bot.live_ledger import LiveLedger
from trade_bot.online_learner import OnlineLogisticLearner
from trade_bot.watchlist import ACTIVE_CRYPTO_SERIES_PREFIXES, build_crypto_watchlist

# REVISED 2026-08-06: was 20 -- measured directly against real closed stop-
# loss trades (59 since 2026-08-05) that the ACTUAL fill routinely lands far
# past the configured stop threshold: median overshoot 7.1 percentage
# points, p75 13.3pp, max 50.1pp, with 34% of trades overshooting by more
# than 10pp. That gap is price moving between polls before a breach is even
# detected, on markets that can swing hard in their final minutes. Halving
# the interval doesn't eliminate this (still 1-2 real API calls per ticker
# per cycle, ~11 tickers), but directly shrinks the window in which price
# can move against an open position undetected.
INTERVAL_SECONDS = 10

# Added 2026-08-06: LiveExecutionEngine previously had no re-entry cooldown
# at all -- see trade_bot/live_engine.py's constructor comment for the real
# data (same ticker stopped out repeatedly within minutes) that motivated
# porting over this already-proven paper-engine mechanism. Same default the
# paper engine has always used.
REENTRY_COOLDOWN_MINUTES = 8.0

# Exposed so app.py/api.py can look up THIS strategy's governor/ledger stats
# by the right name dynamically (governor.evaluate(STRATEGY_NAME)) instead
# of a second hardcoded copy of the string that would silently go stale on
# the next rewrite, the way "crypto_technical" almost did on this one.
STRATEGY_NAME = DataDrivenCryptoStrategy.name

# Replaced 2026-08-04 (CryptoTechnicalStrategy -> DataDrivenCryptoStrategy)
# per explicit user request: "completely rewrite the algorithm based on the
# collected data" -- see docs/ALGORITHM.md for the full writeup. Short
# version: validated against all 329 real closed live trades so far
# (122W/207L, 37.1% win rate, -$136.14), which showed the OLD strategy's core
# premise (momentum continuation, confirmed by two-timeframe agreement)
# barely predicts outcomes at all in a fitted logistic regression, while
# order-book imbalance (never used to drive an entry before) and volatility
# (previously only a size dampener) are the strongest real signals available.
# CryptoTechnicalStrategy itself is UNCHANGED but, as of 2026-08-06, no
# longer runs anywhere by default -- run_paper_trading.py was repurposed
# per a later explicit user request to run DataDrivenCryptoStrategy too
# (tuned more risk-tolerant, uncapped sizing, no hard stops) rather than
# stay an A/B control for this rewrite. See that file's docstring.
#
# Note on why the three 15-min series (KXBTC15M/KXETH15M/KXBNB15M) still
# lean on scalp as their primary mechanism: a 15-min market's whole life is
# too short for a meaningful longer-window read, the same constraint that
# made the old strategy's strict multi-timeframe signal mathematically
# unable to fire there at all (needs t>=12min into the market's life, can't
# enter past min_minutes_to_close=6min remaining). scalp's own 45s window
# sidesteps that; it was real data's best performer of any signal (55% win
# rate) and is kept structurally as-is, just re-costed (larger stake/target)
# to clear fee drag against its small size -- see docs/ALGORITHM.md.
STRATEGY_PARAMS = dict(
    short_window_minutes=3.0,
    long_window_minutes=12.0,
    base_dollars=25.0,
    max_dollars=30.0,
    # Primary direction signal -- see docs/ALGORITHM.md's "why imbalance,
    # not momentum" section. 0.12 is a real, not arbitrary, floor: bucketed
    # against the historical data, |imbalance| below this had no
    # meaningfully different win rate than noise.
    min_imbalance_pct=0.12,
    momentum_conflict_tolerance_pct=1.5,
    # Hard gate (not just a dampener like the old strategy's volatility
    # handling) -- bucketed win rate drops close to monotonically from
    # ~43% under 2pp to ~25-31% above 6pp in the historical data.
    max_volatility_pct=6.5,
    # Absolute predicted-win-probability gate from a batch-fit logistic
    # regression (trade_bot/data_driven_strategy.py's PROBABILITY_MODEL_*
    # constants). Raised 0.43 -> 0.50 on 2026-08-12 per explicit user
    # request, after refit_strategy_model.py's holdout backtest against the
    # (also just-refreshed) 999-trade model showed 0.43 barely beats doing
    # nothing (45.3% win rate on 161/200 held-out trades vs. a 45.0% base
    # rate) while 0.50 shows a real lift (52.9% win rate on 70/200 taken).
    # Checked before shipping: the model's own predictions across all 999
    # real trades range 0.314-0.618 (median 0.447, p90 0.515) -- 0.50 lets
    # through ~16% of historical candidates, meaningfully more selective
    # without starving the bot the way a higher cutoff would (0.55 would
    # have cleared only 2% of the same 999, the same near-total-halt failure
    # mode already hit once with ML_GATE_THRESHOLD on 2026-08-04 -- checked
    # and explicitly rejected for that reason). See refit_strategy_model.py
    # to regenerate all of this later as more real trades accumulate.
    min_win_probability=0.50,
    min_entry_price_pct=45.0,
    max_entry_price_pct=85.0,
    min_minutes_to_close=6.0,
    # REVISED 2026-08-05 -- see trade_bot/data_driven_strategy.py's field
    # comments for the full writeup. Short version: the initial 0.10/0.05
    # tightening (2026-08-04) overcorrected. Real results over the first
    # ~23 hours on this strategy (201 trades) showed 63% of the 87
    # stop-loss exits (standard + scalp) would have WON if held to actual
    # settlement instead of being cut, and in aggregate the stop-loss
    # mechanism cost -$37 relative to just holding on that data. Widened
    # back up in response.
    stop_loss_pct=0.20,
    stop_loss_time_bonus_pct=0.10,
    stop_loss_time_reference_minutes=60.0,
    # Added 2026-08-07 -- see trade_bot/data_driven_strategy.py's field
    # comment for the full writeup. Short version: real stop-loss overshoot
    # on 2026-08-06 was median 11.2pp on the three 15-min series vs 1.8pp
    # on the hourly ones (max 41.1pp vs 14.3pp) -- these markets swing
    # harder near their own close, same reasoning as
    # urgency_minutes_15min_bonus below.
    stop_loss_pct_15min_bonus=0.15,
    trail_activate_pct=0.05,
    trail_giveback_pct=0.03,
    max_profit_pct=0.25,
    urgency_minutes=3.0,
    # Extra buffer on the three 15-min series specifically -- of 41
    # reconciled/forced-settlement trades in the historical data (9.8% win
    # rate, -$89.72, the second-biggest loss source after stop-losses),
    # 14 of 15 were on these three tickers. Defense in depth alongside (not
    # instead of) the bot-watchdog fix for the downtime that caused most of
    # the worst individual cases.
    urgency_minutes_15min_bonus=1.5,
    # DISABLED 2026-08-06 -- see trade_bot/data_driven_strategy.py's field
    # comment for the full investigation. Short version: three tuning
    # rounds couldn't make this profitable, and a trigger-size breakdown
    # showed every bucket losing money, not just the big/noisy ones --
    # there's no configuration of this mechanism found to work yet.
    scalp_enabled=False,
    scalp_window_seconds=45.0,
    scalp_threshold_pct=1.5,
    scalp_dollars=5.0,
    # REVISED 2026-08-06 -- see trade_bot/data_driven_strategy.py's field
    # comment for the full writeup. Short version: an 8% target against the
    # 0.20 stop below needs 71% win rate to break even; real win rate was
    # ~36-42%. 53% of actual scalp stop-loss trades since 2026-08-05 would
    # have won if held, so there's real follow-through to capture.
    scalp_take_profit_pct=0.12,
    # REVISED 2026-08-05: was 0.08, firing on 74% of scalp trades at 5.7%
    # win rate, 68.6% of which would have won if held -- same finding as
    # the standard stop_loss_pct above, worse. scalp_max_hold_minutes
    # already forces an exit regardless of price, so this doesn't undercut
    # "fast in/out", it just stops the stop from deciding most outcomes.
    # REVISED AGAIN 2026-08-06 -- see trade_bot/data_driven_strategy.py's
    # field comment. Short version: independent cross-check via
    # interval_outcomes (actual entry side vs actual settlement, not a log-
    # scraped proxy) found scalp_stop_loss was still the single largest
    # source of "direction was right, still lost money" trades (12 of 25,
    # -$9.87) since the previous widening.
    scalp_stop_loss_pct=0.28,
    scalp_max_hold_minutes=2.5,
    momentum_hold_enabled=True,
    momentum_hold_confirm_minutes=1.0,
    momentum_hold_confirm_pct=0.5,
    momentum_hold_giveback_pct=0.08,
    momentum_hold_max_profit_pct=0.50,
    # No gamble_* parameters -- the mechanism doesn't exist in this
    # strategy at all (not disabled, not present). Real data: 0-for-41,
    # -$31.57, an unambiguous failure. Not carried forward.
)

GOVERNOR_CONFIG = AdaptiveConfig(
    lookback_trades=15,
    min_trades_for_adaptation=8,
    pause_win_rate=0.20,
    reduced_win_rate=0.35,
    pause_expectancy_dollars=-3.0,
)
# Diagnosed 2026-08-03: without a time bound, a pause is permanent -- pausing
# blocks entries, no entries means no new closed trades, and no new closed
# trades means the trailing window can never move past the old data that
# caused the pause (see LiveLedger.get_recent_realized_pnls_within_hours's
# docstring). Old data ages out of the window after this long, letting the
# gate re-open to "not enough data yet" on its own. Shortened from 4h to
# 30min 2026-08-03 per explicit user request -- a shorter memory means a bad
# streak recovers faster, at the real cost of less protection duration after
# one. Reasonable now specifically because the same-cycle correlated-entry
# fix means a single bad market move can no longer produce a large multi-
# trade losing burst in one shot the way it did when this was set to 4h.
GOVERNOR_LOOKBACK_HOURS = 0.5

RISK_LIMITS = LiveRiskLimits(
    max_capital_dollars=100.0,
    # Updated 2026-08-03 per explicit user instruction: bet up to $100 while
    # there's at least $100 available; below that, scale down to 25% of
    # whatever's left rather than getting stuck unable to size a trade, and
    # never risk the whole remaining balance on one trade.
    max_position_size_dollars=100.0,
    low_balance_fraction=0.25,
    daily_stop_loss_dollars=35.0,
    # Disabled 2026-08-04 per explicit user request: keep trading through a
    # losing day instead of auto-pausing, so the online learner keeps
    # getting real labeled examples faster. The user is taking over this
    # control manually via the dashboard/app Stop button instead. The $35
    # figure and its progress bar stay visible in both UIs for awareness --
    # this only controls whether it actually blocks new entries. Per-trade
    # ($100) and total ($100) sizing caps above are unrelated and unchanged.
    daily_kill_switch_enabled=False,
)

# Added 2026-08-03 per explicit user request: a live-updating model, not
# just the governor's coarse win-rate gate. See online_learner.py's
# docstring for the honest framing -- it's a small interpretable per-trade
# model, not a black box, and it stays inert (no effect) below
# MIN_TRADES_FOR_ML_GATE since real trading won't produce enough labeled
# examples to learn anything meaningful for a while. Persisted in
# live_trading.db so it keeps whatever it's learned across restarts.
#
# Threshold restored to 0.40 on 2026-08-11, per the plan above: the model
# now has 903 real learner updates (vs. 48 when it was dropped to 0.0) and
# its output is no longer the degenerate near-uniform ~35% seen back then --
# live predicted win probabilities since 2026-08-09 range 27%-65% (median
# 43%), and a 0.40 gate against that distribution blocks ~34% of candidates,
# not the 97% that forced the original disable. The model's learned feature
# weights now also line up with what the real ledger shows independent of
# the model (e.g. asset_btc +0.30 / asset_eth -0.18: BTC bot-trades have the
# best win rate of the three assets in both the full 959-trade history and
# the most recent 199-trade window, ETH/BNB are both worse) -- evidence it
# has learned a real, not spurious, pattern. Reassess again if this starts
# blocking a large fraction of signals the way it did in the 48-update state.
MIN_TRADES_FOR_ML_GATE = 20
ML_GATE_THRESHOLD = 0.40

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", stream=sys.stdout)


class _StopRequested(BaseException):
    """Raised from the SIGTERM handler. Subclasses BaseException (not
    Exception), same as KeyboardInterrupt/SystemExit -- deliberately so that
    LiveExecutionEngine.run_forever()'s per-cycle `except Exception` can
    never accidentally swallow a shutdown request if the signal happens to
    land while a cycle is mid-flight."""


def _handle_sigterm(signum, frame) -> None:
    raise _StopRequested()


def _require_explicit_confirmation() -> None:
    if os.getenv("LIVE_TRADING_CONFIRMED", "").strip() != CONFIRMATION_PHRASE:
        print(
            "Refusing to start: set LIVE_TRADING_CONFIRMED="
            f"{CONFIRMATION_PHRASE!r} in your environment (or .env) to run this script.\n"
            "This is a real-money trading bot -- this check exists so it never starts by accident."
        )
        sys.exit(1)


def main() -> None:
    import signal

    _require_explicit_confirmation()

    client = KalshiClient()
    if client.settings.use_demo:
        print("Refusing to start: KALSHI_USE_DEMO is true (demo account). Real trading requires the prod account.")
        sys.exit(1)

    signal.signal(signal.SIGTERM, _handle_sigterm)
    write_pidfile()

    ledger = LiveLedger()
    interval_tracker = IntervalTracker()
    # Read-only handle onto the continuous BTC/USD collector's db (see
    # trade_bot/btc_price_history.py) -- the collector process is the only
    # writer, WAL mode makes concurrent reads from this process safe.
    strategy = DataDrivenCryptoStrategy(**STRATEGY_PARAMS, btc_price_history=BtcPriceHistory())
    governor = PerformanceGovernor(
        lambda name, limit: ledger.get_recent_realized_pnls_within_hours(name, limit, GOVERNOR_LOOKBACK_HOURS),
        GOVERNOR_CONFIG,
    )
    learner = OnlineLogisticLearner.from_json(ledger.load_learner_state())

    notify_topic = os.getenv("NTFY_TOPIC", "").strip() or None
    if notify_topic is None:
        print(
            "NTFY_TOPIC not set -- push notifications disabled. To enable: pick a long random\n"
            "topic name (e.g. `openssl rand -hex 16`), set NTFY_TOPIC=<that> in .env, and\n"
            "subscribe to it in the ntfy Android app (https://ntfy.sh, free, no account)."
        )

    engine = LiveExecutionEngine(
        client, strategy, ledger, RISK_LIMITS,
        watchlist_resolver=lambda: build_crypto_watchlist(client),
        allowed_series_prefixes=ACTIVE_CRYPTO_SERIES_PREFIXES,
        governor=governor,
        learner=learner,
        min_trades_for_ml_gate=MIN_TRADES_FOR_ML_GATE,
        ml_gate_threshold=ML_GATE_THRESHOLD,
        notify_topic=notify_topic,
        interval_tracker=interval_tracker,
        reentry_cooldown_minutes=REENTRY_COOLDOWN_MINUTES,
        # Preserves the exact refresh live_engine.py's run_once() used to
        # hardcode inline, before that was generalized 2026-08-12 so the
        # MLB engine could pass its own toggle here instead.
        extra_toggle_refresh=lambda: setattr(strategy, "btc_dip_enabled", ledger.get_btc_dip_enabled()),
        # Added 2026-08-13 per explicit user request: crypto and MLB share
        # the same real Kalshi account/capital -- pause new crypto entries
        # while MLB trading is turned on, rather than have both strategies
        # compete for the same balance at once. Existing open crypto
        # positions still get managed/exited normally either way.
        pause_new_entries_when=lambda: ledger.get_mlb_trading_enabled(),
    )

    print("=" * 70)
    print("LIVE TRADING -- REAL MONEY -- crypto watchlist:", ", ".join(ACTIVE_CRYPTO_SERIES_PREFIXES))
    print(f"Capital cap: ${RISK_LIMITS.max_capital_dollars:.0f}  "
          f"Per-trade cap: ${RISK_LIMITS.max_position_size_dollars:.0f} (or {RISK_LIMITS.low_balance_fraction:.0%} "
          f"of balance if under that)  Daily loss kill-switch: ${RISK_LIMITS.daily_stop_loss_dollars:.0f}")
    print(f"Online learner: resumed with {learner.n_updates} prior learned trades "
          f"(gate activates at {MIN_TRADES_FOR_ML_GATE})")
    print(f"Polling every {INTERVAL_SECONDS}s. Ctrl-C (or the dashboard's Stop button) to stop. "
          "Decisions logged to logs/live_decisions.log")
    print("=" * 70)

    try:
        engine.run_forever(interval_seconds=INTERVAL_SECONDS)
    except (KeyboardInterrupt, _StopRequested):
        print("\nStopped.")
    finally:
        remove_pidfile()
        ledger.close()
        client.close()


if __name__ == "__main__":
    main()
