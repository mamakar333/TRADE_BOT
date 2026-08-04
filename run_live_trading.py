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
from trade_bot.client import KalshiClient
from trade_bot.data_driven_strategy import DataDrivenCryptoStrategy
from trade_bot.live_engine import LiveExecutionEngine, LiveRiskLimits
from trade_bot.live_ledger import LiveLedger
from trade_bot.online_learner import OnlineLogisticLearner
from trade_bot.watchlist import ACTIVE_CRYPTO_SERIES_PREFIXES, build_crypto_watchlist

INTERVAL_SECONDS = 20

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
# CryptoTechnicalStrategy itself is UNCHANGED and still runs, unmodified, in
# run_paper_trading.py -- the explicit A/B control for this rewrite, per the
# same request ("dont change the paper bots algorithm").
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
    # constants), calibrated against a chronological 20% holdout: the full
    # combined gate (imbalance floor + volatility ceiling + this threshold)
    # took 16/63 holdout trades (25%) at 56.2% win rate, vs. 41.3% for all
    # 63 -- real, if modest, out-of-sample separation. See
    # refit_strategy_model.py to regenerate these constants later as more
    # real trades accumulate.
    min_win_probability=0.43,
    min_entry_price_pct=45.0,
    max_entry_price_pct=85.0,
    min_minutes_to_close=6.0,
    # Tightened from the old strategy's 0.15 base / up-to-0.25 widened.
    # Single largest lever the data analysis found: stop-losses fired at a
    # -28.0% average realized return, far wider than winners were ever
    # allowed to run (trailing-stop averaged ~+5-20%) -- losers were simply
    # given much more room than winners, a losing combination at a sub-40%
    # win rate no matter how good the entry signal is.
    stop_loss_pct=0.10,
    stop_loss_time_bonus_pct=0.05,
    stop_loss_time_reference_minutes=60.0,
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
    scalp_enabled=True,
    scalp_window_seconds=45.0,
    scalp_threshold_pct=1.5,
    scalp_dollars=5.0,
    scalp_take_profit_pct=0.08,
    scalp_stop_loss_pct=0.08,
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
# Threshold temporarily dropped to 0.0 on 2026-08-04: after a real losing
# stretch (31% win rate over the last 29 learner-fed trades), the model's
# bias term went sharply negative while its per-feature (momentum) weights
# were still too small after only 48 updates to meaningfully discriminate
# good setups from bad -- so it was outputting a near-uniform ~35% on
# almost every candidate and the 0.40 gate blocked 97% of all signals for
# 6+ hours straight (1081/1117), effectively halting the bot. User chose to
# disable the hard block rather than sit idle waiting for it to recover.
# This does NOT disable the model: predict_proba() below still runs and
# still *dampens* size via ml_multiplier (min sizing floor 0.5x stays in
# effect), and every real close still feeds _update_learner() exactly as
# before -- only the hard veto (`if p_win < threshold: return`) is
# neutralized, since sigmoid output is always >= 0.0. Governor and all hard
# risk limits (capital/position caps, per-trade sizing) are unaffected.
# Restore to 0.40 (or reassess) once the model has enough fresh real
# outcomes to actually differentiate rather than reflect one bad patch.
MIN_TRADES_FOR_ML_GATE = 20
ML_GATE_THRESHOLD = 0.0

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
    strategy = DataDrivenCryptoStrategy(**STRATEGY_PARAMS)
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
