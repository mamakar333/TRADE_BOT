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
from trade_bot.live_engine import LiveExecutionEngine, LiveRiskLimits
from trade_bot.live_ledger import LiveLedger
from trade_bot.online_learner import OnlineLogisticLearner
from trade_bot.strategy import CryptoTechnicalStrategy
from trade_bot.watchlist import ACTIVE_CRYPTO_SERIES_PREFIXES, build_crypto_watchlist

INTERVAL_SECONDS = 20

# Same instrument scope as the paper bot's crypto_technical strategy, but
# sized for a $100 real-money account and with min_minutes_to_close raised
# above the strategy default (5min) -- the single biggest fix from the loss
# diagnosis, and worth being more conservative on with real money than the
# 5min default used in paper trading.
#
# Note on long_window_minutes=12 + min_minutes_to_close=6: on a 15-minute
# rotating market (KXBTC15M/KXETH15M/KXBNB15M) this makes the STRICT signal's
# feasible entry window mathematically empty (needs t>=12min into the
# market's life, but can't enter past t=9min) -- confirmed 2026-08-03. On
# those three series only the micro-bet fallback below can ever fire; the
# strict signal is fully operative on the hourly KXBTCD/KXETHD strikes,
# which have plenty of room (feasible window t=12 to t=54min). Left
# unchanged rather than shrinking long_window_minutes, since these values
# came from the original loss diagnosis and the hourly series now give the
# strict signal a real home without weakening it anywhere.
STRATEGY_PARAMS = dict(
    short_window_minutes=3.0,
    long_window_minutes=12.0,
    threshold_pct=2.0,
    base_dollars=25.0,
    max_dollars=30.0,
    stop_loss_pct=0.15,
    trail_activate_pct=0.05,
    trail_giveback_pct=0.03,
    max_profit_pct=0.25,
    urgency_minutes=3.0,
    max_entry_price_pct=85.0,
    # Added 2026-08-03: win rate climbed steadily with entry price across
    # every cutoff tested on 35 real trades (e.g. below 45c: 1W/15L, 6%;
    # at/above 45c: 3W/17L, 15%) -- buying the side the market already
    # considers the underdog performed consistently worse. Honest caveat:
    # even above this floor, real win rate is still well under breakeven --
    # this removes the worst-performing slice, it is not a fix that makes
    # the signal profitable by itself.
    min_entry_price_pct=45.0,
    min_minutes_to_close=6.0,
    # Added 2026-08-03 per explicit user request: the strict multi-timeframe
    # signal above is intentionally rare (that's what fixed the paper-trading
    # losses), so the account could sit idle for a long time. This is an
    # opt-in looser fallback -- still a real signal (short-term momentum),
    # just a much lower bar -- capped at $1-4 per trade specifically so
    # accepting more/weaker signals can never risk more than pocket change
    # per trade. Same exit logic, same min_minutes_to_close/entry-price
    # guards, same daily kill-switch, same PerformanceGovernor as every other
    # trade this strategy makes.
    micro_bet_enabled=True,
    micro_bet_min_dollars=1.0,
    micro_bet_max_dollars=4.0,
    micro_bet_threshold_pct=0.5,
    # Added 2026-08-03 after diagnosing the first 11 real micro-bet trades
    # (1W/10L, -$8.63): every loss hit the hard stop-loss within 0-1.8min,
    # and the two worst losses (-50%, -26%) came from chasing the two most
    # extreme moves seen (-28.5pp, -48pp/3min) -- almost certainly thin-book
    # spikes that had already happened, not real momentum. Caps how large a
    # move the micro-bet will still treat as signal rather than noise.
    micro_bet_max_move_pct=6.0,
    # Added 2026-08-03: every real loss so far closed within 0-3.3min on a
    # fixed -15% stop-loss that didn't account for how much time was left --
    # a -15% move with 55 minutes left has real room to reverse before the
    # market resolves, the same move with 2 minutes left doesn't. Widens the
    # stop tolerance up to -25% when there's a full hour still ahead,
    # tightening back to -15% as expiry approaches. Catastrophic moves
    # (-30%+) still cut regardless of time left -- this only gives genuinely
    # recoverable-looking drawdowns more room, it doesn't remove the floor.
    stop_loss_time_bonus_pct=0.10,
    stop_loss_time_reference_minutes=60.0,
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
)

# Added 2026-08-03 per explicit user request: a live-updating model, not
# just the governor's coarse win-rate gate. See online_learner.py's
# docstring for the honest framing -- it's a small interpretable per-trade
# model, not a black box, and it stays inert (no effect) below
# MIN_TRADES_FOR_ML_GATE since real trading won't produce enough labeled
# examples to learn anything meaningful for a while. Persisted in
# live_trading.db so it keeps whatever it's learned across restarts.
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
    strategy = CryptoTechnicalStrategy(**STRATEGY_PARAMS)
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
