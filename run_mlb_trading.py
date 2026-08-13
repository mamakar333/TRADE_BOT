"""MLB game-winner REAL-MONEY entry point. Mirrors run_live_trading.py's
structure and safety model exactly, as a fully independent process from the
crypto live bot (own pidfile, own ledger/db, own confirmation phrase) -- not
a sub-mechanism inside the crypto process, matching the existing live/paper
separation precedent rather than the btc_dip_enabled in-process-toggle one.

    uv run python run_mlb_trading.py

Gated OFF by default (MlbGameWinnerStrategy.mlb_trading_enabled starts
False, toggled live from the dashboard/app via LiveLedger.
set_mlb_trading_enabled -- see trade_bot/mlb_strategy.py). Per the explicit
plan this was built against: dollar caps below are deliberately
conservative PLACEHOLDERS, not a real considered decision the way crypto's
$100/$100/$35 was -- revisit before this ever risks real money, informed by
an actual paper-trading track record (this codebase has none yet for MLB).
daily_kill_switch_enabled is True here (unlike crypto's current disabled
state) for the same reason: no track record yet to justify disabling the
one thing that automatically stops a bad day.
"""
from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

from trade_bot.adaptive import AdaptiveConfig, PerformanceGovernor
from trade_bot.client import KalshiClient
from trade_bot.live_engine import LiveExecutionEngine, LiveRiskLimits
from trade_bot.live_ledger import LiveLedger
from trade_bot.mlb_bot_control import CONFIRMATION_ENV_VAR, CONFIRMATION_PHRASE, remove_pidfile, write_pidfile
from trade_bot.mlb_data import MlbDataStore
from trade_bot.mlb_ledger import MlbLedger
from trade_bot.mlb_strategy import MlbGameWinnerStrategy
from trade_bot.mlb_watchlist import KXMLBGAME_SERIES_PREFIX, build_mlb_watchlist, game_key
from trade_bot.online_learner import OnlineLogisticLearner
from trade_bot.mlb_features import FEATURE_NAMES as MLB_FEATURE_NAMES

# Games change on the order of hours, not seconds -- 15min pregame polling
# is plenty (vs. crypto's 10s, tuned for a market that can move meaningfully
# within a single poll gap). See the MLB plan's "Decisions made" section.
INTERVAL_SECONDS = 900

GOVERNOR_CONFIG = AdaptiveConfig(
    lookback_trades=15,
    min_trades_for_adaptation=8,
    pause_win_rate=0.20,
    reduced_win_rate=0.35,
    pause_expectancy_dollars=-3.0,
)
GOVERNOR_LOOKBACK_HOURS = 48.0  # much longer than crypto's 0.5h -- MLB trades close on the order of hours/a day, not minutes

RISK_LIMITS = LiveRiskLimits(
    max_capital_dollars=20.0,
    max_position_size_dollars=10.0,
    low_balance_fraction=0.25,
    daily_stop_loss_dollars=10.0,
    daily_kill_switch_enabled=True,
)

MIN_TRADES_FOR_ML_GATE = 20
ML_GATE_THRESHOLD = 0.0  # inert until there's real MLB trade history to learn from -- same bootstrap as crypto's original rollout

STRATEGY_NAME = MlbGameWinnerStrategy.name

# Own log file, separate from the crypto bot's logs/live_decisions.log --
# LiveExecutionEngine's log_path defaults to that file, and without this
# override MLB and crypto decisions would interleave in the same file
# (confirmed directly during testing), breaking every crypto-only log
# analysis done elsewhere in this codebase and just making both harder to
# read.
LOG_PATH = Path(__file__).resolve().parent / "logs" / "mlb_decisions.log"

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", stream=sys.stdout)


class _StopRequested(BaseException):
    """Same pattern as run_live_trading.py's -- subclasses BaseException so
    a per-cycle `except Exception` in the engine loop can't swallow it."""


def _handle_sigterm(signum, frame) -> None:
    raise _StopRequested()


def _require_explicit_confirmation() -> None:
    if os.getenv(CONFIRMATION_ENV_VAR, "").strip() != CONFIRMATION_PHRASE:
        print(
            f"Refusing to start: set {CONFIRMATION_ENV_VAR}="
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

    toggle_source = LiveLedger()
    ledger = MlbLedger(toggle_source=toggle_source)
    data_store = MlbDataStore()
    strategy = MlbGameWinnerStrategy(data_store=data_store)
    governor = PerformanceGovernor(
        lambda name, limit: ledger.get_recent_realized_pnls_within_hours(name, limit, GOVERNOR_LOOKBACK_HOURS),
        GOVERNOR_CONFIG,
    )
    learner = OnlineLogisticLearner.from_json(ledger.load_learner_state(), feature_names=MLB_FEATURE_NAMES)

    engine = LiveExecutionEngine(
        client, strategy, ledger, RISK_LIMITS,
        watchlist_resolver=lambda: build_mlb_watchlist(client, data_store),
        allowed_series_prefixes=[KXMLBGAME_SERIES_PREFIX],
        log_path=LOG_PATH,
        governor=governor,
        learner=learner,
        min_trades_for_ml_gate=MIN_TRADES_FOR_ML_GATE,
        ml_gate_threshold=ML_GATE_THRESHOLD,
        extra_toggle_refresh=lambda: setattr(strategy, "mlb_trading_enabled", ledger.get_mlb_trading_enabled()),
        underlying_key_fn=game_key,
    )

    print("=" * 70)
    print("MLB TRADING -- REAL MONEY -- series:", KXMLBGAME_SERIES_PREFIX)
    print(f"Capital cap: ${RISK_LIMITS.max_capital_dollars:.0f}  "
          f"Per-trade cap: ${RISK_LIMITS.max_position_size_dollars:.0f} (or {RISK_LIMITS.low_balance_fraction:.0%} "
          f"of balance if under that)  Daily loss kill-switch: ${RISK_LIMITS.daily_stop_loss_dollars:.0f}")
    print(f"mlb_trading_enabled starts as: {ledger.get_mlb_trading_enabled()} (toggle from the dashboard/app)")
    print(f"Polling every {INTERVAL_SECONDS}s. Ctrl-C (or the dashboard's Stop button) to stop.")
    print("=" * 70)

    try:
        engine.run_forever(interval_seconds=INTERVAL_SECONDS)
    except (KeyboardInterrupt, _StopRequested):
        print("\nStopped.")
    finally:
        remove_pidfile()
        ledger.close()
        data_store.close()
        toggle_source.close()
        client.close()


if __name__ == "__main__":
    main()
