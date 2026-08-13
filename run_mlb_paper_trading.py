"""MLB game-winner paper-trading validation loop -- runs the exact same
MlbGameWinnerStrategy/model that run_mlb_trading.py would run for real
money, against the generic simulated engine (trade_bot/engine.py,
SIMULATION_MODE-gated, structurally incapable of placing a real order --
see that module's docstring). This is Phase 6 of the MLB plan: the real
track record this strategy needs before any real-money conversation, since
none exists yet.

    uv run python run_mlb_paper_trading.py

Deliberately always-on (own systemd service, deploy/mlb-paper-trading.service)
rather than a dashboard-controlled start/stop like the live bot -- there's
no real money here and no reason to want it off; the whole point is letting
it accumulate real paper trade outcomes across as much of the remaining
season as possible. Same sizing/thresholds as run_mlb_trading.py's
placeholder config (base_dollars=5/max_dollars=10/min_win_probability=0.52,
all MlbGameWinnerStrategy defaults, not overridden) -- deliberately NOT
tuned more aggressive the way run_paper_trading.py's crypto paper bot is,
since the goal here is an honest read on whether THIS exact configuration
works, not a stress test of a different one.

Own isolated mlb_paper_trading.db (PaperPortfolio), separate from every
other db in this repo -- same isolation principle as everywhere else.
"""
from __future__ import annotations

import logging
import signal
import sys
from pathlib import Path

from trade_bot.client import KalshiClient
from trade_bot.engine import ExecutionEngine, RiskLimits
from trade_bot.mlb_data import MlbDataStore
from trade_bot.mlb_strategy import MlbGameWinnerStrategy
from trade_bot.mlb_watchlist import build_mlb_watchlist
from trade_bot.portfolio import PaperPortfolio

INTERVAL_SECONDS = 900  # same pregame polling cadence as run_mlb_trading.py

ROOT = Path(__file__).resolve().parent
DB_PATH = ROOT / "mlb_paper_trading.db"
# Own log file -- ExecutionEngine's log_path defaults to logs/decisions.log,
# already used by the crypto paper bot; reusing it would interleave the two
# (same class of bug already found and fixed for the live engine's
# logs/live_decisions.log, see run_mlb_trading.py's LOG_PATH).
LOG_PATH = ROOT / "logs" / "mlb_paper_decisions.log"

STARTING_BALANCE = 1000.0

# Generous headroom, not a real constraint -- individual trades are ~$5-10
# per MlbGameWinnerStrategy's own defaults, these exist so the generic
# engine's risk checks never bind and mask what the strategy itself would
# actually do.
RISK_LIMITS = RiskLimits(
    max_position_size_dollars=1000.0,
    max_total_exposure_dollars=1000.0,
    daily_stop_loss_dollars=1000.0,
)

STRATEGY_NAME = MlbGameWinnerStrategy.name

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", stream=sys.stdout)


class _StopRequested(BaseException):
    """Same pattern as every other entry point in this repo -- subclasses
    BaseException so a per-cycle `except Exception` can't swallow it."""


def _handle_sigterm(signum, frame) -> None:
    raise _StopRequested()


def main() -> None:
    signal.signal(signal.SIGTERM, _handle_sigterm)

    client = KalshiClient()
    portfolio = PaperPortfolio(db_path=DB_PATH, starting_balance=STARTING_BALANCE)
    data_store = MlbDataStore()
    # mlb_trading_enabled hardcoded True here, not toggle-refreshed from the
    # ledger -- unlike the live bot, paper has no real-money kill-switch to
    # respect; it should just always run so it actually accumulates a
    # track record.
    strategy = MlbGameWinnerStrategy(data_store=data_store, mlb_trading_enabled=True)

    initial_watchlist = build_mlb_watchlist(client, data_store)
    engine = ExecutionEngine(
        client,
        portfolio,
        strategy,
        initial_watchlist,
        risk_limits=RISK_LIMITS,
        log_path=LOG_PATH,
        watchlist_resolver=lambda: build_mlb_watchlist(client, data_store),
        reentry_cooldown_minutes=0.0,
    )

    print(f"Starting MLB paper-trading loop: every {INTERVAL_SECONDS}s. Ctrl-C to stop.")
    print(f"Starting balance: ${portfolio.get_starting_balance():,.2f}")
    print(f"Initial watchlist ({len(initial_watchlist)} tickers): {', '.join(initial_watchlist)}")
    print(f"Strategy: {STRATEGY_NAME} -- min_win_probability={strategy.min_win_probability}, "
          f"base_dollars={strategy.base_dollars}, max_dollars={strategy.max_dollars}")
    print("Simulation only -- see trade_bot/engine.py's SIMULATION_MODE guard.")
    print("Decisions logged to logs/mlb_paper_decisions.log")

    try:
        engine.run_forever(interval_seconds=INTERVAL_SECONDS)
    except (KeyboardInterrupt, _StopRequested):
        print("\nStopped.")
    finally:
        portfolio.close()
        data_store.close()
        client.close()


if __name__ == "__main__":
    main()
