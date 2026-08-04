"""Phase 2 entry point: runs the simulated execution loop continuously,
24/7, against the same crypto watchlist scope as the live (real-money) bot
-- KXBTC15M/KXETH15M/KXBNB15M (15min) plus KXBTCD/KXETHD (hourly) -- using
the ORIGINAL CryptoTechnicalStrategy configuration, unchanged.

    uv run python run_paper_trading.py

Narrowed 2026-08-04 per explicit user request, specifically so this stays a
clean A/B control: the live bot moved to a new, data-driven strategy (see
docs/ALGORITHM.md) on this exact same market scope, and this paper bot
keeps running the original hand-tuned strategy on that same scope,
unmodified -- so the two are directly comparable going forward, with
nothing else differing between them except the algorithm itself. Previously
traded a much broader universe (every sport category plus extra crypto
tickers); that scope is gone, not just unused -- see git history for the
old sports_strategy/strategy_router if it's ever needed again.

Honest scope: the strategy trades on real market data (price history,
volume, order-book depth) that Kalshi actually provides. It does not have,
and does not claim to have, access to on-chain data or any external feed --
see trade_bot/strategy.py's docstring for what the signal actually is.

Simulation only -- see trade_bot/engine.py's SIMULATION_MODE guard and the
note in trade_bot/client.py (no POST method exists anywhere in this repo,
so no code path here can place a real order).
"""
from __future__ import annotations

import logging
import signal
import sys

from trade_bot.client import KalshiClient
from trade_bot.engine import ExecutionEngine, RiskLimits
from trade_bot.paper_bot_control import remove_pidfile, write_pidfile
from trade_bot.portfolio import PaperPortfolio
from trade_bot.strategy import CryptoTechnicalStrategy
from trade_bot.watchlist import build_crypto_watchlist

INTERVAL_SECONDS = 45

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", stream=sys.stdout)


class _StopRequested(BaseException):
    """Raised from the SIGTERM handler -- same pattern as run_live_trading.py's,
    see its docstring for why this subclasses BaseException rather than
    Exception (so a per-cycle `except Exception` in the engine loop can
    never accidentally swallow a shutdown request)."""


def _handle_sigterm(signum, frame) -> None:
    raise _StopRequested()


# Module-level (not buried in main()) for the same reason run_live_trading.py's
# STRATEGY_PARAMS/RISK_LIMITS are: so app.py/trade_bot/api.py can read the
# exact same numbers actually enforced here, rather than a second,
# driftable copy just for display.
#
# UNCHANGED from the original tuning -- 2026-08-04 request was explicit
# that this algorithm itself stay exactly as-is, only the market scope
# narrows. Tuned from a diagnosis of the first ~165 closed trades
# (2026-07-12/13):
# (1) positions left open through forced settlement on 15min rotating
#     markets averaged ~-$9-10 each and accounted for nearly all realized
#     loss -- fixed with urgency_minutes (exit before forced settlement).
# (2) entries priced near-certain (98-100c) have asymmetric payout that's
#     a bad bet even at a high win rate -- fixed with max_entry_price_pct.
# (3) exits were priced in raw percentage-points of price, which is
#     inconsistent (3pp means a very different return at a 20c vs 80c
#     entry) and was cutting trades before they reached a real profit --
#     reworked to fractional RETURN ON CAPITAL, targeting 5%+ before the
#     trailing lock engages, with room to run to 25% if momentum holds.
# (4) trade sizes were tiny in dollar terms ($4-25 on a $1000 account,
#     because sizing was a fixed contract count regardless of price) --
#     reworked to target a dollar amount per trade directly.
# (5) the same ticker could stop out and immediately re-enter on
#     unchanged conditions, observed whipsawing 10+ times in a row --
#     fixed with a cooldown after any stop-loss exit.
STRATEGY_PARAMS = dict(
    short_window_minutes=3.0,
    long_window_minutes=12.0,
    threshold_pct=2.0,
    base_dollars=150.0,
    max_dollars=300.0,
    stop_loss_pct=0.15,
    trail_activate_pct=0.05,
    trail_giveback_pct=0.03,
    max_profit_pct=0.25,
    urgency_minutes=3.0,
    max_entry_price_pct=85.0,
)

# Aggressive but not reckless: paper money, so we trade meaningfully sized
# positions (10-30% of capital each) -- but risk limits stay on so a
# runaway signal can't blow through the whole account in one cycle.
STARTING_BALANCE = 1000.0
RISK_LIMITS = RiskLimits(
    max_position_size_dollars=300.0,
    max_total_exposure_dollars=800.0,
    daily_stop_loss_dollars=250.0,
)


def main() -> None:
    signal.signal(signal.SIGTERM, _handle_sigterm)
    write_pidfile()

    client = KalshiClient()
    portfolio = PaperPortfolio(starting_balance=STARTING_BALANCE)
    crypto_strategy = CryptoTechnicalStrategy(**STRATEGY_PARAMS)

    initial_watchlist = build_crypto_watchlist(client)
    engine = ExecutionEngine(
        client,
        portfolio,
        crypto_strategy,
        initial_watchlist,
        risk_limits=RISK_LIMITS,
        watchlist_resolver=lambda: build_crypto_watchlist(client),
        reentry_cooldown_minutes=8.0,
    )

    print(f"Starting paper-trading loop: every {INTERVAL_SECONDS}s, 24/7. Ctrl-C to stop.")
    print(f"Starting balance: ${portfolio.get_starting_balance():,.2f} (auto-resets to this if it hits $0)")
    print(f"Initial watchlist ({len(initial_watchlist)} markets): {', '.join(initial_watchlist)}")
    print("Scope: KXBTC15M/KXETH15M/KXBNB15M (15min) + KXBTCD/KXETHD (hourly) -- same as the live bot.")
    print("Algorithm: original CryptoTechnicalStrategy, unchanged -- this is the live bot's A/B control.")
    print("Decisions logged to logs/decisions.log")

    try:
        engine.run_forever(interval_seconds=INTERVAL_SECONDS)
    except (KeyboardInterrupt, _StopRequested):
        print("\nStopped.")
    finally:
        remove_pidfile()
        portfolio.close()
        client.close()


if __name__ == "__main__":
    main()
