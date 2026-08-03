"""Phase 2 entry point: runs the simulated execution loop continuously,
24/7, against a broad watchlist spanning every major sport category plus
rotating 15-minute/hourly crypto markets, using an asset-specific strategy
per ticker.

    uv run python run_paper_trading.py

Honest scope: the crypto and sports strategies both trade on real market
data (price history, volume, order-book depth) that Kalshi actually
provides. Neither has, or claims to have, access to player/coach/injury
data, on-chain data, or any external feed -- see trade_bot/strategy.py's
docstrings for what each signal actually is.

Simulation only -- see trade_bot/engine.py's SIMULATION_MODE guard and the
note in trade_bot/client.py (no POST method exists anywhere in this repo,
so no code path here can place a real order).
"""
from __future__ import annotations

import logging
import sys

from trade_bot.client import KalshiClient
from trade_bot.engine import ExecutionEngine, RiskLimits
from trade_bot.portfolio import PaperPortfolio
from trade_bot.strategy import CryptoTechnicalStrategy, SportsMicrostructureStrategy, Strategy
from trade_bot.watchlist import build_watchlist

INTERVAL_SECONDS = 45
CRYPTO_TICKER_PREFIXES = ("KXBTC", "KXETH", "KXSOL", "KXDOGE", "KXXRP", "KXBNB")

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", stream=sys.stdout)


def is_crypto_ticker(ticker: str) -> bool:
    return ticker.startswith(CRYPTO_TICKER_PREFIXES)


def main() -> None:
    client = KalshiClient()
    portfolio = PaperPortfolio(starting_balance=1000.0)

    # Tuned from a diagnosis of the first ~165 closed trades (2026-07-12/13):
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
    crypto_strategy = CryptoTechnicalStrategy(
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
    sports_strategy = SportsMicrostructureStrategy(
        window_minutes=4.0,
        threshold_pct=2.5,
        volume_surge_ratio=1.4,
        imbalance_threshold=0.15,
        base_dollars=150.0,
        max_dollars=280.0,
        stop_loss_pct=0.15,
        trail_activate_pct=0.05,
        trail_giveback_pct=0.03,
        max_profit_pct=0.25,
        urgency_minutes=5.0,
        max_entry_price_pct=85.0,
    )

    def strategy_router(ticker: str) -> Strategy:
        return crypto_strategy if is_crypto_ticker(ticker) else sports_strategy

    # Aggressive but not reckless: paper money, so we trade meaningfully sized
    # positions (10-30% of capital each) -- but risk limits stay on so a
    # runaway signal can't blow through the whole account in one cycle.
    # $1000 starting balance.
    risk_limits = RiskLimits(
        max_position_size_dollars=300.0,
        max_total_exposure_dollars=800.0,
        daily_stop_loss_dollars=250.0,
    )

    initial_watchlist = build_watchlist(client)
    engine = ExecutionEngine(
        client,
        portfolio,
        crypto_strategy,  # default/fallback strategy
        initial_watchlist,
        risk_limits=risk_limits,
        watchlist_resolver=lambda: build_watchlist(client),
        strategy_router=strategy_router,
        reentry_cooldown_minutes=8.0,
    )

    print(f"Starting paper-trading loop: every {INTERVAL_SECONDS}s, 24/7. Ctrl-C to stop.")
    print(f"Starting balance: ${portfolio.get_starting_balance():,.2f} (auto-resets to this if it hits $0)")
    print(f"Initial watchlist ({len(initial_watchlist)} markets): {', '.join(initial_watchlist)}")
    print("Crypto tickers -> CryptoTechnicalStrategy, everything else -> SportsMicrostructureStrategy.")
    print("Watchlist re-resolves across all sport categories + rotating crypto every cycle.")
    print("Decisions logged to logs/decisions.log")

    try:
        engine.run_forever(interval_seconds=INTERVAL_SECONDS)
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        portfolio.close()
        client.close()


if __name__ == "__main__":
    main()
