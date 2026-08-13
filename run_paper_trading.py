"""Phase 2 entry point: runs the simulated execution loop continuously,
24/7, against the same crypto watchlist scope as the live (real-money) bot
-- KXBTC15M/KXETH15M/KXBNB15M (15min) plus KXBTCD/KXETHD (hourly).

    uv run python run_paper_trading.py

REPURPOSED 2026-08-06 per explicit user request: this used to run the
ORIGINAL CryptoTechnicalStrategy, unchanged, as a clean "old algorithm vs
new algorithm" A/B control for the live bot's data-driven rewrite (see git
history / docs/ALGORITHM.md for that era). It now runs the SAME
DataDrivenCryptoStrategy the live bot runs, built directly on top of
run_live_trading.STRATEGY_PARAMS (so any future live tuning is inherited
here automatically, no parallel manual edit needed) -- with an explicit set
of overrides layered on top that make this bot deliberately MORE
risk-tolerant than live:
  - probability/imbalance/volatility entry gates loosened, so more
    marginal signals qualify
  - entry price band widened
  - stop-losses and profit-taking widened/raised, riding both losers and
    winners further before cutting
  - base_dollars nudged up, and max_dollars/scalp_dollars made effectively
    unbounded (UNCAPPED_DOLLARS below) -- the strategy's own confidence-
    scaled sizing formula decides how big, not a fixed ceiling; the only
    real limit left is whatever cash the paper account actually has
  - full_confidence_margin lowered so that uncapped sizing is actually
    reachable in practice on a real trade, not just a theoretical ceiling
  - no hard stops of any kind (RISK_LIMITS below) and no re-entry cooldown
  - the account can't get stuck at zero: PaperPortfolio.reset_if_busted()
    (called unconditionally every cycle in engine.py's run_once) tops it
    back up to STARTING_BALANCE the moment cash hits $0 or goes negative

This is what makes going uncapped safe to try at all: paper money, on a
self-reloading account, specifically so this algorithm can be watched
running with the safety rails most of the way off -- exactly the kind of
experiment that would never be appropriate on the real-money bot. The two
bots are no longer a same-algorithm-different-scope control; they're now a
same-algorithm-different-risk-appetite comparison instead.

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

import run_live_trading
from trade_bot.btc_price_history import BtcPriceHistory
from trade_bot.client import KalshiClient
from trade_bot.data_driven_strategy import DataDrivenCryptoStrategy
from trade_bot.engine import ExecutionEngine, RiskLimits
from trade_bot.paper_bot_control import remove_pidfile, write_pidfile
from trade_bot.portfolio import PaperPortfolio
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


# "No limit" is deliberately relative to this bot's own reloadable bankroll
# (STARTING_BALANCE below), NOT a disconnected astronomical number.
#
# Diagnosed live 2026-08-06, minutes after first deploying this file with
# max_dollars/scalp_dollars set to $1,000,000: DataDrivenCryptoStrategy's
# sizing formula is `base_dollars + (max_dollars - base_dollars) *
# confidence`, and confidence saturates to 1.0 as soon as the predicted
# win probability clears the gate by more than full_confidence_margin --
# which real signals do almost immediately (observed live margins were
# 13-18 points, versus a 0.06 margin). With max_dollars at a million, that
# means EVERY qualifying trade targeted essentially the full $1,000,000,
# regardless of the account actually having $161. Every single one failed
# with "insufficient cash" and NOT ONE TRADE EVER FILLED -- the bot just
# sat there evaluating forever, the opposite of "risk taker, keep running
# no matter what." Confirmed directly in logs/decisions.log: dozens of
# consecutive BUY decisions all sized in the hundreds of thousands to
# millions, zero fills.
#
# UNCAPPED_DOLLARS below is instead set comfortably above what this
# account can ever actually hold (a $1000 starting/reload balance, plus
# realistic winnings) so a confident trade can size up to -- and the
# portfolio's own insufficient-cash check (PaperPortfolio.open_position)
# will naturally clamp it down to -- essentially the whole account, with
# no fixed ceiling STOPPING it from doing that. That is "no limit on trade
# max amount" in a form that can actually execute.
UNCAPPED_DOLLARS = 2_000.0
SCALP_UNCAPPED_DOLLARS = 500.0

# _size_for_dollars ROUNDS the contract count, so a position targeting
# exactly UNCAPPED_DOLLARS can cost a few cents/dollars MORE than that once
# rounded up to a whole contract -- give RISK_LIMITS real headroom above
# every dollar target in STRATEGY_PARAMS so a rounding remainder can never
# self-trip it (this exact class of bug hit live once already, see above).
RISK_LIMIT_CEILING = 5_000.0

# Module-level (not buried in main()) for the same reason run_live_trading.py's
# STRATEGY_PARAMS/RISK_LIMITS are: so app.py/trade_bot/api.py can read the
# exact same numbers actually enforced here, rather than a second,
# driftable copy just for display.
#
# Built ON TOP OF the live bot's own params (see module docstring) --
# every key below is an intentional override toward more risk, everything
# else (including full_confidence_margin -- see the sizing note above for
# why that one stays untouched) is inherited from
# run_live_trading.STRATEGY_PARAMS as-is.
STRATEGY_PARAMS = {
    **run_live_trading.STRATEGY_PARAMS,
    "min_imbalance_pct": 0.09,
    "max_volatility_pct": 9.0,
    "min_win_probability": 0.38,
    "min_entry_price_pct": 40.0,
    "max_entry_price_pct": 92.0,
    "stop_loss_pct": 0.25,
    "stop_loss_time_bonus_pct": 0.12,
    "trail_giveback_pct": 0.05,
    "max_profit_pct": 0.40,
    "base_dollars": 40.0,
    "max_dollars": UNCAPPED_DOLLARS,
    "scalp_dollars": SCALP_UNCAPPED_DOLLARS,
    "scalp_stop_loss_pct": 0.25,
    "momentum_hold_giveback_pct": 0.12,
    "momentum_hold_max_profit_pct": 0.75,
}

# Reloads to this if the account ever hits $0 (or goes negative) -- see
# PaperPortfolio.reset_if_busted(), called unconditionally every cycle.
STARTING_BALANCE = 1000.0

# No hard stops, per explicit user request -- every ceiling set high enough
# to never bind in practice (see UNCAPPED_DOLLARS/RISK_LIMIT_CEILING above
# for why not literal infinity, and why this uses a distinctly larger
# number than the strategy's own dollar targets). The account auto-
# reloading (above) is what makes this safe to leave uncapped: there's no
# real money and no floor it can get stuck below.
RISK_LIMITS = RiskLimits(
    max_position_size_dollars=RISK_LIMIT_CEILING,
    max_total_exposure_dollars=RISK_LIMIT_CEILING,
    daily_stop_loss_dollars=RISK_LIMIT_CEILING,
)


def main() -> None:
    signal.signal(signal.SIGTERM, _handle_sigterm)
    write_pidfile()

    client = KalshiClient()
    portfolio = PaperPortfolio(starting_balance=STARTING_BALANCE)
    # Same continuous BTC/USD feed live trading reads (see
    # trade_bot/btc_price_history.py) -- read-only here (the collector
    # process is the only writer, WAL mode makes that safe), gives the paper
    # bot's copy of the same strategy the same btc_realized_vol_scaled
    # feature so "same algorithm as live" stays true for this too.
    crypto_strategy = DataDrivenCryptoStrategy(**STRATEGY_PARAMS, btc_price_history=BtcPriceHistory())

    initial_watchlist = build_crypto_watchlist(client)
    engine = ExecutionEngine(
        client,
        portfolio,
        crypto_strategy,
        initial_watchlist,
        risk_limits=RISK_LIMITS,
        watchlist_resolver=lambda: build_crypto_watchlist(client),
        # No re-entry cooldown either -- another friction/safety mechanism
        # removed per the same "no hard stops" request.
        reentry_cooldown_minutes=0.0,
    )

    print(f"Starting paper-trading loop: every {INTERVAL_SECONDS}s, 24/7. Ctrl-C to stop.")
    print(f"Starting balance: ${portfolio.get_starting_balance():,.2f} (auto-resets to this if it hits $0, no matter how many times)")
    print(f"Initial watchlist ({len(initial_watchlist)} markets): {', '.join(initial_watchlist)}")
    print("Scope: KXBTC15M/KXETH15M/KXBNB15M (15min) + KXBTCD/KXETHD (hourly) -- same as the live bot.")
    print("Algorithm: same DataDrivenCryptoStrategy as live, tuned more risk-tolerant -- see module docstring.")
    print("No hard stops: no daily loss limit, no position/exposure cap, no re-entry cooldown.")
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
