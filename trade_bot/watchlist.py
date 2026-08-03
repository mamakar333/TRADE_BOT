"""Resolves a watchlist that includes markets which roll over on a fixed
cadence (15-minute and hourly crypto series) so the bot always has a
currently-tradeable market instead of holding a ticker that already expired,
plus a broad spread across every sport category so it's never sitting idle
just because NBA is in its off-season.

KXBTC15M/KXETH15M are single-market-at-a-time series: exactly one ticker is
open, and it's replaced by a new one every 15 minutes. KXBTCD/KXETHD are
hourly, multi-strike series (many strikes open at once, each expiring on
its own hour). Both confirmed via GET /series (frequency field) 2026-07-12.
"""
from __future__ import annotations

from datetime import datetime, timezone

from .categories import MARKET_CATEGORIES, is_combo_market, list_markets_for_category
from .client import KalshiClient
from .data import list_open_markets

ROTATING_SINGLE = ["KXBTC15M", "KXETH15M"]  # one open market at a time, rotates every 15min
ROTATING_MULTI = ["KXBTCD", "KXETHD"]  # many strikes open at once, rotates hourly
NEAR_ATM_PER_SERIES = 3

# Crypto series short/fast-moving enough for a 3-12min momentum strategy to
# mean anything: KXBTC15M/KXETH15M/KXBNB15M rotate a whole new market every
# 15min, KXBTCD/KXETHD every hour. Deliberately excludes the OTHER crypto
# series in MARKET_CATEGORIES["Crypto"] (KXBTCMAXY, KXETHMAXY, KXSOLMAXMON,
# KXDOGEMAX1, KXXRPMINMON) -- those settle months out (Dec 2026/Jan 2027),
# where a few minutes of price wiggle isn't a meaningful signal for the
# eventual outcome. Trading those would need a genuinely different, slower
# approach, not this one applied somewhere it doesn't fit.
ACTIVE_CRYPTO_ROTATING_SINGLE = ["KXBTC15M", "KXETH15M", "KXBNB15M"]
ACTIVE_CRYPTO_ROTATING_MULTI = ["KXBTCD", "KXETHD"]
ACTIVE_CRYPTO_NEAR_ATM_PER_SERIES = 4
ACTIVE_CRYPTO_SERIES_PREFIXES = [f"{s}-" for s in ACTIVE_CRYPTO_ROTATING_SINGLE + ACTIVE_CRYPTO_ROTATING_MULTI]
# KXBTCD/KXETHD are hourly-resolution, but Kalshi pre-opens many FUTURE
# hours for trading well in advance -- confirmed 2026-08-03 that an
# unfiltered near-ATM pick could land on a strike closing days out (e.g.
# today Aug 3, closing Aug 7). The user explicitly wants only markets
# resolving soon, not long-dated ones traded on this fast-signal strategy,
# so _resolve_near_atm is filtered to this horizon for the live bot.
ACTIVE_CRYPTO_MAX_CLOSE_MINUTES = 70.0

SPORT_CATEGORIES = [c for c in MARKET_CATEGORIES if c != "Crypto"]
PICKS_PER_SPORT = 2  # top-volume, non-combo markets pulled in per sport category
EXTRA_CRYPTO_PICKS = 4  # additional high-volume crypto markets beyond BTC/ETH rotation


def _resolve_rotating_single(client: KalshiClient, series_ticker: str) -> str | None:
    page = list_open_markets(client, series_ticker=series_ticker, status="open", max_pages=1, page_limit=5)
    active = [m for m in page.markets if m.status == "active"]
    return active[0].ticker if active else None


def _close_minutes_from_now(close_time: str | None) -> float | None:
    if not close_time:
        return None
    try:
        close_dt = datetime.fromisoformat(close_time.replace("Z", "+00:00"))
    except ValueError:
        return None
    return (close_dt - datetime.now(timezone.utc)).total_seconds() / 60


def _resolve_near_atm(
    client: KalshiClient, series_ticker: str, count: int, max_close_minutes: float | None = None
) -> list[str]:
    page = list_open_markets(client, series_ticker=series_ticker, status="open", max_pages=2, page_limit=200)
    candidates = [m for m in page.markets if m.status == "active" and m.yes_ask_pct is not None]
    if max_close_minutes is not None:
        candidates = [
            m for m in candidates
            if (mins := _close_minutes_from_now(m.close_time)) is not None and mins <= max_close_minutes
        ]
    near_atm = sorted(candidates, key=lambda m: abs((m.yes_ask_pct or 50) - 50))
    return [m.ticker for m in near_atm[:count]]


def build_crypto_watchlist(client: KalshiClient) -> list[str]:
    """Currently-tradeable tickers across the fast-moving crypto series only
    (see ACTIVE_CRYPTO_* above) -- one ticker per rotating-single series, a
    handful of near-the-money strikes per rotating-multi series. Used by the
    live (real-money) bot, which trades across all of these rather than just
    KXBTC15M."""
    tickers: list[str] = []
    for series in ACTIVE_CRYPTO_ROTATING_SINGLE:
        ticker = _resolve_rotating_single(client, series)
        if ticker:
            tickers.append(ticker)
    for series in ACTIVE_CRYPTO_ROTATING_MULTI:
        tickers.extend(
            _resolve_near_atm(
                client, series, ACTIVE_CRYPTO_NEAR_ATM_PER_SERIES, max_close_minutes=ACTIVE_CRYPTO_MAX_CLOSE_MINUTES
            )
        )
    return tickers


def build_watchlist(client: KalshiClient) -> list[str]:
    """Rotating crypto tickers, extra high-volume crypto markets, and a spread
    of top-volume markets across every sport category -- bounded per category
    so one busy sport (e.g. tennis, which has 100+ live matches at once)
    doesn't crowd out everything else."""
    tickers: list[str] = []

    for series in ROTATING_SINGLE:
        ticker = _resolve_rotating_single(client, series)
        if ticker:
            tickers.append(ticker)
    for series in ROTATING_MULTI:
        tickers.extend(_resolve_near_atm(client, series, NEAR_ATM_PER_SERIES))

    crypto_markets = [m for m in list_markets_for_category(client, "Crypto") if not is_combo_market(m.ticker)]
    crypto_markets.sort(key=lambda m: m.volume, reverse=True)
    for m in crypto_markets:
        if len(tickers) >= len(ROTATING_SINGLE) + len(ROTATING_MULTI) * NEAR_ATM_PER_SERIES + EXTRA_CRYPTO_PICKS:
            break
        if m.ticker not in tickers:
            tickers.append(m.ticker)

    for category in SPORT_CATEGORIES:
        markets = [m for m in list_markets_for_category(client, category) if not is_combo_market(m.ticker)]
        markets.sort(key=lambda m: m.volume, reverse=True)
        for m in markets[:PICKS_PER_SPORT]:
            tickers.append(m.ticker)

    return tickers
