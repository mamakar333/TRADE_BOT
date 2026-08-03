"""Curated sport/asset -> series-ticker groupings for the dashboard's category
browser. Kalshi's own tags (e.g. "Basketball") are too broad -- they mix NBA,
WNBA, college, and dozens of overseas leagues into one bucket -- so this is a
hand-picked set of the primary "who wins this game/match" series per league,
verified against live series data on 2026-07-12.
"""
from __future__ import annotations

from .client import KalshiClient
from .data import MarketSummary, list_open_markets

MARKET_CATEGORIES: dict[str, list[str]] = {
    "NBA": ["KXNBAGAME", "KXNBASUMMERGAME"],
    "MLB": ["KXMLBGAME"],
    "NFL": ["KXNFLGAME"],
    "Soccer": [
        "KXMLSGAME", "KXUELGAME", "KXEPLGAME", "KXLALIGAGAME",
        "KXUCLGAME", "KXSERIEAGAME", "KXBUNDESLIGAGAME", "KXLIGUE1GAME",
    ],
    "Tennis": ["KXATPMATCH", "KXWTAMATCH", "KXITFMATCH", "KXITFWMATCH"],
    "Cricket": [
        "KXT20MATCH", "KXTESTMATCH", "KXCRICKETODIMATCH",
        "KXWT20MATCH", "KXWODIMATCH", "KXWTESTMATCH",
    ],
    "Crypto": [
        "KXBTCD", "KXETHD", "KXBTC15M", "KXETH15M", "KXBTCMAXY", "KXETHMAXY",
        "KXSOLMAXMON", "KXDOGEMAX1", "KXXRPMINMON", "KXBNB15M",
    ],
}

# "Multivariate event" combo/parlay markets bundle many outcomes into one
# ticker and Kalshi renders their title as a comma-joined list of every leg
# (e.g. "yes Pittsburgh,yes Baltimore,yes New York Y,..."), which reads as
# noise rather than a market. Excluded from browsing by default.
COMBO_TICKER_PREFIXES = ("KXMVE",)


def is_combo_market(ticker: str) -> bool:
    return ticker.startswith(COMBO_TICKER_PREFIXES)


def list_markets_for_category(
    client: KalshiClient, category: str, status: str = "open", max_pages_per_series: int = 2
) -> list[MarketSummary]:
    series_list = MARKET_CATEGORIES.get(category, [])
    seen: dict[str, MarketSummary] = {}
    for series_ticker in series_list:
        page = list_open_markets(
            client, series_ticker=series_ticker, status=status, max_pages=max_pages_per_series
        )
        for m in page.markets:
            seen[m.ticker] = m
    return list(seen.values())


def list_all_curated_markets(client: KalshiClient, status: str = "open") -> list[MarketSummary]:
    """Every curated category combined. Used as the *default* Markets view instead
    of an unfiltered exchange-wide scan -- that raw scan is dominated by high-volume
    combo/parlay markets (see is_combo_market), so filtering those out of it can
    leave nothing. This guarantees a clean, relevant default every time."""
    seen: dict[str, MarketSummary] = {}
    for category in MARKET_CATEGORIES:
        for m in list_markets_for_category(client, category, status=status):
            seen[m.ticker] = m
    return list(seen.values())
