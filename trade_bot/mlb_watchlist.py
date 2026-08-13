"""Kalshi-side discovery for KXMLBGAME (per-game MLB moneyline markets).

Critical gotcha #1, confirmed via direct API testing (not assumed): the
bulk `GET /markets?series_ticker=KXMLBGAME` endpoint -- what
trade_bot/data.py's list_open_markets wraps, and what the existing crypto
watchlist code uses for series resolution -- returns EMPTY yes_bid/yes_ask/
volume fields for this market type (0 of 88 open markets had a quote via
that path when tested). Only the per-market detail endpoint (get_market(),
GET /markets/{ticker}) returns real prices -- confirmed with a real example
(bid $0.14/ask $0.15, ~1.07M open interest). So discovery below uses the
bulk endpoint ONLY for ticker/title (never its price fields), then calls
get_market() per candidate for a real quote before including it -- exactly
what LiveExecutionEngine._fetch_snapshot already does correctly per-cycle,
so nothing about the trading loop itself needs to change.

Critical gotcha #2, found and corrected 2026-08-12 after this module's
first version got it backwards: `close_time` on these markets is NOT
"game end plus a small buffer" -- it's a multi-DAY safety buffer in case of
a postponement (confirmed directly: a market for a specific already-
finished game showed close_time three days after the real game). The
ticker's OWN embedded date prefix, by contrast, IS reliable -- confirmed on
two independent real tickers by cross-checking against each market's own
`rules_primary` text (which states the real scheduled date/time in plain
English): ticker "...26AUG11..." matched "originally scheduled for Aug 11,
2026"; ticker "...26AUG14..." matched "Aug 14, 2026". So matching a ticker
to trade_bot/mlb_data.py's real schedule uses the ticker's own embedded
date directly (exact-date lookup, no fuzzy window needed) -- NOT
close_time, which the first version of this module mistakenly relied on
and which silently matched the wrong game in a back-to-back series during
testing (a live, already-started game vs. a different future game between
the same two teams).

Ticker format: KXMLBGAME-{yy}{MON}{dd}{hhmm}{AWAY}{HOME}-{SIDE}, e.g.
KXMLBGAME-26AUG112210KCLAD-KC (Kansas City @ LA Dodgers, officialDate
2026-08-11, this ticker's own side is KC). The {hhmm} component is NOT
used for anything here (unreliable/irrelevant, same as close_time) --
only {yy}{MON}{dd} (the real officialDate) and the two team codes matter.
"""
from __future__ import annotations

import re
from datetime import datetime, timedelta, timezone

from .client import KalshiClient
from .data import get_market, list_open_markets
from .mlb_data import MlbDataStore

KXMLBGAME_SERIES_PREFIX = "KXMLBGAME-"

_TICKER_RE = re.compile(r"^KXMLBGAME-(\d{2})([A-Z]{3})(\d{2})\d{4}([A-Z]+)-([A-Z]+)$")
_MONTHS = {
    "JAN": 1, "FEB": 2, "MAR": 3, "APR": 4, "MAY": 5, "JUN": 6,
    "JUL": 7, "AUG": 8, "SEP": 9, "OCT": 10, "NOV": 11, "DEC": 12,
}


def game_key(ticker: str) -> str:
    """The shared per-game id for both team-side tickers of the same game
    -- everything before the final '-{SIDE}' segment. Used for the
    same-game correlated-exposure guard (don't hold both sides of one
    game)."""
    return ticker.rsplit("-", 1)[0]


def parse_ticker(ticker: str) -> tuple[str, str, str, str] | None:
    """Returns (official_date "YYYY-MM-DD", away_code, home_code, this
    ticker's own side_code), or None if `ticker` doesn't match the expected
    KXMLBGAME format. The two team codes are concatenated with no
    delimiter in the ticker (e.g. "KCLAD"), so splitting them relies on
    already knowing one side's exact code (the ticker's own trailing
    segment), not guessing team-code boundaries blind."""
    m = _TICKER_RE.match(ticker)
    if not m:
        return None
    yy, mon, dd, combined, side = m.groups()
    month_num = _MONTHS.get(mon)
    if month_num is None:
        return None
    official_date = f"20{yy}-{month_num:02d}-{dd}"

    if combined.startswith(side):
        away, home = side, combined[len(side):]
    elif combined.endswith(side):
        away, home = combined[: -len(side)], side
    else:
        return None
    return official_date, away, home, side


def find_game_for_ticker(store: MlbDataStore, ticker: str):
    """Matches a Kalshi ticker to the real schedule row in mlb_data.db, via
    the ticker's own embedded date (see module docstring for why this is
    reliable and close_time is not) plus its two team codes -- an exact
    lookup, no fuzzy window. Returns None if the ticker doesn't parse, the
    team abbreviations aren't known, or no matching game exists yet
    (e.g. the collector hasn't synced that far ahead)."""
    parsed = parse_ticker(ticker)
    if parsed is None:
        return None
    official_date, away_code, home_code, _side = parsed

    away_row = store.get_team_by_abbreviation(away_code)
    home_row = store.get_team_by_abbreviation(home_code)
    if away_row is None or home_row is None:
        return None

    candidates = store.find_games_by_teams(home_row["team_id"], away_row["team_id"], official_date, official_date)
    return candidates[0] if candidates else None


def discover_todays_mlb_tickers(client: KalshiClient) -> list[tuple[str, str]]:
    """Bulk-endpoint discovery ONLY -- ticker/title, never price fields
    (see module docstring for why). Returns (ticker, title) tuples for
    every open KXMLBGAME market."""
    page = list_open_markets(client, series_ticker="KXMLBGAME", status="open", max_pages=10, page_limit=200)
    return [(m.ticker, m.title) for m in page.markets]


def index_tickers_by_game(client: KalshiClient) -> dict[tuple[str, str, str], dict[str, str]]:
    """Reverse of find_game_for_ticker: scans every currently open KXMLBGAME
    market and groups both team-side tickers by (official_date, away_code,
    home_code). Used by the predictions endpoint (trade_bot/api.py) to look
    up the real tradeable ticker for a specific scheduled game -- there's no
    way to construct a ticker directly (the {hhmm} component is an internal
    Kalshi listing detail, not derivable from mlb_data.db), so this builds
    the mapping by discovery instead."""
    index: dict[tuple[str, str, str], dict[str, str]] = {}
    for ticker, _title in discover_todays_mlb_tickers(client):
        parsed = parse_ticker(ticker)
        if parsed is None:
            continue
        official_date, away_code, home_code, side_code = parsed
        key = (official_date, away_code, home_code)
        entry = index.setdefault(key, {})
        if side_code == away_code:
            entry["away_ticker"] = ticker
        elif side_code == home_code:
            entry["home_ticker"] = ticker
    return index


def build_mlb_watchlist(client: KalshiClient, store: MlbDataStore, pregame_window_hours: float = 36.0) -> list[str]:
    """Real tradeable KXMLBGAME tickers for games with first pitch still
    ahead, within the next `pregame_window_hours` -- pregame candidates
    only. Already-open positions on a game whose first pitch has since
    passed stay visible to the engine regardless (LiveExecutionEngine's own
    union-with-open-positions logic in run_once handles that; this
    watchlist doesn't need to). Confirms a real quote exists (via
    get_market(), the per-ticker detail endpoint) before including a
    ticker -- discovery alone isn't enough, since the bulk endpoint can
    list a ticker Kalshi hasn't actually opened trading on yet."""
    now = datetime.now(timezone.utc)
    cutoff = now + timedelta(hours=pregame_window_hours)

    tickers: list[str] = []
    for ticker, _title in discover_todays_mlb_tickers(client):
        game = find_game_for_ticker(store, ticker)
        if game is None:
            continue
        try:
            first_pitch = datetime.fromisoformat(game["game_date_utc"].replace("Z", "+00:00"))
        except (ValueError, TypeError):
            continue

        if not (now < first_pitch <= cutoff):
            continue

        market = get_market(client, ticker)
        if market.get("yes_bid_pct") is None or market.get("yes_ask_pct") is None:
            continue

        tickers.append(ticker)

    return tickers
