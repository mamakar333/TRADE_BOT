"""Thin wrapper around MLB's free, official, unauthenticated Stats API
(statsapi.mlb.com) -- the same category of discovery the Coinbase BTC spot
feed was for crypto (trade_bot/btc_price_history.py): no cost, no API key,
no ToS risk, confirmed live with real current-season data. This is the data
source for trade_bot/mlb_data.py's collector.

Every function here is best-effort: catches network/parse failures and
returns None (never raises), same contract as
btc_price_history.fetch_btc_price_usd -- a bad response here must never take
down the collector loop.
"""
from __future__ import annotations

import logging
import time
from typing import Any

import httpx

logger = logging.getLogger(__name__)

BASE_URL = "https://statsapi.mlb.com/api/v1"
MLB_SPORT_ID = 1

# A real browser-like Accept/User-Agent pair, since some requests without
# one get rejected. Note this does NOT fix everything: /standings with a
# historical `date` param still reliably 406s from this project's deploy
# server regardless of headers (confirmed directly) -- see mlb_data.py's
# module docstring for why that endpoint shape is avoided entirely rather
# than worked around.
_HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; trade-bot/1.0)",
    "Accept": "application/json",
}


_MAX_RETRIES = 4
_RETRY_BACKOFF_SECONDS = 3.0


def _get(client: httpx.Client, path: str, params: dict[str, Any]) -> dict[str, Any] | None:
    """406 from this API has been observed in bursts that look like
    transient rate-limiting from this project's deploy server's IP
    (confirmed: a large burst of calls during testing made even a
    previously-fine endpoint start 406ing, then it recovered) -- retried
    with backoff a few times before giving up, rather than treated as an
    immediate hard failure like a real 4xx would be elsewhere in this repo."""
    last_error: Exception | None = None
    for attempt in range(_MAX_RETRIES):
        try:
            resp = client.get(f"{BASE_URL}{path}", params=params, headers=_HEADERS)
            resp.raise_for_status()
            return resp.json()
        except (httpx.HTTPError, ValueError) as e:
            last_error = e
            is_406 = isinstance(e, httpx.HTTPStatusError) and e.response.status_code == 406
            if not is_406 or attempt == _MAX_RETRIES - 1:
                break
            time.sleep(_RETRY_BACKOFF_SECONDS * (attempt + 1))
    logger.warning("MLB Stats API request failed (%s %s) after retries: %s", path, params, last_error)
    return None


def fetch_schedule(client: httpx.Client, date: str) -> list[dict[str, Any]]:
    """All MLB games scheduled/played on `date` (YYYY-MM-DD), with probable
    pitchers hydrated. Returns [] on any failure or if there are no games
    that day -- both are normal, not errors."""
    data = _get(
        client, "/schedule",
        {"sportId": MLB_SPORT_ID, "date": date, "hydrate": "probablePitcher,team"},
    )
    if not data:
        return []
    dates = data.get("dates", [])
    return dates[0].get("games", []) if dates else []


def fetch_teams(client: httpx.Client) -> list[dict[str, Any]]:
    data = _get(client, "/teams", {"sportId": MLB_SPORT_ID})
    return data.get("teams", []) if data else []


def fetch_pitcher_stats_byrange(
    client: httpx.Client, pitcher_id: int, start_date: str, end_date: str
) -> dict[str, Any] | None:
    """Real, accurate pitching stats for `pitcher_id` accumulated strictly
    within [start_date, end_date] (both YYYY-MM-DD, converted to the API's
    own MM/DD/YYYY format here) -- confirmed working directly (distinct
    from the simpler `stats=season` endpoint, which would leak the
    pitcher's FULL-season numbers into any earlier as-of query). Returns
    None on failure or if the pitcher has no recorded innings in the
    range (e.g. wasn't in the majors yet)."""
    def _mmddyyyy(d: str) -> str:
        y, m, dd = d.split("-")
        return f"{m}/{dd}/{y}"

    data = _get(
        client, f"/people/{pitcher_id}/stats",
        {
            "stats": "byDateRange", "group": "pitching",
            "startDate": _mmddyyyy(start_date), "endDate": _mmddyyyy(end_date),
        },
    )
    if not data:
        return None
    stats_list = data.get("stats", [])
    if not stats_list:
        return None
    splits = stats_list[0].get("splits", [])
    if not splits:
        return None
    return splits[0].get("stat")
