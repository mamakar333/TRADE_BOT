"""Data layer: fetches from KalshiClient and normalizes to percent.

Kalshi's current API quotes contract prices as dollar strings (e.g.
"0.6300" for a contract that costs 63 cents), which is numerically the
implied probability in percent once multiplied by 100 (a 63-cent YES
contract implies a 63% chance). Size/volume/open-interest fields come
back as "_fp" (floating-point share count) strings. All unit
conversion lives here so the UI layer only ever deals in plain
percentages and numbers, never raw dollar strings.

Note: older Kalshi API docs/examples describe integer cents fields
(yes_bid, yes_ask, last_price, volume, open_interest). Those are no
longer present on live responses -- verified against both /markets
and /markets/{ticker} on 2026-07-12. This module reads the current
"_dollars" / "_fp" fields instead.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .client import KalshiClient

PRICE_FIELDS = (
    "yes_bid",
    "yes_ask",
    "no_bid",
    "no_ask",
    "last_price",
    "previous_yes_bid",
    "previous_yes_ask",
    "previous_price",
)

COUNT_FIELDS = ("volume", "volume_24h", "open_interest", "liquidity")


def dollars_to_pct(dollars: str | float | None) -> float | None:
    """Kalshi dollar prices (0.01-0.99) map 1:1 to implied probability percent."""
    if dollars is None or dollars == "":
        return None
    return round(float(dollars) * 100, 2)


def fp_to_count(value: str | float | None) -> float | None:
    """Kalshi "_fp" fields are floating-point share counts."""
    if value is None or value == "":
        return None
    return float(value)


def normalize_market(raw: dict[str, Any]) -> dict[str, Any]:
    market = dict(raw)
    for field in PRICE_FIELDS:
        dollars_key = f"{field}_dollars"
        if dollars_key in market:
            market[f"{field}_pct"] = dollars_to_pct(market[dollars_key])
    for field in COUNT_FIELDS:
        fp_key = f"{field}_fp"
        if fp_key in market:
            market[field] = fp_to_count(market[fp_key])
    return market


def normalize_orderbook(raw: dict[str, Any]) -> dict[str, Any]:
    """`raw` is the `orderbook_fp` object: {"yes_dollars": [...], "no_dollars": [...]}."""
    book: dict[str, Any] = {}
    for side in ("yes", "no"):
        levels = raw.get(f"{side}_dollars") or []
        book[side] = [
            {"price_pct": dollars_to_pct(price), "quantity": fp_to_count(quantity)}
            for price, quantity in levels
        ]
    return book


@dataclass
class MarketSummary:
    ticker: str
    title: str
    yes_bid_pct: float | None
    yes_ask_pct: float | None
    last_price_pct: float | None
    volume: float
    volume_24h: float
    open_interest: float
    status: str
    close_time: str | None = None

    @classmethod
    def from_raw(cls, raw: dict[str, Any]) -> "MarketSummary":
        m = normalize_market(raw)
        return cls(
            ticker=m["ticker"],
            title=m.get("title", ""),
            yes_bid_pct=m.get("yes_bid_pct"),
            yes_ask_pct=m.get("yes_ask_pct"),
            last_price_pct=m.get("last_price_pct"),
            volume=m.get("volume") or 0,
            volume_24h=m.get("volume_24h") or 0,
            open_interest=m.get("open_interest") or 0,
            status=m.get("status", ""),
            close_time=m.get("close_time"),
        )


@dataclass
class MarketPage:
    markets: list[MarketSummary]
    truncated: bool  # True if more markets exist beyond max_pages (Kalshi has no text search,
    # so a keyword filter applied downstream may be missing matches outside this page set).


def list_open_markets(
    client: KalshiClient,
    series_ticker: str | None = None,
    event_ticker: str | None = None,
    status: str = "open",
    max_pages: int = 5,
    page_limit: int = 200,
) -> MarketPage:
    """Paginate GET /markets, filtered server-side by status/series/event.

    Kalshi's API has no free-text search param -- only series_ticker/
    event_ticker/tickers/status. With tens of thousands of open markets
    exchange-wide, an unfiltered or keyword-only fetch is necessarily a
    bounded slice; `series_ticker` is the only filter guaranteed to be
    exhaustive for its scope.
    """
    markets: list[MarketSummary] = []
    cursor: str | None = None

    for _ in range(max_pages):
        params: dict[str, Any] = {"status": status, "limit": page_limit}
        if series_ticker:
            params["series_ticker"] = series_ticker
        if event_ticker:
            params["event_ticker"] = event_ticker
        if cursor:
            params["cursor"] = cursor

        page = client.get("/markets", params=params, signed=False)
        markets.extend(MarketSummary.from_raw(m) for m in page.get("markets", []))

        cursor = page.get("cursor") or None
        if not cursor:
            break

    return MarketPage(markets=markets, truncated=cursor is not None)


def filter_by_keyword(markets: list[MarketSummary], keyword: str) -> list[MarketSummary]:
    if not keyword:
        return markets
    needle = keyword.strip().lower()
    return [
        m
        for m in markets
        if needle in m.title.lower() or needle in m.ticker.lower()
    ]


def get_market(client: KalshiClient, ticker: str) -> dict[str, Any]:
    raw = client.get(f"/markets/{ticker}", signed=False)
    return normalize_market(raw["market"])


def get_market_orderbook(client: KalshiClient, ticker: str) -> dict[str, Any]:
    raw = client.get(f"/markets/{ticker}/orderbook", signed=False)
    return normalize_orderbook(raw["orderbook_fp"])


def get_events(
    client: KalshiClient,
    series_ticker: str | None = None,
    status: str | None = None,
    limit: int = 200,
) -> list[dict[str, Any]]:
    params: dict[str, Any] = {"limit": limit}
    if series_ticker:
        params["series_ticker"] = series_ticker
    if status:
        params["status"] = status
    raw = client.get("/events", params=params, signed=False)
    return raw.get("events", [])


def get_series(client: KalshiClient, series_ticker: str) -> dict[str, Any]:
    raw = client.get(f"/series/{series_ticker}", signed=False)
    return raw.get("series", raw)


def get_portfolio_balance(client: KalshiClient) -> dict[str, Any]:
    """Signed endpoint used only to verify auth in test_connection.py."""
    return client.get("/portfolio/balance", signed=True)


@dataclass
class RealPosition:
    ticker: str
    side: str  # "YES" or "NO"
    quantity: int
    avg_entry_price_pct: float
    realized_pnl_dollars: float
    fees_paid_dollars: float


def get_real_positions(client: KalshiClient, ticker_prefixes: list[str] | None = None) -> list[RealPosition]:
    """Live (real-money) open positions from the exchange itself -- the only
    source of truth for what's actually held. `position_fp` is signed:
    positive = holding YES contracts, negative = holding NO contracts (see
    Kalshi's MarketPosition schema). `market_exposure_dollars` is the current
    cost basis of that position, so dividing by the (absolute) contract count
    recovers the average entry price. Positions are page-cursored; this
    fetches all of them since a live account should only ever hold a small,
    bounded number of positions in the scope this bot manages.

    `ticker_prefixes`, if given, filters to only tickers starting with one of
    them -- used by the live engine to make certain it only ever reasons
    about (and later, manages) positions within its own declared scope (e.g.
    ["KXBTC15M-", "KXETHD-", ...]), never anything else that might exist on
    the user's real account.
    """
    positions: list[RealPosition] = []
    cursor: str | None = None
    while True:
        params: dict[str, Any] = {"limit": 200, "count_filter": "position"}
        if cursor:
            params["cursor"] = cursor
        page = client.get("/portfolio/positions", params=params, signed=True)
        for raw in page.get("market_positions", []):
            ticker = raw.get("ticker", "")
            if ticker_prefixes and not any(ticker.startswith(p) for p in ticker_prefixes):
                continue
            position_fp = float(raw.get("position_fp") or 0)
            if position_fp == 0:
                continue
            exposure = float(raw.get("market_exposure_dollars") or 0)
            quantity = abs(position_fp)
            avg_entry_price_pct = (exposure / quantity) * 100 if quantity else 0.0
            positions.append(
                RealPosition(
                    ticker=ticker,
                    side="YES" if position_fp > 0 else "NO",
                    # max(1, ...), not round(): position_fp can be a small
                    # nonzero fraction left over from a partial IOC close
                    # fill (e.g. 0.3 contracts). round() would floor that to
                    # 0 -- a "position" with 0 contracts that's still real
                    # exposure, gets reconciled into the ledger as open, and
                    # then crashes the exit order (count must be >= 1)
                    # confirmed live 2026-08-03. Rounding up ensures a real
                    # fractional leftover is always fully closeable.
                    quantity=max(1, round(quantity)),
                    avg_entry_price_pct=avg_entry_price_pct,
                    realized_pnl_dollars=float(raw.get("realized_pnl_dollars") or 0),
                    fees_paid_dollars=float(raw.get("fees_paid_dollars") or 0),
                )
            )
        cursor = page.get("cursor") or None
        if not cursor:
            break
    return positions


def get_available_balance_dollars(client: KalshiClient) -> float:
    raw = client.get("/portfolio/balance", signed=True)
    if "balance_dollars" in raw:
        return float(raw["balance_dollars"])
    return float(raw.get("balance", 0)) / 100
