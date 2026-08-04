"""Kalshi trading bot dashboard: live market browser + simulated paper trading.

    uv run streamlit run app.py
"""
from __future__ import annotations

import json
import os
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

# Must be set before pandas/pyarrow import. On this stack (pyarrow's Arrow C++
# core + jemalloc), Streamlit's thread-per-rerun model reliably segfaults the
# *second* script rerun -- confirmed via a minimal repro (bare pyarrow.Table()
# across two AppTest reruns crashes; forcing Arrow's allocator to the system
# malloc instead of jemalloc fixes it). Verified 2026-07-12.
os.environ.setdefault("ARROW_DEFAULT_MEMORY_POOL", "system")

import altair as alt
import pandas as pd
import streamlit as st

import run_live_trading
from trade_bot import bot_control, paper_bot_control
from trade_bot.adaptive import PerformanceGovernor
from trade_bot.categories import MARKET_CATEGORIES, is_combo_market, list_all_curated_markets, list_markets_for_category
from trade_bot.client import KalshiAPIError, KalshiClient
from trade_bot.data import (
    CashFlowSummary,
    MarketSummary,
    filter_by_keyword,
    get_available_balance_dollars,
    get_cash_flow_summary,
    get_market,
    get_market_orderbook,
    get_real_positions,
    list_open_markets,
)
from trade_bot.engine import DEFAULT_LOG_PATH
from trade_bot.fees import taker_fee_dollars
from trade_bot.live_engine import DEFAULT_LOG_PATH as LIVE_LOG_PATH
from trade_bot.live_ledger import LiveLedger, LiveTrade
from trade_bot.online_learner import OnlineLogisticLearner
from trade_bot.portfolio import PaperPortfolio, Position
from trade_bot.strategy import MANUAL_STRATEGY_NAME
from trade_bot.watchlist import build_watchlist

PNL_GOOD = "#0ca30c"
PNL_BAD = "#d03b3b"

REFRESH_SECONDS = 15
LIVE_REFRESH_SECONDS = 10
CARDS_SHOWN = 60

st.set_page_config(page_title="Kalshi Trading Bot", layout="wide")


@st.cache_resource
def get_client() -> KalshiClient:
    return KalshiClient()


@st.cache_resource
def get_portfolio() -> PaperPortfolio:
    return PaperPortfolio()


@st.cache_resource
def get_live_ledger() -> LiveLedger:
    return LiveLedger()


CATEGORY_CACHE_TTL = 45  # seconds


@st.cache_data(ttl=CATEGORY_CACHE_TTL, show_spinner="Loading markets…")
def _cached_all_curated_markets(_client: KalshiClient, status: str) -> list[MarketSummary]:
    # Aggregates ~30 series (900+ markets, 12+ seconds) -- far too heavy to redo
    # on every REFRESH_SECONDS fragment tick, so this is cached separately on its
    # own, longer TTL. _client is underscore-prefixed so Streamlit doesn't try
    # (and fail) to hash an httpx-backed client object as a cache key.
    return list_all_curated_markets(_client, status=status)


@st.cache_data(ttl=CATEGORY_CACHE_TTL, show_spinner="Loading markets…")
def _cached_category_markets(_client: KalshiClient, category: str, status: str) -> list[MarketSummary]:
    return list_markets_for_category(_client, category, status=status)


@st.cache_data(ttl=CATEGORY_CACHE_TTL, show_spinner="Loading bot watchlist…")
def _cached_bot_watchlist(_client: KalshiClient) -> list[str]:
    # build_watchlist() aggregates every sport category + rotating crypto series
    # (~10s+, dozens of API calls) -- called from three different places in this
    # dashboard (market badges, bot status table, manual-trade picker), so it's
    # cached once here rather than paying that cost three times per render.
    return build_watchlist(_client)


CASH_FLOW_CACHE_TTL = 300  # seconds -- deposit/withdrawal history barely changes


@st.cache_data(ttl=CASH_FLOW_CACHE_TTL, show_spinner=False)
def _cached_cash_flow(_client: KalshiClient) -> CashFlowSummary:
    return get_cash_flow_summary(_client)


# ---------------------------------------------------------------------------
# shared helpers
# ---------------------------------------------------------------------------


def _pct_str(pct: float | None) -> str:
    return f"{pct:.0f}%" if pct is not None else "—"


def _mark_price(client: KalshiClient, position: Position) -> float | None:
    try:
        m = get_market(client, position.ticker)
    except KalshiAPIError:
        return None
    return m.get("yes_bid_pct") if position.side == "YES" else m.get("no_bid_pct")


def _position_title(client: KalshiClient, ticker: str) -> str:
    try:
        m = get_market(client, ticker)
    except KalshiAPIError:
        return ticker
    return m.get("title") or ticker


def _build_positions_view(client: KalshiClient, positions: list[Position]) -> tuple[list[dict], float]:
    """Mark-to-market each open position against its current live bid.
    Returns (rows with title/mark/unrealized P&L, total current value of all open positions)."""
    rows = []
    positions_value = 0.0
    for p in positions:
        mark = _mark_price(client, p)
        unrealized = p.quantity * (mark - p.entry_price_pct) / 100 if mark is not None else None
        value = p.quantity * (mark if mark is not None else p.entry_price_pct) / 100
        positions_value += value
        rows.append(
            {
                "ticker": p.ticker,
                "title": _position_title(client, p.ticker),
                "side": p.side,
                "quantity": p.quantity,
                "cost_dollars": p.quantity * p.entry_price_pct / 100,
                "entry_pct": p.entry_price_pct,
                "current_pct": mark,
                "unrealized_pnl": unrealized,
                "opened_at": p.opened_at,
                "strategy_name": p.strategy_name,
            }
        )
    return rows, positions_value


def position_card(row: dict) -> None:
    pnl = row["unrealized_pnl"]
    good = pnl is not None and pnl >= 0
    color = PNL_GOOD if good else (PNL_BAD if pnl is not None else "#898781")
    pnl_str = f"{'+' if good else ''}${pnl:.2f}" if pnl is not None else "n/a"
    status_label = "WINNING" if good else ("LOSING" if pnl is not None else "PENDING")

    with st.container(border=True):
        title = row["title"] if len(row["title"]) <= 80 else row["title"][:80] + "…"
        top = st.columns([4, 1])
        top[0].markdown(f"**{title}**")
        top[1].markdown(
            f"<span style='background:{color}; color:white; padding:2px 10px; border-radius:10px; "
            f"font-size:0.75rem; font-weight:700; float:right'>{status_label}</span>",
            unsafe_allow_html=True,
        )
        st.caption(f"{row['ticker']} · {row['strategy_name']}")

        side_emoji = "🟢" if row["side"] == "YES" else "🔴"
        st.markdown(
            f"{side_emoji} **{row['side']} — ${row['cost_dollars']:.0f}** &nbsp;·&nbsp; "
            f"entry {row['entry_pct']:.0f}% → current {row['current_pct']:.0f}%"
            if row["current_pct"] is not None
            else f"{side_emoji} **{row['side']} — ${row['cost_dollars']:.0f}** &nbsp;·&nbsp; entry {row['entry_pct']:.0f}%"
        )
        if row["current_pct"] is not None:
            st.progress(row["current_pct"] / 100, text=f"{row['side']} now trading at {row['current_pct']:.0f}%")

        st.markdown(
            f"<span style='color:{color}; font-size:1.4rem; font-weight:700'>"
            f"{'▲' if good else '▼'} {pnl_str}</span> unrealized",
            unsafe_allow_html=True,
        )


def _closed_trades_df(trades) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "Ticker": t.ticker,
                "Side": t.side,
                "Amount ($)": t.quantity * t.entry_price_pct / 100,
                "Entry %": t.entry_price_pct,
                "Exit %": t.exit_price_pct,
                "Return %": (t.exit_price_pct - t.entry_price_pct) / t.entry_price_pct * 100,
                "Fees": t.entry_fee + t.exit_fee,
                "Realized P&L": t.realized_pnl,
                "Opened": t.opened_at,
                "Closed": t.closed_at,
                "Reason": t.close_reason,
            }
            for t in trades
        ]
    )


def _style_pnl_column(df: pd.DataFrame, column: str):
    """Green text for a winning row, red for a losing one -- the sign is still
    the primary signal (always visible in the number itself), color is a bonus."""
    return df.style.map(
        lambda v: f"color: {PNL_GOOD}; font-weight: 600" if v >= 0 else f"color: {PNL_BAD}; font-weight: 600",
        subset=[column],
    )


def _read_recent_decisions(limit: int = 300, path: Path = DEFAULT_LOG_PATH) -> list[dict]:
    if not path.exists():
        return []
    lines = path.read_text().splitlines()[-limit:]
    out = []
    for line in reversed(lines):
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return out


def _bot_last_seen(path: Path = DEFAULT_LOG_PATH) -> datetime | None:
    decisions = _read_recent_decisions(limit=1, path=path)
    if not decisions:
        return None
    try:
        return datetime.fromisoformat(decisions[0]["timestamp"])
    except (KeyError, ValueError):
        return None


def _place_manual_trade(client: KalshiClient, portfolio: PaperPortfolio, ticker: str, side: str, dollars: float):
    try:
        m = get_market(client, ticker)
    except KalshiAPIError as e:
        return False, f"Couldn't fetch {ticker}: {e}"
    price = m.get("yes_ask_pct") if side == "YES" else m.get("no_ask_pct")
    if price is None:
        return False, f"No {side} ask available for {ticker} right now."
    quantity = max(1, round(dollars / (price / 100)))
    fee = taker_fee_dollars(quantity, price)
    try:
        portfolio.open_position(ticker, side, quantity, price, fee, MANUAL_STRATEGY_NAME)
    except ValueError as e:
        return False, str(e)
    actual_cost = quantity * price / 100
    return True, f"Bought {quantity} {side} contracts (${actual_cost:.2f}) @ {price:.0f}% (fee ${fee:.4f})"


def _close_manual_position(client: KalshiClient, portfolio: PaperPortfolio, position: Position):
    try:
        m = get_market(client, position.ticker)
    except KalshiAPIError as e:
        return False, f"Couldn't fetch {position.ticker}: {e}"
    price = m.get("yes_bid_pct") if position.side == "YES" else m.get("no_bid_pct")
    if price is None:
        return False, "No bid available to close against right now."
    fee = taker_fee_dollars(position.quantity, price)
    trade = portfolio.close_position(position.ticker, price, fee, close_reason="closed manually")
    return True, f"Closed for ${trade.realized_pnl:+.2f}"


# ---------------------------------------------------------------------------
# Markets tab: card grid + click-through detail page (query-param routed)
# ---------------------------------------------------------------------------


def market_card(m: MarketSummary, bot_tickers: set[str], positions_by_ticker: dict[str, Position]) -> None:
    with st.container(border=True):
        title = m.title if len(m.title) <= 68 else m.title[:68] + "…"
        st.markdown(f"**{title}**")
        st.caption(m.ticker)

        badges = []
        if m.ticker in bot_tickers:
            badges.append("🤖 bot watchlist")
        pos = positions_by_ticker.get(m.ticker)
        if pos:
            badges.append(f"📌 {pos.strategy_name} holds {pos.side} — ${pos.quantity * pos.entry_price_pct / 100:.0f}")
        if badges:
            st.caption(" · ".join(badges))

        yes_pct = m.last_price_pct if m.last_price_pct is not None else m.yes_ask_pct
        st.progress((yes_pct or 0) / 100, text=f"YES {_pct_str(yes_pct)}")

        c1, c2 = st.columns(2)
        c1.metric("Volume", f"{m.volume:,.0f}")
        c2.metric("Status", m.status)

        if st.button("View market →", key=f"card_{m.ticker}", width="stretch"):
            st.query_params["ticker"] = m.ticker
            st.rerun()


@st.fragment(run_every=REFRESH_SECONDS)
def market_grid(category: str, keyword: str, series_ticker: str, status: str, show_combos: bool) -> None:
    client = get_client()
    portfolio = get_portfolio()

    truncated = False
    scanned_count = 0
    default_view = category == "All" and not keyword and not series_ticker
    if category != "All" or default_view:
        # Category browsing (including the "All curated categories" default) hits
        # a curated set of series, each exhaustive and server-side filtered -- no
        # keyword-search limitation here. The default view deliberately isn't a
        # raw unfiltered exchange scan: that's dominated by high-volume combo/
        # parlay markets (see is_combo_market), so filtering those out of it can
        # leave nothing to show.
        try:
            markets = (
                _cached_category_markets(client, category, status=status)
                if category != "All"
                else _cached_all_curated_markets(client, status=status)
            )
        except KalshiAPIError as e:
            st.error(f"Failed to fetch markets: {e}")
            return
        scanned_count = len(markets)
        filtered = filter_by_keyword(markets, keyword)
    else:
        # Kalshi has no free-text search API. series_ticker is filtered server-side and
        # exhaustive; a keyword-only search only scans a bounded page of the ~30k+ open
        # markets exchange-wide, so we scan deeper for it and surface truncation honestly.
        max_pages = 5 if series_ticker else (15 if keyword else 5)
        try:
            page = list_open_markets(client, series_ticker=series_ticker or None, status=status, max_pages=max_pages)
        except KalshiAPIError as e:
            st.error(f"Failed to fetch markets: {e}")
            return
        scanned_count = len(page.markets)
        truncated = page.truncated
        filtered = filter_by_keyword(page.markets, keyword)

    if not show_combos:
        filtered = [m for m in filtered if not is_combo_market(m.ticker)]
    filtered = sorted(filtered, key=lambda m: m.volume, reverse=True)

    refresh_note = (
        f"prices refresh every ~{CATEGORY_CACHE_TTL}s (category browse is a heavy multi-series fetch)"
        if default_view or category != "All"
        else f"refreshes every {REFRESH_SECONDS}s"
    )
    st.caption(
        f"{len(filtered)} of {scanned_count} scanned markets match · showing top "
        f"{min(CARDS_SHOWN, len(filtered))} by volume · {refresh_note}"
    )
    if truncated:
        scope = f"series {series_ticker!r}" if series_ticker else "the whole exchange"
        st.caption(
            f"⚠️ More open markets exist beyond the {scanned_count} scanned for {scope} "
            "— Kalshi's API has no text search, so a keyword match outside this batch won't "
            "show up. Use Series ticker or a Category for guaranteed full coverage."
        )

    if not filtered:
        st.info("No markets match the current filters.")
        return

    try:
        bot_tickers = set(_cached_bot_watchlist(client))
    except KalshiAPIError:
        bot_tickers = set()
    positions_by_ticker = {p.ticker: p for p in portfolio.get_all_open_positions()}

    shown = filtered[:CARDS_SHOWN]
    # A stable, explicit key forces Streamlit to treat this as the *same*
    # container across reruns and fully replace its contents -- without it, a
    # fragment rerun that produces fewer cards than last time (e.g. filtering
    # 60 -> 1) can leave the previous run's extra cards visually stranded.
    with st.container(key="market_grid_container"):
        cols = st.columns(3)
        for i, m in enumerate(shown):
            with cols[i % 3]:
                market_card(m, bot_tickers, positions_by_ticker)


@st.fragment(run_every=REFRESH_SECONDS)
def market_detail_page(ticker: str) -> None:
    client = get_client()
    portfolio = get_portfolio()

    if st.button("← Back to Markets"):
        st.query_params.clear()
        st.rerun()

    try:
        market = get_market(client, ticker)
        orderbook = get_market_orderbook(client, ticker)
    except KalshiAPIError as e:
        st.error(f"Failed to fetch {ticker}: {e}")
        return

    st.title(market.get("title", ticker))
    st.caption(ticker)

    yes_pct = market.get("last_price_pct") or market.get("yes_ask_pct") or market.get("yes_bid_pct")
    no_pct = None if yes_pct is None else 100 - yes_pct
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Implied YES", _pct_str(yes_pct))
    col2.metric("Implied NO", _pct_str(no_pct))
    col3.metric("Volume", f"{market.get('volume', 0):,.0f}")
    col4.metric("Status", market.get("status", "n/a"))

    position = portfolio.get_open_position(ticker)
    if position:
        mark = _mark_price(client, position)
        unrealized = position.quantity * (mark - position.entry_price_pct) / 100 if mark is not None else None
        st.info(
            f"📌 **{position.strategy_name}** holds **{position.side} — "
            f"${position.quantity * position.entry_price_pct / 100:.0f}** @ entry {position.entry_price_pct:.0f}%"
            + (f" · mark {mark:.0f}% · unrealized P&L ${unrealized:+.2f}" if mark is not None else "")
        )

    history = portfolio.get_price_history(ticker, limit=500)
    if len(history) >= 2:
        hist_df = pd.DataFrame(history)
        hist_df["timestamp"] = pd.to_datetime(hist_df["timestamp"])
        chart = (
            alt.Chart(hist_df)
            .mark_line(color="#2a78d6", strokeWidth=2)
            .encode(
                x=alt.X("timestamp:T", title=None, axis=alt.Axis(format="%H:%M")),
                y=alt.Y("last_price_pct:Q", title="Last price (%)", scale=alt.Scale(domain=[0, 100])),
                tooltip=[
                    alt.Tooltip("timestamp:T", title="Time"),
                    alt.Tooltip("last_price_pct:Q", title="Price %"),
                ],
            )
        )
        st.altair_chart(chart.properties(height=250), width="stretch")
    else:
        st.caption(
            "No price history yet — this market isn't on the bot's watchlist, so history hasn't "
            "been recorded. It'll start accumulating once the bot (or you) trade it."
        )

    st.divider()
    book_col1, book_col2 = st.columns(2)
    for col, side in ((book_col1, "yes"), (book_col2, "no")):
        with col:
            st.write(f"{side.upper()} order book")
            levels = orderbook.get(side, [])
            if levels:
                st.dataframe(pd.DataFrame(levels), hide_index=True, width="stretch")
            else:
                st.caption("No resting orders on this side.")

    st.divider()
    st.subheader("Practice trade")
    if position:
        st.caption("You already have an open position in this market (one position per market, bot or manual).")
        if st.button("Close position", key=f"close_{ticker}"):
            ok, msg = _close_manual_position(client, portfolio, position)
            (st.success if ok else st.error)(msg)
            st.rerun()
    else:
        with st.form(key=f"trade_form_{ticker}"):
            side = st.radio("Side", ["YES", "NO"], horizontal=True)
            dollars = st.number_input("Amount to bet ($)", min_value=1.0, max_value=1000.0, value=100.0, step=10.0)
            ask = market.get("yes_ask_pct") if side == "YES" else market.get("no_ask_pct")
            est_qty = max(1, round(dollars / (ask / 100))) if ask else 0
            st.caption(f"Current {side} ask: {_pct_str(ask)} · ≈{est_qty} contracts + fees")
            submitted = st.form_submit_button("Buy (paper)", type="primary")
        if submitted:
            ok, msg = _place_manual_trade(client, portfolio, ticker, side, float(dollars))
            (st.success if ok else st.error)(msg)
            st.rerun()


def markets_tab() -> None:
    ticker = st.query_params.get("ticker")
    if ticker:
        market_detail_page(ticker)
        return

    with st.sidebar:
        st.header("Filters")
        category = st.selectbox("Category", options=["All"] + list(MARKET_CATEGORIES.keys()), index=0)
        keyword = st.text_input("Keyword (matches title or ticker)", value="")
        series_ticker = st.text_input("Series ticker", value="", disabled=category != "All")
        status = st.selectbox("Status", options=["open", "closed", "settled"], index=0)
        show_combos = st.checkbox(
            "Include combo/multi-team markets", value=False,
            help="Multivariate 'combo' markets bundle many outcomes into one confusing title "
                 "(e.g. \"yes Boston,yes Chicago,yes...\"). Hidden by default.",
        )
    market_grid(category, keyword, series_ticker, status, show_combos)


# ---------------------------------------------------------------------------
# Paper Trading tab
# ---------------------------------------------------------------------------


def _paper_bot_control_panel() -> None:
    """Manual start/stop for the paper bot -- added 2026-08-04 per explicit
    request ("a start button for paper bot for somereason if it stops").
    No confirmation step needed (unlike the live bot's): nothing here can
    ever place a real order, see trade_bot/engine.py's SIMULATION_MODE
    guard. Deliberately no auto-restart-on-crash either, matching what was
    actually asked for -- manual control, not a second watchdog."""
    running, pid = paper_bot_control.is_running()
    c1, c2, c3 = st.columns([2, 1, 1])
    with c1:
        if running:
            started = paper_bot_control.started_at()
            uptime = ""
            if started:
                secs = time.time() - started
                uptime = f" · up {secs / 60:.0f}min" if secs < 3600 else f" · up {secs / 3600:.1f}h"
            st.success(f"🟢 Paper bot process RUNNING (PID {pid}){uptime}")
        else:
            st.error("🔴 Paper bot process STOPPED")
    with c2:
        if st.button("▶️ Start", width="stretch", disabled=running, key="paper_start_button"):
            ok, msg = paper_bot_control.start()
            (st.success if ok else st.error)(msg)
            time.sleep(0.5)
            st.rerun()
    with c3:
        if st.button("🛑 Stop", width="stretch", disabled=not running, key="paper_stop_button"):
            ok, msg = paper_bot_control.stop()
            (st.success if ok else st.warning)(msg)
            time.sleep(0.5)
            st.rerun()


def bot_status_section(client: KalshiClient, portfolio: PaperPortfolio) -> None:
    st.subheader("Bot Status")

    _paper_bot_control_panel()

    last_seen = _bot_last_seen()
    if last_seen is None:
        st.warning("No decisions logged yet — the execution loop (`run_paper_trading.py`) hasn't run.")
    else:
        age_seconds = (datetime.now(timezone.utc) - last_seen).total_seconds()
        if age_seconds < REFRESH_SECONDS * 6:
            st.success(f"🟢 Loop is live — last cycle {age_seconds:.0f}s ago")
        else:
            st.warning(f"🟡 Loop looks stopped — last cycle was {age_seconds / 60:.1f} min ago")

    reset_events = portfolio.get_reset_events()
    if reset_events:
        st.caption(f"⚠️ Account auto-reset to starting balance {len(reset_events)}× after hitting $0 (most recent: {reset_events[-1]['timestamp']})")

    try:
        watchlist = _cached_bot_watchlist(client)
    except KalshiAPIError as e:
        st.error(f"Couldn't resolve current watchlist: {e}")
        watchlist = []

    if watchlist:
        rows = []
        positions_by_ticker = {p.ticker: p for p in portfolio.get_all_open_positions()}
        for t in watchlist:
            try:
                m = get_market(client, t)
            except KalshiAPIError:
                continue
            pos = positions_by_ticker.get(t)
            mark = (m.get("yes_bid_pct") if pos and pos.side == "YES" else m.get("no_bid_pct")) if pos else None
            unrealized = pos.quantity * (mark - pos.entry_price_pct) / 100 if pos and mark is not None else None
            rows.append(
                {
                    "Ticker": t,
                    "Title": (m.get("title") or "")[:60],
                    "YES %": m.get("last_price_pct") or m.get("yes_ask_pct"),
                    "Status": m.get("status"),
                    "Bot Position": f"{pos.side} — ${pos.quantity * pos.entry_price_pct / 100:.0f}" if pos else "—",
                    "Live P&L": unrealized,
                }
            )
        watchlist_df = pd.DataFrame(rows)
        styled = watchlist_df.style.map(
            lambda v: (f"color: {PNL_GOOD}; font-weight: 600" if v >= 0 else f"color: {PNL_BAD}; font-weight: 600")
            if pd.notna(v)
            else "",
            subset=["Live P&L"],
        )
        st.dataframe(styled, hide_index=True, width="stretch")


def bot_activity_feed() -> None:
    st.subheader("Recent Bot Activity")
    show_holds = st.checkbox("Show HOLD decisions too", value=False)

    decisions = _read_recent_decisions(limit=300)
    decisions = [d for d in decisions if d.get("event") != "decision" or show_holds or d.get("signal") != "HOLD"]
    if not decisions:
        st.info("No activity logged yet.")
        return

    rows = []
    for d in decisions[:100]:
        event = d.get("event", "")
        if event == "decision":
            label = f"{d.get('signal')}"
        elif event == "fill":
            label = f"FILLED {d.get('action', '').upper()} {d.get('side', '')}"
        elif event == "risk_blocked":
            label = "RISK BLOCKED"
        elif event == "account_reset":
            label = "ACCOUNT RESET"
        else:
            label = event.upper()
        rows.append(
            {
                "Time": d.get("timestamp", "")[:19].replace("T", " "),
                "Event": label,
                "Ticker": d.get("ticker", ""),
                "Reason": d.get("reason", ""),
            }
        )
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch", height=350)


def manual_trading_section(client: KalshiClient, portfolio: PaperPortfolio) -> None:
    st.subheader("Manual Trading (practice)")

    try:
        watchlist = _cached_bot_watchlist(client)
    except KalshiAPIError:
        watchlist = []

    with st.form("manual_trade_form"):
        c1, c2, c3 = st.columns([2, 1, 1])
        ticker = c1.selectbox("Market (bot watchlist)", options=watchlist) if watchlist else c1.text_input("Ticker")
        side = c2.radio("Side", ["YES", "NO"], horizontal=True)
        dollars = c3.number_input("Amount ($)", min_value=1.0, max_value=1000.0, value=100.0, step=10.0)
        submitted = st.form_submit_button("Buy (paper)", type="primary")
    if submitted and ticker:
        ok, msg = _place_manual_trade(client, portfolio, ticker, side, float(dollars))
        (st.success if ok else st.error)(msg)
        st.rerun()

    manual_positions = [p for p in portfolio.get_all_open_positions() if p.strategy_name == MANUAL_STRATEGY_NAME]
    if manual_positions:
        st.caption("Your open manual positions:")
        for p in manual_positions:
            mark = _mark_price(client, p)
            unrealized = p.quantity * (mark - p.entry_price_pct) / 100 if mark is not None else None
            cols = st.columns([3, 1, 1, 1])
            cols[0].write(f"{p.ticker} — {p.side} ${p.quantity * p.entry_price_pct / 100:.0f} @ {p.entry_price_pct:.0f}%")
            cols[1].write(f"mark {mark:.0f}%" if mark is not None else "n/a")
            cols[2].write(f"P&L ${unrealized:+.2f}" if unrealized is not None else "n/a")
            if cols[3].button("Close", key=f"manual_close_{p.ticker}"):
                ok, msg = _close_manual_position(client, portfolio, p)
                (st.success if ok else st.error)(msg)
                st.rerun()


@st.fragment(run_every=REFRESH_SECONDS)
def paper_trading_tab() -> None:
    client = get_client()
    portfolio = get_portfolio()

    cash = portfolio.get_cash_balance()
    starting = portfolio.get_starting_balance()
    open_positions = portfolio.get_all_open_positions()
    closed_trades = portfolio.get_closed_trades()

    position_rows, positions_value = _build_positions_view(client, open_positions)
    total_equity = cash + positions_value
    total_return_pct = (total_equity - starting) / starting * 100 if starting else 0.0
    realized_total = sum(t.realized_pnl for t in closed_trades)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric(
        "Total Equity", f"${total_equity:,.2f}", f"{total_equity - starting:+,.2f} ({total_return_pct:+.1f}%)"
    )
    col2.metric("Cash Balance", f"${cash:,.2f}")
    col3.metric("Open Positions", f"{len(open_positions)}")
    col4.metric("Realized P&L (all-time)", f"${realized_total:+,.2f}")

    st.divider()
    bot_status_section(client, portfolio)

    st.divider()
    bot_activity_feed()

    st.divider()
    manual_trading_section(client, portfolio)

    st.divider()
    st.subheader("Equity Curve")
    equity_curve = portfolio.get_equity_curve()
    if len(equity_curve) < 2:
        st.info("Not enough equity history yet — run the simulation loop to build up the curve.")
    else:
        curve_df = pd.DataFrame(equity_curve)
        curve_df["timestamp"] = pd.to_datetime(curve_df["timestamp"])
        line = (
            alt.Chart(curve_df)
            .mark_line(color="#2a78d6", strokeWidth=2)
            .encode(
                x=alt.X("timestamp:T", title=None, axis=alt.Axis(format="%b %d, %H:%M")),
                y=alt.Y("total_equity:Q", title="Equity ($)", scale=alt.Scale(zero=False)),
                tooltip=[
                    alt.Tooltip("timestamp:T", title="Time"),
                    alt.Tooltip("total_equity:Q", title="Equity", format="$.2f"),
                ],
            )
        )
        baseline = (
            alt.Chart(pd.DataFrame({"y": [starting]}))
            .mark_rule(color="#898781", strokeDash=[4, 4])
            .encode(y="y:Q")
        )
        st.altair_chart((line + baseline).properties(height=300), width="stretch")
        st.caption(f"Dashed line = starting balance (${starting:,.2f}) · resets to this if cash hits $0")

    bot_positions = [r for r in position_rows if r["strategy_name"] != MANUAL_STRATEGY_NAME]
    manual_positions = [r for r in position_rows if r["strategy_name"] == MANUAL_STRATEGY_NAME]
    bot_trades = [t for t in closed_trades if t.strategy_name != MANUAL_STRATEGY_NAME]
    manual_trades = [t for t in closed_trades if t.strategy_name == MANUAL_STRATEGY_NAME]

    st.divider()
    st.header(f"🤖 Bot Positions & Trades ({len(bot_positions)} open)")
    _render_position_group(bot_positions)
    _render_trades_group(bot_trades)

    st.divider()
    st.header(f"👤 Your Manual Positions & Trades ({len(manual_positions)} open)")
    _render_position_group(manual_positions)
    _render_trades_group(manual_trades)


def _render_position_group(rows: list[dict]) -> None:
    if not rows:
        st.info("No open positions.")
        return
    winning = sum(1 for r in rows if (r["unrealized_pnl"] or 0) >= 0)
    st.caption(f"{winning} winning / {len(rows) - winning} losing right now")
    cols = st.columns(2)
    for i, row in enumerate(rows):
        with cols[i % 2]:
            position_card(row)


def _render_trades_group(trades: list) -> None:
    st.subheader(f"Trade History ({len(trades)})")
    if not trades:
        st.info("No closed trades yet.")
        return
    wins = [t for t in trades if t.realized_pnl > 0]
    losses = [t for t in trades if t.realized_pnl <= 0]
    win_rate = len(wins) / len(trades) * 100
    avg_win = sum(t.realized_pnl for t in wins) / len(wins) if wins else 0.0
    avg_loss = sum(t.realized_pnl for t in losses) / len(losses) if losses else 0.0

    c1, c2, c3 = st.columns(3)
    c1.metric("Win Rate", f"{win_rate:.0f}%", f"{len(wins)}W / {len(losses)}L")
    c2.metric("Avg Win", f"${avg_win:,.2f}")
    c3.metric("Avg Loss", f"${avg_loss:,.2f}")

    trades_df = _closed_trades_df(trades)
    st.dataframe(_style_pnl_column(trades_df, "Realized P&L"), hide_index=True, width="stretch")


# ---------------------------------------------------------------------------
# Live Trading tab: real-money bot control + monitoring
# ---------------------------------------------------------------------------


def _live_trades_df(trades: list[LiveTrade]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "Ticker": t.ticker,
                "Side": t.side,
                "Qty": t.quantity,
                "Entry %": t.entry_price_pct,
                "Exit %": t.exit_price_pct,
                "Fees": (t.entry_fee or 0) + (t.exit_fee or 0),
                "Realized P&L": t.realized_pnl,
                "Opened": t.opened_at,
                "Closed": t.closed_at,
                "Reason": t.close_reason,
            }
            for t in trades
        ]
    )


def _live_kill_switch_panel() -> None:
    st.subheader("⚡ Bot Control")
    running, pid = bot_control.is_running()

    c1, c2, c3 = st.columns([2, 1, 2])
    with c1:
        if running:
            started = bot_control.started_at()
            uptime = ""
            if started:
                secs = time.time() - started
                uptime = f" · up {secs / 60:.0f}min" if secs < 3600 else f" · up {secs / 3600:.1f}h"
            st.success(f"🟢 RUNNING (PID {pid}){uptime}")
        else:
            st.error("🔴 STOPPED")

    with c2:
        if st.button("🛑 STOP", width="stretch", type="primary", disabled=not running):
            ok, msg = bot_control.stop()
            (st.success if ok else st.warning)(msg)
            time.sleep(0.5)
            st.rerun()

    with c3:
        with st.popover("▶️ Start Bot", width="stretch", disabled=running):
            st.warning("This places **real orders with real money** on your Kalshi account.")
            st.caption(
                f"Capital cap ${run_live_trading.RISK_LIMITS.max_capital_dollars:.0f} · "
                f"per-trade up to ${run_live_trading.RISK_LIMITS.max_position_size_dollars:.0f} "
                f"(or {run_live_trading.RISK_LIMITS.low_balance_fraction:.0%} of balance if under that) · "
                f"daily kill-switch ${run_live_trading.RISK_LIMITS.daily_stop_loss_dollars:.0f} · "
                f"{len(run_live_trading.ACTIVE_CRYPTO_SERIES_PREFIXES)} crypto series "
                f"({', '.join(p.rstrip('-') for p in run_live_trading.ACTIVE_CRYPTO_SERIES_PREFIXES)})"
            )
            confirmed = st.checkbox("I understand this places real orders with real money.", key="live_start_confirm")
            if st.button("Confirm & Start", type="primary", disabled=not confirmed, key="live_start_button"):
                ok, msg = bot_control.start()
                (st.success if ok else st.error)(msg)
                time.sleep(0.5)
                st.rerun()


def _live_summary_section(client: KalshiClient, ledger: LiveLedger) -> None:
    try:
        balance = get_available_balance_dollars(client)
    except KalshiAPIError as e:
        balance = None
        st.error(f"Couldn't fetch real balance: {e}")

    total_realized = ledger.get_total_realized_pnl()
    today_start = datetime.now(timezone.utc).date().isoformat() + "T00:00:00+00:00"
    today_pnl = ledger.get_realized_pnl_since(today_start)
    now = datetime.now(timezone.utc)
    pnl_24h = ledger.get_realized_pnl_since((now - timedelta(hours=24)).isoformat())
    pnl_7d = ledger.get_realized_pnl_since((now - timedelta(days=7)).isoformat())
    # Exchange, not the local ledger, is the source of truth for "how many
    # positions are open" -- a ledger row can go stale if its ticker rotated
    # off the watchlist before being reconciled (fixed at the root in
    # live_engine.py's run_once(), but this stays exchange-truth regardless
    # so the count is never wrong even for the one cycle before that runs).
    try:
        open_position_count = len(get_real_positions(client, ticker_prefixes=run_live_trading.ACTIVE_CRYPTO_SERIES_PREFIXES))
    except KalshiAPIError:
        open_position_count = len(ledger.get_all_open_trades())

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Real Balance", f"${balance:,.2f}" if balance is not None else "n/a")
    c2.metric("Today's P&L", f"${today_pnl:+,.2f}")
    c3.metric("Open Positions", f"{open_position_count}")
    c4.metric("Historical (all-time) P&L", f"${total_realized:+,.2f}")

    c5, c6 = st.columns(2)
    c5.metric("Last 24h P&L", f"${pnl_24h:+,.2f}")
    c6.metric("Last 7d P&L", f"${pnl_7d:+,.2f}")

    try:
        cash_flow = _cached_cash_flow(client)
        c7, c8, c9 = st.columns(3)
        c7.metric("Total Deposited (net)", f"${cash_flow.total_deposited_net_dollars:,.2f}")
        c8.metric("Deposit Fees Paid", f"${cash_flow.total_deposit_fees_dollars:,.2f}")
        c9.metric("Total Withdrawn", f"${cash_flow.total_withdrawn_dollars:,.2f}")
        st.caption(
            f"{cash_flow.deposit_count} deposits totaling ${cash_flow.total_deposited_gross_dollars:,.2f} gross "
            f"(${cash_flow.total_deposit_fees_dollars:,.2f} in fees) · trading P&L is separate from this, "
            "shown above"
        )
    except KalshiAPIError as e:
        st.caption(f"Couldn't fetch deposit history: {e}")

    limit = run_live_trading.RISK_LIMITS.daily_stop_loss_dollars
    enabled = run_live_trading.RISK_LIMITS.daily_kill_switch_enabled
    used_fraction = min(1.0, -today_pnl / limit) if today_pnl < 0 else 0.0
    st.progress(
        used_fraction,
        text=(
            f"${-today_pnl:.2f} of ${limit:.0f} daily loss reference used"
            if today_pnl < 0
            else f"${today_pnl:+.2f} today"
        )
        + ("" if enabled else " (kill-switch disabled — informational only, manual control via Stop)"),
    )
    if today_pnl <= -limit:
        if enabled:
            st.error("🛑 Daily loss kill-switch ENGAGED — no new entries until UTC midnight (open positions can still close).")
        else:
            st.warning(f"⚠️ Today's loss has passed the ${limit:.0f} reference figure — kill-switch is disabled, still trading. Stop manually if you want to pause.")

    governor = PerformanceGovernor(
        lambda name, limit: ledger.get_recent_realized_pnls_within_hours(name, limit, run_live_trading.GOVERNOR_LOOKBACK_HOURS),
        run_live_trading.GOVERNOR_CONFIG,
    )
    gd = governor.evaluate(run_live_trading.STRATEGY_NAME)
    if gd.multiplier <= 0:
        st.warning(f"🐢 **Adaptive governor: PAUSED** — {gd.reason}")
    elif gd.multiplier < 1.0:
        st.info(f"🐢 **Adaptive governor: SIZE HALVED** — {gd.reason}")
    else:
        st.caption(f"🐢 Adaptive governor — {gd.reason}")

    learner = OnlineLogisticLearner.from_json(ledger.load_learner_state())
    min_trades = run_live_trading.MIN_TRADES_FOR_ML_GATE
    if learner.n_updates < min_trades:
        st.caption(f"🧠 Online learner — {learner.n_updates}/{min_trades} learned trades so far, not gating yet")
    else:
        top_weights = sorted(learner.weights.items(), key=lambda kv: abs(kv[1]), reverse=True)[:3]
        weights_str = ", ".join(f"{k}={v:+.2f}" for k, v in top_weights)
        st.info(
            f"🧠 **Online learner active** — {learner.n_updates} learned trades, "
            f"gating entries below {run_live_trading.ML_GATE_THRESHOLD:.0%} predicted win probability. "
            f"Strongest weights: {weights_str}"
        )


def _live_position_section(client: KalshiClient) -> None:
    st.subheader("Current Positions")
    try:
        positions = get_real_positions(client, ticker_prefixes=run_live_trading.ACTIVE_CRYPTO_SERIES_PREFIXES)
    except KalshiAPIError as e:
        st.error(f"Couldn't fetch positions: {e}")
        return
    if not positions:
        st.info("No open positions right now.")
        return

    for p in positions:
        try:
            m = get_market(client, p.ticker)
        except KalshiAPIError:
            m = {}
        mark = m.get("yes_bid_pct") if p.side == "YES" else m.get("no_bid_pct")
        unrealized = p.quantity * (mark - p.avg_entry_price_pct) / 100 if mark is not None else None
        minutes_left = None
        close_time = m.get("close_time")
        if close_time:
            try:
                ct = datetime.fromisoformat(close_time.replace("Z", "+00:00"))
                minutes_left = (ct - datetime.now(timezone.utc)).total_seconds() / 60
            except ValueError:
                pass

        good = unrealized is not None and unrealized >= 0
        color = PNL_GOOD if good else (PNL_BAD if unrealized is not None else "#898781")
        status_label = "WINNING" if good else ("LOSING" if unrealized is not None else "PENDING")
        with st.container(border=True):
            top = st.columns([4, 1])
            top[0].markdown(f"**{m.get('title', p.ticker)}**")
            top[1].markdown(
                f"<span style='background:{color}; color:white; padding:2px 10px; border-radius:10px; "
                f"font-size:0.75rem; font-weight:700; float:right'>{status_label}</span>",
                unsafe_allow_html=True,
            )
            st.caption(
                f"{p.ticker} · crypto_technical"
                + (f" · {minutes_left:.1f}min to close" if minutes_left is not None else "")
            )
            side_emoji = "🟢" if p.side == "YES" else "🔴"
            cost = p.quantity * p.avg_entry_price_pct / 100
            st.markdown(
                f"{side_emoji} **{p.side} — {p.quantity} contracts (${cost:.2f})** &nbsp;·&nbsp; "
                f"entry {p.avg_entry_price_pct:.1f}%" + (f" → mark {mark:.1f}%" if mark is not None else "")
            )
            if mark is not None:
                st.progress(mark / 100, text=f"{p.side} now trading at {mark:.1f}%")
            if unrealized is not None:
                st.markdown(
                    f"<span style='color:{color}; font-size:1.4rem; font-weight:700'>"
                    f"{'▲' if good else '▼'} {'+' if good else ''}${unrealized:.2f}</span> unrealized",
                    unsafe_allow_html=True,
                )


def _live_activity_feed() -> None:
    st.subheader("Recent Bot Activity")
    show_holds = st.checkbox("Show HOLD decisions too", value=False, key="live_show_holds")

    all_decisions = _read_recent_decisions(limit=300, path=LIVE_LOG_PATH)
    decisions = [d for d in all_decisions if d.get("event") != "decision" or show_holds or d.get("signal") != "HOLD"]
    if not decisions:
        if all_decisions:
            st.info(
                f"The bot has logged {len(all_decisions)} recent decisions, all HOLD so far -- "
                "check \"Show HOLD decisions too\" above to see its reasoning."
            )
        else:
            st.info("No activity logged yet -- the bot hasn't run a cycle.")
        return

    event_labels = {
        "fill": lambda d: f"✅ FILLED {d.get('action', '').upper()} {d.get('side', '')}",
        "order_error": lambda d: "⚠️ ORDER ERROR",
        "order_not_filled": lambda d: "ORDER NOT FILLED (FOK)",
        "risk_blocked": lambda d: "⛔ RISK BLOCKED",
        "governor_paused": lambda d: "🐢 GOVERNOR PAUSED",
        "reconciled_untracked_position": lambda d: "🔄 RECONCILED (found untracked position)",
        "reconciled_missing_position": lambda d: "🔄 RECONCILED (position closed elsewhere)",
        "balance_check_failed": lambda d: "⚠️ BALANCE CHECK FAILED",
    }

    rows = []
    for d in decisions[:150]:
        event = d.get("event", "")
        label = event_labels.get(event, lambda d: d.get("signal", event.upper()))(d)
        rows.append(
            {
                "Time": d.get("timestamp", "")[:19].replace("T", " "),
                "Event": label,
                "Ticker": d.get("ticker", ""),
                "Reason": d.get("reason") or d.get("error") or d.get("note", ""),
            }
        )
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch", height=350)


def _live_trades_section(ledger: LiveLedger) -> None:
    st.subheader("Trade History")
    trades = ledger.get_closed_trades(limit=500)

    latest_id = trades[0].id if trades else None
    prev_id = st.session_state.get("_live_last_trade_id")
    if latest_id is not None and prev_id is not None and latest_id != prev_id:
        newest = trades[0]
        emoji = "🎉" if (newest.realized_pnl or 0) >= 0 else "💸"
        st.toast(f"{emoji} {newest.ticker} closed for ${newest.realized_pnl:+.2f}", icon=emoji)
    st.session_state["_live_last_trade_id"] = latest_id

    if not trades:
        st.info("No closed live trades yet.")
        return

    wins = [t for t in trades if (t.realized_pnl or 0) > 0]
    losses = [t for t in trades if (t.realized_pnl or 0) <= 0]
    win_rate = len(wins) / len(trades) * 100
    avg_win = sum(t.realized_pnl for t in wins) / len(wins) if wins else 0.0
    avg_loss = sum(t.realized_pnl for t in losses) / len(losses) if losses else 0.0

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Trades", f"{len(trades)}")
    c2.metric("Win Rate", f"{win_rate:.0f}%", f"{len(wins)}W / {len(losses)}L")
    c3.metric("Avg Win", f"${avg_win:,.2f}")
    c4.metric("Avg Loss", f"${avg_loss:,.2f}")

    chrono = sorted(trades, key=lambda t: t.closed_at or "")
    cum = 0.0
    curve_rows = []
    for t in chrono:
        cum += t.realized_pnl or 0.0
        curve_rows.append({"timestamp": t.closed_at, "cumulative_pnl": cum})
    if len(curve_rows) >= 2:
        curve_df = pd.DataFrame(curve_rows)
        curve_df["timestamp"] = pd.to_datetime(curve_df["timestamp"])
        line = (
            alt.Chart(curve_df)
            .mark_line(color="#2a78d6", strokeWidth=2)
            .encode(
                x=alt.X("timestamp:T", title=None, axis=alt.Axis(format="%b %d, %H:%M")),
                y=alt.Y("cumulative_pnl:Q", title="Cumulative Realized P&L ($)"),
                tooltip=[
                    alt.Tooltip("timestamp:T", title="Time"),
                    alt.Tooltip("cumulative_pnl:Q", title="Cumulative P&L", format="$.2f"),
                ],
            )
        )
        zero_line = alt.Chart(pd.DataFrame({"y": [0]})).mark_rule(color="#898781", strokeDash=[4, 4]).encode(y="y:Q")
        st.altair_chart((line + zero_line).properties(height=250), width="stretch")

    win_tab, loss_tab = st.tabs([f"✅ Wins ({len(wins)})", f"❌ Losses ({len(losses)})"])
    with win_tab:
        if wins:
            wins_df = _live_trades_df(sorted(wins, key=lambda t: t.closed_at or "", reverse=True))
            st.dataframe(_style_pnl_column(wins_df, "Realized P&L"), hide_index=True, width="stretch")
        else:
            st.info("No winning trades yet.")
    with loss_tab:
        if losses:
            losses_df = _live_trades_df(sorted(losses, key=lambda t: t.closed_at or "", reverse=True))
            st.dataframe(_style_pnl_column(losses_df, "Realized P&L"), hide_index=True, width="stretch")
        else:
            st.info("No losing trades yet.")


def _live_events_section(ledger: LiveLedger) -> None:
    events = ledger.get_recent_events(limit=50)
    if not events:
        return
    with st.expander(f"⚠️ System events ({len(events)})"):
        rows = [
            {"Time": e["timestamp"][:19].replace("T", " "), "Event": e["event"], "Detail": e["detail"]}
            for e in events
        ]
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")


@st.fragment(run_every=LIVE_REFRESH_SECONDS)
def live_trading_tab() -> None:
    client = get_client()
    ledger = get_live_ledger()

    _live_kill_switch_panel()
    series_list = ", ".join(p.rstrip("-") for p in run_live_trading.ACTIVE_CRYPTO_SERIES_PREFIXES)
    st.caption(f"Scope: {series_list} · refreshes every {LIVE_REFRESH_SECONDS}s")

    st.divider()
    _live_summary_section(client, ledger)

    st.divider()
    _live_position_section(client)

    st.divider()
    _live_activity_feed()

    st.divider()
    _live_trades_section(ledger)
    _live_events_section(ledger)


def main() -> None:
    st.title("Kalshi Trading Bot")

    # Deliberately not st.tabs(): tabs render BOTH tabs' bodies on every script
    # run (they're CSS-only visibility), so filtering on Markets was silently
    # also re-running Paper Trading's ~15 API calls every time, adding several
    # seconds of latency. Query-param routing only executes the active view.
    current_tab = st.query_params.get("tab", "markets")

    nav_col1, nav_col2, nav_col3, _ = st.columns([1, 1, 1, 3])
    if nav_col1.button(
        "📊 Markets", width="stretch", type="primary" if current_tab == "markets" else "secondary"
    ):
        st.query_params["tab"] = "markets"
        st.rerun()
    if nav_col2.button(
        "💰 Paper Trading", width="stretch", type="primary" if current_tab == "paper" else "secondary"
    ):
        st.query_params["tab"] = "paper"
        st.rerun()
    if nav_col3.button(
        "⚡ Live Trading", width="stretch", type="primary" if current_tab == "live" else "secondary"
    ):
        st.query_params["tab"] = "live"
        st.rerun()
    st.divider()

    if current_tab == "paper":
        st.caption("Simulated paper trading, no real orders placed")
        paper_trading_tab()
    elif current_tab == "live":
        st.caption("⚠️ REAL MONEY — real orders are placed against your actual Kalshi account")
        live_trading_tab()
    else:
        markets_tab()


if __name__ == "__main__":
    main()
