"""Read-mostly JSON API for the native Android app. The web dashboard
(app.py) is completely unaffected by this -- this is a separate, additional
interface onto the same real data (LiveLedger, the real Kalshi account,
bot_control), built specifically because a native UI needs structured JSON,
not an HTML page.

Auth is NOT handled here on purpose -- Caddy's basic_auth already gates
every path on the dashboard's domain, including /api/*, so this trusts that
anything reaching it has already been authenticated at the edge. This must
never be exposed directly to the internet: deploy/api.service binds
127.0.0.1 only, same pattern as the dashboard (see deploy/dashboard.service).
"""
from __future__ import annotations

import json
from pathlib import Path
from datetime import datetime, timedelta, timezone

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

import run_live_trading as r
import run_paper_trading as rp
from trade_bot import bot_control, paper_bot_control
from trade_bot.adaptive import PerformanceGovernor
from trade_bot.client import KalshiAPIError, KalshiClient
from trade_bot.data import CashFlowSummary, get_available_balance_dollars, get_cash_flow_summary, get_market, get_real_positions
from trade_bot.btc_price_history import BtcPriceHistory, WINDOWS as BTC_PRICE_WINDOWS
from trade_bot.interval_tracker import IntervalTracker, TRACKED_SERIES_PREFIXES
from trade_bot.live_ledger import LiveLedger
from trade_bot.mlb_bot_control import is_running as mlb_is_running
from trade_bot.mlb_data import MlbDataStore
from trade_bot.mlb_features import build_mlb_features
from trade_bot.mlb_ledger import MlbLedger
from trade_bot.mlb_model import predict_mlb_win_probability
from trade_bot.mlb_watchlist import index_tickers_by_game
from trade_bot.online_learner import OnlineLogisticLearner
from trade_bot.portfolio import PaperPortfolio
from trade_bot.push import push_status, save_device_token

app = FastAPI(title="Kalshi Trade Bot API")

_client: KalshiClient | None = None

# Deposit/withdrawal history barely changes and costs a real (paginated)
# Kalshi API call -- cached separately from everything else so a 10s status
# poll doesn't refetch it every time.
_cash_flow_cache: tuple[float, CashFlowSummary] | None = None
_CASH_FLOW_CACHE_TTL_SECONDS = 300.0


def get_cash_flow_cached() -> CashFlowSummary:
    global _cash_flow_cache
    now = datetime.now(timezone.utc).timestamp()
    if _cash_flow_cache is not None and now - _cash_flow_cache[0] < _CASH_FLOW_CACHE_TTL_SECONDS:
        return _cash_flow_cache[1]
    summary = get_cash_flow_summary(get_client())
    _cash_flow_cache = (now, summary)
    return summary
_ledger: LiveLedger | None = None
_paper_portfolio: PaperPortfolio | None = None
_interval_tracker: IntervalTracker | None = None


def get_client() -> KalshiClient:
    global _client
    if _client is None:
        _client = KalshiClient()
    return _client


def get_ledger() -> LiveLedger:
    global _ledger
    if _ledger is None:
        _ledger = LiveLedger()
    return _ledger


_mlb_ledger: MlbLedger | None = None


def get_mlb_ledger() -> MlbLedger:
    global _mlb_ledger
    if _mlb_ledger is None:
        _mlb_ledger = MlbLedger(toggle_source=get_ledger())
    return _mlb_ledger


_mlb_data_store: MlbDataStore | None = None


def get_mlb_data_store() -> MlbDataStore:
    global _mlb_data_store
    if _mlb_data_store is None:
        _mlb_data_store = MlbDataStore()
    return _mlb_data_store


def get_interval_tracker() -> IntervalTracker:
    global _interval_tracker
    if _interval_tracker is None:
        _interval_tracker = IntervalTracker()
    return _interval_tracker


def get_paper_portfolio() -> PaperPortfolio:
    global _paper_portfolio
    if _paper_portfolio is None:
        _paper_portfolio = PaperPortfolio()
    return _paper_portfolio


_btc_price_history: BtcPriceHistory | None = None


def get_btc_price_history() -> BtcPriceHistory:
    global _btc_price_history
    if _btc_price_history is None:
        _btc_price_history = BtcPriceHistory()
    return _btc_price_history


LIVE_LOG_PATH = Path(__file__).resolve().parent.parent / "logs" / "live_decisions.log"
PAPER_LOG_PATH = Path(__file__).resolve().parent.parent / "logs" / "decisions.log"
# Bounds the cost of an "initial load" (offset=-1) to one bounded read
# regardless of how large the log has grown over a 24/7 run -- a JSONL
# decision line here is ~200-400 bytes, so this comfortably covers a couple
# hundred lines without ever reading the whole file.
_LOG_TAIL_INITIAL_BYTES = 131_072


def _tail_log(path: Path, offset: int, limit: int) -> tuple[list[dict], int]:
    """Returns (entries, next_offset).

    offset < 0 (or stale/past EOF, e.g. the log rotated): "give me the last
    `limit` lines" -- seeks back a bounded window rather than reading the
    whole file. offset >= 0: "give me every complete line written since
    that byte position" -- true incremental tailing for repeated polling.

    Only ever returns COMPLETE (newline-terminated) lines and only ever
    advances next_offset past what was actually consumed -- a line the bot
    is still mid-write on is left for the next call rather than returned
    truncated or double-counted.
    """
    if not path.exists():
        return [], 0
    size = path.stat().st_size
    using_fallback = not (0 <= offset <= size)
    start = max(0, size - _LOG_TAIL_INITIAL_BYTES) if using_fallback else offset

    with open(path, "rb") as f:
        f.seek(start)
        raw = f.read()

    kept_start = start
    if using_fallback and start > 0:
        # Only the bounded fallback window can start mid-line -- a
        # previously-returned offset is always an exact line boundary
        # (next_offset is only ever set right after a newline), so this
        # trimming must never run for an incremental (offset >= 0) call --
        # doing so was a real bug: it silently dropped the first new line
        # of every poll after the first.
        nl = raw.find(b"\n")
        if nl == -1:
            # No newline anywhere in this whole window -- a single line
            # longer than _LOG_TAIL_INITIAL_BYTES, not realistic for these
            # JSONL events. Nothing usable this call.
            return [], start
        raw = raw[nl + 1:]
        kept_start = start + nl + 1

    complete_end = raw.rfind(b"\n")
    if complete_end == -1:
        return [], kept_start  # only a partial trailing line so far -- nothing to return yet

    next_offset = kept_start + complete_end + 1
    entries: list[dict] = []
    for line in raw[:complete_end].split(b"\n"):
        line = line.strip()
        if not line:
            continue
        try:
            entries.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return entries[-limit:], next_offset


def _today_start_iso() -> str:
    return datetime.now(timezone.utc).date().isoformat() + "T00:00:00+00:00"


class StatusResponse(BaseModel):
    bot_running: bool
    pid: int | None
    uptime_seconds: float | None
    balance_dollars: float | None
    total_realized_pnl: float
    today_realized_pnl: float
    pnl_24h: float
    pnl_7d: float
    daily_loss_limit: float
    daily_kill_switch_enabled: bool
    open_position_count: int
    governor_multiplier: float
    governor_reason: str
    learner_n_updates: int
    learner_min_trades: int
    learner_active: bool
    scope: list[str]
    total_deposited_gross_dollars: float
    total_deposit_fees_dollars: float
    total_deposited_net_dollars: float
    total_withdrawn_dollars: float


class PositionResponse(BaseModel):
    ticker: str
    title: str
    side: str
    quantity: int
    entry_price_pct: float
    mark_pct: float | None
    unrealized_pnl: float | None
    minutes_to_close: float | None


class TradeResponse(BaseModel):
    ticker: str
    side: str
    quantity: int
    entry_price_pct: float
    exit_price_pct: float | None
    fees: float
    realized_pnl: float | None
    opened_at: str
    closed_at: str | None
    close_reason: str | None


class TradesPageResponse(BaseModel):
    trades: list[TradeResponse]
    has_more: bool


class MlbTradesPageResponse(BaseModel):
    trades: list[TradeResponse]
    has_more: bool


class MlbGamePredictionResponse(BaseModel):
    game_pk: int
    away_team: str
    home_team: str
    away_abbreviation: str
    home_abbreviation: str
    first_pitch_utc: str
    status: str
    away_win_probability: float
    home_win_probability: float
    # None if Kalshi hasn't opened a market for this game yet, or no ticker
    # was found for that side -- the prediction itself is still shown either
    # way, just nothing to manually trade against yet.
    away_ticker: str | None
    home_ticker: str | None
    away_market_price_pct: float | None
    home_market_price_pct: float | None


class MlbPredictionsResponse(BaseModel):
    date: str
    games: list[MlbGamePredictionResponse]


class MlbStatusResponse(BaseModel):
    bot_running: bool
    pid: int | None
    mlb_trading_enabled: bool
    total_realized_pnl: float
    open_position_count: int


class LogPageResponse(BaseModel):
    # Generic passthrough of whatever fields each decision-log JSON line
    # has (they vary by event type: decision/fill/skip/risk_blocked/
    # ml_gated/governor_paused/cycle_error/...) -- the client displays
    # these as formatted log lines, it doesn't need a rigid per-event-type
    # schema to do that.
    entries: list[dict]
    next_offset: int


class BotActionResponse(BaseModel):
    ok: bool
    message: str


class RegisterTokenRequest(BaseModel):
    token: str


class PushStatusResponse(BaseModel):
    firebase_configured: bool
    device_registered: bool


class IntervalOutcomeResponse(BaseModel):
    ticker: str
    asset: str
    last_seen_at: str
    close_time: str | None
    last_yes_bid_pct: float | None
    bot_lean: str | None
    last_predicted_probability: float | None
    last_signal: str | None
    bot_traded: bool
    trade_side: str | None
    trade_realized_pnl: float | None
    result: str | None
    settled_at: str | None
    lean_correct: bool | None
    trade_correct: bool | None


class IntervalOutcomesPageResponse(BaseModel):
    intervals: list[IntervalOutcomeResponse]


class IntervalOutcomeSummaryResponse(BaseModel):
    settled_count: int
    lean_evaluated_count: int
    lean_correct_count: int
    traded_count: int
    trade_correct_count: int


class BtcPricePointResponse(BaseModel):
    t: str
    price: float


class BtcPriceHistoryResponse(BaseModel):
    window: str
    points: list[BtcPricePointResponse]


class BtcPriceAvailabilityResponse(BaseModel):
    # e.g. {"5m": true, "15m": true, "1h": false, ...} -- see
    # BtcPriceHistory.get_availability's doc for why a window can be
    # unavailable even once collection has started.
    windows: dict[str, bool]


class RiskSettingsResponse(BaseModel):
    small_bets_only: bool
    btc_dip_enabled: bool
    mlb_trading_enabled: bool


class RiskSettingsRequest(BaseModel):
    # All optional (default: don't touch) so a caller can flip just one
    # toggle without having to already know the others' current values --
    # added btc_dip_enabled 2026-08-07, mlb_trading_enabled 2026-08-12,
    # without breaking existing callers that only ever sent small_bets_only.
    small_bets_only: bool | None = None
    btc_dip_enabled: bool | None = None
    mlb_trading_enabled: bool | None = None


@app.get("/api/status", response_model=StatusResponse)
def status() -> StatusResponse:
    ledger = get_ledger()
    running, pid = bot_control.is_running()
    started = bot_control.started_at()
    uptime = (datetime.now(timezone.utc).timestamp() - started) if started else None

    try:
        balance = get_available_balance_dollars(get_client())
    except KalshiAPIError:
        balance = None

    total_pnl = ledger.get_total_realized_pnl()
    today_pnl = ledger.get_realized_pnl_since(_today_start_iso())
    now = datetime.now(timezone.utc)
    pnl_24h = ledger.get_realized_pnl_since((now - timedelta(hours=24)).isoformat())
    pnl_7d = ledger.get_realized_pnl_since((now - timedelta(days=7)).isoformat())

    # The real exchange, not the local ledger, is the source of truth for
    # "how many positions are open right now" -- found live 2026-08-04 that
    # a ledger row can go stale (its ticker rotated off the watchlist before
    # the engine got a chance to reconcile it against a since-closed real
    # position), which would silently overcount here. Also fixed at the
    # root in live_engine.py's run_once() so stale rows get cleaned up
    # rather than just papered over here, but this stays exchange-truth
    # regardless so the count is never wrong even for the one cycle before
    # that reconciliation runs.
    try:
        open_position_count = len(get_real_positions(get_client(), ticker_prefixes=r.ACTIVE_CRYPTO_SERIES_PREFIXES))
    except KalshiAPIError:
        open_position_count = len(ledger.get_all_open_trades())

    try:
        cash_flow = get_cash_flow_cached()
    except KalshiAPIError:
        cash_flow = CashFlowSummary(0.0, 0.0, 0.0, 0.0, 0, 0)

    governor = PerformanceGovernor(
        lambda name, limit: ledger.get_recent_realized_pnls_within_hours(name, limit, r.GOVERNOR_LOOKBACK_HOURS),
        r.GOVERNOR_CONFIG,
    )
    gd = governor.evaluate(r.STRATEGY_NAME)
    learner = OnlineLogisticLearner.from_json(ledger.load_learner_state())

    return StatusResponse(
        bot_running=running,
        pid=pid,
        uptime_seconds=uptime,
        balance_dollars=balance,
        total_realized_pnl=total_pnl,
        today_realized_pnl=today_pnl,
        pnl_24h=pnl_24h,
        pnl_7d=pnl_7d,
        daily_loss_limit=r.RISK_LIMITS.daily_stop_loss_dollars,
        daily_kill_switch_enabled=r.RISK_LIMITS.daily_kill_switch_enabled,
        open_position_count=open_position_count,
        governor_multiplier=gd.multiplier,
        governor_reason=gd.reason,
        total_deposited_gross_dollars=cash_flow.total_deposited_gross_dollars,
        total_deposit_fees_dollars=cash_flow.total_deposit_fees_dollars,
        total_deposited_net_dollars=cash_flow.total_deposited_net_dollars,
        total_withdrawn_dollars=cash_flow.total_withdrawn_dollars,
        learner_n_updates=learner.n_updates,
        learner_min_trades=r.MIN_TRADES_FOR_ML_GATE,
        learner_active=learner.n_updates >= r.MIN_TRADES_FOR_ML_GATE,
        scope=r.ACTIVE_CRYPTO_SERIES_PREFIXES,
    )


@app.get("/api/positions", response_model=list[PositionResponse])
def positions() -> list[PositionResponse]:
    client = get_client()
    try:
        real_positions = get_real_positions(client, ticker_prefixes=r.ACTIVE_CRYPTO_SERIES_PREFIXES)
    except KalshiAPIError as e:
        raise HTTPException(status_code=502, detail=str(e))

    out = []
    for p in real_positions:
        try:
            m = get_market(client, p.ticker)
        except KalshiAPIError:
            m = {}
        mark = m.get("yes_bid_pct") if p.side == "YES" else m.get("no_bid_pct")
        unrealized = p.quantity * (mark - p.avg_entry_price_pct) / 100 if mark is not None else None
        minutes_to_close = None
        close_time = m.get("close_time")
        if close_time:
            try:
                ct = datetime.fromisoformat(close_time.replace("Z", "+00:00"))
                minutes_to_close = (ct - datetime.now(timezone.utc)).total_seconds() / 60
            except ValueError:
                pass
        out.append(
            PositionResponse(
                ticker=p.ticker,
                title=m.get("title") or p.ticker,
                side=p.side,
                quantity=p.quantity,
                entry_price_pct=p.avg_entry_price_pct,
                mark_pct=mark,
                unrealized_pnl=unrealized,
                minutes_to_close=minutes_to_close,
            )
        )
    return out


@app.get("/api/trades", response_model=TradesPageResponse)
def trades(limit: int = 10, offset: int = 0) -> TradesPageResponse:
    """Real server-side pagination for the Android app's History tab --
    fetches one extra row beyond `limit` to know whether there's a next
    page, without a separate COUNT query."""
    rows = get_ledger().get_closed_trades(limit=limit + 1, offset=offset)
    has_more = len(rows) > limit
    page = rows[:limit]
    return TradesPageResponse(
        trades=[
            TradeResponse(
                ticker=t.ticker,
                side=t.side,
                quantity=t.quantity,
                entry_price_pct=t.entry_price_pct,
                exit_price_pct=t.exit_price_pct,
                fees=(t.entry_fee or 0) + (t.exit_fee or 0),
                realized_pnl=t.realized_pnl,
                opened_at=t.opened_at,
                closed_at=t.closed_at,
                close_reason=t.close_reason,
            )
            for t in page
        ],
        has_more=has_more,
    )


@app.get("/api/mlb/predictions", response_model=MlbPredictionsResponse)
def mlb_predictions(date: str | None = None) -> MlbPredictionsResponse:
    """Every scheduled game on `date` (default: today, UTC) with the
    pregame model's predicted win probability for both teams -- shown
    regardless of whether mlb_trading_enabled is on, so the prediction is
    visible even when the bot itself won't act on it (per explicit user
    request: "if bot doesn't place bets I can do manually"). Market price
    is best-effort -- None whenever Kalshi hasn't opened trading on that
    specific game yet."""
    target_date = date or datetime.now(timezone.utc).strftime("%Y-%m-%d")
    store = get_mlb_data_store()
    games = store.get_games_on_date(target_date)

    ticker_index = index_tickers_by_game(get_client())

    results: list[MlbGamePredictionResponse] = []
    for g in games:
        away_row = store.get_team_by_id(g["away_team_id"])
        home_row = store.get_team_by_id(g["home_team_id"])
        if away_row is None or home_row is None:
            continue

        away_features = build_mlb_features(
            store, g["away_team_id"], g["home_team_id"], g["game_date"], False,
            pitcher_id=g["away_probable_pitcher_id"], opponent_pitcher_id=g["home_probable_pitcher_id"],
        )
        home_features = build_mlb_features(
            store, g["home_team_id"], g["away_team_id"], g["game_date"], True,
            pitcher_id=g["home_probable_pitcher_id"], opponent_pitcher_id=g["away_probable_pitcher_id"],
        )
        away_prob = predict_mlb_win_probability(away_features)
        home_prob = predict_mlb_win_probability(home_features)

        key = (target_date, away_row["abbreviation"], home_row["abbreviation"])
        tickers = ticker_index.get(key, {})
        away_ticker = tickers.get("away_ticker")
        home_ticker = tickers.get("home_ticker")

        away_price = home_price = None
        try:
            if away_ticker:
                away_price = get_market(get_client(), away_ticker).get("yes_ask_pct")
            if home_ticker:
                home_price = get_market(get_client(), home_ticker).get("yes_ask_pct")
        except KalshiAPIError:
            pass

        results.append(
            MlbGamePredictionResponse(
                game_pk=g["game_pk"],
                away_team=away_row["name"],
                home_team=home_row["name"],
                away_abbreviation=away_row["abbreviation"],
                home_abbreviation=home_row["abbreviation"],
                first_pitch_utc=g["game_date_utc"],
                status=g["status"],
                away_win_probability=away_prob,
                home_win_probability=home_prob,
                away_ticker=away_ticker,
                home_ticker=home_ticker,
                away_market_price_pct=away_price,
                home_market_price_pct=home_price,
            )
        )

    results.sort(key=lambda r: r.first_pitch_utc)
    return MlbPredictionsResponse(date=target_date, games=results)


@app.get("/api/mlb/status", response_model=MlbStatusResponse)
def mlb_status() -> MlbStatusResponse:
    """Own process, own ledger (mlb_trading.db) -- see run_mlb_trading.py.
    Not mode-scoped like the crypto/paper split: MLB is a single real-money
    strategy, no paper A/B variant exists for it yet."""
    ledger = get_mlb_ledger()
    running, pid = mlb_is_running()
    return MlbStatusResponse(
        bot_running=running,
        pid=pid,
        mlb_trading_enabled=ledger.get_mlb_trading_enabled(),
        total_realized_pnl=ledger.get_total_realized_pnl(),
        open_position_count=len(ledger.get_all_open_trades()),
    )


@app.get("/api/mlb/trades", response_model=MlbTradesPageResponse)
def mlb_trades(limit: int = 10, offset: int = 0) -> MlbTradesPageResponse:
    """Same server-side pagination shape as /api/trades -- see that
    endpoint's docstring. Separate endpoint (not a mode param on the
    existing one) since MLB trades live in a fully isolated ledger/db, not
    a live/paper variant of the same one."""
    rows = get_mlb_ledger().get_closed_trades(limit=limit + 1, offset=offset)
    has_more = len(rows) > limit
    page = rows[:limit]
    return MlbTradesPageResponse(
        trades=[
            TradeResponse(
                ticker=t.ticker,
                side=t.side,
                quantity=t.quantity,
                entry_price_pct=t.entry_price_pct,
                exit_price_pct=t.exit_price_pct,
                fees=(t.entry_fee or 0) + (t.exit_fee or 0),
                realized_pnl=t.realized_pnl,
                opened_at=t.opened_at,
                closed_at=t.closed_at,
                close_reason=t.close_reason,
            )
            for t in page
        ],
        has_more=has_more,
    )


@app.get("/api/intervals", response_model=IntervalOutcomesPageResponse)
def intervals(limit: int = 100, asset: str | None = None) -> IntervalOutcomesPageResponse:
    """Every evaluated 15-minute BTC/ETH/BNB interval (traded or not), with
    the bot's final order-book-imbalance lean/predicted probability next to
    the actual Kalshi settlement result once known -- see
    trade_bot/interval_tracker.py. Backs the Android app's Intervals tab."""
    if asset is not None and asset not in TRACKED_SERIES_PREFIXES.values():
        raise HTTPException(status_code=400, detail=f"unknown asset {asset!r}")
    rows = get_interval_tracker().get_recent(limit=limit, asset=asset)
    return IntervalOutcomesPageResponse(
        intervals=[
            IntervalOutcomeResponse(
                ticker=r.ticker,
                asset=r.asset,
                last_seen_at=r.last_seen_at,
                close_time=r.close_time,
                last_yes_bid_pct=r.last_yes_bid_pct,
                bot_lean=r.bot_lean,
                last_predicted_probability=r.last_predicted_probability,
                last_signal=r.last_signal,
                bot_traded=bool(r.bot_traded),
                trade_side=r.trade_side,
                trade_realized_pnl=r.trade_realized_pnl,
                result=r.result,
                settled_at=r.settled_at,
                lean_correct=None if r.lean_correct is None else bool(r.lean_correct),
                trade_correct=None if r.trade_correct is None else bool(r.trade_correct),
            )
            for r in rows
        ],
    )


@app.get("/api/intervals/summary", response_model=IntervalOutcomeSummaryResponse)
def intervals_summary() -> IntervalOutcomeSummaryResponse:
    stats = get_interval_tracker().get_summary_stats()
    return IntervalOutcomeSummaryResponse(
        settled_count=stats["settled_count"] or 0,
        lean_evaluated_count=stats["lean_evaluated_count"] or 0,
        lean_correct_count=stats["lean_correct_count"] or 0,
        traded_count=stats["traded_count"] or 0,
        trade_correct_count=stats["trade_correct_count"] or 0,
    )


@app.get("/api/btc-price/availability", response_model=BtcPriceAvailabilityResponse)
def btc_price_availability() -> BtcPriceAvailabilityResponse:
    """Which chart windows have enough real elapsed collection time to be
    worth showing yet -- see BtcPriceHistory.get_availability. Not
    live/paper-scoped: this is independent of both trading bots entirely."""
    return BtcPriceAvailabilityResponse(windows=get_btc_price_history().get_availability())


@app.get("/api/btc-price/history", response_model=BtcPriceHistoryResponse)
def btc_price_history_endpoint(window: str = "1h") -> BtcPriceHistoryResponse:
    if window not in BTC_PRICE_WINDOWS:
        raise HTTPException(
            status_code=400, detail=f"unknown window {window!r}, must be one of {list(BTC_PRICE_WINDOWS)}"
        )
    points = get_btc_price_history().get_history(window)
    return BtcPriceHistoryResponse(window=window, points=[BtcPricePointResponse(**p) for p in points])


@app.post("/api/notifications/register", response_model=BotActionResponse)
def register_notification_token(body: RegisterTokenRequest) -> BotActionResponse:
    """Called by the Android app on launch (and whenever FCM hands it a new
    token) so trade_bot/push.py knows where to actually send native push
    notifications -- see that module's docstring. Single-device deployment:
    always overwrites, no per-device list."""
    save_device_token(body.token)
    return BotActionResponse(ok=True, message="Registered for push notifications.")


@app.get("/api/push/status", response_model=PushStatusResponse)
def get_push_status() -> PushStatusResponse:
    """Diagnostic-only, never affects trading -- lets the dashboard/app show
    WHY native push notifications aren't arriving instead of that being a
    silent SSH-only mystery (see trade_bot/push.py's push_status doc)."""
    status = push_status()
    return PushStatusResponse(**status)


@app.get("/api/risk-settings", response_model=RiskSettingsResponse)
def risk_settings() -> RiskSettingsResponse:
    """Live-only (real money) toggles -- see LiveLedger.get_small_bets_only/
    get_btc_dip_enabled. Not mirrored under /api/paper: paper trading is an
    unthrottled A/B control of the original strategy, these have no
    meaning there."""
    ledger = get_ledger()
    return RiskSettingsResponse(
        small_bets_only=ledger.get_small_bets_only(),
        btc_dip_enabled=ledger.get_btc_dip_enabled(),
        mlb_trading_enabled=ledger.get_mlb_trading_enabled(),
    )


@app.post("/api/risk-settings", response_model=RiskSettingsResponse)
def set_risk_settings(body: RiskSettingsRequest) -> RiskSettingsResponse:
    ledger = get_ledger()
    if body.small_bets_only is not None:
        ledger.set_small_bets_only(body.small_bets_only)
    if body.btc_dip_enabled is not None:
        ledger.set_btc_dip_enabled(body.btc_dip_enabled)
    if body.mlb_trading_enabled is not None:
        ledger.set_mlb_trading_enabled(body.mlb_trading_enabled)
    return RiskSettingsResponse(
        small_bets_only=ledger.get_small_bets_only(),
        btc_dip_enabled=ledger.get_btc_dip_enabled(),
        mlb_trading_enabled=ledger.get_mlb_trading_enabled(),
    )


@app.post("/api/bot/start", response_model=BotActionResponse)
def start_bot() -> BotActionResponse:
    ok, message = bot_control.start()
    return BotActionResponse(ok=ok, message=message)


@app.post("/api/bot/stop", response_model=BotActionResponse)
def stop_bot() -> BotActionResponse:
    ok, message = bot_control.stop()
    return BotActionResponse(ok=ok, message=message)


# ---------------------------------------------------------------------------
# Paper trading -- same response shapes as the /api/* endpoints above so the
# Android app can reuse identical UI code for both, just pointed at a
# different base path. See docs/ALGORITHM.md: the paper bot runs the
# ORIGINAL (unchanged) strategy on the live bot's exact market scope, as an
# explicit A/B control -- these endpoints exist so that comparison can
# happen from the phone too, not just by hand on the server.
# ---------------------------------------------------------------------------


@app.get("/api/paper/status", response_model=StatusResponse)
def paper_status() -> StatusResponse:
    portfolio = get_paper_portfolio()
    running, pid = paper_bot_control.is_running()
    started = paper_bot_control.started_at()
    uptime = (datetime.now(timezone.utc).timestamp() - started) if started else None

    total_pnl = portfolio.get_total_realized_pnl()
    today_pnl = portfolio.get_realized_pnl_since(_today_start_iso())
    now = datetime.now(timezone.utc)
    pnl_24h = portfolio.get_realized_pnl_since((now - timedelta(hours=24)).isoformat())
    pnl_7d = portfolio.get_realized_pnl_since((now - timedelta(days=7)).isoformat())

    return StatusResponse(
        bot_running=running,
        pid=pid,
        uptime_seconds=uptime,
        balance_dollars=portfolio.get_cash_balance(),
        total_realized_pnl=total_pnl,
        today_realized_pnl=today_pnl,
        pnl_24h=pnl_24h,
        pnl_7d=pnl_7d,
        daily_loss_limit=rp.RISK_LIMITS.daily_stop_loss_dollars,
        # paper's engine.py always enforces its own daily_stop_loss_dollars (no
        # on/off toggle exists the way LiveRiskLimits has one) -- "enabled" here
        # just means the mechanism is live, not that it will ever actually bind.
        # As of 2026-08-06 it's deliberately set to UNCAPPED_DOLLARS (no hard
        # stops, per explicit user request) so in practice it never fires.
        daily_kill_switch_enabled=True,
        open_position_count=len(portfolio.get_all_open_positions()),
        # No governor/learner wired up for paper trading (out of scope for
        # the A/B control -- see docs/ALGORITHM.md) -- same "not configured"
        # shape live_engine.py itself falls back to when governor is None.
        governor_multiplier=1.0,
        governor_reason="no governor configured for paper trading",
        learner_n_updates=0,
        learner_min_trades=0,
        learner_active=False,
        # Cash-flow concept (deposits/withdrawals) doesn't apply to a
        # simulated account funded by a fixed starting balance.
        total_deposited_gross_dollars=0.0,
        total_deposit_fees_dollars=0.0,
        total_deposited_net_dollars=0.0,
        total_withdrawn_dollars=0.0,
        scope=list(r.ACTIVE_CRYPTO_SERIES_PREFIXES),
    )


@app.get("/api/paper/positions", response_model=list[PositionResponse])
def paper_positions() -> list[PositionResponse]:
    client = get_client()
    portfolio = get_paper_portfolio()
    out = []
    for pos in portfolio.get_all_open_positions():
        try:
            m = get_market(client, pos.ticker)
        except KalshiAPIError:
            m = {}
        mark = m.get("yes_bid_pct") if pos.side == "YES" else m.get("no_bid_pct")
        unrealized = pos.quantity * (mark - pos.entry_price_pct) / 100 if mark is not None else None
        minutes_to_close = None
        close_time = m.get("close_time")
        if close_time:
            try:
                ct = datetime.fromisoformat(close_time.replace("Z", "+00:00"))
                minutes_to_close = (ct - datetime.now(timezone.utc)).total_seconds() / 60
            except ValueError:
                pass
        out.append(
            PositionResponse(
                ticker=pos.ticker,
                title=m.get("title") or pos.ticker,
                side=pos.side,
                quantity=pos.quantity,
                entry_price_pct=pos.entry_price_pct,
                mark_pct=mark,
                unrealized_pnl=unrealized,
                minutes_to_close=minutes_to_close,
            )
        )
    return out


@app.get("/api/paper/trades", response_model=TradesPageResponse)
def paper_trades(limit: int = 10, offset: int = 0) -> TradesPageResponse:
    rows = get_paper_portfolio().get_closed_trades(limit=limit + 1, offset=offset)
    has_more = len(rows) > limit
    page = rows[:limit]
    return TradesPageResponse(
        trades=[
            TradeResponse(
                ticker=t.ticker,
                side=t.side,
                quantity=t.quantity,
                entry_price_pct=t.entry_price_pct,
                exit_price_pct=t.exit_price_pct,
                fees=(t.entry_fee or 0) + (t.exit_fee or 0),
                realized_pnl=t.realized_pnl,
                opened_at=t.opened_at,
                closed_at=t.closed_at,
                close_reason=t.close_reason,
            )
            for t in page
        ],
        has_more=has_more,
    )


@app.post("/api/paper/bot/start", response_model=BotActionResponse)
def start_paper_bot() -> BotActionResponse:
    ok, message = paper_bot_control.start()
    return BotActionResponse(ok=ok, message=message)


@app.post("/api/paper/bot/stop", response_model=BotActionResponse)
def stop_paper_bot() -> BotActionResponse:
    ok, message = paper_bot_control.stop()
    return BotActionResponse(ok=ok, message=message)


# ---------------------------------------------------------------------------
# Live decision-thinking feed ("logcat" tab) -- real-time-ish view of what
# the bot is currently evaluating, accepting, and rejecting, for both bots.
# Polling-based (not a websocket): the Android client passes back
# next_offset from the previous call to get only what's new since then.
# ---------------------------------------------------------------------------


@app.get("/api/live/log", response_model=LogPageResponse)
def live_log(offset: int = -1, limit: int = 200) -> LogPageResponse:
    entries, next_offset = _tail_log(LIVE_LOG_PATH, offset, limit)
    return LogPageResponse(entries=entries, next_offset=next_offset)


@app.get("/api/paper/log", response_model=LogPageResponse)
def paper_log(offset: int = -1, limit: int = 200) -> LogPageResponse:
    entries, next_offset = _tail_log(PAPER_LOG_PATH, offset, limit)
    return LogPageResponse(entries=entries, next_offset=next_offset)
