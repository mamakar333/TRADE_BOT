package com.example.androidkalshibot.data

data class BotStatus(
    val botRunning: Boolean,
    val pid: Int?,
    val uptimeSeconds: Double?,
    val balanceDollars: Double?,
    val totalRealizedPnl: Double,
    val todayRealizedPnl: Double,
    val pnl24h: Double,
    val pnl7d: Double,
    val dailyLossLimit: Double,
    val dailyKillSwitchEnabled: Boolean,
    val openPositionCount: Int,
    val governorMultiplier: Double,
    val governorReason: String,
    val learnerNUpdates: Int,
    val learnerMinTrades: Int,
    val learnerActive: Boolean,
    val scope: List<String>,
    val totalDepositedGrossDollars: Double,
    val totalDepositFeesDollars: Double,
    val totalDepositedNetDollars: Double,
    val totalWithdrawnDollars: Double,
)

data class Position(
    val ticker: String,
    val title: String,
    val side: String,
    val quantity: Int,
    val entryPricePct: Double,
    val markPct: Double?,
    val unrealizedPnl: Double?,
    val minutesToClose: Double?,
)

data class Trade(
    val ticker: String,
    val side: String,
    val quantity: Int,
    val entryPricePct: Double,
    val exitPricePct: Double?,
    val fees: Double,
    val realizedPnl: Double?,
    val openedAt: String,
    val closedAt: String?,
    val closeReason: String?,
)

data class TradesPage(val trades: List<Trade>, val hasMore: Boolean)

data class BotActionResult(val ok: Boolean, val message: String)

/** Which bot's data a request targets -- LIVE hits the existing "/api"
 * endpoints (unchanged, so the original app keeps working untouched), PAPER
 * hits the mirrored endpoints under "/api/paper" added 2026-08-04. See
 * docs/ALGORITHM.md for why the paper bot exists as a live A/B control. */
enum class BotMode(val apiPrefix: String) {
    LIVE("/api"),
    PAPER("/api/paper"),
}

/** One line from the bot's decision log (logcat-style "thinking" feed) --
 * deliberately a loose passthrough (raw JSON fields as strings) rather
 * than a rigid per-event-type schema, since decision/fill/skip/risk_blocked/
 * ml_gated/cycle_error/... events all carry different fields. The UI only
 * ever needs to format-and-display these, not deeply parse every variant. */
data class LogEntry(val raw: Map<String, Any?>) {
    val event: String? get() = raw["event"] as? String
    val ticker: String? get() = raw["ticker"] as? String
    val signal: String? get() = raw["signal"] as? String
    val reason: String? get() = raw["reason"] as? String
    val timestamp: String? get() = raw["timestamp"] as? String
}

data class LogPage(val entries: List<LogEntry>, val nextOffset: Int)

/** One tracked 15-minute BTC/ETH/BNB interval -- recorded every cycle
 * regardless of whether the bot ever traded it, so the bot's directional
 * lean (and, once available, its predicted win probability) can be
 * compared against the market's actual settlement result. Backs the
 * Intervals tab. See trade_bot/interval_tracker.py. */
data class Interval(
    val ticker: String,
    val asset: String,
    val lastSeenAt: String,
    val closeTime: String?,
    val lastYesBidPct: Double?,
    val botLean: String?,
    val lastPredictedProbability: Double?,
    val lastSignal: String?,
    val botTraded: Boolean,
    val tradeSide: String?,
    val tradeRealizedPnl: Double?,
    val result: String?,
    val settledAt: String?,
    val leanCorrect: Boolean?,
    val tradeCorrect: Boolean?,
)

data class IntervalSummary(
    val settledCount: Int,
    val leanEvaluatedCount: Int,
    val leanCorrectCount: Int,
    val tradedCount: Int,
    val tradeCorrectCount: Int,
)

/** Diagnostic-only snapshot of whether native push (trade_bot/push.py) can
 * actually deliver anything right now -- surfaced in the UI specifically
 * because this failure mode used to be a silent SSH-only mystery: a device
 * can register successfully (deviceRegistered=true) while the server still
 * has no Firebase credential at all (firebaseConfigured=false), in which
 * case nothing is ever sent and nothing ever said why. */
data class PushStatus(val firebaseConfigured: Boolean, val deviceRegistered: Boolean)

/** One bucketed BTC/USD price sample -- see trade_bot/btc_price_history.py.
 * Collected independent of Kalshi and both trading bots, every second,
 * from a public exchange feed (Kalshi's own API has no live index price). */
data class BtcPricePoint(val t: String, val price: Double)

data class BtcPriceHistory(val window: String, val points: List<BtcPricePoint>)

/** Which chart windows (5m/15m/1h/3h/1d/3d/1w/1mo) have enough real
 * elapsed collection time to be worth showing yet -- a window key is
 * absent from `windows` only if the server hasn't heard of it, but every
 * currently-defined window is always present with true/false. */
data class BtcPriceAvailability(val windows: Map<String, Boolean>)

/** One scheduled MLB game, with the pregame model's predicted win
 * probability for both teams -- see trade_bot/mlb_strategy.py. Shown
 * regardless of whether mlb_trading_enabled is on, so a prediction is
 * visible even on cycles/games the bot itself won't act on (manual-bet
 * fallback, per explicit user request). Ticker/market price fields are
 * null whenever Kalshi hasn't opened a market for that side of the game
 * yet -- the prediction itself is still meaningful either way. */
data class MlbGamePrediction(
    val gamePk: Int,
    val awayTeam: String,
    val homeTeam: String,
    val awayAbbreviation: String,
    val homeAbbreviation: String,
    val firstPitchUtc: String,
    val status: String,
    val awayWinProbability: Double,
    val homeWinProbability: Double,
    val awayTicker: String?,
    val homeTicker: String?,
    val awayMarketPricePct: Double?,
    val homeMarketPricePct: Double?,
)

data class MlbPredictions(val date: String, val games: List<MlbGamePrediction>)
