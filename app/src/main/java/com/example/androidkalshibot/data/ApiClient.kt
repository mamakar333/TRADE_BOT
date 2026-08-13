package com.example.androidkalshibot.data

import android.util.Base64
import org.json.JSONArray
import org.json.JSONObject
import java.io.BufferedReader
import java.net.HttpURLConnection
import java.net.URL

/**
 * Deliberately built on plain java.net.HttpURLConnection + org.json instead
 * of Retrofit/OkHttp/kotlinx.serialization -- both are already part of the
 * Android platform, so this adds zero new Gradle dependencies. That matters
 * here specifically because these changes can't be compile-tested locally
 * (no Android SDK/Gradle toolchain available in this environment); fewer
 * new dependency versions to resolve means fewer ways for a build to fail
 * for reasons unrelated to the actual code.
 *
 * Auth: HTTP Basic, same credentials as the web dashboard's Caddy login --
 * see trade_bot/api.py's docstring for why the API itself doesn't also
 * check auth (Caddy already gates every path on the domain).
 */
class ApiException(message: String) : Exception(message)

class ApiClient(private val baseUrl: String, private val username: String, private val password: String) {

    private fun authHeader(): String {
        val raw = "$username:$password"
        return "Basic " + Base64.encodeToString(raw.toByteArray(Charsets.UTF_8), Base64.NO_WRAP)
    }

    private fun request(path: String, method: String = "GET", body: String? = null): String {
        val url = URL(baseUrl.trimEnd('/') + path)
        val conn = url.openConnection() as HttpURLConnection
        try {
            conn.requestMethod = method
            conn.setRequestProperty("Authorization", authHeader())
            conn.setRequestProperty("Accept", "application/json")
            conn.connectTimeout = 10_000
            conn.readTimeout = 10_000
            if (method == "POST") {
                conn.doOutput = true
                val bodyBytes = (body ?: "").toByteArray(Charsets.UTF_8)
                if (body != null) {
                    conn.setRequestProperty("Content-Type", "application/json")
                }
                conn.setRequestProperty("Content-Length", bodyBytes.size.toString())
                conn.outputStream.use { it.write(bodyBytes) }
            }
            val code = conn.responseCode
            val stream = if (code in 200..299) conn.inputStream else conn.errorStream
            val responseBody = stream?.bufferedReader()?.use(BufferedReader::readText) ?: ""
            if (code !in 200..299) {
                throw ApiException("HTTP $code: $responseBody")
            }
            return responseBody
        } finally {
            conn.disconnect()
        }
    }

    fun getStatus(mode: BotMode = BotMode.LIVE): BotStatus {
        val j = JSONObject(request("${mode.apiPrefix}/status"))
        return BotStatus(
            botRunning = j.getBoolean("bot_running"),
            pid = j.optIntOrNull("pid"),
            uptimeSeconds = j.optDoubleOrNull("uptime_seconds"),
            balanceDollars = j.optDoubleOrNull("balance_dollars"),
            totalRealizedPnl = j.getDouble("total_realized_pnl"),
            todayRealizedPnl = j.getDouble("today_realized_pnl"),
            pnl24h = j.getDouble("pnl_24h"),
            pnl7d = j.getDouble("pnl_7d"),
            dailyLossLimit = j.getDouble("daily_loss_limit"),
            dailyKillSwitchEnabled = j.getBoolean("daily_kill_switch_enabled"),
            openPositionCount = j.getInt("open_position_count"),
            governorMultiplier = j.getDouble("governor_multiplier"),
            governorReason = j.getString("governor_reason"),
            learnerNUpdates = j.getInt("learner_n_updates"),
            learnerMinTrades = j.getInt("learner_min_trades"),
            learnerActive = j.getBoolean("learner_active"),
            scope = j.getJSONArray("scope").toStringList(),
            totalDepositedGrossDollars = j.getDouble("total_deposited_gross_dollars"),
            totalDepositFeesDollars = j.getDouble("total_deposit_fees_dollars"),
            totalDepositedNetDollars = j.getDouble("total_deposited_net_dollars"),
            totalWithdrawnDollars = j.getDouble("total_withdrawn_dollars"),
        )
    }

    fun getPositions(mode: BotMode = BotMode.LIVE): List<Position> {
        val arr = JSONArray(request("${mode.apiPrefix}/positions"))
        return (0 until arr.length()).map { i ->
            val j = arr.getJSONObject(i)
            Position(
                ticker = j.getString("ticker"),
                title = j.getString("title"),
                side = j.getString("side"),
                quantity = j.getInt("quantity"),
                entryPricePct = j.getDouble("entry_price_pct"),
                markPct = j.optDoubleOrNull("mark_pct"),
                unrealizedPnl = j.optDoubleOrNull("unrealized_pnl"),
                minutesToClose = j.optDoubleOrNull("minutes_to_close"),
            )
        }
    }

    fun getTrades(mode: BotMode = BotMode.LIVE, limit: Int = 10, offset: Int = 0): TradesPage {
        val j = JSONObject(request("${mode.apiPrefix}/trades?limit=$limit&offset=$offset"))
        val arr = j.getJSONArray("trades")
        val trades = (0 until arr.length()).map { i ->
            val t = arr.getJSONObject(i)
            Trade(
                ticker = t.getString("ticker"),
                side = t.getString("side"),
                quantity = t.getInt("quantity"),
                entryPricePct = t.getDouble("entry_price_pct"),
                exitPricePct = t.optDoubleOrNull("exit_price_pct"),
                fees = t.getDouble("fees"),
                realizedPnl = t.optDoubleOrNull("realized_pnl"),
                openedAt = t.getString("opened_at"),
                closedAt = t.optStringOrNull("closed_at"),
                closeReason = t.optStringOrNull("close_reason"),
            )
        }
        return TradesPage(trades = trades, hasMore = j.getBoolean("has_more"))
    }

    fun startBot(mode: BotMode = BotMode.LIVE): BotActionResult {
        val j = JSONObject(request("${mode.apiPrefix}/bot/start", "POST"))
        return BotActionResult(j.getBoolean("ok"), j.getString("message"))
    }

    fun stopBot(mode: BotMode = BotMode.LIVE): BotActionResult {
        val j = JSONObject(request("${mode.apiPrefix}/bot/stop", "POST"))
        return BotActionResult(j.getBoolean("ok"), j.getString("message"))
    }

    /** Logcat-style decision feed. offset=-1 (default) means "give me the
     * most recent `limit` lines"; pass back a previous call's nextOffset to
     * get only what's new since then -- true incremental polling, not a
     * full re-fetch. Note the server's log endpoints are /api/live/log and
     * /api/paper/log specifically (not /api/paper/... for paper the way
     * every other endpoint here is), since "live"/"paper" are the log
     * selector itself, not a bot-scope prefix -- see trade_bot/api.py. */
    fun getLog(mode: BotMode = BotMode.LIVE, offset: Int = -1, limit: Int = 200): LogPage {
        val path = if (mode == BotMode.LIVE) "/api/live/log" else "/api/paper/log"
        val j = JSONObject(request("$path?offset=$offset&limit=$limit"))
        val arr = j.getJSONArray("entries")
        val entries = (0 until arr.length()).map { i -> LogEntry(arr.getJSONObject(i).toMap()) }
        return LogPage(entries = entries, nextOffset = j.getInt("next_offset"))
    }

    /** Tells the server which device to send native push notifications to
     * (see trade_bot/push.py) -- one bot-wide endpoint, not live/paper
     * -scoped, since it's the same physical phone either way. */
    fun registerPushToken(token: String): BotActionResult {
        val payload = JSONObject().put("token", token).toString()
        val j = JSONObject(request("/api/notifications/register", "POST", body = payload))
        return BotActionResult(j.getBoolean("ok"), j.getString("message"))
    }

    /** Every tracked 15-minute BTC/ETH/BNB interval the live bot has
     * evaluated, traded or not -- see trade_bot/interval_tracker.py. Not
     * live/paper-scoped (only the real bot's own market scan feeds it), so
     * no BotMode parameter, same as registerPushToken above. `asset` must be
     * one of "BTC"/"ETH"/"BNB" (server 400s on anything else) or null for
     * all three. */
    fun getIntervals(limit: Int = 100, asset: String? = null): List<Interval> {
        val assetParam = if (asset != null) "&asset=$asset" else ""
        val j = JSONObject(request("/api/intervals?limit=$limit$assetParam"))
        val arr = j.getJSONArray("intervals")
        return (0 until arr.length()).map { i ->
            val row = arr.getJSONObject(i)
            Interval(
                ticker = row.getString("ticker"),
                asset = row.getString("asset"),
                lastSeenAt = row.getString("last_seen_at"),
                closeTime = row.optStringOrNull("close_time"),
                lastYesBidPct = row.optDoubleOrNull("last_yes_bid_pct"),
                botLean = row.optStringOrNull("bot_lean"),
                lastPredictedProbability = row.optDoubleOrNull("last_predicted_probability"),
                lastSignal = row.optStringOrNull("last_signal"),
                botTraded = row.getBoolean("bot_traded"),
                tradeSide = row.optStringOrNull("trade_side"),
                tradeRealizedPnl = row.optDoubleOrNull("trade_realized_pnl"),
                result = row.optStringOrNull("result"),
                settledAt = row.optStringOrNull("settled_at"),
                leanCorrect = row.optBooleanOrNull("lean_correct"),
                tradeCorrect = row.optBooleanOrNull("trade_correct"),
            )
        }
    }

    fun getIntervalsSummary(): IntervalSummary {
        val j = JSONObject(request("/api/intervals/summary"))
        return IntervalSummary(
            settledCount = j.getInt("settled_count"),
            leanEvaluatedCount = j.getInt("lean_evaluated_count"),
            leanCorrectCount = j.getInt("lean_correct_count"),
            tradedCount = j.getInt("traded_count"),
            tradeCorrectCount = j.getInt("trade_correct_count"),
        )
    }

    /** Live-only (real money) risk toggle -- see trade_bot/live_engine.py's
     * SMALL_BETS_ONLY_CAP_DOLLARS. Not mode-scoped: paper trading is an
     * unthrottled A/B control of the original strategy and has no concept
     * of this toggle, so these are only ever called while BotMode.LIVE is
     * selected (see MainScreen.kt). */
    fun getSmallBetsOnly(): Boolean {
        val j = JSONObject(request("/api/risk-settings"))
        return j.getBoolean("small_bets_only")
    }

    fun setSmallBetsOnly(enabled: Boolean): Boolean {
        val payload = JSONObject().put("small_bets_only", enabled).toString()
        val j = JSONObject(request("/api/risk-settings", "POST", body = payload))
        return j.getBoolean("small_bets_only")
    }

    /** Live-only, KXBTC15M-only practice/test toggle -- see
     * trade_bot/data_driven_strategy.py's btc_dip_enabled field comment.
     * Same non-mode-scoped reasoning as getSmallBetsOnly above. The POST
     * body only ever sends this one field -- /api/risk-settings leaves
     * small_bets_only untouched when it's omitted (see
     * RiskSettingsRequest in trade_bot/api.py). */
    fun getBtcDipEnabled(): Boolean {
        val j = JSONObject(request("/api/risk-settings"))
        return j.getBoolean("btc_dip_enabled")
    }

    fun setBtcDipEnabled(enabled: Boolean): Boolean {
        val payload = JSONObject().put("btc_dip_enabled", enabled).toString()
        val j = JSONObject(request("/api/risk-settings", "POST", body = payload))
        return j.getBoolean("btc_dip_enabled")
    }

    /** Live-only toggle for the MLB game-winner strategy -- a fully
     * separate real-money process from the crypto bot (see
     * trade_bot/mlb_strategy.py, run_mlb_trading.py), not a crypto
     * sub-mechanism. Same non-mode-scoped reasoning as getBtcDipEnabled
     * above -- this toggle has no meaning for paper trading. */
    fun getMlbTradingEnabled(): Boolean {
        val j = JSONObject(request("/api/risk-settings"))
        return j.getBoolean("mlb_trading_enabled")
    }

    fun setMlbTradingEnabled(enabled: Boolean): Boolean {
        val payload = JSONObject().put("mlb_trading_enabled", enabled).toString()
        val j = JSONObject(request("/api/risk-settings", "POST", body = payload))
        return j.getBoolean("mlb_trading_enabled")
    }

    /** Live-only, same non-mode-scoped reasoning as risk-settings above --
     * paper trading never sends push notifications at all. */
    fun getPushStatus(): PushStatus {
        val j = JSONObject(request("/api/push/status"))
        return PushStatus(
            firebaseConfigured = j.getBoolean("firebase_configured"),
            deviceRegistered = j.getBoolean("device_registered"),
        )
    }

    /** Independent of Kalshi and both trading bots entirely -- not mode-
     * scoped, see trade_bot/btc_price_history.py's docstring. */
    fun getBtcPriceAvailability(): BtcPriceAvailability {
        val j = JSONObject(request("/api/btc-price/availability"))
        val windowsJson = j.getJSONObject("windows")
        val windows = mutableMapOf<String, Boolean>()
        val keys = windowsJson.keys()
        while (keys.hasNext()) {
            val key = keys.next()
            windows[key] = windowsJson.getBoolean(key)
        }
        return BtcPriceAvailability(windows)
    }

    fun getBtcPriceHistory(window: String): BtcPriceHistory {
        val j = JSONObject(request("/api/btc-price/history?window=$window"))
        val arr = j.getJSONArray("points")
        val points = (0 until arr.length()).map { i ->
            val p = arr.getJSONObject(i)
            BtcPricePoint(t = p.getString("t"), price = p.getDouble("price"))
        }
        return BtcPriceHistory(window = j.getString("window"), points = points)
    }

    /** Not mode-scoped (no BotMode param): MLB predictions exist
     * independent of live/paper, same reasoning as BTC price above --
     * see trade_bot/api.py's mlb_predictions endpoint. */
    fun getMlbPredictions(date: String? = null): MlbPredictions {
        val path = if (date != null) "/api/mlb/predictions?date=$date" else "/api/mlb/predictions"
        val j = JSONObject(request(path))
        val arr = j.getJSONArray("games")
        val games = (0 until arr.length()).map { i ->
            val g = arr.getJSONObject(i)
            MlbGamePrediction(
                gamePk = g.getInt("game_pk"),
                awayTeam = g.getString("away_team"),
                homeTeam = g.getString("home_team"),
                awayAbbreviation = g.getString("away_abbreviation"),
                homeAbbreviation = g.getString("home_abbreviation"),
                firstPitchUtc = g.getString("first_pitch_utc"),
                status = g.getString("status"),
                awayWinProbability = g.getDouble("away_win_probability"),
                homeWinProbability = g.getDouble("home_win_probability"),
                awayTicker = g.optStringOrNull("away_ticker"),
                homeTicker = g.optStringOrNull("home_ticker"),
                awayMarketPricePct = g.optDoubleOrNull("away_market_price_pct"),
                homeMarketPricePct = g.optDoubleOrNull("home_market_price_pct"),
            )
        }
        return MlbPredictions(date = j.getString("date"), games = games)
    }
}

private fun JSONObject.optIntOrNull(key: String): Int? = if (isNull(key)) null else optInt(key)
private fun JSONObject.optDoubleOrNull(key: String): Double? = if (isNull(key)) null else optDouble(key)
private fun JSONObject.optStringOrNull(key: String): String? = if (isNull(key)) null else optString(key)
private fun JSONObject.optBooleanOrNull(key: String): Boolean? = if (isNull(key)) null else optBoolean(key)
private fun JSONArray.toStringList(): List<String> = (0 until length()).map { getString(it) }

private fun JSONObject.toMap(): Map<String, Any?> {
    val map = mutableMapOf<String, Any?>()
    val iter = keys()
    while (iter.hasNext()) {
        val key = iter.next()
        val value = get(key)
        map[key] = when (value) {
            JSONObject.NULL -> null
            is JSONObject -> value.toMap()
            is JSONArray -> (0 until value.length()).map { value.get(it) }
            else -> value
        }
    }
    return map
}
