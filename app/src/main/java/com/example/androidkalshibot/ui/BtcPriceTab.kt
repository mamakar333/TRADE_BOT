package com.example.androidkalshibot.ui

import androidx.compose.foundation.Canvas
import androidx.compose.foundation.horizontalScroll
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.PaddingValues
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.rememberScrollState
import androidx.compose.material3.Button
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.Path
import androidx.compose.ui.graphics.drawscope.Stroke
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import com.example.androidkalshibot.data.ApiClient
import com.example.androidkalshibot.data.BtcPricePoint
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.delay
import kotlinx.coroutines.isActive
import kotlinx.coroutines.withContext

// Canonical order + labels, matching trade_bot/btc_price_history.py's
// WINDOWS dict and app.py's _BTC_PRICE_WINDOW_LABELS exactly.
private val BTC_PRICE_WINDOWS = listOf(
    "5m" to "5 min", "15m" to "15 min", "1h" to "1 hour", "3h" to "3 hours",
    "1d" to "1 day", "3d" to "3 days", "1w" to "1 week", "1mo" to "1 month",
)
private const val BTC_PRICE_POLL_INTERVAL_MS = 15_000L
private val BTC_ORANGE = Color(0xFFF7931A)

/** Continuous BTC/USD price, collected every second independent of Kalshi
 * and both trading bots -- see trade_bot/btc_price_history.py's docstring
 * for why this is the one feature in the app backed by a non-Kalshi data
 * source. Not mode-scoped (no BotMode parameter): the same one price
 * history applies regardless of which bot tab is selected elsewhere. */
@Composable
fun BtcPriceTab(client: ApiClient) {
    var availableWindows by remember(client) { mutableStateOf<List<String>>(emptyList()) }
    var selectedWindow by remember(client) { mutableStateOf<String?>(null) }
    var points by remember(client) { mutableStateOf<List<BtcPricePoint>>(emptyList()) }
    var loading by remember(client) { mutableStateOf(true) }
    var error by remember(client) { mutableStateOf<String?>(null) }

    // Availability polling and window selection are independent of which
    // window's chart data is currently loaded -- kept in their own effect
    // (keyed only on client) so they don't restart every time selectedWindow
    // changes.
    LaunchedEffect(client) {
        while (isActive) {
            try {
                val availability = withContext(Dispatchers.IO) { client.getBtcPriceAvailability() }
                val nowAvailable = BTC_PRICE_WINDOWS.map { it.first }.filter { availability.windows[it] == true }
                availableWindows = nowAvailable
                if (selectedWindow == null || selectedWindow !in nowAvailable) {
                    selectedWindow = nowAvailable.firstOrNull()
                }
                error = null
            } catch (e: Exception) {
                error = e.message ?: "Failed to load BTC price data"
            } finally {
                loading = false
            }
            delay(BTC_PRICE_POLL_INTERVAL_MS)
        }
    }

    // Chart data fetch is keyed on selectedWindow too -- tapping a different
    // window button must cancel any in-flight poll for the old window and
    // fetch the new one immediately, not wait for the next 15s tick (that
    // was the bug: previously this lived in the effect above, keyed only on
    // client, so a tap only took effect on whatever poll cycle happened to
    // land next).
    LaunchedEffect(client, selectedWindow) {
        val window = selectedWindow ?: return@LaunchedEffect
        while (isActive) {
            try {
                val history = withContext(Dispatchers.IO) { client.getBtcPriceHistory(window) }
                points = history.points
                error = null
            } catch (e: Exception) {
                error = e.message ?: "Failed to load BTC price data"
            }
            delay(BTC_PRICE_POLL_INTERVAL_MS)
        }
    }

    if (loading && availableWindows.isEmpty() && error == null) {
        Box(modifier = Modifier.fillMaxSize()) {
            CircularProgressIndicator(modifier = Modifier.align(Alignment.Center))
        }
        return
    }

    LazyColumn(
        modifier = Modifier.fillMaxSize(),
        contentPadding = PaddingValues(16.dp),
        verticalArrangement = Arrangement.spacedBy(12.dp),
    ) {
        item(key = "header") {
            Text("₿ BTC/USD Price History", style = MaterialTheme.typography.titleMedium, fontWeight = FontWeight.Bold)
        }
        item(key = "caption") {
            Text(
                "Collected every second from a public exchange feed, independent of Kalshi and both trading " +
                    "bots. Longer windows appear automatically once enough time has passed to plot them.",
                style = MaterialTheme.typography.bodySmall,
                color = MaterialTheme.colorScheme.onSurfaceVariant,
            )
        }
        error?.let { e -> item(key = "error") { ErrorCard(e) } }
        if (availableWindows.isEmpty() && error == null) {
            item(key = "empty") {
                EmptyHint("No data collected yet -- the collector just started. Check back in a few minutes.")
            }
        } else {
            item(key = "window_filters") {
                Row(
                    modifier = Modifier.horizontalScroll(rememberScrollState()),
                    horizontalArrangement = Arrangement.spacedBy(8.dp),
                ) {
                    BTC_PRICE_WINDOWS.filter { it.first in availableWindows }.forEach { (key, label) ->
                        if (selectedWindow == key) {
                            Button(onClick = { selectedWindow = key }) { Text(label) }
                        } else {
                            OutlinedButton(onClick = { selectedWindow = key }) { Text(label) }
                        }
                    }
                }
            }
            if (points.size < 2) {
                item(key = "not_enough") { EmptyHint("Not enough data yet for this window.") }
            } else {
                item(key = "chart") {
                    ElevatedCardContainer {
                        PriceLineChart(points = points, modifier = Modifier.fillMaxWidth().height(220.dp))
                    }
                }
                item(key = "stats") { PriceStatsRow(points) }
            }
        }
    }
}

@Composable
private fun PriceStatsRow(points: List<BtcPricePoint>) {
    val prices = points.map { it.price }
    val latest = prices.last()
    val high = prices.max()
    val low = prices.min()
    Row(horizontalArrangement = Arrangement.spacedBy(12.dp), modifier = Modifier.fillMaxWidth()) {
        StatTile("Latest", "$${"%,.2f".format(latest)}", Color.Unspecified, Modifier.weight(1f))
        StatTile("High", "$${"%,.2f".format(high)}", PnlGood, Modifier.weight(1f))
        StatTile("Low", "$${"%,.2f".format(low)}", PnlBad, Modifier.weight(1f))
    }
}

@Composable
private fun PriceLineChart(points: List<BtcPricePoint>, modifier: Modifier = Modifier) {
    val prices = points.map { it.price }
    val minPrice = prices.min()
    val maxPrice = prices.max()
    val priceRange = (maxPrice - minPrice).let { if (it <= 0.0) 1.0 else it }

    Canvas(modifier = modifier.padding(8.dp)) {
        val stepX = if (points.size > 1) size.width / (points.size - 1).toFloat() else 0f
        val path = Path()
        points.forEachIndexed { index, point ->
            val x = index * stepX
            val normalized = ((point.price - minPrice) / priceRange).toFloat()
            val y = size.height - normalized * size.height
            if (index == 0) path.moveTo(x, y) else path.lineTo(x, y)
        }
        drawPath(path, color = BTC_ORANGE, style = Stroke(width = 4f))

        // Faint baseline at the low so the eye has a reference point --
        // this is a line chart, not a candlestick, deliberately simple.
        drawLine(
            color = BTC_ORANGE.copy(alpha = 0.15f),
            start = Offset(0f, size.height),
            end = Offset(size.width, size.height),
            strokeWidth = 1f,
        )
    }
}
