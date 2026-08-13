package com.example.androidkalshibot.ui

import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.PaddingValues
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.material3.Button
import androidx.compose.material3.Card
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.IconButton
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.rememberCoroutineScope
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import com.example.androidkalshibot.data.ApiClient
import com.example.androidkalshibot.data.Interval
import com.example.androidkalshibot.data.IntervalSummary
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.isActive
import kotlinx.coroutines.launch
import kotlinx.coroutines.delay
import kotlinx.coroutines.withContext

private val ASSET_FILTERS = listOf(null, "BTC", "ETH", "BNB")
private const val INTERVALS_LIMIT = 100

/** Every tracked 15-minute BTC/ETH/BNB interval the live bot has evaluated,
 * traded or not, next to the market's actual settlement result once known.
 * Added 2026-08-05 per explicit user request: the ~200-trade ledger alone
 * is too small a sample to reason about the algorithm's real directional
 * accuracy from -- this tab shows the much larger "final decision vs actual
 * outcome" dataset behind trade_bot/interval_tracker.py, whether or not a
 * given interval ever became a real trade. Not live/paper-scoped, same as
 * OverviewTab's push-token registration -- only the real bot's own market
 * scan feeds this table. */
@Composable
fun IntervalsTab(client: ApiClient) {
    var intervals by remember(client) { mutableStateOf<List<Interval>>(emptyList()) }
    var summary by remember(client) { mutableStateOf<IntervalSummary?>(null) }
    var selectedAsset by remember(client) { mutableStateOf<String?>(null) }
    var loading by remember(client) { mutableStateOf(true) }
    var error by remember(client) { mutableStateOf<String?>(null) }
    val scope = rememberCoroutineScope()

    suspend fun load(asset: String?) {
        try {
            val fetchedIntervals = withContext(Dispatchers.IO) { client.getIntervals(limit = INTERVALS_LIMIT, asset = asset) }
            val fetchedSummary = withContext(Dispatchers.IO) { client.getIntervalsSummary() }
            intervals = fetchedIntervals
            summary = fetchedSummary
            error = null
        } catch (e: Exception) {
            error = e.message ?: "Failed to load interval data"
        } finally {
            loading = false
        }
    }

    LaunchedEffect(client, selectedAsset) {
        loading = true
        while (isActive) {
            load(selectedAsset)
            delay(REFRESH_INTERVAL_MS)
        }
    }

    if (loading && intervals.isEmpty() && error == null) {
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
            Row(modifier = Modifier.fillMaxWidth(), verticalAlignment = Alignment.CenterVertically) {
                Text("Intervals", style = MaterialTheme.typography.titleMedium, fontWeight = FontWeight.Bold, modifier = Modifier.weight(1f))
                IconButton(onClick = { scope.launch { load(selectedAsset) } }) { Text("⟳") }
            }
        }
        summary?.let { s -> item(key = "summary") { IntervalSummaryRow(s) } }
        item(key = "filters") {
            Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                ASSET_FILTERS.forEach { asset ->
                    val label = asset ?: "All"
                    if (selectedAsset == asset) {
                        Button(onClick = { selectedAsset = asset }) { Text(label) }
                    } else {
                        OutlinedButton(onClick = { selectedAsset = asset }) { Text(label) }
                    }
                }
            }
        }
        error?.let { e -> item(key = "error") { ErrorCard(e) } }
        if (intervals.isEmpty() && error == null) {
            item(key = "empty") { EmptyHint("No intervals recorded yet -- these fill in as the live bot's cycles run.") }
        }
        items(intervals, key = { it.ticker }) { i -> IntervalRow(i) }
    }
}

@Composable
private fun IntervalSummaryRow(s: IntervalSummary) {
    val leanAccuracyPct = if (s.leanEvaluatedCount > 0) 100.0 * s.leanCorrectCount / s.leanEvaluatedCount else null
    val tradeAccuracyPct = if (s.tradedCount > 0) 100.0 * s.tradeCorrectCount / s.tradedCount else null
    Row(horizontalArrangement = Arrangement.spacedBy(12.dp), modifier = Modifier.fillMaxWidth()) {
        StatTile(
            label = "Settled",
            value = "${s.settledCount}",
            valueColor = MaterialTheme.colorScheme.onSurface,
            modifier = Modifier.weight(1f),
        )
        StatTile(
            label = "Lean accuracy",
            value = leanAccuracyPct?.let { "%.0f%%".format(it) } ?: "n/a",
            valueColor = pnlColor(leanAccuracyPct?.let { it - 50.0 }),
            modifier = Modifier.weight(1f),
        )
        StatTile(
            label = "Trade accuracy",
            value = tradeAccuracyPct?.let { "%.0f%%".format(it) } ?: "n/a",
            valueColor = pnlColor(tradeAccuracyPct?.let { it - 50.0 }),
            modifier = Modifier.weight(1f),
        )
    }
}

@Composable
private fun IntervalRow(i: Interval) {
    Card(modifier = Modifier.fillMaxWidth()) {
        Column(modifier = Modifier.fillMaxWidth().padding(horizontal = 16.dp, vertical = 12.dp)) {
            Row(modifier = Modifier.fillMaxWidth(), verticalAlignment = Alignment.CenterVertically) {
                Text(i.ticker, style = MaterialTheme.typography.bodyMedium, fontWeight = FontWeight.Medium, modifier = Modifier.weight(1f))
                if (i.result == null) Badge("PENDING", WarnColor)
            }
            Row(modifier = Modifier.fillMaxWidth(), verticalAlignment = Alignment.CenterVertically) {
                val leanText = i.botLean?.let { lean ->
                    "Lean $lean" + (i.lastPredictedProbability?.let { " (%.0f%%)".format(it * 100) } ?: "")
                } ?: "No lean yet"
                Text(leanText, style = MaterialTheme.typography.bodySmall, modifier = Modifier.weight(1f))
                i.leanCorrect?.let { correct ->
                    Badge(if (correct) "LEAN ✓" else "LEAN ✗", correctnessColor(correct))
                }
            }
            if (i.botTraded) {
                Row(modifier = Modifier.fillMaxWidth(), verticalAlignment = Alignment.CenterVertically) {
                    Text(
                        "Traded ${i.tradeSide ?: "?"}" + (i.tradeRealizedPnl?.let { " · ${signedMoney(it)}" } ?: ""),
                        style = MaterialTheme.typography.bodySmall,
                        color = pnlColor(i.tradeRealizedPnl),
                        modifier = Modifier.weight(1f),
                    )
                    i.tradeCorrect?.let { correct ->
                        Badge(if (correct) "TRADE ✓" else "TRADE ✗", correctnessColor(correct))
                    }
                }
            }
            Text(
                "Result: ${i.result ?: "pending"}" + (i.lastYesBidPct?.let { "  ·  last bid ${"%.1f".format(it)}%" } ?: ""),
                style = MaterialTheme.typography.bodySmall,
                color = MaterialTheme.colorScheme.onSurfaceVariant,
            )
        }
    }
}
