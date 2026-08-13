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
import com.example.androidkalshibot.data.BotMode
import com.example.androidkalshibot.data.Trade
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext

private const val HISTORY_PAGE_SIZE = 10

@Composable
fun HistoryTab(client: ApiClient, mode: BotMode) {
    var trades by remember(client, mode) { mutableStateOf<List<Trade>>(emptyList()) }
    var hasMore by remember(client, mode) { mutableStateOf(false) }
    var loading by remember(client, mode) { mutableStateOf(true) }
    var loadingMore by remember(client, mode) { mutableStateOf(false) }
    var error by remember(client, mode) { mutableStateOf<String?>(null) }
    val scope = rememberCoroutineScope()

    suspend fun loadFirstPage() {
        loading = true
        try {
            val page = withContext(Dispatchers.IO) { client.getTrades(mode, limit = HISTORY_PAGE_SIZE, offset = 0) }
            trades = page.trades
            hasMore = page.hasMore
            error = null
        } catch (e: Exception) {
            error = e.message ?: "Failed to load trade history"
        } finally {
            loading = false
        }
    }

    suspend fun loadMore() {
        if (loadingMore || !hasMore) return
        loadingMore = true
        try {
            val page = withContext(Dispatchers.IO) { client.getTrades(mode, limit = HISTORY_PAGE_SIZE, offset = trades.size) }
            trades = trades + page.trades
            hasMore = page.hasMore
            error = null
        } catch (e: Exception) {
            error = e.message ?: "Failed to load more trades"
        } finally {
            loadingMore = false
        }
    }

    LaunchedEffect(client, mode) { loadFirstPage() }

    val wins = trades.count { (it.realizedPnl ?: 0.0) > 0 }
    val losses = trades.size - wins

    if (loading) {
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
                Column(Modifier.weight(1f)) {
                    Text("Trade History (${trades.size}${if (hasMore) "+" else ""})", style = MaterialTheme.typography.titleMedium, fontWeight = FontWeight.Bold)
                    if (trades.isNotEmpty()) {
                        Text("$wins W / $losses L so far", style = MaterialTheme.typography.bodySmall)
                    }
                }
                IconButton(onClick = { scope.launch { loadFirstPage() } }) { Text("⟳") }
            }
        }
        error?.let { e -> item(key = "error") { ErrorCard(e) } }
        if (trades.isEmpty() && error == null) {
            item(key = "empty") { EmptyHint("No closed trades yet.") }
        }
        items(trades, key = { "${it.openedAt}_${it.ticker}" }) { t -> TradeRow(t) }
        if (hasMore) {
            item(key = "load_more") {
                Box(modifier = Modifier.fillMaxWidth().padding(vertical = 16.dp), contentAlignment = Alignment.Center) {
                    if (loadingMore) {
                        CircularProgressIndicator()
                    } else {
                        OutlinedButton(onClick = { scope.launch { loadMore() } }) { Text("Load $HISTORY_PAGE_SIZE more") }
                    }
                }
            }
        }
    }
}

@Composable
private fun TradeRow(t: Trade) {
    val color = pnlColor(t.realizedPnl)
    Card(modifier = Modifier.fillMaxWidth()) {
        Row(
            modifier = Modifier.fillMaxWidth().padding(horizontal = 16.dp, vertical = 12.dp),
            verticalAlignment = Alignment.CenterVertically,
        ) {
            Column(Modifier.weight(1f)) {
                Text(t.ticker, style = MaterialTheme.typography.bodyMedium, fontWeight = FontWeight.Medium)
                Text(
                    "${t.side} · entry ${"%.1f".format(t.entryPricePct)}%" +
                        (t.exitPricePct?.let { " → exit ${"%.1f".format(it)}%" } ?: ""),
                    style = MaterialTheme.typography.bodySmall,
                )
                t.closeReason?.let {
                    Text(it, style = MaterialTheme.typography.bodySmall, maxLines = 2)
                }
            }
            Text(
                signedMoney(t.realizedPnl),
                color = color,
                fontWeight = FontWeight.Bold,
                style = MaterialTheme.typography.titleMedium,
            )
        }
    }
}
