package com.example.androidkalshibot.ui

import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.PaddingValues
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.material3.Card
import androidx.compose.material3.CardDefaults
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import com.example.androidkalshibot.data.Position

private const val POSITIONS_PAGE_SIZE = 5

@Composable
fun PositionsTab(positions: List<Position>) {
    var visibleCount by remember { mutableStateOf(POSITIONS_PAGE_SIZE) }
    val shown = positions.take(visibleCount)

    LazyColumn(
        modifier = Modifier.fillMaxWidth(),
        contentPadding = PaddingValues(16.dp),
        verticalArrangement = Arrangement.spacedBy(12.dp),
    ) {
        item(key = "header") {
            Text("Open Positions (${positions.size})", style = MaterialTheme.typography.titleMedium, fontWeight = FontWeight.Bold)
        }
        if (positions.isEmpty()) {
            item(key = "empty") { EmptyHint("No open positions right now.") }
        } else {
            items(shown, key = { "pos_${it.ticker}" }) { p -> PositionCard(p) }
            if (visibleCount < positions.size) {
                item(key = "load_more") {
                    Box(modifier = Modifier.fillMaxWidth(), contentAlignment = Alignment.Center) {
                        OutlinedButton(onClick = { visibleCount += POSITIONS_PAGE_SIZE }) {
                            Text("Load ${minOf(POSITIONS_PAGE_SIZE, positions.size - visibleCount)} more (${positions.size - visibleCount} remaining)")
                        }
                    }
                }
            }
        }
    }
}

@Composable
private fun PositionCard(p: Position) {
    val pnl = p.unrealizedPnl
    val color = pnlColor(pnl)
    Card(
        modifier = Modifier.fillMaxWidth(),
        colors = CardDefaults.cardColors(
            containerColor = if (pnl != null && pnl >= 0) MaterialTheme.colorScheme.primaryContainer else MaterialTheme.colorScheme.surface,
        ),
    ) {
        Column(Modifier.padding(16.dp)) {
            Row(verticalAlignment = Alignment.CenterVertically) {
                Text(p.title, fontWeight = FontWeight.Bold, style = MaterialTheme.typography.bodyLarge, modifier = Modifier.weight(1f))
                Badge(if (pnl == null) "PENDING" else if (pnl >= 0) "WINNING" else "LOSING", color)
            }
            Text(p.ticker, style = MaterialTheme.typography.bodySmall)
            Row(modifier = Modifier.padding(top = 8.dp), verticalAlignment = Alignment.CenterVertically) {
                Text(
                    "${if (p.side == "YES") "🟢" else "🔴"} ${p.side} · ${p.quantity} contracts",
                    style = MaterialTheme.typography.bodyMedium,
                )
            }
            Text(
                "entry ${"%.1f".format(p.entryPricePct)}%" +
                    (p.markPct?.let { " → mark ${"%.1f".format(it)}%" } ?: "") +
                    (p.minutesToClose?.let { " · ${"%.0f".format(it)}min to close" } ?: ""),
                style = MaterialTheme.typography.bodySmall,
            )
            Text(
                "${if ((pnl ?: 0.0) >= 0) "▲" else "▼"} ${signedMoney(pnl)} unrealized",
                color = color,
                fontWeight = FontWeight.Bold,
                style = MaterialTheme.typography.titleMedium,
                modifier = Modifier.padding(top = 8.dp),
            )
        }
    }
}
