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
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Switch
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.unit.dp
import com.example.androidkalshibot.data.ApiClient
import com.example.androidkalshibot.data.BotMode
import com.example.androidkalshibot.data.LogEntry
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.delay
import kotlinx.coroutines.isActive
import kotlinx.coroutines.withContext

private const val LOG_POLL_INTERVAL_MS = 2_500L
private const val MAX_LOG_LINES_KEPT = 500

/** Logcat-style live feed of the bot's own decision log -- what it's
 * currently evaluating, accepting, and rejecting, for whichever bot
 * (live/paper) is selected up in MainScreen. Polls trade_bot/api.py's
 * /api/live/log or /api/paper/log, using each response's next_offset so
 * repeated polls only ever fetch what's new (see _tail_log's docstring in
 * api.py) rather than re-fetching and re-rendering the whole history every
 * 2.5s. */
@Composable
fun LogTab(client: ApiClient, mode: BotMode) {
    var lines by remember(client, mode) { mutableStateOf<List<LogEntry>>(emptyList()) }
    var nextOffset by remember(client, mode) { mutableStateOf(-1) }
    var loading by remember(client, mode) { mutableStateOf(true) }
    var error by remember(client, mode) { mutableStateOf<String?>(null) }
    var paused by remember(client, mode) { mutableStateOf(false) }

    LaunchedEffect(client, mode) {
        // A fresh start for every distinct (client, mode) combination --
        // otherwise switching from Live to Paper (or back) would carry a
        // stale offset into the OTHER bot's log file and could silently
        // return nothing for a long time.
        lines = emptyList()
        nextOffset = -1
        loading = true
        while (isActive) {
            if (!paused) {
                try {
                    val page = withContext(Dispatchers.IO) { client.getLog(mode, offset = nextOffset) }
                    if (page.entries.isNotEmpty()) {
                        // Newest first, matching the History tab's convention.
                        lines = (page.entries.reversed() + lines).take(MAX_LOG_LINES_KEPT)
                    }
                    nextOffset = page.nextOffset
                    error = null
                } catch (e: Exception) {
                    error = e.message ?: "Failed to load log"
                } finally {
                    loading = false
                }
            }
            delay(LOG_POLL_INTERVAL_MS)
        }
    }

    Column(modifier = Modifier.fillMaxSize()) {
        Row(
            modifier = Modifier.fillMaxWidth().padding(horizontal = 16.dp, vertical = 8.dp),
            verticalAlignment = Alignment.CenterVertically,
        ) {
            Text(
                "Bot thinking (${lines.size})",
                style = MaterialTheme.typography.titleMedium,
                modifier = Modifier.weight(1f),
            )
            Text(if (paused) "Paused" else "Live", style = MaterialTheme.typography.bodySmall)
            Switch(checked = !paused, onCheckedChange = { paused = !it })
        }
        error?.let { e -> ErrorCard(e) }
        if (loading && lines.isEmpty()) {
            Box(modifier = Modifier.fillMaxSize()) {
                CircularProgressIndicator(modifier = Modifier.align(Alignment.Center))
            }
        } else if (lines.isEmpty()) {
            Box(modifier = Modifier.fillMaxWidth().padding(16.dp)) {
                EmptyHint("No decisions logged yet.")
            }
        } else {
            LazyColumn(
                modifier = Modifier.fillMaxSize(),
                contentPadding = PaddingValues(horizontal = 12.dp, vertical = 8.dp),
                verticalArrangement = Arrangement.spacedBy(2.dp),
            ) {
                items(lines, key = { "${it.timestamp}_${it.event}_${it.ticker}" }) { entry ->
                    LogLine(entry)
                }
            }
        }
    }
}

@Composable
private fun LogLine(entry: LogEntry) {
    val event = entry.event ?: ""
    val signal = entry.signal ?: ""
    val (color, prefix) = when {
        event == "fill" -> PnlGood to "FILL"
        event == "decision" && (signal == "BUY_YES" || signal == "BUY_NO") -> PnlGood to "BUY "
        event == "decision" && signal == "SELL" -> Color(0xFF3B82F6) to "SELL"
        event == "ml_gated" -> WarnColor to "GATE"
        event == "risk_blocked" || event == "governor_paused" -> WarnColor to "RISK"
        event == "skip" -> Color.Gray to "SKIP"
        event == "cycle_error" || event == "fetch_error" || event == "order_error" -> PnlBad to "ERR "
        event == "decision" -> Color.Gray to "HOLD"
        else -> MaterialTheme.colorScheme.onSurface to event.uppercase().take(4).padEnd(4)
    }
    val time = entry.timestamp?.let { formatLogTime(it) } ?: "--:--:--"
    val tail = listOfNotNull(entry.ticker, signal.takeIf { it.isNotBlank() }, entry.reason).joinToString(" ")
    Text(
        "$time $prefix $tail",
        color = color,
        fontFamily = FontFamily.Monospace,
        style = MaterialTheme.typography.bodySmall,
    )
}

/** "2026-08-04T17:08:44.678062+00:00" -> "17:08:44". Plain string slicing
 * on purpose -- the timestamp format is entirely our own backend's
 * (datetime.isoformat()), so this is simpler and safer than pulling in a
 * date-parsing dependency for a fixed, known shape. */
private fun formatLogTime(iso: String): String {
    val t = iso.indexOf('T')
    if (t == -1 || iso.length < t + 9) return iso
    return iso.substring(t + 1, t + 9)
}
