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
import com.example.androidkalshibot.data.MlbGamePrediction
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.delay
import kotlinx.coroutines.isActive
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext

/** Every scheduled MLB game today with the pregame model's predicted win
 * probability for both teams -- see trade_bot/mlb_strategy.py. Added
 * 2026-08-13 per explicit user request: shown regardless of whether the
 * mlb_trading_enabled toggle (OverviewTab) is on, specifically so a
 * prediction is visible even on games the bot itself won't act on --
 * "if bot doesn't place bets I can do manually" -- read-only, no order
 * placement here; this only surfaces what the model thinks and what
 * Kalshi is currently pricing it at. */
@Composable
fun MlbTab(client: ApiClient) {
    var games by remember(client) { mutableStateOf<List<MlbGamePrediction>>(emptyList()) }
    var loading by remember(client) { mutableStateOf(true) }
    var error by remember(client) { mutableStateOf<String?>(null) }
    val scope = rememberCoroutineScope()

    suspend fun load() {
        try {
            val predictions = withContext(Dispatchers.IO) { client.getMlbPredictions() }
            games = predictions.games
            error = null
        } catch (e: Exception) {
            error = e.message ?: "Failed to load MLB predictions"
        } finally {
            loading = false
        }
    }

    LaunchedEffect(client) {
        loading = true
        while (isActive) {
            load()
            delay(REFRESH_INTERVAL_MS)
        }
    }

    if (loading && games.isEmpty() && error == null) {
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
                Text("⚾ Today's MLB Games", style = MaterialTheme.typography.titleMedium, fontWeight = FontWeight.Bold, modifier = Modifier.weight(1f))
                IconButton(onClick = { scope.launch { load() } }) { Text("⟳") }
            }
        }
        item(key = "caption") {
            Text(
                "Pregame model prediction vs. current Kalshi market price. Read-only -- toggle MLB trading "
                    + "on the Overview tab if you want the bot to act on these automatically.",
                style = MaterialTheme.typography.bodySmall,
                color = MaterialTheme.colorScheme.onSurfaceVariant,
            )
        }
        error?.let { e -> item(key = "error") { ErrorCard(e) } }
        if (games.isEmpty() && error == null) {
            item(key = "empty") { EmptyHint("No MLB games scheduled today, or the collector hasn't synced yet.") }
        }
        items(games, key = { it.gamePk }) { g -> MlbGameCard(g) }
    }
}

@Composable
private fun MlbGameCard(g: MlbGamePrediction) {
    Card(modifier = Modifier.fillMaxWidth()) {
        Column(modifier = Modifier.fillMaxWidth().padding(horizontal = 16.dp, vertical = 12.dp)) {
            Row(modifier = Modifier.fillMaxWidth(), verticalAlignment = Alignment.CenterVertically) {
                Text(
                    "${g.awayAbbreviation} @ ${g.homeAbbreviation}",
                    style = MaterialTheme.typography.bodyMedium,
                    fontWeight = FontWeight.Bold,
                    modifier = Modifier.weight(1f),
                )
                if (g.status != "Preview") Badge(g.status.uppercase(), WarnColor)
            }
            Text(
                formatFirstPitch(g.firstPitchUtc),
                style = MaterialTheme.typography.bodySmall,
                color = MaterialTheme.colorScheme.onSurfaceVariant,
            )
            TeamPredictionRow(g.awayTeam, g.awayWinProbability, g.awayMarketPricePct, favored = g.awayWinProbability > g.homeWinProbability)
            TeamPredictionRow(g.homeTeam, g.homeWinProbability, g.homeMarketPricePct, favored = g.homeWinProbability > g.awayWinProbability)
        }
    }
}

@Composable
private fun TeamPredictionRow(team: String, winProbability: Double, marketPricePct: Double?, favored: Boolean) {
    Row(modifier = Modifier.fillMaxWidth(), verticalAlignment = Alignment.CenterVertically) {
        Text(
            team,
            style = MaterialTheme.typography.bodySmall,
            fontWeight = if (favored) FontWeight.Bold else FontWeight.Normal,
            modifier = Modifier.weight(1f),
        )
        Text(
            "model %.0f%%".format(winProbability * 100),
            style = MaterialTheme.typography.bodySmall,
            fontWeight = if (favored) FontWeight.Bold else FontWeight.Normal,
            color = if (favored) PnlGood else MaterialTheme.colorScheme.onSurface,
        )
        Text(
            marketPricePct?.let { "  ·  market %.0f¢".format(it) } ?: "  ·  no market yet",
            style = MaterialTheme.typography.bodySmall,
            color = MaterialTheme.colorScheme.onSurfaceVariant,
        )
    }
}

/** "2026-08-13T20:05:00Z" -> "Aug 13, 8:05 PM UTC" -- kept in UTC rather
 * than converting to device-local time zone, deliberately simple (no
 * java.time zone-database dependency question to worry about across
 * Android API levels for a first version of this tab). */
private fun formatFirstPitch(utcIso: String): String {
    return try {
        val datePart = utcIso.substring(0, 10)
        val timePart = utcIso.substring(11, 16)
        val (hourStr, minute) = timePart.split(":")
        var hour = hourStr.toInt()
        val ampm = if (hour >= 12) "PM" else "AM"
        if (hour == 0) hour = 12 else if (hour > 12) hour -= 12
        "$datePart, $hour:$minute $ampm UTC"
    } catch (e: Exception) {
        utcIso
    }
}
