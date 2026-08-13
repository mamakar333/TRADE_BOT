package com.example.androidkalshibot.ui

import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.Card
import androidx.compose.material3.CardDefaults
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp

// Semantic trading colors -- deliberately separate from the app's brand
// theme (ui/theme/Color.kt) so P&L always reads the same regardless of
// light/dark mode or any future theme change.
val PnlGood = Color(0xFF0CA30C)
val PnlBad = Color(0xFFD03B3B)
val WarnColor = Color(0xFFB08800)

const val REFRESH_INTERVAL_MS = 10_000L

fun pnlColor(value: Double?): Color = if (value == null) Color.Gray else if (value >= 0) PnlGood else PnlBad
fun correctnessColor(correct: Boolean?): Color = when (correct) { null -> Color.Gray; true -> PnlGood; false -> PnlBad }
fun money(value: Double?): String = if (value == null) "n/a" else "%,.2f".format(value)
fun signedMoney(value: Double?): String = if (value == null) "n/a" else "%+,.2f".format(value)

@Composable
fun ErrorCard(message: String) {
    Card(
        colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.errorContainer),
        modifier = Modifier.fillMaxWidth(),
    ) {
        Column(Modifier.padding(16.dp)) {
            Text("Couldn't reach the server", fontWeight = FontWeight.Bold, color = MaterialTheme.colorScheme.onErrorContainer)
            Text(message, color = MaterialTheme.colorScheme.onErrorContainer, style = MaterialTheme.typography.bodySmall)
        }
    }
}

@Composable
fun StatTile(label: String, value: String, valueColor: Color, modifier: Modifier = Modifier, containerColor: Color? = null) {
    Card(
        modifier = modifier,
        colors = if (containerColor != null) CardDefaults.cardColors(containerColor = containerColor) else CardDefaults.cardColors(),
    ) {
        Column(Modifier.padding(16.dp)) {
            Text(label, style = MaterialTheme.typography.labelMedium)
            Text(
                value,
                style = MaterialTheme.typography.titleLarge,
                fontWeight = FontWeight.Bold,
                color = if (valueColor == Color.Unspecified) MaterialTheme.colorScheme.onSurface else valueColor,
            )
        }
    }
}

@Composable
fun Badge(text: String, color: Color) {
    Surface(color = color, shape = RoundedCornerShape(50)) {
        Text(
            text,
            color = Color.White,
            fontWeight = FontWeight.Bold,
            style = MaterialTheme.typography.labelSmall,
            modifier = Modifier.padding(horizontal = 10.dp, vertical = 4.dp),
        )
    }
}

@Composable
fun StatusDot(color: Color) {
    Surface(color = color, shape = RoundedCornerShape(50), modifier = Modifier.padding(2.dp)) {
        Box(modifier = Modifier.padding(6.dp))
    }
}

@Composable
fun ElevatedCardContainer(containerColor: Color? = null, content: @Composable () -> Unit) {
    Card(
        modifier = Modifier.fillMaxWidth(),
        colors = if (containerColor != null) CardDefaults.cardColors(containerColor = containerColor) else CardDefaults.cardColors(),
    ) {
        Column(Modifier.padding(16.dp)) {
            content()
        }
    }
}

@Composable
fun EmptyHint(text: String) {
    Text(text, style = MaterialTheme.typography.bodyMedium, color = MaterialTheme.colorScheme.onSurfaceVariant)
}
