package com.example.androidkalshibot.ui

import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.PaddingValues
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.material3.Button
import androidx.compose.material3.ButtonDefaults
import androidx.compose.material3.LinearProgressIndicator
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.Switch
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import com.example.androidkalshibot.data.BotStatus
import com.example.androidkalshibot.data.PushStatus
import kotlin.math.min

@Composable
fun OverviewTab(
    status: BotStatus?,
    error: String?,
    actionInProgress: Boolean,
    onStart: () -> Unit,
    onStop: () -> Unit,
    // Null in Paper mode -- see MainScreen.kt's refresh(), this toggle only
    // means anything for the live (real-money) bot.
    smallBetsOnly: Boolean? = null,
    onToggleSmallBetsOnly: (Boolean) -> Unit = {},
    // Null in Paper mode -- KXBTC15M-only practice/test toggle, live-only.
    btcDipEnabled: Boolean? = null,
    onToggleBtcDipEnabled: (Boolean) -> Unit = {},
    // Null in Paper mode -- MLB is its own separate real-money process
    // (see trade_bot/mlb_strategy.py), not a crypto sub-mechanism, but this
    // toggle still only has meaning while viewing the live (real-money) bot.
    mlbTradingEnabled: Boolean? = null,
    onToggleMlbTradingEnabled: (Boolean) -> Unit = {},
    // Null in Paper mode -- push notifications are a live-only concept.
    pushStatus: PushStatus? = null,
) {
    LazyColumn(
        modifier = Modifier.fillMaxWidth(),
        contentPadding = PaddingValues(16.dp),
        verticalArrangement = Arrangement.spacedBy(12.dp),
    ) {
        error?.let { e -> item(key = "error") { ErrorCard(e) } }
        status?.let { s ->
            item(key = "kill_switch") { KillSwitchCard(s, actionInProgress, onStart, onStop) }
            smallBetsOnly?.let { enabled ->
                item(key = "small_bets_only") { SmallBetsOnlyCard(enabled, onToggleSmallBetsOnly) }
            }
            btcDipEnabled?.let { enabled ->
                item(key = "btc_dip") { BtcDipCard(enabled, onToggleBtcDipEnabled) }
            }
            mlbTradingEnabled?.let { enabled ->
                item(key = "mlb_trading") { MlbTradingCard(enabled, onToggleMlbTradingEnabled) }
            }
            pushStatus?.let { ps -> item(key = "push_status") { PushStatusCard(ps) } }
            item(key = "balance") { BalanceCards(s) }
            item(key = "pnl_windows") { PnlWindowCards(s) }
            item(key = "cash_flow") { CashFlowCard(s) }
            item(key = "daily_loss") { DailyLossCard(s) }
            item(key = "governor") { GovernorCard(s) }
            item(key = "learner") { LearnerCard(s) }
        }
    }
}

@Composable
private fun KillSwitchCard(status: BotStatus, actionInProgress: Boolean, onStart: () -> Unit, onStop: () -> Unit) {
    ElevatedCardContainer(containerColor = if (status.botRunning) MaterialTheme.colorScheme.primaryContainer else null) {
        Row(verticalAlignment = Alignment.CenterVertically, modifier = Modifier.fillMaxWidth()) {
            StatusDot(color = if (status.botRunning) PnlGood else PnlBad)
            Column(Modifier.padding(start = 12.dp).weight(1f)) {
                Text(
                    if (status.botRunning) "RUNNING" else "STOPPED",
                    fontWeight = FontWeight.Bold,
                    style = MaterialTheme.typography.titleMedium,
                )
                status.uptimeSeconds?.let { uptime ->
                    if (status.botRunning) {
                        val mins = uptime / 60
                        val label = if (mins < 60) "up ${mins.toInt()}min" else "up %.1fh".format(mins / 60)
                        Text(label, style = MaterialTheme.typography.bodySmall)
                    }
                }
            }
        }
        Row(
            modifier = Modifier.fillMaxWidth().padding(top = 12.dp),
            horizontalArrangement = Arrangement.spacedBy(12.dp),
        ) {
            Button(
                onClick = onStop,
                enabled = status.botRunning && !actionInProgress,
                colors = ButtonDefaults.buttonColors(containerColor = PnlBad),
                modifier = Modifier.weight(1f),
            ) { Text("STOP") }
            OutlinedButton(
                onClick = onStart,
                enabled = !status.botRunning && !actionInProgress,
                modifier = Modifier.weight(1f),
            ) { Text("START") }
        }
    }
}

@Composable
private fun SmallBetsOnlyCard(enabled: Boolean, onToggle: (Boolean) -> Unit) {
    ElevatedCardContainer {
        Row(verticalAlignment = Alignment.CenterVertically, modifier = Modifier.fillMaxWidth()) {
            Column(Modifier.weight(1f)) {
                Text("🔒 Small bets only", style = MaterialTheme.typography.labelMedium, fontWeight = FontWeight.Bold)
                Text(
                    if (enabled) "ON -- every new entry capped near \$5" else "OFF -- normal sizing, more risk per trade",
                    style = MaterialTheme.typography.bodySmall,
                )
            }
            Switch(checked = enabled, onCheckedChange = onToggle)
        }
    }
}

@Composable
private fun BtcDipCard(enabled: Boolean, onToggle: (Boolean) -> Unit) {
    ElevatedCardContainer {
        Row(verticalAlignment = Alignment.CenterVertically, modifier = Modifier.fillMaxWidth()) {
            Column(Modifier.weight(1f)) {
                Text("🪙 BTC 15-min dip-buy", style = MaterialTheme.typography.labelMedium, fontWeight = FontWeight.Bold)
                Text(
                    if (enabled) "ON -- buys a beaten-down side (20-30c) early, $2-3 stake, practice/test"
                    else "OFF -- normal strategy only",
                    style = MaterialTheme.typography.bodySmall,
                )
            }
            Switch(checked = enabled, onCheckedChange = onToggle)
        }
    }
}

@Composable
private fun MlbTradingCard(enabled: Boolean, onToggle: (Boolean) -> Unit) {
    ElevatedCardContainer {
        Row(verticalAlignment = Alignment.CenterVertically, modifier = Modifier.fillMaxWidth()) {
            Column(Modifier.weight(1f)) {
                Text("⚾ MLB game-winner trading", style = MaterialTheme.typography.labelMedium, fontWeight = FontWeight.Bold)
                Text(
                    if (enabled) "ON -- buys the model's pregame pick when it clears the gate"
                    else "OFF -- MLB process (if running) makes no trades",
                    style = MaterialTheme.typography.bodySmall,
                )
            }
            Switch(checked = enabled, onCheckedChange = onToggle)
        }
    }
}

@Composable
private fun PushStatusCard(status: PushStatus) {
    val (label, color, detail) = when {
        status.firebaseConfigured && status.deviceRegistered ->
            Triple("ACTIVE", PnlGood, "Firebase is configured and this device is registered.")
        status.firebaseConfigured ->
            Triple("NO DEVICE", WarnColor, "Firebase is configured, but no device has registered yet.")
        else ->
            Triple("NOT CONFIGURED", Color.Gray, "Server has no Firebase credential yet -- see docs/FCM_SETUP.md step 3.")
    }
    ElevatedCardContainer {
        Row(verticalAlignment = Alignment.CenterVertically) {
            Text("🔔 Native push", style = MaterialTheme.typography.labelMedium, modifier = Modifier.weight(1f))
            Badge(label, color)
        }
        Text(detail, style = MaterialTheme.typography.bodySmall, modifier = Modifier.padding(top = 4.dp))
    }
}

@Composable
private fun BalanceCards(status: BotStatus) {
    Row(horizontalArrangement = Arrangement.spacedBy(12.dp), modifier = Modifier.fillMaxWidth()) {
        StatTile(
            "Balance", "$${money(status.balanceDollars)}", Color.Unspecified, Modifier.weight(1f),
            containerColor = MaterialTheme.colorScheme.secondaryContainer,
        )
        StatTile("Today's P&L", "$${signedMoney(status.todayRealizedPnl)}", pnlColor(status.todayRealizedPnl), Modifier.weight(1f))
    }
    Row(horizontalArrangement = Arrangement.spacedBy(12.dp), modifier = Modifier.fillMaxWidth().padding(top = 12.dp)) {
        StatTile("Historical P&L (all-time)", "$${signedMoney(status.totalRealizedPnl)}", pnlColor(status.totalRealizedPnl), Modifier.weight(1f))
        StatTile("Open Positions", "${status.openPositionCount}", Color.Unspecified, Modifier.weight(1f))
    }
}

@Composable
private fun PnlWindowCards(status: BotStatus) {
    Row(horizontalArrangement = Arrangement.spacedBy(12.dp), modifier = Modifier.fillMaxWidth()) {
        StatTile("Last 24h P&L", "$${signedMoney(status.pnl24h)}", pnlColor(status.pnl24h), Modifier.weight(1f))
        StatTile("Last 7d P&L", "$${signedMoney(status.pnl7d)}", pnlColor(status.pnl7d), Modifier.weight(1f))
    }
}

@Composable
private fun CashFlowCard(status: BotStatus) {
    ElevatedCardContainer(containerColor = MaterialTheme.colorScheme.tertiaryContainer) {
        Text("💵 Cash flow (separate from trading P&L)", style = MaterialTheme.typography.labelMedium)
        Row(
            modifier = Modifier.fillMaxWidth().padding(top = 8.dp),
            horizontalArrangement = Arrangement.SpaceBetween,
        ) {
            Column {
                Text("Deposited (net)", style = MaterialTheme.typography.bodySmall)
                Text("$${money(status.totalDepositedNetDollars)}", fontWeight = FontWeight.Bold)
            }
            Column {
                Text("Deposit fees", style = MaterialTheme.typography.bodySmall)
                Text("$${money(status.totalDepositFeesDollars)}", fontWeight = FontWeight.Bold)
            }
            Column {
                Text("Withdrawn", style = MaterialTheme.typography.bodySmall)
                Text("$${money(status.totalWithdrawnDollars)}", fontWeight = FontWeight.Bold)
            }
        }
    }
}

@Composable
private fun DailyLossCard(status: BotStatus) {
    ElevatedCardContainer {
        val used = if (status.todayRealizedPnl < 0) -status.todayRealizedPnl else 0.0
        val fraction = if (status.dailyLossLimit > 0) min(1.0, used / status.dailyLossLimit).toFloat() else 0f
        Text(
            if (status.dailyKillSwitchEnabled) "Daily loss kill-switch" else "Daily loss reference (kill-switch disabled)",
            style = MaterialTheme.typography.labelMedium,
        )
        LinearProgressIndicator(
            progress = { fraction },
            modifier = Modifier.fillMaxWidth().padding(vertical = 8.dp),
            color = if (fraction >= 1f && status.dailyKillSwitchEnabled) PnlBad else MaterialTheme.colorScheme.primary,
        )
        Text(
            if (status.todayRealizedPnl < 0)
                "$${"%.2f".format(used)} of $${"%.0f".format(status.dailyLossLimit)}" +
                    (if (status.dailyKillSwitchEnabled) " used" else " (informational only)")
            else
                "$${signedMoney(status.todayRealizedPnl)} today" +
                    (if (status.dailyKillSwitchEnabled) " -- kill-switch not engaged" else ""),
            style = MaterialTheme.typography.bodySmall,
        )
        if (fraction >= 1f) {
            if (status.dailyKillSwitchEnabled) {
                Text(
                    "Kill-switch ENGAGED -- no new entries until UTC midnight",
                    color = PnlBad,
                    fontWeight = FontWeight.Bold,
                    style = MaterialTheme.typography.bodySmall,
                )
            } else {
                Text(
                    "Past the reference figure -- kill-switch disabled, still trading. Stop manually to pause.",
                    color = WarnColor,
                    fontWeight = FontWeight.Bold,
                    style = MaterialTheme.typography.bodySmall,
                )
            }
        }
    }
}

@Composable
private fun GovernorCard(status: BotStatus) {
    ElevatedCardContainer {
        val label = when {
            status.governorMultiplier <= 0.0 -> "PAUSED"
            status.governorMultiplier < 1.0 -> "SIZE HALVED"
            else -> "FULL SIZE"
        }
        val color = when {
            status.governorMultiplier <= 0.0 -> PnlBad
            status.governorMultiplier < 1.0 -> WarnColor
            else -> PnlGood
        }
        Row(verticalAlignment = Alignment.CenterVertically) {
            Text("🐢 Adaptive governor", style = MaterialTheme.typography.labelMedium, modifier = Modifier.weight(1f))
            Badge(label, color)
        }
        Text(status.governorReason, style = MaterialTheme.typography.bodySmall, modifier = Modifier.padding(top = 4.dp))
    }
}

@Composable
private fun LearnerCard(status: BotStatus) {
    ElevatedCardContainer {
        Row(verticalAlignment = Alignment.CenterVertically) {
            Text("🧠 Online learner", style = MaterialTheme.typography.labelMedium, modifier = Modifier.weight(1f))
            Badge(if (status.learnerActive) "ACTIVE" else "WARMING UP", if (status.learnerActive) PnlGood else Color.Gray)
        }
        Text(
            "${status.learnerNUpdates}/${status.learnerMinTrades} learned trades",
            style = MaterialTheme.typography.bodySmall,
            modifier = Modifier.padding(top = 4.dp),
        )
    }
}
