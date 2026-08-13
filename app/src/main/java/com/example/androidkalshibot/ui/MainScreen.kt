package com.example.androidkalshibot.ui

import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.DropdownMenu
import androidx.compose.material3.DropdownMenuItem
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.IconButton
import androidx.compose.material3.Scaffold
import androidx.compose.material3.Tab
import androidx.compose.material3.TabRow
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.material3.TopAppBar
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.rememberCoroutineScope
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import com.example.androidkalshibot.data.ApiClient
import com.example.androidkalshibot.data.BotMode
import com.example.androidkalshibot.data.BotStatus
import com.example.androidkalshibot.data.Position
import com.example.androidkalshibot.data.PushStatus
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.delay
import kotlinx.coroutines.isActive
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext

// Primary row stays visible at all times; MENU_TAB_TITLES moved behind a
// dropdown 2026-08-13 per explicit user request -- 7 tabs in one TabRow
// was too cramped on a phone. selectedTab keeps indexing the CONCATENATION
// of both lists (0..2 = primary, 3..6 = menu), unchanged from before this
// split -- only how the row is drawn changed, not the content dispatch
// below.
private val PRIMARY_TAB_TITLES = listOf("Overview", "Positions", "History")
private val MENU_TAB_TITLES = listOf("Intervals", "BTC Price", "MLB", "Log")

/** Everything below this used to be one long scrollable screen -- split
 * into tabs (Overview / Positions / History / Log) for easier navigation,
 * and 2026-08-04 gained a second axis: which BOT (live real-money vs paper
 * A/B control, see docs/ALGORITHM.md) the four tabs are showing. Status and
 * positions are fetched once here (per selected mode) and shared with the
 * Overview/Positions tabs; History and Log each own their own independent,
 * paginated/polling fetch (see HistoryTab.kt/LogTab.kt) so their state
 * naturally resets when the mode switch changes which server data they
 * should be showing. */
@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun MainScreen(baseUrl: String, username: String, password: String, onOpenSettings: () -> Unit) {
    val client = remember(baseUrl, username, password) { ApiClient(baseUrl, username, password) }
    var selectedMode by remember { mutableStateOf(BotMode.LIVE) }
    var status by remember { mutableStateOf<BotStatus?>(null) }
    var positions by remember { mutableStateOf<List<Position>>(emptyList()) }
    var error by remember { mutableStateOf<String?>(null) }
    var loading by remember { mutableStateOf(true) }
    var actionInProgress by remember { mutableStateOf(false) }
    var selectedTab by remember { mutableStateOf(0) }
    var menuExpanded by remember { mutableStateOf(false) }
    // Null in Paper mode -- all only mean anything for the live
    // (real-money) bot, see ApiClient.getSmallBetsOnly/getBtcDipEnabled/
    // getPushStatus's doc.
    var smallBetsOnly by remember { mutableStateOf<Boolean?>(null) }
    var btcDipEnabled by remember { mutableStateOf<Boolean?>(null) }
    var mlbTradingEnabled by remember { mutableStateOf<Boolean?>(null) }
    var pushStatus by remember { mutableStateOf<PushStatus?>(null) }
    val scope = rememberCoroutineScope()

    suspend fun refresh(mode: BotMode) {
        try {
            val s = withContext(Dispatchers.IO) { client.getStatus(mode) }
            val p = withContext(Dispatchers.IO) { client.getPositions(mode) }
            status = s
            positions = p
            if (mode == BotMode.LIVE) {
                smallBetsOnly = withContext(Dispatchers.IO) { client.getSmallBetsOnly() }
                btcDipEnabled = withContext(Dispatchers.IO) { client.getBtcDipEnabled() }
                mlbTradingEnabled = withContext(Dispatchers.IO) { client.getMlbTradingEnabled() }
                pushStatus = withContext(Dispatchers.IO) { client.getPushStatus() }
            } else {
                smallBetsOnly = null
                btcDipEnabled = null
                mlbTradingEnabled = null
                pushStatus = null
            }
            error = null
        } catch (e: Exception) {
            error = e.message ?: "Unknown error contacting server"
        } finally {
            loading = false
        }
    }

    LaunchedEffect(client, selectedMode) {
        // Clear stale data from whichever mode was showing before -- without
        // this, switching Live -> Paper would briefly keep showing the live
        // balance/positions under the Paper header until the first fetch
        // for the new mode lands.
        status = null
        positions = emptyList()
        loading = true
        while (isActive) {
            refresh(selectedMode)
            delay(REFRESH_INTERVAL_MS)
        }
    }

    Scaffold(
        topBar = {
            Column {
                TopAppBar(
                    title = { Text(if (selectedMode == BotMode.LIVE) "Kalshi Bot — LIVE" else "Kalshi Bot — Paper") },
                    actions = {
                        IconButton(onClick = { scope.launch { refresh(selectedMode) } }) { Text("⟳") }
                        IconButton(onClick = onOpenSettings) { Text("⚙") }
                    },
                )
                TabRow(selectedTabIndex = selectedMode.ordinal) {
                    Tab(
                        selected = selectedMode == BotMode.LIVE,
                        onClick = { selectedMode = BotMode.LIVE },
                        text = { Text("🔴 LIVE (real money)") },
                    )
                    Tab(
                        selected = selectedMode == BotMode.PAPER,
                        onClick = { selectedMode = BotMode.PAPER },
                        text = { Text("📝 PAPER (simulated)") },
                    )
                }
                Row(modifier = Modifier.fillMaxWidth()) {
                    // selectedTabIndex must always be a valid index into THIS
                    // row's own children (0..2) -- coerced rather than using
                    // selectedTab directly, since selectedTab can be 3..6
                    // while a menu destination is showing, which would crash
                    // TabRow's indicator positioning.
                    TabRow(
                        selectedTabIndex = selectedTab.coerceIn(0, PRIMARY_TAB_TITLES.size - 1),
                        modifier = Modifier.weight(1f),
                    ) {
                        PRIMARY_TAB_TITLES.forEachIndexed { index, title ->
                            Tab(
                                selected = selectedTab == index,
                                onClick = { selectedTab = index },
                                text = { Text(title) },
                            )
                        }
                    }
                    Box {
                        val menuActiveIndex = selectedTab - PRIMARY_TAB_TITLES.size
                        val menuLabel = MENU_TAB_TITLES.getOrNull(menuActiveIndex) ?: "More"
                        TextButton(onClick = { menuExpanded = true }) { Text("$menuLabel ▾") }
                        DropdownMenu(expanded = menuExpanded, onDismissRequest = { menuExpanded = false }) {
                            MENU_TAB_TITLES.forEachIndexed { index, title ->
                                DropdownMenuItem(
                                    text = { Text(title) },
                                    onClick = {
                                        selectedTab = PRIMARY_TAB_TITLES.size + index
                                        menuExpanded = false
                                    },
                                )
                            }
                        }
                    }
                }
            }
        },
    ) { padding ->
        Box(modifier = Modifier.fillMaxSize().padding(padding)) {
            if (loading && status == null) {
                CircularProgressIndicator(modifier = Modifier.align(Alignment.Center))
            } else {
                when (selectedTab) {
                    0 -> OverviewTab(
                        status = status,
                        error = error,
                        actionInProgress = actionInProgress,
                        onStart = {
                            scope.launch {
                                actionInProgress = true
                                runCatching { withContext(Dispatchers.IO) { client.startBot(selectedMode) } }
                                refresh(selectedMode)
                                actionInProgress = false
                            }
                        },
                        onStop = {
                            scope.launch {
                                actionInProgress = true
                                runCatching { withContext(Dispatchers.IO) { client.stopBot(selectedMode) } }
                                refresh(selectedMode)
                                actionInProgress = false
                            }
                        },
                        smallBetsOnly = smallBetsOnly,
                        onToggleSmallBetsOnly = { enabled ->
                            scope.launch {
                                runCatching { withContext(Dispatchers.IO) { client.setSmallBetsOnly(enabled) } }
                                refresh(selectedMode)
                            }
                        },
                        btcDipEnabled = btcDipEnabled,
                        onToggleBtcDipEnabled = { enabled ->
                            scope.launch {
                                runCatching { withContext(Dispatchers.IO) { client.setBtcDipEnabled(enabled) } }
                                refresh(selectedMode)
                            }
                        },
                        mlbTradingEnabled = mlbTradingEnabled,
                        onToggleMlbTradingEnabled = { enabled ->
                            scope.launch {
                                runCatching { withContext(Dispatchers.IO) { client.setMlbTradingEnabled(enabled) } }
                                refresh(selectedMode)
                            }
                        },
                        pushStatus = pushStatus,
                    )
                    1 -> PositionsTab(positions = positions)
                    2 -> HistoryTab(client = client, mode = selectedMode)
                    3 -> IntervalsTab(client = client)
                    4 -> BtcPriceTab(client = client)
                    5 -> MlbTab(client = client)
                    6 -> LogTab(client = client, mode = selectedMode)
                }
            }
        }
    }
}
