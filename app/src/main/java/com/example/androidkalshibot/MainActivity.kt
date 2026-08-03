package com.example.androidkalshibot

import android.annotation.SuppressLint
import android.content.Context
import android.net.Uri
import android.net.http.SslError
import android.os.Bundle
import android.webkit.HttpAuthHandler
import android.webkit.SslErrorHandler
import android.webkit.WebResourceRequest
import android.webkit.WebView
import android.webkit.WebViewClient
import androidx.activity.ComponentActivity
import androidx.activity.compose.BackHandler
import androidx.activity.compose.setContent
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.text.KeyboardOptions
import androidx.compose.material3.Button
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.IconButton
import androidx.compose.material3.LinearProgressIndicator
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.Scaffold
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.material3.TopAppBar
import androidx.compose.runtime.Composable
import androidx.compose.runtime.mutableStateListOf
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.runtime.getValue
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.input.KeyboardType
import androidx.compose.ui.text.input.PasswordVisualTransformation
import androidx.compose.ui.unit.dp
import androidx.compose.ui.viewinterop.AndroidView
import androidx.security.crypto.EncryptedSharedPreferences
import androidx.security.crypto.MasterKey
import com.example.androidkalshibot.ui.theme.AndroidKalshiBotTheme

/**
 * This app does NOT run the trading bot -- the bot and its risk limits run
 * on your server, exactly as designed. This is only a locked-down viewer
 * for the dashboard already protected by Caddy's HTTPS + password gate
 * (see deploy/README.md). It never talks to Kalshi directly and holds no
 * trading credentials -- only the dashboard URL and the login you set for
 * Caddy's basic auth, stored in Android's EncryptedSharedPreferences
 * (backed by the hardware keystore, not plain text, not backed up).
 */
private const val PREFS_FILE = "secure_dashboard_prefs"
private const val KEY_URL = "dashboard_url"
private const val KEY_USER = "dashboard_user"
private const val KEY_PASS = "dashboard_pass"

private fun encryptedPrefs(context: Context) =
    EncryptedSharedPreferences.create(
        context,
        PREFS_FILE,
        MasterKey.Builder(context).setKeyScheme(MasterKey.KeyScheme.AES256_GCM).build(),
        EncryptedSharedPreferences.PrefKeyEncryptionScheme.AES256_SIV,
        EncryptedSharedPreferences.PrefValueEncryptionScheme.AES256_GCM,
    )

class MainActivity : ComponentActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        // Removed enableEdgeToEdge() 2026-08-03: debug log evidence showed the
        // page loading successfully with zero errors, which points at a layout
        // sizing problem rather than a load/auth failure -- edge-to-edge +
        // Scaffold inset handling collapsing the content area to zero size is
        // a known way to get exactly "loaded fine, renders nothing".
        setContent {
            AndroidKalshiBotTheme {
                AppRoot()
            }
        }
    }
}

@Composable
fun AppRoot() {
    val context = LocalContext.current
    val prefs = remember { encryptedPrefs(context) }

    var url by remember { mutableStateOf(prefs.getString(KEY_URL, "") ?: "") }
    var user by remember { mutableStateOf(prefs.getString(KEY_USER, "") ?: "") }
    var pass by remember { mutableStateOf(prefs.getString(KEY_PASS, "") ?: "") }
    var showSettings by remember { mutableStateOf(url.isBlank()) }

    if (showSettings) {
        SettingsScreen(
            initialUrl = url,
            initialUser = user,
            initialPass = pass,
            canCancel = url.isNotBlank(),
            onSave = { newUrl, newUser, newPass ->
                prefs.edit()
                    .putString(KEY_URL, newUrl)
                    .putString(KEY_USER, newUser)
                    .putString(KEY_PASS, newPass)
                    .apply()
                url = newUrl; user = newUser; pass = newPass
                showSettings = false
            },
            onCancel = { showSettings = false },
        )
    } else {
        DashboardScreen(url = url, user = user, pass = pass, onOpenSettings = { showSettings = true })
    }
}

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun SettingsScreen(
    initialUrl: String,
    initialUser: String,
    initialPass: String,
    canCancel: Boolean,
    onSave: (String, String, String) -> Unit,
    onCancel: () -> Unit,
) {
    var url by remember { mutableStateOf(initialUrl) }
    var user by remember { mutableStateOf(initialUser) }
    var pass by remember { mutableStateOf(initialPass) }

    Scaffold(topBar = { TopAppBar(title = { Text("Dashboard connection") }) }) { padding ->
        Column(
            modifier = Modifier.fillMaxSize().padding(padding).padding(24.dp),
            verticalArrangement = Arrangement.spacedBy(16.dp),
        ) {
            Text(
                "Enter the HTTPS address of your dashboard (from deploy/README.md, " +
                    "e.g. https://203-0-113-5.nip.io) and the login you set for it. " +
                    "Stored encrypted on this device only.",
                style = MaterialTheme.typography.bodyMedium,
            )
            OutlinedTextField(
                value = url, onValueChange = { url = it },
                label = { Text("Dashboard URL") }, modifier = Modifier.fillMaxWidth(),
                keyboardOptions = KeyboardOptions(keyboardType = KeyboardType.Uri),
                singleLine = true,
            )
            OutlinedTextField(
                value = user, onValueChange = { user = it },
                label = { Text("Username") }, modifier = Modifier.fillMaxWidth(), singleLine = true,
            )
            OutlinedTextField(
                value = pass, onValueChange = { pass = it },
                label = { Text("Password") }, modifier = Modifier.fillMaxWidth(), singleLine = true,
                visualTransformation = PasswordVisualTransformation(),
            )
            Button(
                onClick = { onSave(url.trim(), user.trim(), pass) },
                enabled = url.trim().startsWith("https://"),
                modifier = Modifier.fillMaxWidth(),
            ) { Text("Save & connect") }
            if (!url.trim().startsWith("https://") && url.isNotBlank()) {
                Text(
                    "URL must start with https:// -- this app refuses plain http on purpose.",
                    color = MaterialTheme.colorScheme.error,
                    style = MaterialTheme.typography.bodySmall,
                )
            }
            if (canCancel) {
                TextButton(onClick = onCancel, modifier = Modifier.fillMaxWidth()) { Text("Cancel") }
            }
        }
    }
}

@OptIn(ExperimentalMaterial3Api::class)
@SuppressLint("SetJavaScriptEnabled")
@Composable
fun DashboardScreen(url: String, user: String, pass: String, onOpenSettings: () -> Unit) {
    val allowedHost = remember(url) { runCatching { Uri.parse(url).host }.getOrNull() }
    var progress by remember { mutableStateOf(0) }
    var webViewRef by remember { mutableStateOf<WebView?>(null) }
    // Debug visibility -- added 2026-08-03 after two blind fix attempts for a
    // blank-screen report that only reproduced on one physical device.
    // Instead of guessing again, this surfaces what the WebView itself sees:
    // every JS console message/error and every network-level load failure,
    // right in the UI, no computer or logcat access needed. Also enables
    // chrome://inspect (see loadLog note below) for full remote DevTools.
    val loadLog = remember { mutableStateListOf<String>() }
    fun log(line: String) {
        loadLog.add(line)
        if (loadLog.size > 200) loadLog.removeAt(0)
    }
    var showDebugLog by remember { mutableStateOf(false) }

    BackHandler(enabled = webViewRef?.canGoBack() == true) {
        webViewRef?.goBack()
    }

    Scaffold(
        topBar = {
            TopAppBar(
                title = { Text("Kalshi Trading Bot") },
                actions = {
                    if (loadLog.isNotEmpty()) {
                        TextButton(onClick = { showDebugLog = !showDebugLog }) {
                            Text("🐛 ${loadLog.size}")
                        }
                    }
                    IconButton(onClick = { webViewRef?.reload() }) { Text("⟳") }
                    IconButton(onClick = onOpenSettings) { Text("⚙") }
                },
            )
        },
    ) { padding ->
        Column(modifier = Modifier.fillMaxSize().padding(padding)) {
            if (progress in 1..99) {
                LinearProgressIndicator(
                    progress = { progress / 100f },
                    modifier = Modifier.fillMaxWidth(),
                )
            }
            if (showDebugLog) {
                Column(
                    modifier = Modifier.fillMaxWidth().padding(8.dp),
                ) {
                    Text("Debug log (newest first) -- send this to Claude if something's wrong:", style = MaterialTheme.typography.labelMedium)
                    loadLog.asReversed().forEach { line ->
                        Text(line, style = MaterialTheme.typography.bodySmall)
                    }
                }
            }
            AndroidView(
                modifier = Modifier.fillMaxSize(),
                factory = { context ->
                    WebView(context).apply {
                        settings.javaScriptEnabled = true
                        settings.domStorageEnabled = true
                        settings.useWideViewPort = true
                        settings.loadWithOverviewMode = true
                        // Full remote DevTools via chrome://inspect on a USB-connected
                        // computer -- the most reliable way to see exactly what's
                        // happening if the log panel below isn't enough.
                        WebView.setWebContentsDebuggingEnabled(true)
                        // Pre-seed credentials instead of only answering the HTTP auth
                        // challenge reactively (below). Confirmed via server-side access
                        // logs 2026-08-03: on a real device, plain HTTP requests (health
                        // checks) carried the Authorization header fine via the reactive
                        // handler, but the WebSocket upgrade Streamlit needs to render
                        // anything never appeared in the logs at all -- WebView doesn't
                        // reliably carry auth credentials over to a WS handshake unless
                        // they're already in its credential store *before* the request.
                        // Without this, the page loads its JS shell (so it looks alive)
                        // but never receives real content -- a blank white screen.
                        allowedHost?.let { h ->
                            if (user.isNotBlank()) {
                                // Realm must match Caddy's exactly (confirmed via
                                // `curl -I` -> `WWW-Authenticate: Basic realm="restricted"`,
                                // Caddy's basic_auth default) or WebView's credential
                                // lookup won't match it and this pre-seeding is a no-op.
                                android.webkit.WebViewDatabase.getInstance(context)
                                    .setHttpAuthUsernamePassword(h, "restricted", user, pass)
                            }
                        }
                        webChromeClient = object : android.webkit.WebChromeClient() {
                            override fun onProgressChanged(view: WebView?, newProgress: Int) {
                                progress = newProgress
                            }

                            // Every JS console.log/warn/error, right in the app UI --
                            // this is where a real rendering bug (as opposed to a
                            // network/auth problem, already ruled out via server logs)
                            // would show up.
                            override fun onConsoleMessage(msg: android.webkit.ConsoleMessage?): Boolean {
                                if (msg != null) {
                                    log("[console/${msg.messageLevel()}] ${msg.message()} (${msg.sourceId()}:${msg.lineNumber()})")
                                }
                                return true
                            }
                        }
                        webViewClient = object : WebViewClient() {
                            // Auto-supply the saved Caddy basic-auth credentials instead of
                            // showing the OS auth dialog every time.
                            override fun onReceivedHttpAuthRequest(
                                view: WebView?, handler: HttpAuthHandler?, host: String?, realm: String?,
                            ) {
                                log("[auth] challenge for host=$host realm=$realm")
                                if (handler != null && user.isNotBlank()) {
                                    handler.proceed(user, pass)
                                } else {
                                    handler?.cancel()
                                }
                            }

                            // Never load a different host than the one configured -- this
                            // viewer only ever talks to your own dashboard.
                            override fun shouldOverrideUrlLoading(
                                view: WebView?, request: WebResourceRequest?,
                            ): Boolean {
                                val reqHost = request?.url?.host
                                return allowedHost != null && reqHost != null && reqHost != allowedHost
                            }

                            // Never silently accept an invalid/self-signed certificate --
                            // a real Let's Encrypt cert (per deploy/README.md) always validates
                            // normally, so this should never fire in normal operation.
                            override fun onReceivedSslError(
                                view: WebView?, handler: SslErrorHandler?, error: SslError?,
                            ) {
                                log("[ssl error] ${error?.primaryError} for ${error?.url}")
                                handler?.cancel()
                            }

                            // Network-level failures (as opposed to HTTP status codes,
                            // which don't land here) -- e.g. connection reset, timeout.
                            override fun onReceivedError(
                                view: WebView?,
                                request: WebResourceRequest?,
                                error: android.webkit.WebResourceError?,
                            ) {
                                if (request?.isForMainFrame == true) {
                                    log("[load error] ${error?.errorCode} ${error?.description} loading ${request.url}")
                                }
                            }

                            // Logs every top-level navigation with a timestamp -- directly
                            // confirms or rules out whether the page is silently reloading
                            // in a loop (suspected from server-side access-log patterns).
                            override fun onPageStarted(view: WebView?, pageUrl: String?, favicon: android.graphics.Bitmap?) {
                                val t = java.text.SimpleDateFormat("HH:mm:ss", java.util.Locale.US).format(java.util.Date())
                                log("[nav] page started @ $t: $pageUrl")
                            }

                            override fun onPageFinished(view: WebView?, pageUrl: String?) {
                                val t = java.text.SimpleDateFormat("HH:mm:ss", java.util.Locale.US).format(java.util.Date())
                                // Direct proof, not another guess: if this ever prints
                                // 0x0 (or near it), the WebView really is being laid out
                                // with no visible area -- confirms the sizing theory
                                // outright instead of inferring it from load timing.
                                log("[nav] page finished @ $t: $pageUrl (view size ${view?.width}x${view?.height})")
                            }
                        }
                        loadUrl(url)
                    }.also { webViewRef = it }
                },
            )
        }
    }
}
