package com.example.androidkalshibot

import android.Manifest
import android.os.Build
import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.compose.setContent
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.platform.LocalContext
import com.example.androidkalshibot.data.SecurePrefs
import com.example.androidkalshibot.push.RegisterTokenHelper
import com.example.androidkalshibot.ui.MainScreen
import com.example.androidkalshibot.ui.SettingsScreen
import com.example.androidkalshibot.ui.theme.AndroidKalshiBotTheme

/**
 * Fully native UI (Jetpack Compose + Material3) talking to the server's
 * JSON API (trade_bot/api.py) over plain HTTPS -- no WebView anywhere.
 * The web dashboard (Streamlit, app.py) is untouched and still reachable
 * from any browser at the same URL; this app is an independent, native
 * client onto the same real data and the same real bot control.
 */
class MainActivity : ComponentActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContent {
            AndroidKalshiBotTheme {
                AppRoot(SecurePrefs(applicationContext))
            }
        }
    }
}

@Composable
fun AppRoot(prefs: SecurePrefs) {
    val context = LocalContext.current
    var url by remember { mutableStateOf(prefs.url) }
    var username by remember { mutableStateOf(prefs.username) }
    var password by remember { mutableStateOf(prefs.password) }
    var showSettings by remember { mutableStateOf(url.isBlank()) }

    // Android 13+ (API 33) requires this runtime permission before ANY
    // notification -- including one delivered via Firebase Cloud Messaging
    // -- can actually be shown; the manifest declaration alone only makes
    // the permission requestable, it doesn't grant it. No explicit result
    // handling needed: if denied, FCM pushes still arrive and get handed to
    // KalshiFirebaseMessagingService, they just won't produce a visible
    // system notification -- nothing else in the app depends on this.
    val notificationPermissionLauncher = rememberLauncherForActivityResult(
        ActivityResultContracts.RequestPermission()
    ) { }

    LaunchedEffect(Unit) {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.TIRAMISU) {
            notificationPermissionLauncher.launch(Manifest.permission.POST_NOTIFICATIONS)
        }
    }

    // Registers (or re-registers) this device for push notifications
    // whenever real server credentials are available -- covers first setup
    // and any later change of server/account from Settings. See
    // RegisterTokenHelper and trade_bot/push.py.
    LaunchedEffect(url, username, password) {
        if (url.isNotBlank()) {
            RegisterTokenHelper.registerCurrentToken(context)
        }
    }

    if (showSettings) {
        SettingsScreen(
            initialUrl = url,
            initialUser = username,
            initialPass = password,
            canCancel = url.isNotBlank(),
            onSave = { newUrl, newUser, newPass ->
                prefs.saveAll(newUrl, newUser, newPass)
                url = newUrl
                username = newUser
                password = newPass
                showSettings = false
            },
            onCancel = { showSettings = false },
        )
    } else {
        MainScreen(
            baseUrl = url,
            username = username,
            password = password,
            onOpenSettings = { showSettings = true },
        )
    }
}
