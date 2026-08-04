package com.example.androidkalshibot

import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import com.example.androidkalshibot.data.SecurePrefs
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
    var url by remember { mutableStateOf(prefs.url) }
    var username by remember { mutableStateOf(prefs.username) }
    var password by remember { mutableStateOf(prefs.password) }
    var showSettings by remember { mutableStateOf(url.isBlank()) }

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
