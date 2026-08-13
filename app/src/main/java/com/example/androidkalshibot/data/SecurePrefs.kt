package com.example.androidkalshibot.data

import android.content.Context
import androidx.security.crypto.EncryptedSharedPreferences
import androidx.security.crypto.MasterKey

private const val PREFS_FILE = "secure_dashboard_prefs"
private const val KEY_URL = "dashboard_url"
private const val KEY_USER = "dashboard_user"
private const val KEY_PASS = "dashboard_pass"

/** Keystore-backed encrypted storage for the server URL and Caddy login --
 * never plaintext, never backed up (see AndroidManifest's allowBackup=false). */
class SecurePrefs(context: Context) {
    private val prefs = EncryptedSharedPreferences.create(
        context,
        PREFS_FILE,
        MasterKey.Builder(context).setKeyScheme(MasterKey.KeyScheme.AES256_GCM).build(),
        EncryptedSharedPreferences.PrefKeyEncryptionScheme.AES256_SIV,
        EncryptedSharedPreferences.PrefValueEncryptionScheme.AES256_GCM,
    )

    val url: String get() = prefs.getString(KEY_URL, "") ?: ""
    val username: String get() = prefs.getString(KEY_USER, "") ?: ""
    val password: String get() = prefs.getString(KEY_PASS, "") ?: ""

    fun saveAll(url: String, username: String, password: String) {
        prefs.edit()
            .putString(KEY_URL, url)
            .putString(KEY_USER, username)
            .putString(KEY_PASS, password)
            .apply()
    }
}
