package com.example.androidkalshibot.push

import android.content.Context
import com.example.androidkalshibot.data.ApiClient
import com.example.androidkalshibot.data.SecurePrefs
import com.google.firebase.messaging.FirebaseMessaging

/** Shared by two call sites -- AppRoot (on every app launch/credential
 * change, in MainActivity.kt) and KalshiFirebaseMessagingService.onNewToken
 * (whenever FCM issues a fresh token, e.g. after a reinstall) -- both need
 * to tell the server the same thing: "push notifications for this trading
 * account should go to this device". */
object RegisterTokenHelper {

    /** Fetches the current FCM token (may already be cached by the SDK, may
     * trigger a fresh registration with Google's servers) and registers it.
     * Best-effort throughout -- a failure here must never surface as a
     * visible error, notifications are inherently non-critical. */
    fun registerCurrentToken(context: Context) {
        FirebaseMessaging.getInstance().token.addOnCompleteListener { task ->
            if (!task.isSuccessful) return@addOnCompleteListener
            val token = task.result ?: return@addOnCompleteListener
            Thread {
                try {
                    register(context, token)
                } catch (_: Exception) {
                    // Silent by design, same contract as everywhere else in
                    // this push pipeline.
                }
            }.start()
        }
    }

    /** Network call -- must run off the main thread (callers already do). */
    fun register(context: Context, token: String) {
        val prefs = SecurePrefs(context)
        if (prefs.url.isBlank()) return  // not configured yet -- nothing to register against
        val client = ApiClient(prefs.url, prefs.username, prefs.password)
        client.registerPushToken(token)
    }
}
