package com.example.androidkalshibot.push

import android.app.NotificationChannel
import android.app.NotificationManager
import android.app.PendingIntent
import android.content.Context
import android.content.Intent
import android.os.Build
import androidx.core.app.NotificationCompat
import com.example.androidkalshibot.MainActivity
import com.google.firebase.messaging.FirebaseMessagingService
import com.google.firebase.messaging.RemoteMessage

private const val CHANNEL_ID = "trade_events"
private const val CHANNEL_NAME = "Trade events"

/** Receives FCM pushes (trade_bot/push.py on the server -- fires on every
 * real position open/close and on watchdog auto-restarts) and shows them as
 * a real OS notification, including while the app is backgrounded or fully
 * killed -- that's the whole point of FCM over a polling-based in-app
 * alert, see docs/FCM_SETUP.md for why this exists instead of (well,
 * alongside -- ntfy.sh stays on as a fallback) the ntfy-based channel. */
class KalshiFirebaseMessagingService : FirebaseMessagingService() {

    override fun onMessageReceived(message: RemoteMessage) {
        val title = message.notification?.title ?: "Kalshi Trading Bot"
        val body = message.notification?.body ?: return
        showNotification(applicationContext, title, body)
    }

    override fun onNewToken(token: String) {
        // Best-effort -- if this particular attempt fails (server briefly
        // unreachable, no server URL configured yet), the next app launch's
        // AppRoot-level registration call covers it too. Never throws into
        // FCM's own callback regardless.
        Thread {
            try {
                RegisterTokenHelper.register(applicationContext, token)
            } catch (_: Exception) {
                // Silent by design -- same "never affects anything else"
                // contract the server side holds itself to.
            }
        }.start()
    }
}

private fun showNotification(context: Context, title: String, body: String) {
    val manager = context.getSystemService(Context.NOTIFICATION_SERVICE) as NotificationManager
    if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
        val channel = NotificationChannel(CHANNEL_ID, CHANNEL_NAME, NotificationManager.IMPORTANCE_HIGH)
        manager.createNotificationChannel(channel)
    }
    val intent = Intent(context, MainActivity::class.java).apply {
        flags = Intent.FLAG_ACTIVITY_NEW_TASK or Intent.FLAG_ACTIVITY_CLEAR_TOP
    }
    val pendingIntent = PendingIntent.getActivity(
        context, 0, intent,
        PendingIntent.FLAG_UPDATE_CURRENT or PendingIntent.FLAG_IMMUTABLE,
    )
    val notification = NotificationCompat.Builder(context, CHANNEL_ID)
        .setContentTitle(title)
        .setContentText(body)
        .setStyle(NotificationCompat.BigTextStyle().bigText(body))
        // Reuses the app's own launcher icon -- functional and guaranteed
        // to exist, though a dedicated simple monochrome icon would look
        // more correct once Android auto-tints it for the status bar.
        .setSmallIcon(com.example.androidkalshibot.R.mipmap.ic_launcher)
        .setPriority(NotificationCompat.PRIORITY_HIGH)
        .setAutoCancel(true)
        .setContentIntent(pendingIntent)
        .build()
    manager.notify(System.currentTimeMillis().toInt(), notification)
}
