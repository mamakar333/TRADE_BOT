# Setting up native push notifications (Firebase Cloud Messaging)

Everything code-side is already built and deployed. What's left needs your
Google account, so it has to happen here first before the Android app will
even build. Two files come out of this: one goes in the Android project,
one goes on the server (never through chat).

## 1. Create the Firebase project

1. Go to [https://console.firebase.google.com](https://console.firebase.google.com) and sign in with your Google account.
2. Click **Add project**. Name it anything (e.g. "Kalshi Trading Bot") -- this name is never shown to you or anyone else, it's just a label in your console.
3. You can decline Google Analytics for this project (not needed).
4. Wait for it to finish provisioning.

## 2. Register the Android app

1. In the new project's dashboard, click the **Android icon** to add an Android app.
2. **Android package name**: enter exactly
   ```
   com.example.androidkalshibot
   ```
   This must match exactly -- it's the `applicationId` in `app/build.gradle.kts`.
3. Nickname/SHA-1 are optional, skip them.
4. Click **Register app**.
5. Download the file it offers you: **`google-services.json`**.
6. Place that file at:
   ```
   app/google-services.json
   ```
   (i.e. directly inside the `app/` folder, next to `build.gradle.kts` -- same level as `src/`.)
7. Skip the "Add Firebase SDK" and "Add initialization code" steps shown in the console -- that's already done in this codebase (see `app/build.gradle.kts` and `gradle/libs.versions.toml`).

**The Android app will not build until this file is present.** If you try to build before this step, you'll see an error like `File google-services.json is missing`, which is expected and just means this step isn't done yet.

## 3. Get a service-account key (for the server to send pushes)

1. In the Firebase console, click the gear icon next to **Project Overview** → **Project settings**.
2. Go to the **Service accounts** tab.
3. Click **Generate new private key**. Confirm. A JSON file downloads.
4. This file is a real credential (it lets whoever holds it send push notifications as your project) -- treat it like any other secret. **Don't paste its contents into chat.**
5. Transfer it directly to the server instead:
   ```
   scp ~/Downloads/<the-downloaded-file>.json kalshi-bot-server:/home/tradebot/TRADE_BOT/firebase-service-account.json
   ```
   (adjust the local filename to whatever it actually downloaded as)
6. On the server, add this line to `/home/tradebot/TRADE_BOT/.env`:
   ```
   FIREBASE_SERVICE_ACCOUNT_PATH=/home/tradebot/TRADE_BOT/firebase-service-account.json
   ```
7. Make sure the file is only readable by the `tradebot` user (matches how every other secret on this server is already handled):
   ```
   ssh kalshi-bot-server "chown tradebot:tradebot /home/tradebot/TRADE_BOT/firebase-service-account.json && chmod 600 /home/tradebot/TRADE_BOT/firebase-service-account.json"
   ```
8. Restart the live bot, dashboard, and api services so they pick up the new `.env` value (ask me to do this once the file is in place -- restarting the live bot needs the same open-position check every other restart in this project gets).

## 4. Rebuild the Android app

Once `app/google-services.json` exists, rebuild in Android Studio as usual. On first launch, the app will:
- Ask for notification permission (Android 13+) -- allow it.
- Silently register this device with the server (`RegisterTokenHelper`, see `MainActivity.kt`).

## 5. Verify

Once both the service-account key is on the server and the app has launched at least once:
- `curl http://127.0.0.1:8502/api/notifications/register` having been called successfully means the app registered -- check `logs/fcm_device_token.json` exists on the server.
- The next real trade fill (or a watchdog restart) will trigger a push. To force a manual test, ask me and I'll fire one from the server directly.

## What happens if this isn't finished

Nothing breaks. `trade_bot/push.py` is inert (a silent no-op) until
`FIREBASE_SERVICE_ACCOUNT_PATH` is set and points at a real file -- trading
is completely unaffected either way, same contract the existing
ntfy.sh-based notifications already hold themselves to. The ntfy channel
stays on in parallel regardless, so you're not down to zero notifications
while this is in progress.
