"""Native push notifications delivered directly to the custom Android app,
via Firebase Cloud Messaging (FCM) -- added 2026-08-04 per explicit user
request ("build real push into my own app"), supplementing (not replacing)
the ntfy.sh-based channel in notifications.py.

Requires a Firebase project (see docs/FCM_SETUP.md for the one-time setup
only the account owner can do -- creating the project and downloading a
service-account key is tied to their Google account, not something this
code can do on its own) and that key's path in .env as
FIREBASE_SERVICE_ACCOUNT_PATH. Entirely inert (every function here becomes
a silent no-op) until that's configured -- same "never affects trading"
contract notifications.py already holds itself to, checked the same way:
best-effort, catches everything, never raises into a trading code path.

Single-device deployment (one phone), so the registered FCM token is just a
JSON file on disk (TOKEN_FILE below), not a database table -- consistent
with how this codebase already treats other single-value OS-process-
adjacent state (see bot_control.PID_FILE, bot_control.DESIRED_STATE_FILE).
Re-registering (POST /api/notifications/register) simply overwrites it;
FCM tokens can rotate (app reinstall, data clear, Google Play Services
update), and there is deliberately no way here to be subscribed from a
previous, now-stale token -- the newest registration always wins.
"""
from __future__ import annotations

import json
import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)

TOKEN_FILE = Path(__file__).resolve().parent.parent / "logs" / "fcm_device_token.json"

_firebase_app = None
_firebase_init_attempted = False


def _get_firebase_app():
    """Lazy, cached, and safe to call every time even when unconfigured --
    the common case (not set up yet, or the Android side hasn't been built
    yet) must cost nothing beyond one os.getenv() check per call."""
    global _firebase_app, _firebase_init_attempted
    if _firebase_app is not None or _firebase_init_attempted:
        return _firebase_app
    _firebase_init_attempted = True

    service_account_path = os.getenv("FIREBASE_SERVICE_ACCOUNT_PATH", "").strip()
    if not service_account_path:
        # Diagnosed 2026-08-05: this branch used to return silently, so a
        # fully unconfigured setup produced zero log evidence anywhere --
        # the only way to tell push was dead was to SSH in and check env
        # vars by hand. _firebase_init_attempted above means this still
        # only logs once per process lifetime, not once per trade.
        logger.info(
            "FIREBASE_SERVICE_ACCOUNT_PATH not set -- native push notifications disabled "
            "(ntfy fallback in notifications.py still active). See docs/FCM_SETUP.md step 3."
        )
        return None
    if not Path(service_account_path).exists():
        logger.warning("FIREBASE_SERVICE_ACCOUNT_PATH is set but the file doesn't exist: %s", service_account_path)
        return None

    try:
        import firebase_admin
        from firebase_admin import credentials

        cred = credentials.Certificate(service_account_path)
        _firebase_app = firebase_admin.initialize_app(cred)
        logger.info("Firebase Cloud Messaging initialized -- native push notifications active.")
    except Exception as e:
        logger.warning("Firebase init failed (native push disabled, ntfy fallback still active): %s", e)
        _firebase_app = None
    return _firebase_app


def save_device_token(token: str) -> None:
    """Called by the /api/notifications/register endpoint whenever the
    Android app has a token to report (first launch, or FCM issued a new
    one). Overwrites unconditionally -- see module docstring."""
    TOKEN_FILE.parent.mkdir(parents=True, exist_ok=True)
    TOKEN_FILE.write_text(json.dumps({"token": token}))


def _load_device_token() -> str | None:
    try:
        return json.loads(TOKEN_FILE.read_text()).get("token")
    except (FileNotFoundError, ValueError, KeyError):
        return None


def push_status() -> dict:
    """Purely informational (never affects trading) -- lets the dashboard
    and app show WHY native push isn't arriving instead of it being a
    silent dead end only diagnosable by SSHing in and checking env vars,
    which is exactly what happened live 2026-08-05: the device had
    registered fine, but the server had no Firebase credential at all and
    said nothing about it anywhere."""
    return {
        "firebase_configured": _get_firebase_app() is not None,
        "device_registered": _load_device_token() is not None,
    }


def send_push(title: str, message: str) -> None:
    """Best-effort and silent on failure -- identical contract to
    notifications.send_notification, deliberately: a notification must
    never affect trading, regardless of which channel it's on."""
    app = _get_firebase_app()
    if app is None:
        return
    token = _load_device_token()
    if not token:
        logger.info(
            "Firebase is configured but no device token is registered yet -- open the Android "
            "app once (POST /api/notifications/register fires automatically on launch)."
        )
        return
    try:
        from firebase_admin import messaging

        msg = messaging.Message(
            notification=messaging.Notification(title=title, body=message),
            token=token,
        )
        messaging.send(msg, app=app)
    except Exception as e:
        logger.warning("FCM push failed (ntfy fallback still active): %s", e)
