"""Push notifications for real trade events, via ntfy.sh (https://ntfy.sh) --
free, no account required, no Firebase/APNs project to set up. The server
POSTs a short message to a private topic; you subscribe to that same topic
in ntfy's official Android app (Play Store or F-Droid) to get a real OS-
level push notification, even when the app is closed.

Deliberately NOT built as a persistent connection inside the custom
Android dashboard-viewer app (app/) -- a connection kept alive 24/7 in the
background on a phone hits the exact same battery-management problem that
makes running the actual trading engine on a phone unreliable (see
deploy/README.md's reasoning for why the bot itself runs on a server, not
the phone). ntfy's own app already correctly implements real push delivery
via Firebase Cloud Messaging under the hood -- that's the only mechanism
Android has for reliably waking a backgrounded/killed app, and building
that integration ourselves would mean either the same unreliable
persistent-connection trap, or standing up our own Firebase project. Using
ntfy's existing app sidesteps both.
"""
from __future__ import annotations

import logging

import httpx

logger = logging.getLogger(__name__)


def send_notification(topic: str | None, title: str, message: str, priority: str = "default") -> None:
    """Best-effort and silent on failure -- a notification never affects
    trading. `priority`: min/low/default/high/urgent (ntfy's scale)."""
    if not topic:
        return
    try:
        httpx.post(
            f"https://ntfy.sh/{topic}",
            data=message.encode("utf-8"),
            headers={"Title": title, "Priority": priority},
            timeout=5.0,
        )
    except httpx.HTTPError as e:
        logger.warning("ntfy notification failed (trading continues regardless): %s", e)
