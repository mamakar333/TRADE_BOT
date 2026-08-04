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
    """Best-effort and silent on failure -- a notification must never affect
    trading. `priority`: min/low/default/high/urgent (ntfy's scale).

    Bug found live 2026-08-04: emoji in `title` (e.g. an emoji win/loss
    marker) crashed with UnicodeEncodeError when written as a raw HTTP
    header on this server's locale -- HTTP header values are expected to be
    ASCII/Latin-1, unlike the body, which is explicitly UTF-8 encoded below
    and has no such restriction. That exception is a plain Python
    UnicodeEncodeError, not an httpx.HTTPError, so the except clause below
    never caught it -- every real trade close crashed the poll cycle right
    after successfully recording the close (the close itself was already
    safe; only whatever else that cycle would have done next was lost).
    Fixed at the root here (ASCII-safe title, catch-all except) rather than
    just at the one call site that happened to trip it, since this
    function's whole contract is "never affects trading" -- that has to
    hold regardless of what any future caller passes as a title.
    """
    if not topic:
        return
    try:
        safe_title = title.encode("ascii", errors="replace").decode("ascii")
        httpx.post(
            f"https://ntfy.sh/{topic}",
            data=message.encode("utf-8"),
            headers={"Title": safe_title, "Priority": priority},
            timeout=5.0,
        )
    except Exception as e:
        logger.warning("ntfy notification failed (trading continues regardless): %s", e)
