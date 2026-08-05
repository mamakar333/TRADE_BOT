"""Push notifications for real trade events, via ntfy.sh (https://ntfy.sh) --
free, no account required. The server POSTs a short message to a private
topic; subscribing to that topic in ntfy's own Android app (Play Store or
F-Droid) delivers a real OS-level push notification, even when the app is
closed.

Originally the ONLY notification channel, specifically to avoid standing up
a Firebase project (see git history pre-2026-08-04). Superseded 2026-08-04
by trade_bot/push.py (native FCM push straight to the custom Android app,
per explicit user request) -- kept running here in parallel as a fallback
channel, not removed, since it's simple, already ~92% reliable in practice,
and costs nothing to leave on. See push.py's docstring for the FCM side.
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
        resp = httpx.post(
            f"https://ntfy.sh/{topic}",
            data=message.encode("utf-8"),
            headers={"Title": safe_title, "Priority": priority},
            timeout=5.0,
        )
        # Found live 2026-08-04: httpx does NOT raise on a non-2xx status
        # unless raise_for_status() is called -- the response here was never
        # checked at all, so ntfy.sh rejecting a send (49 of 632 sends this
        # week were HTTP 429, rate-limited) was completely invisible. Not
        # calling raise_for_status() (that would just get caught below and
        # logged the same way) -- explicit status check instead, so the log
        # message can name the actual cause.
        if resp.status_code != 200:
            logger.warning(
                "ntfy notification not delivered (HTTP %s): %s (trading continues regardless)",
                resp.status_code, resp.text[:200],
            )
    except Exception as e:
        logger.warning("ntfy notification failed (trading continues regardless): %s", e)
