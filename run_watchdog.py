"""Periodic liveness check for the live trading bot -- invoked by
bot-watchdog.timer (systemd), roughly every 3 minutes.

The bot has no auto-restart of its own by design: it's a detached
subprocess (see bot_control.start()), not its own systemd service with
Restart=always, specifically so that pressing Stop actually stops it rather
than fighting a supervisor that keeps bringing it back. That's the right
behavior for an intentional stop, but it also means an *unexpected* death
(a crash, or a stop that never got followed up with a restart) leaves it
silently down for however long until someone happens to notice -- a real
~75min gap on 2026-08-03 night was found and diagnosed this way after the
fact, with no crash/error in the logs at all, just silence.

This script closes that gap without undermining manual control:
watchdog_check() only ever acts when the operator's last action was Start,
never after an explicit Stop (see bot_control.DESIRED_STATE_FILE). It also
never restarts across a full server reboot -- deploy/README.md documents
that as a deliberate safety decision ("a reboot silently resuming
real-money trading without you noticing is worse than it staying off"),
and watchdog_check() checks system boot time specifically to preserve it.
Silent when there's nothing to do; logs and sends a push notification
(best-effort, same as every other trade notification) only on the rare
cycle it actually has to restart something, since real-money operators
should know their bot went down and came back even if they were asleep
for it.

    uv run python run_watchdog.py
"""
import os
from datetime import datetime, timezone

from trade_bot import bot_control
from trade_bot.notifications import send_notification
from trade_bot.push import send_push

EVENT_LOG = bot_control.ROOT / "logs" / "watchdog_events.log"


def main() -> None:
    acted, message = bot_control.watchdog_check()
    if not acted:
        return  # quiet on every normal check -- nothing needed doing

    EVENT_LOG.parent.mkdir(parents=True, exist_ok=True)
    with open(EVENT_LOG, "a") as f:
        f.write(f"{datetime.now(timezone.utc).isoformat()} {message}\n")

    notify_topic = os.getenv("NTFY_TOPIC", "").strip() or None
    send_notification(
        notify_topic,
        "Bot watchdog restarted the trading bot",
        message,
        priority="high",
    )
    send_push("Bot watchdog restarted the trading bot", message)


if __name__ == "__main__":
    main()
