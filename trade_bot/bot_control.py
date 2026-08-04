"""Process control for the live (real-money) trading bot -- the dashboard's
kill switch talks to it through this module, and `run_live_trading.py` uses
it to register/deregister itself. There's no other IPC between the two
processes: a PID file plus POSIX signals.

Stopping is deliberately frictionless (SIGTERM, escalating to SIGKILL if it
doesn't exit in time) -- that's the point of a kill switch. Starting is not
gated here; the caller (the dashboard) is responsible for getting explicit
confirmation from the operator before calling `start()`, since this module
sets `LIVE_TRADING_CONFIRMED` on the child process itself.
"""
from __future__ import annotations

import os
import signal
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PID_FILE = ROOT / "logs" / "live_trading.pid"
STDOUT_LOG = ROOT / "logs" / "live_stdout.log"
CONFIRMATION_PHRASE = "yes-i-understand-real-money-is-at-risk"
# Tracks operator INTENT, separate from PID_FILE's "is it actually running"
# -- written "running" by start(), "stopped" by stop(), read by
# watchdog_check() (see below). Found live 2026-08-04: the bot has no
# auto-restart on its own by design (so a manual Stop actually sticks,
# rather than a systemd Restart=always fighting the kill switch), which
# means an unexpected death (crash, an operator-triggered stop that never
# got followed up with a restart, etc.) leaves it silently dead for however
# long until someone happens to notice -- a real ~75min gap was found this
# way. The watchdog (run_watchdog.py, invoked periodically by
# bot-watchdog.timer) restores automatically, but ONLY when this file says
# "running" -- an explicit Stop always wins and stays stopped.
#
# Deliberately does NOT survive a full server reboot resuming trading --
# deploy/README.md documents that as an explicit safety decision ("a reboot
# silently resuming real-money trading without you noticing is worse than
# it staying off"), made before this file existed. watchdog_check() checks
# system boot time against this file's mtime specifically to preserve that:
# a reboot after the last Start resets intent to stopped instead of
# restarting. See _system_boot_time().
DESIRED_STATE_FILE = ROOT / "logs" / "bot_desired_state"


def write_pidfile() -> None:
    PID_FILE.parent.mkdir(parents=True, exist_ok=True)
    PID_FILE.write_text(str(os.getpid()))


def remove_pidfile() -> None:
    try:
        PID_FILE.unlink()
    except FileNotFoundError:
        pass


def is_running() -> tuple[bool, int | None]:
    if not PID_FILE.exists():
        return False, None
    try:
        pid = int(PID_FILE.read_text().strip())
    except ValueError:
        return False, None
    try:
        os.kill(pid, 0)  # signal 0: existence check only, doesn't actually signal the process
    except ProcessLookupError:
        return False, None
    except PermissionError:
        # Process exists but isn't ours to signal -- still "running" for display purposes.
        return True, pid
    return True, pid


def started_at() -> float | None:
    """Rough process start time, via the pidfile's mtime (written once, at startup)."""
    try:
        return PID_FILE.stat().st_mtime
    except FileNotFoundError:
        return None


def _set_desired_state(running: bool) -> None:
    DESIRED_STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    DESIRED_STATE_FILE.write_text("running" if running else "stopped")


def desired_state_is_running() -> bool:
    try:
        return DESIRED_STATE_FILE.read_text().strip() == "running"
    except FileNotFoundError:
        # No explicit Start has ever recorded intent (e.g. a fresh install)
        # -- stay inert rather than guessing the operator wants it running.
        return False


def start() -> tuple[bool, str]:
    # Set BEFORE the early-return too: if the operator hits Start while it's
    # already running (or the watchdog calls this), intent should still end
    # up recorded as "running" regardless of which branch returns.
    _set_desired_state(True)

    running, pid = is_running()
    if running:
        return False, f"Already running (PID {pid})."

    STDOUT_LOG.parent.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["LIVE_TRADING_CONFIRMED"] = CONFIRMATION_PHRASE
    with open(STDOUT_LOG, "a") as out:
        subprocess.Popen(
            [sys.executable, str(ROOT / "run_live_trading.py")],
            cwd=str(ROOT),
            env=env,
            stdout=out,
            stderr=subprocess.STDOUT,
            start_new_session=True,  # survives the dashboard process exiting/restarting
        )

    for _ in range(20):
        time.sleep(0.25)
        running, pid = is_running()
        if running:
            return True, f"Started (PID {pid})."
    return False, "Launched the process but it didn't confirm startup within 5s -- check logs/live_stdout.log."


def stop(timeout_seconds: float = 5.0) -> tuple[bool, str]:
    # Set BEFORE anything else, same reasoning as start(): an explicit Stop
    # must always be recorded as intent immediately, even if is_running()
    # below finds nothing to actually kill -- otherwise the watchdog could
    # win a race and bring it back up right after.
    _set_desired_state(False)

    running, pid = is_running()
    if not running:
        remove_pidfile()  # clean up a stale file from a process that died without cleaning up after itself
        return False, "Not running."

    os.kill(pid, signal.SIGTERM)
    waited = 0.0
    while waited < timeout_seconds:
        time.sleep(0.25)
        waited += 0.25
        still_running, _ = is_running()
        if not still_running:
            return True, f"Stopped (PID {pid})."

    try:
        os.kill(pid, signal.SIGKILL)
    except ProcessLookupError:
        return True, f"Stopped (PID {pid})."
    remove_pidfile()
    return True, f"Force-killed (PID {pid}) after it didn't exit within {timeout_seconds:.0f}s."


def _system_boot_time() -> float | None:
    """Unix timestamp of the last boot, via /proc/uptime -- Linux-only (the
    server), avoids adding a new dependency (psutil) for one number. None on
    a dev machine without /proc/uptime -- callers should treat that as "may
    have rebooted, be conservative" rather than assuming no reboot."""
    try:
        with open("/proc/uptime") as f:
            uptime_seconds = float(f.read().split()[0])
        return time.time() - uptime_seconds
    except (FileNotFoundError, ValueError, IndexError):
        return None


def watchdog_check() -> tuple[bool, str]:
    """Called periodically by bot-watchdog.timer -- brings the bot back if
    it's supposed to be running (last operator action was Start, not Stop)
    but currently isn't. Never overrides an explicit Stop: only acts when
    desired_state_is_running() is True. Returns (acted, message) -- acted is
    False on every normal check where nothing needed doing, so the caller
    can stay quiet and only log/notify on the rare cycle this actually
    fires.

    Deliberately refuses to restart across a full server reboot -- see
    DESIRED_STATE_FILE's docstring. If the system booted more recently than
    the last recorded Start, treat that as "the machine went down and came
    back", reset intent to stopped, and stay off exactly like a cold start
    would, rather than silently resuming real-money trading unattended."""
    if not desired_state_is_running():
        return False, "desired state is stopped -- nothing to do"

    boot_time = _system_boot_time()
    try:
        desired_set_at = DESIRED_STATE_FILE.stat().st_mtime
    except FileNotFoundError:
        desired_set_at = None
    if boot_time is not None and desired_set_at is not None and boot_time > desired_set_at:
        _set_desired_state(False)
        return False, "system rebooted since the last Start -- staying off (never auto-resume across a reboot)"

    running, pid = is_running()
    if running:
        return False, f"already running (PID {pid})"
    ok, message = start()
    return ok, f"watchdog restarted the bot after finding it unexpectedly down: {message}"
