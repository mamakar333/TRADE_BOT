"""Process control for the paper-trading bot -- mirrors bot_control.py's
PID-file + SIGTERM pattern for the live (real-money) bot, minus the
real-money confirmation step: nothing this process does can ever place a
real order (trade_bot/engine.py's SIMULATION_MODE guard, and
trade_bot/client.py has no POST method anywhere in this codebase), so
there's no equivalent of CONFIRMATION_PHRASE to check here.

Separate module (not a parameterized version of bot_control.py) so the two
bots' process lifecycles can never be accidentally cross-wired -- a bug in
one PID file can't ever affect the other bot's start/stop behavior.

Deliberately does NOT get the live bot's watchdog treatment (see
bot_control.DESIRED_STATE_FILE / run_watchdog.py): the paper bot is meant to
be manually restarted with a button if it stops, per explicit request, not
auto-recovered. Nothing stops that being added later the same way if it
turns out to matter -- just not built ahead of being asked for.
"""
from __future__ import annotations

import os
import signal
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PID_FILE = ROOT / "logs" / "paper_trading.pid"
STDOUT_LOG = ROOT / "logs" / "paper_stdout.log"


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
        return True, pid
    return True, pid


def started_at() -> float | None:
    """Rough process start time, via the pidfile's mtime (written once, at startup)."""
    try:
        return PID_FILE.stat().st_mtime
    except FileNotFoundError:
        return None


def start() -> tuple[bool, str]:
    running, pid = is_running()
    if running:
        return False, f"Already running (PID {pid})."

    STDOUT_LOG.parent.mkdir(parents=True, exist_ok=True)
    with open(STDOUT_LOG, "a") as out:
        subprocess.Popen(
            [sys.executable, str(ROOT / "run_paper_trading.py")],
            cwd=str(ROOT),
            stdout=out,
            stderr=subprocess.STDOUT,
            start_new_session=True,  # survives the dashboard/api process exiting/restarting
        )

    for _ in range(20):
        time.sleep(0.25)
        running, pid = is_running()
        if running:
            return True, f"Started (PID {pid})."
    return False, "Launched the process but it didn't confirm startup within 5s -- check logs/paper_stdout.log."


def stop(timeout_seconds: float = 5.0) -> tuple[bool, str]:
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
