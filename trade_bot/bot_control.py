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


def start() -> tuple[bool, str]:
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
