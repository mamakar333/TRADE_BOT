"""Process control for the MLB (real-money) trading bot -- mirrors
trade_bot/bot_control.py's structure exactly (own pidfile, own desired-state
file, own confirmation phrase/env var), a fully independent process from the
crypto live bot with its own start/stop lifecycle, not a sub-mechanism
inside it. See that module's docstring for the full reasoning on why
stopping is frictionless (SIGTERM escalating to SIGKILL) while starting
requires explicit confirmation.

watchdog_check() is included for interface parity with bot_control.py but
is NOT yet wired to a systemd timer (no mlb-watchdog.timer exists) -- an
unexpected crash won't auto-restart until that's added; acceptable for the
initial paper-trading rollout, worth adding before this ever manages real
money unattended.
"""
from __future__ import annotations

import os
import signal
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PID_FILE = ROOT / "logs" / "mlb_trading.pid"
STDOUT_LOG = ROOT / "logs" / "mlb_stdout.log"
CONFIRMATION_PHRASE = "yes-i-understand-mlb-real-money-is-at-risk"
CONFIRMATION_ENV_VAR = "MLB_TRADING_CONFIRMED"
# Same "operator intent, separate from is-it-actually-running" reasoning as
# bot_control.py's DESIRED_STATE_FILE -- also deliberately does NOT survive
# a full server reboot, same safety decision.
DESIRED_STATE_FILE = ROOT / "logs" / "mlb_bot_desired_state"


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
        os.kill(pid, 0)
    except ProcessLookupError:
        return False, None
    except PermissionError:
        return True, pid
    return True, pid


def started_at() -> float | None:
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
        return False


def start() -> tuple[bool, str]:
    _set_desired_state(True)

    running, pid = is_running()
    if running:
        return False, f"Already running (PID {pid})."

    STDOUT_LOG.parent.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env[CONFIRMATION_ENV_VAR] = CONFIRMATION_PHRASE
    with open(STDOUT_LOG, "a") as out:
        subprocess.Popen(
            [sys.executable, str(ROOT / "run_mlb_trading.py")],
            cwd=str(ROOT),
            env=env,
            stdout=out,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )

    for _ in range(20):
        time.sleep(0.25)
        running, pid = is_running()
        if running:
            return True, f"Started (PID {pid})."
    return False, "Launched the process but it didn't confirm startup within 5s -- check logs/mlb_stdout.log."


def stop(timeout_seconds: float = 5.0) -> tuple[bool, str]:
    _set_desired_state(False)

    running, pid = is_running()
    if not running:
        remove_pidfile()
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
    try:
        with open("/proc/uptime") as f:
            uptime_seconds = float(f.read().split()[0])
        return time.time() - uptime_seconds
    except (FileNotFoundError, ValueError, IndexError):
        return None


def watchdog_check() -> tuple[bool, str]:
    """See module docstring -- not yet invoked by any timer."""
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
