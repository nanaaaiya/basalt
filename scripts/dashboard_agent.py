#!/usr/bin/env python3
"""Small, always-on agent that lets the VIO Dashboard remotely
start/stop the basalt_oak_d_vio VIO/SLAM process on this Pi5, instead of
an operator SSHing in and running it by hand.

Connects OUT to the dashboard backend's control port, mirroring how
basalt_oak_d_vio's own DashboardClient connects out for flight telemetry
-- same direction, same newline-delimited-JSON convention, just a
separate port/connection dedicated to process control, so this agent's
own availability doesn't depend on whether a flight is currently
running (see VIO_Dashboard's backend/app/control_agent.py, the other
end of this connection, for the full protocol description).

Meant to run persistently (see dashboard-agent.service in this same
directory for a systemd unit to keep it running across reboots/crashes),
auto-reconnecting on its own if the backend restarts or the network
blips. Stdlib only, deliberately -- nothing to install on the Pi5 beyond
Python itself.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import signal
from typing import Optional

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger("dashboard_agent")

BACKEND_HOST = os.environ.get("DASHBOARD_HOST", "192.168.1.135")
CONTROL_PORT = int(os.environ.get("DASHBOARD_CONTROL_PORT", "8766"))
DATA_PORT = os.environ.get("DASHBOARD_DATA_PORT", "8765")

BASALT_DIR = os.environ.get("BASALT_DIR", os.path.expanduser("~/vio_ws/basalt"))
CALIB_PATH = os.environ.get(
    "CALIB_PATH", os.path.expanduser("~/vio_ws/basalt/results/calibration_final.json")
)
# The flight configuration used throughout this project's live testing:
# loop closure + occupancy mapping on, headless since it's launched
# remotely with nobody at a display. Override via LAUNCH_FLAGS (a plain
# space-separated string) for a different default.
LAUNCH_FLAGS = os.environ.get(
    "LAUNCH_FLAGS",
    "--show-gui false --online-loop-closure true "
    "--enable-occupancy-mapping true --occupancy-depth-stride 2 --occupancy-rate-hz 8",
).split()

# How long a graceful SIGINT gets to finish saving the run log and
# closing the dashboard connection before escalating to SIGKILL -- this
# process has been observed getting stuck in an unkillable device-
# reconnect loop after a hardware crash, so the fallback is a real
# necessity, not just defensive padding.
STOP_GRACE_SECONDS = 10
RECONNECT_DELAY_SECONDS = 5

_proc: Optional[asyncio.subprocess.Process] = None


def _status_message() -> dict:
    running = _proc is not None and _proc.returncode is None
    return {"type": "status", "running": running, "pid": _proc.pid if running else None}


async def _start_process() -> None:
    global _proc
    if _proc is not None and _proc.returncode is None:
        return  # already running -- start is idempotent

    release_dir = os.path.join(BASALT_DIR, "build", "release")
    log_path = os.path.join(release_dir, "dashboard_launch.log")
    log_file = open(log_path, "ab")
    _proc = await asyncio.create_subprocess_exec(
        "./basalt_oak_d_vio",
        "--cam-calib",
        CALIB_PATH,
        *LAUNCH_FLAGS,
        "--dashboard-host",
        BACKEND_HOST,
        "--dashboard-port",
        DATA_PORT,
        cwd=release_dir,
        stdout=log_file,
        stderr=asyncio.subprocess.STDOUT,
    )
    log.info("started basalt_oak_d_vio, pid=%d (log: %s)", _proc.pid, log_path)


async def _stop_process() -> None:
    global _proc
    if _proc is None or _proc.returncode is not None:
        return  # already stopped -- stop is idempotent

    log.info("sending SIGINT to pid=%d", _proc.pid)
    _proc.send_signal(signal.SIGINT)
    try:
        await asyncio.wait_for(_proc.wait(), timeout=STOP_GRACE_SECONDS)
        log.info("stopped gracefully")
    except asyncio.TimeoutError:
        log.warning("did not exit within %ds of SIGINT, sending SIGKILL", STOP_GRACE_SECONDS)
        _proc.kill()
        await _proc.wait()


async def _handle_connection(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
    async def send_status() -> None:
        writer.write((json.dumps(_status_message()) + "\n").encode())
        await writer.drain()

    await send_status()  # initial state, so the backend has something without asking

    while True:
        line = await reader.readline()
        if not line:
            break
        try:
            message = json.loads(line)
        except json.JSONDecodeError:
            continue

        action = message.get("action")
        if action == "start":
            await _start_process()
        elif action == "stop":
            await _stop_process()
        # "status" (or anything else) falls through to just reporting
        # current state, same as start/stop do after acting.
        await send_status()


async def main() -> None:
    while True:
        try:
            log.info("connecting to %s:%d", BACKEND_HOST, CONTROL_PORT)
            reader, writer = await asyncio.open_connection(BACKEND_HOST, CONTROL_PORT)
            log.info("connected")
            try:
                await _handle_connection(reader, writer)
            finally:
                writer.close()
        except (ConnectionRefusedError, OSError) as exc:
            log.warning("connection failed: %s", exc)
        log.info("reconnecting in %ds", RECONNECT_DELAY_SECONDS)
        await asyncio.sleep(RECONNECT_DELAY_SECONDS)


if __name__ == "__main__":
    asyncio.run(main())
