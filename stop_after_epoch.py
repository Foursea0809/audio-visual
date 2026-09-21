"""Stop a VoxMM training process only after its current epoch checkpoint exists."""
from __future__ import annotations

import argparse
import ctypes
import os
import re
import signal
import time
from pathlib import Path


def latest_epoch(log_path: Path) -> int:
    matches = re.findall(r"epoch\s+(\d+):", log_path.read_text(encoding="utf-8", errors="replace"))
    return int(matches[-1]) if matches else 0


PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
PROCESS_TERMINATE = 0x0001


def open_process(pid: int, access: int):
    handle = ctypes.windll.kernel32.OpenProcess(access, False, pid)
    return handle if handle else None


def process_exists(pid: int) -> bool:
    handle = open_process(pid, PROCESS_QUERY_LIMITED_INFORMATION)
    if not handle:
        return False
    ctypes.windll.kernel32.CloseHandle(handle)
    return True


def stop_process(pid: int) -> None:
    handle = open_process(pid, PROCESS_TERMINATE)
    if not handle or not ctypes.windll.kernel32.TerminateProcess(handle, 0):
        raise OSError(f"Could not stop training process {pid}.")
    ctypes.windll.kernel32.CloseHandle(handle)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pid", type=int, required=True)
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--poll-seconds", type=float, default=10.0)
    args = parser.parse_args()
    completed = latest_epoch(args.log)
    print(f"Waiting for epoch after {completed} to finish.", flush=True)
    while True:
        if not process_exists(args.pid):
            print("Training process already stopped.", flush=True)
            return
        current = latest_epoch(args.log)
        if current > completed and args.checkpoint.is_file():
            stop_process(args.pid)
            print(f"Epoch {current} checkpoint recorded; stopped PID {args.pid}.", flush=True)
            return
        time.sleep(args.poll_seconds)


if __name__ == "__main__":
    main()
