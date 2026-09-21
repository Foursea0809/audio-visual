"""Stop a GPU training PID after sustained unsafe GTX 1050 temperature."""
from __future__ import annotations

import argparse
import ctypes
import subprocess
import time


PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
PROCESS_TERMINATE = 0x0001


def process_handle(pid: int, access: int):
    handle = ctypes.windll.kernel32.OpenProcess(access, False, pid)
    return handle if handle else None


def exists(pid: int) -> bool:
    handle = process_handle(pid, PROCESS_QUERY_LIMITED_INFORMATION)
    if not handle:
        return False
    ctypes.windll.kernel32.CloseHandle(handle)
    return True


def terminate(pid: int) -> None:
    handle = process_handle(pid, PROCESS_TERMINATE)
    if not handle or not ctypes.windll.kernel32.TerminateProcess(handle, 0):
        raise OSError(f"Could not stop PID {pid}")
    ctypes.windll.kernel32.CloseHandle(handle)


def temperature() -> int:
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=temperature.gpu", "--format=csv,noheader,nounits"],
        check=True, capture_output=True, text=True,
    )
    return int(result.stdout.strip().splitlines()[0])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pid", type=int, required=True)
    parser.add_argument("--threshold", type=int, default=87)
    parser.add_argument("--consecutive", type=int, default=3)
    parser.add_argument("--interval", type=int, default=30)
    args = parser.parse_args()
    hot = 0
    while exists(args.pid):
        value = temperature()
        hot = hot + 1 if value >= args.threshold else 0
        print(f"temperature={value}C consecutive_hot={hot}", flush=True)
        if hot >= args.consecutive:
            terminate(args.pid)
            print(f"Stopped PID {args.pid}: temperature remained >= {args.threshold}C.", flush=True)
            return
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
