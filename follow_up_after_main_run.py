"""
follow_up_after_main_run.py
---------------------------
Sleep-loops until the main v1 fetch process exits, then invokes
fetch_v1_overnight.py once. The follow-up sweep skips every cell that
already has _SUCCESS, so it cleans up any cells whose _SUCCESS was
deliberately removed (e.g. africa/transport after a partial wifi-outage
yield).

Usage
-----
python follow_up_after_main_run.py --wait-pid 32032
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

try:
    import psutil  # standard on this env
except ImportError:
    psutil = None


def is_alive(pid: int) -> bool:
    if psutil is not None:
        try:
            p = psutil.Process(pid)
            return p.is_running() and p.status() != psutil.STATUS_ZOMBIE
        except Exception:
            return False
    # Fallback for Windows without psutil — use tasklist
    out = subprocess.run(
        ["tasklist", "/FI", f"PID eq {pid}"],
        capture_output=True, text=True
    )
    return str(pid) in out.stdout


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--wait-pid", type=int, required=True,
                    help="PID of the main fetch process to wait for.")
    ap.add_argument("--poll-secs", type=int, default=120,
                    help="How often to poll for the main process (default 120s).")
    ap.add_argument("--driver", type=Path,
                    default=Path(__file__).resolve().parent / "fetch_v1_overnight.py")
    args = ap.parse_args()

    print(f"[{datetime.now():%Y-%m-%d %H:%M:%S}] waiting for PID {args.wait_pid} to exit "
          f"(poll every {args.poll_secs}s)", flush=True)

    while is_alive(args.wait_pid):
        time.sleep(args.poll_secs)

    print(f"[{datetime.now():%Y-%m-%d %H:%M:%S}] PID {args.wait_pid} is gone — "
          f"launching follow-up sweep", flush=True)

    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"

    result = subprocess.run(
        [sys.executable, str(args.driver)],
        env=env,
        cwd=str(args.driver.parent),
    )

    print(f"[{datetime.now():%Y-%m-%d %H:%M:%S}] follow-up sweep exited "
          f"with code {result.returncode}", flush=True)


if __name__ == "__main__":
    main()
