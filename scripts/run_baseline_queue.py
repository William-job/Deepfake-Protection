"""
Queue runner: runs baseline models sequentially.
Launch after CAR multi-seed training finishes.
Usage: python run_baseline_queue.py
"""
import subprocess
import sys
import time

BASELINES = [
    ("EfficientNet-B3", ["python", "baseline_full.py", "--model", "efficientnet_b3",
                          "--epochs", "15", "--batch_size", "4", "--lr", "0.0001"]),
    ("MesoNet", ["python", "baseline_full.py", "--model", "mesonet",
                  "--epochs", "15", "--batch_size", "8", "--lr", "0.0001"]),
    ("EfficientNet-B0", ["python", "baseline_full.py", "--model", "efficientnet_b0",
                          "--epochs", "15", "--batch_size", "4", "--lr", "0.0001"]),
]

LOG_FILE = "results/baselines/queue_log.txt"

def main():
    with open(LOG_FILE, "a") as log:
        log.write(f"\n{'='*60}\nQueue started: {time.ctime()}\n{'='*60}\n")

    for name, cmd in BASELINES:
        print(f"\n{'='*60}")
        print(f"Starting {name}...")
        print(f"Command: {' '.join(cmd)}")
        print(f"{'='*60}")

        start_time = time.time()
        result = subprocess.run(cmd, capture_output=False, text=True)
        elapsed = (time.time() - start_time) / 3600

        status = "OK" if result.returncode == 0 else f"FAILED (code {result.returncode})"
        msg = f"\n{name}: {status} in {elapsed:.1f}h"
        print(msg)

        with open(LOG_FILE, "a") as log:
            log.write(f"{time.ctime()} | {msg}\n")

        if result.returncode != 0:
            print(f"Stopping queue — {name} failed.")
            sys.exit(1)

    print("\nAll baselines completed!")
    with open(LOG_FILE, "a") as log:
        log.write(f"{time.ctime()} | All baselines completed.\n")

if __name__ == "__main__":
    main()
