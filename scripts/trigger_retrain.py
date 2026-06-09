#!/usr/bin/env python3
"""
Submits 50 analyst feedback labels against the 5 anomaly flags to trigger
the automatic retraining threshold, then polls /health until a new model
version appears. Run this while the stack is up.

Usage:
    python scripts/trigger_retrain.py
"""

import time
import httpx

API = "http://localhost:8000"
ANALYST = "analyst_retrain_test"
FLAG_IDS = [1, 2, 3, 4, 5]
TARGET = 50


def current_feedback_count() -> int:
    r = httpx.get(f"{API}/health", timeout=5)
    r.raise_for_status()
    return r.json().get("feedback_queue_depth", 0)


def current_model_version() -> str:
    r = httpx.get(f"{API}/health", timeout=5)
    r.raise_for_status()
    versions = r.json().get("model_versions", {})
    return versions.get("IT", "none")


def submit_feedback(flag_id: int, label: str) -> int:
    r = httpx.patch(
        f"{API}/flags/{flag_id}/feedback",
        json={"analyst_id": ANALYST, "label": label},
        timeout=5,
    )
    r.raise_for_status()
    return r.json().get("feedback_count", 0)


def main():
    print("Checking current state...")
    before_count = current_feedback_count()
    before_version = current_model_version()
    print(f"  Feedback queue : {before_count}/50")
    print(f"  Model version  : {before_version}")

    needed = max(0, TARGET - before_count)
    print(f"\nSubmitting {needed} feedback labels to reach {TARGET}...")

    labels = ["true_positive", "true_positive", "true_positive", "false_positive", "escalated"]
    submitted = 0
    while submitted < needed:
        flag_id = FLAG_IDS[submitted % len(FLAG_IDS)]
        label   = labels[submitted % len(labels)]
        count   = submit_feedback(flag_id, label)
        submitted += 1
        print(f"  [{submitted:>3}/{needed}] flag={flag_id} label={label:15s}  queue={count}/50")
        if count >= TARGET:
            print(f"\n  Threshold reached at submission {submitted}.")
            break

    print("\nPolling for retraining to complete (up to 120s)...")
    deadline = time.time() + 120
    while time.time() < deadline:
        version = current_model_version()
        if version != before_version:
            print(f"\n  New model version detected: {version}")
            print("  Retraining completed successfully.")
            break
        time.sleep(5)
        print(f"  Waiting... current version still {version}")
    else:
        print("\n  Timed out waiting for new version.")
        print("  Check retrainer logs: docker compose logs retrainer")
        return

    # Final health check
    r = httpx.get(f"{API}/health", timeout=5)
    health = r.json()
    print("\nFinal health state:")
    print(f"  Status         : {health['status']}")
    print(f"  Feedback queue : {health['feedback_queue_depth']}")
    print(f"  Model versions : {health['model_versions']}")


if __name__ == "__main__":
    main()
