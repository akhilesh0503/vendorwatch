"""
VendorWatch retrainer service — runs as a standalone container.

Startup:
  1. bootstrap() — trains initial models if /models volume is empty.

Scheduled (hourly):
  2. Check feedback count since last retrain.
  3. If count >= FEEDBACK_RETRAIN_THRESHOLD, call train_all().
"""

import logging
import time

import psycopg2
from apscheduler.schedulers.blocking import BlockingScheduler

from src.config import get_settings
from src.services.trainer import bootstrap, feedback_since_last_retrain, train_all

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [retrainer] %(message)s",
)
log = logging.getLogger(__name__)
settings = get_settings()


def _check_and_retrain() -> None:
    try:
        conn  = psycopg2.connect(settings.SYNC_DATABASE_URL)
        count = feedback_since_last_retrain(conn)
        conn.close()
        log.info("Feedback since last retrain: %d (threshold=%d)", count, settings.FEEDBACK_RETRAIN_THRESHOLD)
        if count >= settings.FEEDBACK_RETRAIN_THRESHOLD:
            log.info("Threshold reached — starting retraining.")
            train_all(reason="feedback_threshold")
        else:
            log.info("Below threshold — skipping retrain.")
    except Exception as exc:
        log.error("Scheduled retrain check failed: %s", exc)


def main() -> None:
    log.info("Retrainer service starting…")
    bootstrap()

    scheduler = BlockingScheduler()
    scheduler.add_job(
        _check_and_retrain,
        trigger="interval",
        hours=1,
        id="feedback_retrain",
        max_instances=1,
        coalesce=True,
    )
    log.info("APScheduler running — checking for retrains every hour.")
    scheduler.start()


if __name__ == "__main__":
    main()
