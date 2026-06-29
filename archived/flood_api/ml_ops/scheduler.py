# flood_api/ml_ops/scheduler.py
"""Background scheduler for automatic retraining triggers."""

import logging
import os

logger = logging.getLogger(__name__)

# Use APScheduler for background scheduling
try:
    from apscheduler.schedulers.background import BackgroundScheduler
    SCHEDULER_AVAILABLE = True
except ImportError:
    SCHEDULER_AVAILABLE = False
    logger.warning("APScheduler not installed; background scheduler disabled")

scheduler = None


def start_scheduler():
    """Initialize background scheduler for retraining checks."""
    global scheduler

    if not SCHEDULER_AVAILABLE:
        logger.warning("Scheduler not available (APScheduler not installed)")
        return

    if scheduler and scheduler.running:
        logger.info("Scheduler already running")
        return

    try:
        from flood_api.ml_ops.retraining_trigger import trigger_retraining_if_needed

        scheduler = BackgroundScheduler()

        # Run retraining trigger every 6 hours
        scheduler.add_job(
            trigger_retraining_if_needed,
            "interval",
            hours=6,
            id="retrain_trigger",
            name="Check for retraining trigger",
            max_instances=1,  # Prevent concurrent runs
        )

        scheduler.start()
        logger.info("✅ Background scheduler started (retraining trigger: every 6h)")

    except Exception as e:
        logger.exception(f"Failed to start scheduler: {e}")


def stop_scheduler():
    """Stop background scheduler."""
    global scheduler
    if scheduler and scheduler.running:
        scheduler.shutdown()
        logger.info("Background scheduler stopped")


def get_scheduler_status():
    """Return current scheduler status."""
    if not SCHEDULER_AVAILABLE:
        return {"status": "not_available", "reason": "APScheduler not installed"}

    if scheduler is None:
        return {"status": "not_initialized"}

    if scheduler.running:
        jobs = [
            {
                "id": job.id,
                "name": job.name,
                "next_run_time": str(job.next_run_time),
            }
            for job in scheduler.get_jobs()
        ]
        return {"status": "running", "jobs": jobs}
    else:
        return {"status": "stopped"}
