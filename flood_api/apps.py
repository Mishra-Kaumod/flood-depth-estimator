from django.apps import AppConfig
import os
import logging

logger = logging.getLogger(__name__)


class FloodApiConfig(AppConfig):
    default_auto_field = "django.db.models.BigAutoField"
    name = "flood_api"

    def ready(self):
        """Start scheduler when Django app is ready."""
        # Skip in migrations or during initial setup
        if os.environ.get("RUN_MAIN") != "true":
            return

        try:
            from flood_api.ml_ops.scheduler import start_scheduler
            start_scheduler()
        except Exception as e:
            logger.warning(f"Could not start scheduler: {e}")

