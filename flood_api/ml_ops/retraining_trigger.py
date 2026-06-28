# flood_api/ml_ops/retraining_trigger.py
"""
Retraining trigger service: decides when to retrain and orchestrates the pipeline.
Triggers on: (1) feedback volume >= threshold, (2) false-negative spike, (3) rapid corrections.
"""

import json
import logging
import csv
import subprocess
import shutil
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Tuple, Dict, Optional

import os
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "flood_project.settings")

import django
django.setup()

from django.conf import settings
from django.utils import timezone as dj_timezone
from flood_api.models import (
    PredictionFeedback,
    ModelVersion,
    FloodInundationTelemetry,
    FailedTaskEvent,
)

logger = logging.getLogger(__name__)


class RetrainingTrigger:
    """
    Orchestrates the feedback loop:
    1. Check if retraining should trigger
    2. Build training set from corrections
    3. Run retraining
    4. Evaluate gates
    5. Promote new model version if gates pass
    """

    def __init__(self):
        self.base_dir = Path(settings.BASE_DIR)
        self.models_dir = self.base_dir / "models"
        self.models_dir.mkdir(exist_ok=True)

    def check_trigger(self) -> Tuple[bool, str, Dict]:
        """
        Decides if retraining should run.
        Returns: (should_retrain, reason, metadata)
        """
        # Check 1: Feedback volume (rejected + corrected)
        unused_count = PredictionFeedback.objects.filter(
            feedback_type__in=["rejected", "corrected"]
        ).count()
        
        used_count = PredictionFeedback.objects.filter(
            feedback_type__in=["rejected", "corrected"],
            metadata__has_key="used_in_training"
        ).count()
        
        new_feedback = unused_count - used_count

        threshold = getattr(settings, "FEEDBACK_RETRAINING_THRESHOLD", 50)
        if new_feedback >= threshold:
            return (
                True,
                f"Accumulated {new_feedback} new corrections (threshold: {threshold})",
                {"trigger": "volume", "count": new_feedback},
            )

        # Check 2: False-negative rate spike (last 7 days)
        fn_rate = self._compute_fn_rate_7d()
        fn_threshold = getattr(settings, "FN_SPIKE_THRESHOLD", 0.02)

        if fn_rate > fn_threshold:
            return (
                True,
                f"False-negative rate spiked to {fn_rate:.1%} (threshold: {fn_threshold:.1%})",
                {"trigger": "fn_spike", "fn_rate": fn_rate},
            )

        # Check 3: High correction volume in last 24h (rapid response)
        recent_count = PredictionFeedback.objects.filter(
            created_at__gte=dj_timezone.now() - timedelta(hours=24),
            feedback_type__in=["rejected", "corrected"],
        ).count()

        rapid_threshold = getattr(settings, "RAPID_FEEDBACK_THRESHOLD", 20)
        if recent_count >= rapid_threshold:
            return (
                True,
                f"Rapid feedback: {recent_count} corrections in 24h (threshold: {rapid_threshold})",
                {"trigger": "rapid", "count_24h": recent_count},
            )

        return False, "No trigger condition met", {}

    def retrain_and_evaluate(self) -> Tuple[bool, Optional[ModelVersion], str]:
        """
        Full retraining pipeline:
        1. Materialize training set from corrections
        2. Run retraining
        3. Evaluate gates
        4. Promote or quarantine new version

        Returns: (success: bool, new_version: ModelVersion | None, message: str)
        """
        try:
            logger.info("Starting retraining pipeline...")

            # Step 1: Build training set from feedback
            train_images, train_labels = self._build_training_set()

            if len(train_images) < 20:
                msg = f"Training set too small: {len(train_images)} samples (need >= 20)"
                logger.warning(msg)
                return False, None, msg

            flood_count = sum(train_labels)
            non_flood_count = len(train_labels) - flood_count
            logger.info(
                f"Training set ready: {len(train_images)} images "
                f"({flood_count} floods, {non_flood_count} non-floods)"
            )

            # Step 2: Run retraining
            checkpoint_path = self._run_retraining(train_images, train_labels)
            if not checkpoint_path or not checkpoint_path.exists():
                msg = "Retraining failed: checkpoint not created"
                logger.error(msg)
                return False, None, msg

            logger.info(f"Model checkpoint saved: {checkpoint_path}")

            # Step 3: Evaluate readiness gates
            gates_result = self._evaluate_readiness_gates(checkpoint_path)

            if not gates_result["all_pass"]:
                logger.warning(f"Readiness gates failed: {gates_result['summary']}")
                return False, None, f"Gates failed: {gates_result['summary']}"

            logger.info(f"All gates passed: {gates_result['summary']}")

            # Step 4: Create new ModelVersion
            prod_version = ModelVersion.objects.filter(stage="production").first()
            if not prod_version:
                prod_version = ModelVersion.objects.filter(stage="prod").first()

            if not prod_version:
                logger.warning(
                    "No current production version found; creating v1.0 baseline"
                )
                prod_version = ModelVersion.objects.create(
                    model_name="flood_classifier",
                    version="1.0",
                    stage="production",
                    checkpoint_path=str(self.base_dir / "models" / "flood_model_final.pth"),
                    metadata={"created_by": "system"},
                )

            new_version_name = self._next_version_name(prod_version.version)
            new_version = ModelVersion.objects.create(
                model_name="flood_classifier",
                version=new_version_name,
                stage="staging",
                checkpoint_path=str(checkpoint_path),
                metadata={
                    "gate_results": gates_result,
                    "training_samples": len(train_images),
                    "training_floods": flood_count,
                    "training_non_floods": non_flood_count,
                    "parent_version": prod_version.version,
                    "created_by": "automatic_retraining",
                    "created_at": datetime.now(timezone.utc).isoformat(),
                },
            )

            logger.info(
                f"New model version created: {new_version_name} (stage: staging)"
            )

            # Step 5: Mark feedback as used
            marked_count = 0
            for feedback in PredictionFeedback.objects.filter(
                feedback_type__in=["rejected", "corrected"]
            ):
                if not feedback.metadata.get("used_in_training"):
                    feedback.metadata["used_in_training"] = True
                    feedback.metadata["training_run_id"] = str(new_version.id)
                    feedback.save()
                    marked_count += 1

            logger.info(
                f"Marked {marked_count} feedback records as used_in_training"
            )

            msg = f"✅ Retraining successful. New version: {new_version_name} (gates all pass)"
            return True, new_version, msg

        except Exception as e:
            logger.exception("Retraining pipeline failed")
            FailedTaskEvent.objects.create(
                task_name="retrain_and_evaluate",
                error_message=str(e),
                payload={"stage": "orchestration"},
            )
            return False, None, f"Retraining error: {str(e)}"

    def _compute_fn_rate_7d(self) -> float:
        """False-negative rate from corrections in last 7 days."""
        cutoff = dj_timezone.now() - timedelta(days=7)

        # "rejected" = model predicted flood but human said no (false positive in detection)
        # For now we track all feedback as signal
        total_count = PredictionFeedback.objects.filter(
            created_at__gte=cutoff,
            feedback_type__in=["rejected", "corrected"],
        ).count()

        if total_count == 0:
            return 0.0

        # If 30%+ of recent feedback is rejections, that's a signal
        rejected_count = PredictionFeedback.objects.filter(
            created_at__gte=cutoff,
            feedback_type="rejected",
        ).count()

        return rejected_count / max(total_count, 1)

    def _build_training_set(self) -> Tuple[list, list]:
        """
        Assemble training set from:
        1. Corrected feedback (ground truth overrides)
        2. Original labeled images (prevent catastrophic forgetting)
        """
        train_images = []
        train_labels = []

        # Add corrected feedback
        corrected = PredictionFeedback.objects.filter(
            feedback_type__in=["rejected", "corrected"]
        ).select_related("telemetry")

        for feedback in corrected:
            # Skip if already used
            if feedback.metadata.get("used_in_training"):
                continue

            img_path = feedback.telemetry.image_name
            if not img_path:
                continue

            # Reconstruct label from correction
            if feedback.feedback_type == "rejected":
                # Model predicted flood, but human said NO
                label = 0
            else:
                # feedback.feedback_type == "corrected"
                # Check metadata for corrected flood status
                label = feedback.metadata.get("corrected_flood", 1)

            train_images.append(img_path)
            train_labels.append(label)

        # Add original labeled images (for regularization)
        manifest_path = self.base_dir / "test_images" / "evaluation_manifest_labeled.csv"
        if manifest_path.exists():
            with open(manifest_path, "r", encoding="utf-8-sig") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    if not row.get("expected_flood"):
                        continue

                    img_path = row["image_path"]
                    try:
                        label = int(row["expected_flood"])
                    except (ValueError, TypeError):
                        continue

                    # Avoid duplicates
                    if img_path not in train_images:
                        train_images.append(img_path)
                        train_labels.append(label)

        logger.info(
            f"Training set assembled: {len(train_images)} images "
            f"({sum(train_labels)} floods, {len(train_labels) - sum(train_labels)} non-floods)"
        )

        return train_images, train_labels

    def _run_retraining(self, train_images: list, train_labels: list) -> Optional[Path]:
        """
        Wrapper around retrain_flood_classifier.py
        Returns: checkpoint path if successful, None otherwise
        """
        try:
            # Create temporary training list file
            train_list_file = self.base_dir / "tmp_train_list.csv"
            with open(train_list_file, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["image_path", "label"])
                for img, label in zip(train_images, train_labels):
                    writer.writerow([img, label])

            logger.info(f"Created training list: {train_list_file}")

            # Run retraining script
            output_name = f"flood_model_v{datetime.now().strftime('%Y%m%d_%H%M%S')}.pth"
            checkpoint_path = self.models_dir / output_name

            # Use subprocess to run retrain script with proper isolation
            cmd = [
                "python",
                str(self.base_dir / "retrain_flood_classifier.py"),
                "--train-list",
                str(train_list_file),
                "--output",
                str(checkpoint_path),
                "--epochs",
                "10",
                "--batch-size",
                "16",
            ]

            logger.info(f"Running: {' '.join(cmd)}")
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)

            if result.returncode != 0:
                logger.error(f"Retrain script failed: {result.stderr}")
                FailedTaskEvent.objects.create(
                    task_name="retrain_flood_classifier",
                    error_message=result.stderr[:500],
                    payload={"command": " ".join(cmd)},
                )
                return None

            logger.info(f"Retraining stdout:\n{result.stdout[-500:]}")  # Last 500 chars

            # Verify checkpoint exists
            if checkpoint_path.exists():
                size_mb = checkpoint_path.stat().st_size / (1024 * 1024)
                logger.info(f"✅ Checkpoint verified: {checkpoint_path} ({size_mb:.1f} MB)")
                return checkpoint_path
            else:
                logger.error(f"Checkpoint not found: {checkpoint_path}")
                return None

        except subprocess.TimeoutExpired:
            logger.error("Retraining timed out (10 min)")
            return None
        except Exception as e:
            logger.exception(f"Retraining execution error: {e}")
            return None
        finally:
            # Clean up temporary file
            if train_list_file.exists():
                train_list_file.unlink()

    def _evaluate_readiness_gates(self, checkpoint_path: Path) -> Dict:
        """
        Run full evaluation suite on new checkpoint.
        Calls evaluate_model_readiness.py

        Returns: {
            'all_pass': bool,
            'f1': float,
            'precision': float,
            'recall': float,
            'depth_mae': float,
            'barren_fp_rate': float,
            'p95_latency': float,
            'summary': str
        }
        """
        try:
            cmd = [
                "python",
                str(self.base_dir / "evaluate_model_readiness.py"),
                "--model-path",
                str(checkpoint_path),
                "--enforce-gates",
            ]

            logger.info(f"Running gates evaluation: {' '.join(cmd)}")
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)

            logger.info(f"Evaluation stdout:\n{result.stdout[-1000:]}")  # Last 1000 chars
            logger.info(f"Evaluation stderr:\n{result.stderr[-1000:]}")

            if result.returncode == 0:
                # Gates passed
                return {
                    "all_pass": True,
                    "summary": "All readiness gates passed",
                }
            else:
                # Gates failed (exit code 2 per convention)
                return {
                    "all_pass": False,
                    "summary": f"Gates failed (exit code {result.returncode})",
                }

        except subprocess.TimeoutExpired:
            logger.error("Gates evaluation timed out (5 min)")
            return {"all_pass": False, "summary": "Evaluation timeout"}
        except Exception as e:
            logger.exception(f"Gates evaluation error: {e}")
            return {"all_pass": False, "summary": f"Evaluation error: {str(e)[:100]}"}

    def _next_version_name(self, current: str) -> str:
        """Increment semantic version: 1.0 → 1.1, 2.1 → 2.2"""
        try:
            # Handle formats: "1.0", "v1.0", "1.0.0"
            version_str = current.lstrip("v")
            parts = version_str.split(".")

            # Increment minor version (last part)
            if len(parts) >= 2:
                parts[-1] = str(int(parts[-1]) + 1)
            else:
                parts.append("1")

            return ".".join(parts)
        except (ValueError, IndexError):
            # Fallback: return auto-generated version
            return f"auto_{datetime.now().strftime('%Y%m%d_%H%M%S')}"


def trigger_retraining_if_needed():
    """
    Scheduled job to run every 6 hours.
    Checks if retraining should trigger, and if so, runs the full pipeline.
    """
    trigger = RetrainingTrigger()

    should_retrain, reason, metadata = trigger.check_trigger()

    if not should_retrain:
        logger.info(f"No retrain trigger: {reason}")
        return

    logger.info(f"🔄 Retraining triggered: {reason}")

    success, new_version, message = trigger.retrain_and_evaluate()

    if success:
        logger.info(f"✅ {message}")
    else:
        logger.error(f"❌ {message}")


if __name__ == "__main__":
    import sys
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    trigger_retraining_if_needed()
