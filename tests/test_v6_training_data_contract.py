import csv
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from src.v6_training_data_contract import MANIFEST_COLUMNS, ManifestRow, quarantine_rows


def row(image_id, scene="FLOOD_20_50", depth="35", session=None, group=None, sha=""):
    return {
        "image_id": image_id, "filename": f"{image_id}.jpg", "sha256": sha,
        "scene_type": scene, "depth_cm": depth, "label_confidence": "HIGH",
        "measurement_source": "instrumented", "source_session_id": session or f"session-{image_id}",
        "group_id": group or f"group-{image_id}", "physical_reference": "car",
        "depth_eligible": "true", "classification_only": "false", "quarantine_reason": "",
        "visual_near_duplicate_status": "not_run",
    }


class V6TrainingDataContractTests(unittest.TestCase):
    def test_dry_is_classification_only_and_depth_remains_missing(self):
        raw = row("dry", scene="DRY", depth="")
        raw.update({"depth_eligible": "false", "classification_only": "true"})
        parsed = ManifestRow.from_mapping(raw)
        self.assertIsNone(parsed.depth_cm)
        self.assertTrue(parsed.classification_only)

    def test_zero_depth_flood_is_quarantined(self):
        raw = row("bad", depth="0")
        valid, quarantined = quarantine_rows([raw])
        self.assertEqual(valid, [])
        self.assertIn("zero/flood conflict", quarantined[0]["quarantine_reason"])

    def test_invalid_scene_is_quarantined(self):
        raw = row("bad-scene", scene="NOT_A_SCENE")
        valid, quarantined = quarantine_rows([raw])
        self.assertEqual(valid, [])
        self.assertIn("invalid scene_type", quarantined[0]["quarantine_reason"])

    def test_prepare_is_deterministic_and_group_safe_without_images(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest = root / "manifest.csv"
            rows = [row(f"row-{index}", session="shared-session" if index in {0, 1} else None) for index in range(12)]
            # This pair has distinct declared sessions but the same content hash.
            # Exact duplicate grouping must keep it in one split.
            rows[0]["sha256"] = "a" * 64
            rows[1]["sha256"] = "a" * 64
            rows[1]["source_session_id"] = "different-session"
            with manifest.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=MANIFEST_COLUMNS); writer.writeheader(); writer.writerows(rows)
            outputs = []
            for name in ("one", "two"):
                output = root / name
                subprocess.run([
                    sys.executable, "-m", "scripts.prepare_v6_training_dataset", "--manifest", str(manifest),
                    "--output-dir", str(output), "--seed", "42", "--validation-fraction", "0.2", "--challenge-fraction", "0.2",
                ], check=True, cwd=Path(__file__).resolve().parents[1])
                outputs.append(output)
            self.assertEqual((outputs[0] / "train_manifest.csv").read_text(), (outputs[1] / "train_manifest.csv").read_text())
            leakage = json.loads((outputs[0] / "leakage_check.json").read_text())
            self.assertTrue(leakage["passed"])
            self.assertEqual(leakage["visual_near_duplicate_status"], "not_run")
            audit = json.loads((outputs[0] / "duplicate_and_group_audit.json").read_text())
            self.assertTrue(any(item["kind"] == "exact_sha256_duplicate" for item in audit))
            frozen = json.loads((outputs[0] / "challenge_freeze.json").read_text())
            self.assertEqual(frozen["split"], "CHALLENGE")
            self.assertIn("manifest_hash", frozen)

    def test_future_evaluation_rejects_challenge_without_opt_in(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            predictions = root / "predictions.csv"
            predictions.write_text(
                "split,image_id,actual_depth_cm,v5_final_depth_cm,v6_primary_depth_cm,v6_refined_depth_cm\n"
                "CHALLENGE,synthetic,35,30,32,\n", encoding="utf-8"
            )
            completed = subprocess.run([
                sys.executable, "-m", "scripts.evaluate_v6_future_template", "--predictions", str(predictions),
                "--output", str(root / "result.json"),
            ], cwd=Path(__file__).resolve().parents[1], capture_output=True, text=True)
            self.assertNotEqual(completed.returncode, 0)
            self.assertIn("CHALLENGE rows are forbidden", completed.stderr)

            candidate_freeze = root / "candidate_freeze.json"
            candidate_freeze.write_text("{}", encoding="utf-8")
            permitted = subprocess.run([
                sys.executable, "-m", "scripts.evaluate_v6_future_template", "--predictions", str(predictions),
                "--output", str(root / "final_result.json"), "--mode", "final_evaluation",
                "--candidate-freeze", str(candidate_freeze),
            ], cwd=Path(__file__).resolve().parents[1], capture_output=True, text=True)
            self.assertEqual(permitted.returncode, 0, permitted.stderr)
