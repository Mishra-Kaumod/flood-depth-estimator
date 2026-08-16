import os
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from depth_teachers import TeacherEnsemble


class TestDepthTeachers(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.img_path = Path(self.tmp.name) / "sample.png"

        h, w = 32, 32
        image = np.zeros((h, w, 3), dtype=np.uint8)
        image[..., 0] = np.linspace(10, 200, w, dtype=np.uint8)
        image[..., 1] = np.linspace(20, 160, h, dtype=np.uint8)[:, None]
        image[..., 2] = 80
        Image.fromarray(image, mode="RGB").save(self.img_path)

        self.map_a = np.tile(np.linspace(0.1, 1.0, w, dtype=np.float32), (h, 1))
        self.map_b = np.tile(np.linspace(1.0, 0.1, w, dtype=np.float32), (h, 1))
        self.map_c = np.full((h, w), 0.5, dtype=np.float32)

    def tearDown(self) -> None:
        self.tmp.cleanup()

    @staticmethod
    def _predict_from_payload(payload, _image_rgb):
        return payload["depth_map"]

    def _ensemble(self, failing=None, custom_maps=None):
        failing = failing or set()
        custom_maps = custom_maps or {}

        default_maps = {
            "DepthAnythingV2": self.map_a,
            "DepthPro": self.map_b,
            "Metric3D": self.map_c,
        }
        default_maps.update(custom_maps)

        overrides = {}
        for name in TeacherEnsemble.TEACHER_NAMES:
            if name in failing:
                def _bad_loader(name=name):
                    raise RuntimeError(f"{name} unavailable")

                overrides[name] = {
                    "model": f"mock://{name}",
                    "loader": _bad_loader,
                    "predictor": self._predict_from_payload,
                }
            else:
                depth_map = default_maps[name]

                def _loader(dm=depth_map):
                    return {"depth_map": dm}

                overrides[name] = {
                    "model": f"mock://{name}",
                    "loader": _loader,
                    "predictor": self._predict_from_payload,
                }

        return TeacherEnsemble(
            device="cpu",
            teacher_overrides=overrides,
            lazy_load=False,
            allow_download=False,
        )

    def test_initialization(self):
        ens = self._ensemble()
        self.assertEqual(ens.device, "cpu")
        self.assertEqual(set(ens.status.keys()), set(TeacherEnsemble.TEACHER_NAMES))

    def test_status_schema(self):
        ens = self._ensemble()
        status = ens.status
        for name in TeacherEnsemble.TEACHER_NAMES:
            self.assertIn("available", status[name])
            self.assertIn("model", status[name])
            self.assertIn("error", status[name])
            self.assertTrue(status[name]["available"])

    def test_missing_teacher(self):
        ens = self._ensemble(failing={"Metric3D"})
        out = ens.predict(self.img_path)
        self.assertFalse(out["teachers"]["Metric3D"]["available"])
        self.assertIn("unavailable", out["teachers"]["Metric3D"]["error"])

    def test_valid_image_predict(self):
        ens = self._ensemble()
        out = ens.predict(self.img_path)
        self.assertIn("teachers", out)
        self.assertIn("ensemble", out)
        self.assertIn("features", out)
        self.assertIn("DepthAnythingV2", out["teachers"])

    def test_valid_water_mask(self):
        ens = self._ensemble()
        mask = np.zeros((32, 32), dtype=np.uint8)
        mask[16:, :] = 1
        out = ens.predict(self.img_path, water_mask=mask)
        self.assertTrue(out["water_region_valid"])
        self.assertIn("water_depth_mean", out["teachers"]["DepthPro"])

    def test_empty_water_mask(self):
        ens = self._ensemble()
        mask = np.zeros((32, 32), dtype=np.uint8)
        out = ens.predict(self.img_path, water_mask=mask)
        self.assertFalse(out["water_region_valid"])
        t = out["teachers"]["DepthAnythingV2"]
        self.assertAlmostEqual(t["water_depth_mean"], t["global_mean"], places=5)

    def test_invalid_water_mask_shape(self):
        ens = self._ensemble()
        bad_mask = np.ones((20, 20), dtype=np.uint8)
        out = ens.predict(self.img_path, water_mask=bad_mask)
        self.assertFalse(out["water_region_valid"])

    def test_teacher_disagreement(self):
        grid_x = np.tile(np.linspace(0.0, 1.0, 32, dtype=np.float32), (32, 1))
        grid_y = np.tile(np.linspace(0.0, 1.0, 32, dtype=np.float32)[:, None], (1, 32))
        checker = ((np.indices((32, 32)).sum(axis=0) % 2).astype(np.float32))
        maps = {
            "DepthAnythingV2": grid_x,
            "DepthPro": 1.0 - grid_y,
            "Metric3D": checker,
        }
        ens = self._ensemble(custom_maps=maps)
        out = ens.predict(self.img_path)
        self.assertLess(out["ensemble"]["teacher_agreement"], 0.8)

    def test_cpu_mode(self):
        ens = self._ensemble()
        self.assertEqual(ens.device, "cpu")

    def test_output_schema(self):
        ens = self._ensemble()
        out = ens.predict(self.img_path)

        required_top = {
            "global_mean", "global_median", "p10", "p25", "p50", "p75", "p90",
            "water_depth_mean", "water_depth_median", "teacher_mean", "teacher_median",
            "teacher_spread", "teacher_std", "teacher_agreement", "valid_pixel_ratio"
        }
        self.assertTrue(required_top.issubset(out["features"].keys()))

        per_teacher_required = {
            "global_mean", "global_median", "p10", "p25", "p50", "p75", "p90",
            "water_depth_mean", "water_depth_median", "water_p10", "water_p25", "water_p50", "water_p75", "water_p90",
            "valid_pixel_ratio"
        }
        for name in TeacherEnsemble.TEACHER_NAMES:
            row = out["teachers"][name]
            if row.get("available"):
                self.assertTrue(per_teacher_required.issubset(row.keys()))

    def test_diagnose_schema(self):
        ens = self._ensemble()
        d = ens.diagnose()
        self.assertIn("python_version", d)
        self.assertIn("torch_version", d)
        self.assertIn("teacher_status", d)
        self.assertIn("cache_paths", d)


@unittest.skipUnless(os.getenv("DEPTH_TEACHER_INTEGRATION") == "1", "integration test disabled")
class TestDepthTeachersIntegration(unittest.TestCase):
    def test_real_teacher_integration(self):
        repo_root = Path(__file__).resolve().parents[1]
        candidates = list((repo_root / "test_images").glob("*.jpg")) + list((repo_root / "test_images").glob("*.png"))
        self.assertTrue(len(candidates) > 0, "No test image found under test_images/")
        image_path = candidates[0]

        teachers = TeacherEnsemble(device="cpu", lazy_load=True)
        result = teachers.predict(image_path)
        self.assertIn("teachers", result)
        self.assertIn("features", result)

        # Do not assert all teachers available; only assert schema and at least one executed or failed with explicit error.
        for name in TeacherEnsemble.TEACHER_NAMES:
            self.assertIn(name, result["teachers"])
            self.assertIn("available", result["teachers"][name])
            self.assertIn("error", result["teachers"][name])


if __name__ == "__main__":
    unittest.main()
