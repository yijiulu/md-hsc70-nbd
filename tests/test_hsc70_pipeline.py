"""Static and geometry unit tests for the Hsc70 NBD MD pipeline."""

from __future__ import annotations

import ast
import importlib.util
import re
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = REPO_ROOT / "scripts"


class TestHsc70Pipeline(unittest.TestCase):
    def test_all_python_scripts_parse(self) -> None:
        py_files = sorted(SCRIPTS_DIR.glob("*.py"))
        self.assertGreaterEqual(len(py_files), 10)
        for p in py_files:
            ast.parse(p.read_text(encoding="utf-8"), filename=str(p))

    def test_mdp_and_param_files_present(self) -> None:
        for fn in ("ions.mdp", "em.mdp", "nvt.mdp", "npt.mdp", "prod.mdp"):
            self.assertTrue((REPO_ROOT / "mdp" / fn).is_file(), f"Missing mdp/{fn}")
        for fn in ("ATP.prep", "ADP.prep", "frcmod.phos"):
            self.assertTrue((REPO_ROOT / "params" / fn).is_file(), f"Missing params/{fn}")

    def test_movie_template_is_english_and_has_placeholder(self) -> None:
        html = (SCRIPTS_DIR / "movie_template.html").read_text(encoding="utf-8")
        self.assertEqual(html.count("/*MOVIE_DATA*/null"), 1)
        self.assertIsNone(re.search(r"[\u4e00-\u9fff]", html))

    def test_geometry_helpers_if_numpy_installed(self) -> None:
        if importlib.util.find_spec("numpy") is None:
            self.skipTest("numpy not installed in base interpreter")
        import numpy as np

        spec = importlib.util.spec_from_file_location("nbd_defs", SCRIPTS_DIR / "nbd_defs.py")
        assert spec and spec.loader
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)

        # Orthogonal angle & Kabsch identity check
        a = np.array([1.0, 0.0, 0.0])
        b = np.array([0.0, 0.0, 0.0])
        c = np.array([0.0, 1.0, 0.0])
        self.assertAlmostEqual(mod.angle(a, b, c), 90.0, places=6)

        pts = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 3.0]])
        shifted = pts + np.array([5.0, -3.0, 2.0])
        rot = mod.kabsch(shifted, pts)
        fitted = mod.fit_apply(shifted, rot)
        self.assertAlmostEqual(mod.rmsd(fitted, pts), 0.0, places=6)


if __name__ == "__main__":
    unittest.main()
