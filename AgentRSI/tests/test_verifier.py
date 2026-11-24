"""Host unit tests for AgentRSI verifier v1 (metrics, gate, requirements, end-to-end grading)."""
from __future__ import annotations

import json
import math
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import average_precision_score as sk_ap

AGENT_ROOT = Path(__file__).resolve().parents[1]
if str(AGENT_ROOT) not in sys.path:
    sys.path.insert(0, str(AGENT_ROOT))

from verifier.metrics import (  # noqa: E402
    enrichment_at_top_frac,
    estate_metrics,
    evaluate_gate,
    exact_ap,
    paired_block_bootstrap,
    recall_at_top_frac,
)
from verifier.runner import grade_submission, validate_datasets, validate_requirements  # noqa: E402


class MetricsUnitTests(unittest.TestCase):
    def test_exact_ap_matches_sklearn(self) -> None:
        rng = np.random.default_rng(42)
        for _ in range(10):
            y = rng.integers(0, 2, size=120)
            y[0] = 1
            s = rng.normal(size=120)
            self.assertAlmostEqual(exact_ap(y, s), float(sk_ap(y, s)), places=12)

            # Heavy score ties
            s_ties = np.round(s * 2.0) / 2.0
            self.assertAlmostEqual(exact_ap(y, s_ties), float(sk_ap(y, s_ties)), places=12)

    def test_exact_ap_constant_and_zero_pos(self) -> None:
        y = np.array([0, 1, 0, 0, 1, 0, 0, 0])
        self.assertAlmostEqual(exact_ap(y, np.zeros_like(y, dtype=float)), 0.25, places=12)
        self.assertEqual(exact_ap(np.zeros(10, dtype=int), np.arange(10, dtype=float)), 0.0)
        self.assertEqual(exact_ap(np.array([]), np.array([])), 0.0)

    def test_estate_metrics_nop_neutral(self) -> None:
        y = np.array([0] * 90 + [1] * 10)
        m = estate_metrics(y, np.zeros(100))
        self.assertAlmostEqual(m["prevalence"], 0.10, places=12)
        self.assertAlmostEqual(m["pr_auc"], 0.10, places=12)
        self.assertAlmostEqual(m["pr_auc_lift"], 0.0, places=12)
        self.assertAlmostEqual(m["recall_at_25"], 0.25, places=12)
        self.assertAlmostEqual(m["enrichment_top10"], 1.0, places=12)
        self.assertAlmostEqual(m["roc_auc"], 0.5, places=12)

    def test_recall_and_enrichment_perfect_ranking(self) -> None:
        y = np.array([0] * 95 + [1] * 5)
        scores = np.arange(100, dtype=float)
        self.assertAlmostEqual(recall_at_top_frac(y, scores, 0.25), 1.0, places=12)
        self.assertAlmostEqual(enrichment_at_top_frac(y, scores, 0.10), 10.0, places=12)

    def test_paired_block_bootstrap_identity_and_gain(self) -> None:
        rng = np.random.default_rng(7)
        y_e, ref_e, cand_e, blk_e = {}, {}, {}, {}
        for e in ("AirHitam", "Palong", "Serting"):
            y = (rng.random(120) < 0.12).astype(int)
            y[:15] = 1
            ref = rng.normal(size=120)
            y_e[e] = y
            ref_e[e] = ref
            cand_e[e] = ref + 2.5 * y
            blk_e[e] = np.repeat(np.arange(12), 10)
        weights = {"AirHitam": 1.0, "Palong": 1.0, "Serting": 1.0}

        same = paired_block_bootstrap(y_e, ref_e, ref_e, blk_e, weights, B=100, seed=1)
        self.assertEqual(same["comp_delta"], 0.0)
        self.assertEqual(same["comp_ci_low"], 0.0)
        self.assertEqual(same["comp_ci_high"], 0.0)

        gain = paired_block_bootstrap(y_e, cand_e, ref_e, blk_e, weights, B=100, seed=1)
        self.assertGreater(gain["comp_delta"], 0.2)
        self.assertGreater(gain["comp_ci_low"], 0.0)
        self.assertEqual(gain["n_valid_draws"], 100)

    def test_evaluate_gate_all_conditions(self) -> None:
        per_ok = {"AirHitam": 0.04, "Palong": 0.05, "Serting": 0.03}
        self.assertTrue(evaluate_gate(0.04, 0.01, per_ok, 0.03, -0.03)["passed"])
        # Exact boundary passes
        self.assertTrue(
            evaluate_gate(0.03, 1e-6, {"AirHitam": 0.06, "Palong": 0.06, "Serting": -0.03}, 0.03, -0.03)[
                "passed"
            ]
        )
        # CI lower bound <= 0 fails
        res_ci = evaluate_gate(0.05, 0.0, per_ok, 0.03, -0.03)
        self.assertFalse(res_ci["passed"])
        self.assertEqual(len(res_ci["failed_reasons"]), 1)
        # Composite below floor fails
        res_floor = evaluate_gate(0.0299, 0.005, per_ok, 0.03, -0.03)
        self.assertFalse(res_floor["passed"])
        # Single-estate non-degradation fails
        res_deg = evaluate_gate(0.05, 0.01, {"AirHitam": 0.09, "Palong": 0.09, "Serting": -0.031}, 0.03, -0.03)
        self.assertFalse(res_deg["passed"])
        self.assertEqual(res_deg["worst_estate"], "Serting")
        # Single-estate dominance fails breadth rule (only 1 of 3 improved)
        res_dom = evaluate_gate(
            0.043, 0.01, {"AirHitam": -0.01, "Palong": 0.15, "Serting": -0.01}, 0.03, -0.03, min_improved_estates=2
        )
        self.assertFalse(res_dom["passed"])
        self.assertEqual(res_dom["n_improved_estates"], 1)
        self.assertTrue(any("min_improved_estates" in r for r in res_dom["failed_reasons"]))

    def test_validate_requirements(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "requirements.txt"
            p.write_text("# comment\n  \ntabulate==0.9.0  # inline\nlightgbm==4.5.0\n")
            self.assertEqual(validate_requirements(p), ["tabulate==0.9.0", "lightgbm==4.5.0"])

            for bad in (
                "/sealed/labels_master.csv",
                "file:///sealed/labels_master.csv",
                "-e /tmp/evil",
                "../evil.whl",
                "tabulate",
                "tabulate>=0.9.0",
            ):
                p.write_text(bad + "\n")
                with self.assertRaises(ValueError, msg=bad):
                    validate_requirements(p)


class HostGradingIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.tmp_sealed_obj = tempfile.TemporaryDirectory()
        cls.sealed_dir = Path(cls.tmp_sealed_obj.name)
        for src in (
            AGENT_ROOT / ".sealed" / "labels_master.csv",
            AGENT_ROOT / "config" / "split_assignment.csv",
            AGENT_ROOT / "config" / "split.yaml",
            AGENT_ROOT / "config" / "verifier.yaml",
            AGENT_ROOT / ".work" / "agent_view" / "L_labelled.csv",
            AGENT_ROOT / ".work" / "agent_view" / "unlabelled.csv",
        ):
            (cls.sealed_dir / src.name).write_bytes(src.read_bytes())
        cls.verifier_cfg = yaml.safe_load((AGENT_ROOT / "config" / "verifier.yaml").read_text())

    @classmethod
    def tearDownClass(cls) -> None:
        cls.tmp_sealed_obj.cleanup()

    def test_starter_pipeline_deterministic_and_zero_delta(self) -> None:
        with tempfile.TemporaryDirectory() as td1, tempfile.TemporaryDirectory() as td2:
            out1, out2 = Path(td1), Path(td2)
            r1, vr1, c1 = grade_submission(
                sealed_dir=self.sealed_dir,
                submission_dir=AGENT_ROOT / "starter",
                datasets_dir=out1 / "missing_datasets",
                output_dir=out1 / "verifier",
                verifier_cfg=self.verifier_cfg,
                run_user="runner",
            )
            r2, vr2, c2 = grade_submission(
                sealed_dir=self.sealed_dir,
                submission_dir=AGENT_ROOT / "starter",
                datasets_dir=out2 / "missing_datasets",
                output_dir=out2 / "verifier",
                verifier_cfg=self.verifier_cfg,
                run_user="runner",
            )
            self.assertEqual(r1["submission_valid"], 1)
            self.assertEqual(r1["v_delta_vs_incumbent"], 0.0)
            self.assertEqual(c1["composite"]["comp_delta"], 0.0)
            self.assertFalse(c1["gate_passed"])
            for k in ("score", "v_lift_AirHitam", "v_lift_Palong", "v_lift_Serting", "v_delta_vs_coords"):
                self.assertEqual(r1[k], r2[k])
            for v in r1.values():
                self.assertIsInstance(v, (int, float))
                self.assertTrue(math.isfinite(v))

            # V report must never contain T numbers or certification keys
            v_text = (out1 / "verifier" / "v_report.json").read_text()
            self.assertNotIn("gate_passed", v_text)
            self.assertNotIn("certification", v_text)

            # Verify leak checker passes on agent-visible outputs (reward.json + v_report.json)
            leak_dir = out1 / "agent_visible"
            leak_dir.mkdir()
            for fn in ("reward.json", "v_report.json"):
                (leak_dir / fn).write_text((out1 / "verifier" / fn).read_text())
            proc = subprocess.run(
                [
                    sys.executable,
                    str(AGENT_ROOT / "src" / "DataPipeline" / "002-CheckLeaks.py"),
                    "--no-view",
                    "--scan",
                    str(leak_dir),
                ],
                capture_output=True,
                text=True,
            )
            self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)

    def test_nop_neutral_on_all_three_estates(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            reward, v_rep, cert = grade_submission(
                sealed_dir=self.sealed_dir,
                submission_dir=root / "nonexistent_submission",
                datasets_dir=root / "datasets",
                output_dir=root / "verifier",
                verifier_cfg=self.verifier_cfg,
            )
            self.assertEqual(v_rep["status"], "nop")
            self.assertEqual(reward["submission_valid"], 0)
            self.assertEqual(reward["score"], 0.0)
            for e in ("AirHitam", "Palong", "Serting"):
                self.assertEqual(reward[f"v_lift_{e}"], 0.0)
                self.assertAlmostEqual(v_rep["estates"][e]["recall_at_25"], 0.25, places=12)
                self.assertAlmostEqual(v_rep["estates"][e]["enrichment_top10"], 1.0, places=12)
            self.assertFalse(cert["gate_passed"])

    def test_bad_shape_and_nan_fail_safely(self) -> None:
        for body in (
            "import numpy as np\ndef fit_predict(l, u, ctx): return np.zeros(5)\n",
            "import numpy as np\ndef fit_predict(l, u, ctx): return np.full(len(u), np.nan)\n",
            "def fit_predict(l, u, ctx): raise RuntimeError('boom')\n",
        ):
            with tempfile.TemporaryDirectory() as td:
                root = Path(td)
                sub = root / "submission"
                sub.mkdir()
                (sub / "pipeline.py").write_text(body)
                reward, v_rep, cert = grade_submission(
                    sealed_dir=self.sealed_dir,
                    submission_dir=sub,
                    datasets_dir=root / "datasets",
                    output_dir=root / "verifier",
                    verifier_cfg=self.verifier_cfg,
                )
                self.assertEqual(v_rep["status"], "failed")
                self.assertEqual(reward["submission_valid"], 0)
                self.assertEqual(reward["score"], 0.0)
                self.assertFalse(cert["gate_passed"])

    def test_dataset_audit_and_symlink_rejection(self) -> None:
        l_df = pd.read_csv(self.sealed_dir / "L_labelled.csv", usecols=["estate", "Long", "Lat"])
        u_df = pd.read_csv(self.sealed_dir / "unlabelled.csv", usecols=["estate", "Long", "Lat"])
        all_coords = pd.concat([l_df, u_df], ignore_index=True)
        expected = set(zip(all_coords["estate"], all_coords["Long"], all_coords["Lat"]))

        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            ds_dir = root / "datasets" / "extra"
            ds_dir.mkdir(parents=True)
            ok_df = all_coords.copy()
            ok_df["extra_feat"] = 1.0
            ok_df.to_csv(ds_dir / "features.csv", index=False)
            ok_res = validate_datasets(root / "datasets", expected, max_mb=512)
            self.assertTrue(ok_res["valid"])
            self.assertIn("extra", ok_res["datasets"])

            # Forbidden column fails audit and grading
            bad_df = ok_df.copy()
            bad_df["label"] = "Healthy"
            bad_df.to_csv(ds_dir / "features.csv", index=False)
            bad_res = validate_datasets(root / "datasets", expected, max_mb=512)
            self.assertFalse(bad_res["valid"])

            sub = root / "submission"
            sub.mkdir()
            (sub / "pipeline.py").write_text((AGENT_ROOT / "starter" / "pipeline.py").read_text())
            (sub / "leak_link").symlink_to(self.sealed_dir / "labels_master.csv")
            reward, v_rep, cert = grade_submission(
                sealed_dir=self.sealed_dir,
                submission_dir=sub,
                datasets_dir=root / "empty_ds",
                output_dir=root / "verifier",
                verifier_cfg=self.verifier_cfg,
            )
            self.assertEqual(v_rep["status"], "failed")
            self.assertEqual(reward["submission_valid"], 0)
            self.assertIn("Symlink", v_rep["error"])

    def test_dynamic_promoted_incumbent_zero_delta_on_self(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            inc_sub = root / "incumbent_sub"
            inc_sub.mkdir()
            # Custom non-starter incumbent using NDVI + Long
            (inc_sub / "pipeline.py").write_text(
                "import numpy as np\n"
                "from sklearn.linear_model import LogisticRegression\n"
                "from sklearn.pipeline import make_pipeline\n"
                "from sklearn.preprocessing import StandardScaler\n"
                "def fit_predict(labeled, unlabeled, ctx):\n"
                "    tr = labeled[labeled['label'].isin(['Healthy', 'Unhealthy'])]\n"
                "    y = (tr['label'] == 'Unhealthy').astype(int).to_numpy()\n"
                "    m = make_pipeline(StandardScaler(), LogisticRegression(class_weight='balanced', max_iter=500))\n"
                "    m.fit(tr[['NDVI', 'Long']].to_numpy(float), y)\n"
                "    return np.asarray(m.decision_function(unlabeled[['NDVI', 'Long']].to_numpy(float)), dtype=np.float64)\n"
            )
            (inc_sub / "requirements.txt").write_text("")
            reward, v_rep, cert = grade_submission(
                sealed_dir=self.sealed_dir,
                submission_dir=inc_sub,
                datasets_dir=root / "empty_ds",
                output_dir=root / "verifier",
                verifier_cfg=self.verifier_cfg,
                incumbent_dir=inc_sub,
                incumbent_datasets_dir=root / "empty_inc_ds",
            )
            self.assertEqual(reward["submission_valid"], 1)
            self.assertEqual(reward["v_delta_vs_incumbent"], 0.0)
            self.assertEqual(cert["composite"]["comp_delta"], 0.0)
            self.assertFalse(cert["gate_passed"])


if __name__ == "__main__":
    unittest.main()
