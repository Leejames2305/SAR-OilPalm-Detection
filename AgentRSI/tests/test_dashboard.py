"""Unit tests for AgentRSI M4 dashboard generation, verification, and collect integration."""
from __future__ import annotations

import json
import re
import sys
import tempfile
import unittest
from pathlib import Path

import yaml

AGENT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = AGENT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from libbuilder.collect import collect_session  # noqa: E402
from libbuilder.dashboard import (  # noqa: E402
    _DashboardHTMLInspector,
    generate_dashboard,
    verify_dashboard,
)
from libbuilder.events import append_event  # noqa: E402
from libbuilder.lineage import default_lineage, save_lineage  # noqa: E402


def _write_session_fixture(
    sessions_root: Path,
    sid: str,
    *,
    epoch: int,
    parent_id: str,
    v_score: float,
    v_estates: tuple[float, float, float],
    v_delta_inc: float,
    t_delta: float,
    t_ci_low: float,
    t_ci_high: float,
    worst_estate: str,
    worst_delta: float,
    gate_passed: bool,
    dry_session: bool = False,
    cost_usd: float = 0.25,
    runtime_sec: float = 4.5,
    started_at: str = "2026-11-23T05:00:00Z",
    smells: list[str] | None = None,
    report_text: str = "# Report\nNominal run.\n",
    handoff_text: str = "## State of play\nReady.\n",
) -> None:
    sdir = sessions_root / sid
    (sdir / "submission").mkdir(parents=True, exist_ok=True)
    (sdir / "submission" / "pipeline.py").write_text("def fit_predict(l, u, ctx): return [0.0]*len(u)\n")
    (sdir / "REPORT.md").write_text(report_text)
    (sdir / "HANDOFF.md").write_text(handoff_text)
    (sdir / "meta.json").write_text(
        json.dumps({
            "session_id": sid,
            "job_id": f"job-{sid}",
            "parent_id": parent_id,
            "dry_session": dry_session,
            "agent": "codex",
            "model": "gpt-6.1-terra",
            "started_at": started_at,
            "finished_at": started_at,
            "cost_usd": cost_usd,
            "smells": smells or [],
            "handoff_missing_sections": [],
        })
    )
    (sdir / "reward.json").write_text(
        json.dumps({
            "score": v_score,
            "v_lift_AirHitam": v_estates[0],
            "v_lift_Palong": v_estates[1],
            "v_lift_Serting": v_estates[2],
            "v_delta_vs_incumbent": v_delta_inc,
            "v_delta_vs_coords": 0.01,
            "submission_valid": 1,
            "runtime_sec": runtime_sec,
        })
    )
    (sdir / "v_report.json").write_text(
        json.dumps({
            "status": "ok",
            "submission_valid": True,
            "estates": {
                est: {
                    "pr_auc_lift": val,
                    "recall_at_25": 0.4,
                    "delta_vs_incumbent": {"delta": v_delta_inc},
                }
                for est, val in zip(("AirHitam", "Palong", "Serting"), v_estates)
            },
        })
    )
    (sdir / "certification.json").write_text(
        json.dumps({
            "epoch": epoch,
            "status": "ok",
            "submission_valid": True,
            "gate_passed": gate_passed,
            "composite": {
                "comp_delta": t_delta,
                "ci_low": t_ci_low,
                "ci_high": t_ci_high,
            },
            "gate": {
                "passed": gate_passed,
                "comp_delta": t_delta,
                "comp_ci_low": t_ci_low,
                "effect_floor": 0.03,
                "non_degradation_floor": -0.03,
                "worst_estate": worst_estate,
                "worst_estate_delta": worst_delta,
                "failed_reasons": [] if gate_passed else [f"composite delta {t_delta:+.6f} < +0.0300"],
            },
            "estates": {
                est: {
                    "cand_metrics": {"pr_auc_lift": t_delta + 0.05},
                    "delta_vs_incumbent": {"delta": t_delta, "ci_low": t_ci_low, "ci_high": t_ci_high},
                }
                for est in ("AirHitam", "Palong", "Serting")
            },
        })
    )


class DashboardTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.split_cfg = yaml.safe_load((AGENT_ROOT / "config" / "split.yaml").read_text())
        cls.verifier_cfg = yaml.safe_load((AGENT_ROOT / "config" / "verifier.yaml").read_text())

    def test_real_s000_dry_dashboard_exact_match(self) -> None:
        sessions_root = AGENT_ROOT / "sessions"
        lineage_path = AGENT_ROOT / "state" / "lineage.json"
        events_path = AGENT_ROOT / "state" / "events.jsonl"
        self.assertTrue((sessions_root / "s000-dry" / "reward.json").exists())

        with tempfile.TemporaryDirectory() as td:
            out = Path(td) / "dashboard" / "index.html"
            generate_dashboard(
                AGENT_ROOT,
                output_path=out,
                sessions_root=sessions_root,
                lineage_path=lineage_path,
                events_path=events_path,
            )
            errs = verify_dashboard(
                out,
                sessions_root=sessions_root,
                lineage_path=lineage_path,
                events_path=events_path,
            )
            self.assertEqual(errs, [], f"Verification mismatches: {errs}")
            html_txt = out.read_text(encoding="utf-8")
            self.assertIn("s000-dry", html_txt)
            self.assertIn("search signal", html_txt.lower())
            self.assertIn("+0.03", html_txt)
            self.assertIn('id="toggle-baseline"', html_txt)
            self.assertIn('id="baseline-preset"', html_txt)
            self.assertIn('id="baseline-value"', html_txt)
            self.assertIn('data-ref="comparison-baseline"', html_txt)
            self.assertIn('data-metric="v_delta_vs_baseline"', html_txt)
            self.assertIn("NDVI Starter", html_txt)
            self.assertIn("renderMarkdown", html_txt)
            self.assertIn('data-mode="rendered"', html_txt)
            self.assertIn('data-mode="raw"', html_txt)

    def test_multi_epoch_fixture_charts_events_and_no_drift(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            sessions_root = root / "sessions"
            lp = root / "state" / "lineage.json"
            ep = root / "state" / "events.jsonl"
            out = root / "dashboard" / "index.html"

            # Epoch 1: s001 fail, s002 promoted, s003 dry (smell/review)
            _write_session_fixture(
                sessions_root, "s001", epoch=1, parent_id="starter",
                v_score=0.062, v_estates=(0.050, 0.055, 0.081), v_delta_inc=0.007531,
                t_delta=0.012, t_ci_low=-0.014, t_ci_high=0.038,
                worst_estate="AirHitam", worst_delta=-0.011, gate_passed=False,
                started_at="2026-11-23T01:00:00Z",
                report_text="# Bogus claim\nMy score was 0.999999!\n<script>alert(1)</script>\n",
                handoff_text="## State of play\nClaimed score 0.999999 <script>alert(1)</script>\n",
            )
            _write_session_fixture(
                sessions_root, "s002", epoch=1, parent_id="s001",
                v_score=0.095, v_estates=(0.088, 0.091, 0.106), v_delta_inc=0.040531,
                t_delta=0.041, t_ci_low=0.012, t_ci_high=0.070,
                worst_estate="Palong", worst_delta=0.025, gate_passed=True,
                started_at="2026-11-23T02:00:00Z",
            )
            _write_session_fixture(
                sessions_root, "s003", epoch=1, parent_id="s002",
                v_score=0.089, v_estates=(0.082, 0.085, 0.100), v_delta_inc=-0.006,
                t_delta=-0.005, t_ci_low=-0.025, t_ci_high=0.015,
                worst_estate="Serting", worst_delta=-0.020, gate_passed=False,
                dry_session=True, started_at="2026-11-23T03:00:00Z",
                smells=["submission/pipeline.py: URL https://example.org"],
            )
            # Epoch 2: s004 fail, s005 promoted
            _write_session_fixture(
                sessions_root, "s004", epoch=2, parent_id="s002",
                v_score=0.098, v_estates=(0.090, 0.094, 0.110), v_delta_inc=0.003,
                t_delta=0.015, t_ci_low=-0.005, t_ci_high=0.036,
                worst_estate="AirHitam", worst_delta=0.002, gate_passed=False,
                started_at="2026-11-24T01:00:00Z",
            )
            _write_session_fixture(
                sessions_root, "s005", epoch=2, parent_id="s004",
                v_score=0.132, v_estates=(0.120, 0.128, 0.148), v_delta_inc=0.037,
                t_delta=0.045, t_ci_low=0.018, t_ci_high=0.075,
                worst_estate="Palong", worst_delta=0.031, gate_passed=True,
                started_at="2026-11-24T02:00:00Z",
            )

            lin = default_lineage(self.split_cfg, self.verifier_cfg)
            lin["epoch"] = 2
            lin["t_attempts_used"] = 2
            lin["epochs"].append({"epoch": 2, "split_sha256": "deadbeef", "started_at": "2026-11-24T00:00:00Z"})
            lin["incumbent"] = {
                "id": "s005",
                "source_dir": "sessions/s005/submission",
                "epoch": 2,
                "v_score": 0.132,
                "per_estate_v_lift": {"AirHitam": 0.120, "Palong": 0.128, "Serting": 0.148},
            }
            lin["daily_spend_usd"] = {"2026-11-23": 0.75, "2026-11-24": 0.50}
            lin["sessions"] = {
                "s001": {"epoch": 1, "parent_id": "starter", "promoted": False, "dry_session": False, "collected_at": "2026-11-23T01:10:00Z"},
                "s002": {"epoch": 1, "parent_id": "s001", "promoted": True, "dry_session": False, "collected_at": "2026-11-23T02:10:00Z"},
                "s003": {"epoch": 1, "parent_id": "s002", "promoted": False, "dry_session": True, "collected_at": "2026-11-23T03:10:00Z"},
                "s004": {"epoch": 2, "parent_id": "s002", "promoted": False, "dry_session": False, "collected_at": "2026-11-24T01:10:00Z"},
                "s005": {"epoch": 2, "parent_id": "s004", "promoted": True, "dry_session": False, "collected_at": "2026-11-24T02:10:00Z"},
            }
            save_lineage(lp, lin)

            append_event(ep, session="s002", epoch=1, event_type="promoted", summary="Promoted s002", ref="sessions/s002/REPORT.md")
            append_event(ep, session="s003", epoch=1, event_type="review_flag", summary="Review flag for s003", ref="sessions/s003/meta.json")
            append_event(ep, session="epoch-2", epoch=2, event_type="split_refresh", summary="Split refreshed for epoch 2")
            append_event(ep, session="s005", epoch=2, event_type="promoted", summary="Promoted s005", ref="sessions/s005/REPORT.md")

            generate_dashboard(
                AGENT_ROOT,
                output_path=out,
                sessions_root=sessions_root,
                lineage_path=lp,
                events_path=ep,
            )
            errs = verify_dashboard(out, sessions_root=sessions_root, lineage_path=lp, events_path=ep)
            self.assertEqual(errs, [], f"Verification mismatches: {errs}")

            html_txt = out.read_text(encoding="utf-8")
            # Raw <script>alert(1)</script> from agent text must NEVER execute unescaped
            self.assertNotIn("<script>alert(1)</script>", html_txt)

            # Bogus 0.999999 in REPORT.md/HANDOFF.md must never appear in #sessions-table cells
            table_match = re.search(r'<table id="sessions-table">.*?</table>', html_txt, re.DOTALL)
            self.assertIsNotNone(table_match)
            self.assertNotIn("0.999999", table_match.group(0))

            # Direct relative links to REPORT.md / HANDOFF.md present
            self.assertIn("../sessions/s002/REPORT.md", html_txt)
            self.assertIn("../sessions/s002/HANDOFF.md", html_txt)

            # Customisable baseline presets present
            self.assertIn("NDVI Starter", html_txt)
            self.assertIn("Coords-only", html_txt)
            self.assertIn("POC Best (SAR+Spatial)", html_txt)

            # Explicit check on epoch boundary separation across polylines
            insp = _DashboardHTMLInspector()
            insp.feed(html_txt)
            v_polylines = [a for tag, a in insp.svgs["chart-v"] if tag == "polyline"]
            t_polylines = [a for tag, a in insp.svgs["chart-t"] if tag == "polyline"]
            self.assertTrue(any(a.get("data-epoch") == "1" for a in v_polylines))
            self.assertTrue(any(a.get("data-epoch") == "2" for a in v_polylines))
            self.assertEqual(len(t_polylines), 2)

    def test_empty_sessions_state(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            sessions_root = root / "sessions"
            sessions_root.mkdir()
            lp = root / "state" / "lineage.json"
            ep = root / "state" / "events.jsonl"
            lin = default_lineage(self.split_cfg, self.verifier_cfg)
            save_lineage(lp, lin)
            ep.parent.mkdir(parents=True, exist_ok=True)
            ep.write_text("")
            out = root / "dashboard" / "index.html"

            generate_dashboard(
                AGENT_ROOT,
                output_path=out,
                sessions_root=sessions_root,
                lineage_path=lp,
                events_path=ep,
            )
            errs = verify_dashboard(out, sessions_root=sessions_root, lineage_path=lp, events_path=ep)
            self.assertEqual(errs, [])
            self.assertIn("No collected sessions", out.read_text(encoding="utf-8"))

    def test_collect_auto_regenerates_dashboard(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            job_dir = root / "jobs" / "s010" / "2026-11-23__14-00-00"
            trial_dir = job_dir / "s010__xyz"
            (trial_dir / "verifier").mkdir(parents=True)
            (trial_dir / "artifacts" / "app" / "submission").mkdir(parents=True)
            (trial_dir / "artifacts" / "app" / "report").mkdir(parents=True)
            (trial_dir / "agent").mkdir(parents=True)

            (job_dir / "result.json").write_text(json.dumps({"stats": {"cost_usd": 0.15}}))
            (trial_dir / "result.json").write_text(
                json.dumps({
                    "started_at": "2026-11-23T06:00:00Z",
                    "finished_at": "2026-11-23T06:04:00Z",
                    "agent_info": {"name": "codex", "model_info": {"name": "gpt-6.1-terra"}},
                    "agent_result": {"cost_usd": 0.15, "n_input_tokens": 50, "n_output_tokens": 20},
                    "exception_info": None,
                })
            )
            (trial_dir / "agent" / "codex.log").write_text("clean\n")
            (trial_dir / "artifacts" / "app" / "submission" / "pipeline.py").write_text(
                "def fit_predict(l, u, ctx): return [0.0]*len(u)\n"
            )
            (trial_dir / "artifacts" / "app" / "submission" / "requirements.txt").write_text("")
            (trial_dir / "artifacts" / "app" / "report" / "REPORT.md").write_text("# S010\nAuto dashboard test.\n")
            (trial_dir / "artifacts" / "app" / "report" / "HANDOFF.md").write_text(
                "## State of play\nOk.\n## What I tried\nTest.\n## Dead ends\nNone.\n"
                "## Open leads, ranked\n1. SAR.\n## How to resume\nRun.\n## Artifact index\n- `/app/submission/pipeline.py`\n"
            )
            (trial_dir / "verifier" / "reward.json").write_text(
                json.dumps({
                    "score": 0.092,
                    "v_lift_AirHitam": 0.088,
                    "v_lift_Palong": 0.090,
                    "v_lift_Serting": 0.098,
                    "v_delta_vs_incumbent": 0.037531,
                    "v_delta_vs_coords": 0.02,
                    "submission_valid": 1,
                    "runtime_sec": 3.2,
                })
            )
            (trial_dir / "verifier" / "v_report.json").write_text(
                json.dumps({"status": "ok", "submission_valid": True, "composite": {"score": 0.092}})
            )
            (trial_dir / "verifier" / "certification.json").write_text(
                json.dumps({
                    "epoch": 1,
                    "gate_passed": True,
                    "composite": {"comp_delta": 0.038, "ci_low": 0.009, "ci_high": 0.067},
                    "gate": {
                        "passed": True,
                        "effect_floor": 0.03,
                        "non_degradation_floor": -0.03,
                        "worst_estate": "Palong",
                        "worst_estate_delta": 0.015,
                        "failed_reasons": [],
                    },
                })
            )

            envp = root / ".env"
            envp.write_text("API_KEY=sk-fake-secret-999999\nDAILY_SPEND_CAP_USD=5\n")
            lp = root / "state" / "lineage.json"
            ep = root / "state" / "events.jsonl"
            sessions_root = root / "sessions"

            res = collect_session(
                "s010",
                job_dir=job_dir,
                jobs_root=root / "jobs",
                sessions_root=sessions_root,
                lineage_path=lp,
                events_path=ep,
                env_path=envp,
            )
            dash_path = Path(res["dashboard_path"])
            self.assertEqual(dash_path, root / "dashboard" / "index.html")
            self.assertTrue(dash_path.exists())
            errs = verify_dashboard(dash_path, sessions_root=sessions_root, lineage_path=lp, events_path=ep)
            self.assertEqual(errs, [], f"Auto-regenerated dashboard verification failed: {errs}")

    def test_yymmdd_long_session_ids_and_compact_ticks(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            sessions_root = root / "sessions"
            lp = root / "state" / "lineage.json"
            ep = root / "state" / "events.jsonl"
            out = root / "dashboard" / "index.html"

            ids = [
                ("261123-s001-data-expansion", 0.060, "2026-11-23T01:00:00Z"),
                ("261123-s002-literature", 0.068, "2026-11-23T02:00:00Z"),
                ("261124-s003-ablation", 0.075, "2026-11-24T01:00:00Z"),
            ]
            lin = default_lineage(self.split_cfg, self.verifier_cfg)
            for sid, v_sc, ts in ids:
                _write_session_fixture(
                    sessions_root, sid, epoch=1, parent_id="starter",
                    v_score=v_sc, v_estates=(v_sc, v_sc, v_sc), v_delta_inc=v_sc - 0.054469,
                    t_delta=0.01, t_ci_low=-0.01, t_ci_high=0.03,
                    worst_estate="AirHitam", worst_delta=0.0, gate_passed=False,
                    started_at=ts,
                )
                lin["sessions"][sid] = {"epoch": 1, "parent_id": "starter", "promoted": False, "collected_at": ts}
            save_lineage(lp, lin)
            ep.parent.mkdir(parents=True, exist_ok=True)
            ep.write_text("")

            generate_dashboard(AGENT_ROOT, output_path=out, sessions_root=sessions_root, lineage_path=lp, events_path=ep)
            errs = verify_dashboard(out, sessions_root=sessions_root, lineage_path=lp, events_path=ep)
            self.assertEqual(errs, [], f"Verification mismatches: {errs}")

            html_txt = out.read_text(encoding="utf-8")
            self.assertIn("<title>261123-s001-data-expansion</title>261123-s001</text>", html_txt)
            self.assertIn("<title>261124-s003-ablation</title>261124-s003</text>", html_txt)
            self.assertIn("<strong>261123-s001-data-expansion</strong>", html_txt)
            self.assertNotIn('transform="rotate(-40 ', html_txt)

    def test_many_sessions_angled_ticks_and_grid_containment_css(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            sessions_root = root / "sessions"
            lp = root / "state" / "lineage.json"
            ep = root / "state" / "events.jsonl"
            out = root / "dashboard" / "index.html"

            lin = default_lineage(self.split_cfg, self.verifier_cfg)
            for idx in range(1, 16):
                sid = f"261127-s{idx:03d}-synthesis"
                ts = f"2026-11-27T{idx:02d}:00:00Z"
                _write_session_fixture(
                    sessions_root, sid, epoch=1, parent_id="starter",
                    v_score=0.05 + idx * 0.01, v_estates=(0.05, 0.05, 0.05),
                    v_delta_inc=idx * 0.01, t_delta=0.01, t_ci_low=-0.01, t_ci_high=0.03,
                    worst_estate="AirHitam", worst_delta=0.0, gate_passed=False, started_at=ts,
                )
                lin["sessions"][sid] = {"epoch": 1, "parent_id": "starter", "promoted": False, "collected_at": ts}
            save_lineage(lp, lin)
            ep.parent.mkdir(parents=True, exist_ok=True)
            ep.write_text("")

            generate_dashboard(AGENT_ROOT, output_path=out, sessions_root=sessions_root, lineage_path=lp, events_path=ep)
            errs = verify_dashboard(out, sessions_root=sessions_root, lineage_path=lp, events_path=ep)
            self.assertEqual(errs, [], f"Verification mismatches: {errs}")

            html_txt = out.read_text(encoding="utf-8")
            self.assertIn("x-tick-angled", html_txt)
            self.assertIn("transform=\"rotate(-40 ", html_txt)
            self.assertIn("grid-template-columns: minmax(0, 1fr) minmax(0, 1.25fr);", html_txt)
            self.assertIn("min-width: 0;", html_txt)


if __name__ == "__main__":
    unittest.main()
