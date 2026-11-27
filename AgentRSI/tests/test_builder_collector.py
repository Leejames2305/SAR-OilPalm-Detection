"""Host unit tests for AgentRSI M3 builder, runner guards, handoff, and collector."""
from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import yaml

AGENT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = AGENT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from libbuilder.build_task import (  # noqa: E402
    build_task,
    infer_lens_from_session_id,
    make_session_id,
    next_session_seq,
)
from libbuilder.collect import collect_session  # noqa: E402
from libbuilder.env_utils import load_dotenv, resolve_gateway  # noqa: E402
from libbuilder.events import load_events  # noqa: E402
from libbuilder.lineage import default_lineage, save_lineage  # noqa: E402
from libbuilder.render_handoff import render_handoff_section, validate_handoff  # noqa: E402
from libbuilder.run_session import build_harbor_command, run_session  # noqa: E402


def _make_mock_job(
    jobs_root: Path,
    session_id: str,
    job_id: str,
    *,
    gate_passed: bool = False,
    v_score: float = 0.061,
    comp_delta: float = 0.01,
    cost_usd: float = 0.25,
    pipeline_code: str = "def fit_predict(l, u, ctx): return [0.0]*len(u)\n",
    handoff_text: str | None = None,
    extra_log_text: str = "clean run\n",
) -> Path:
    job_dir = jobs_root / session_id / job_id
    trial_dir = job_dir / f"{session_id}__abc123"
    (trial_dir / "verifier").mkdir(parents=True)
    (trial_dir / "artifacts" / "app" / "submission").mkdir(parents=True)
    (trial_dir / "artifacts" / "app" / "report").mkdir(parents=True)
    (trial_dir / "agent").mkdir(parents=True)

    (job_dir / "result.json").write_text(json.dumps({"stats": {"cost_usd": cost_usd}}))
    (trial_dir / "result.json").write_text(
        json.dumps({
            "started_at": "2026-11-23T05:00:00Z",
            "finished_at": "2026-11-23T05:05:00Z",
            "agent_info": {"name": "claude-code", "model_info": {"name": "claude-haiku-5-5"}},
            "agent_result": {"cost_usd": cost_usd, "n_input_tokens": 100, "n_output_tokens": 50},
            "exception_info": None,
        })
    )
    (trial_dir / "agent" / "claude.log").write_text(extra_log_text)
    (trial_dir / "artifacts" / "app" / "submission" / "pipeline.py").write_text(pipeline_code)
    (trial_dir / "artifacts" / "app" / "submission" / "requirements.txt").write_text("")
    (trial_dir / "artifacts" / "app" / "report" / "REPORT.md").write_text("# Report\nAll good.\n")
    default_handoff = (
        "## State of play\nTested.\n"
        "## What I tried\nBaseline.\n"
        "## Dead ends\nNone.\n"
        "## Open leads, ranked\n1. SAR deltas.\n"
        "## How to resume\nRun cv.\n"
        "## Artifact index\n- `/app/submission/pipeline.py`\n"
    )
    (trial_dir / "artifacts" / "app" / "report" / "HANDOFF.md").write_text(handoff_text or default_handoff)

    reward = {
        "score": v_score,
        "v_lift_AirHitam": v_score,
        "v_lift_Palong": v_score,
        "v_lift_Serting": v_score,
        "v_delta_vs_incumbent": v_score - 0.054469,
        "v_delta_vs_coords": 0.01,
        "submission_valid": 1,
        "runtime_sec": 3.5,
    }
    v_rep = {"status": "ok", "submission_valid": True, "error": None, "composite": {"score": v_score}}
    cert = {
        "gate_passed": gate_passed,
        "composite": {"comp_delta": comp_delta, "ci_low": 0.005 if gate_passed else -0.01},
        "gate": {"passed": gate_passed, "failed_reasons": [] if gate_passed else ["below floor"]},
    }
    (trial_dir / "verifier" / "reward.json").write_text(json.dumps(reward))
    (trial_dir / "verifier" / "v_report.json").write_text(json.dumps(v_rep))
    (trial_dir / "verifier" / "certification.json").write_text(json.dumps(cert))
    return job_dir


class BuilderCollectorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.split_cfg = yaml.safe_load((AGENT_ROOT / "config" / "split.yaml").read_text())
        cls.verifier_cfg = yaml.safe_load((AGENT_ROOT / "config" / "verifier.yaml").read_text())

    def test_dotenv_and_gateway_urls(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / ".env"
            p.write_text(
                "# comment\n"
                "BASE_URL=https://example.invalid/v1  # trailing comment\n"
                "API_KEY='sk-test-1234567890' # quoted\n"
                "DAILY_SPEND_CAP_USD=5 # dollars\n"
            )
            env = load_dotenv(p)
            self.assertEqual(env["BASE_URL"], "https://example.invalid/v1")
            self.assertEqual(env["API_KEY"], "sk-test-1234567890")
            self.assertEqual(env["DAILY_SPEND_CAP_USD"], "5")

            gw_codex = resolve_gateway("codex", env)
            self.assertEqual(gw_codex["base_url"], "https://example.invalid/v1")
            self.assertEqual(gw_codex["api_format"], "openai_responses")
            self.assertEqual(gw_codex["api_key_name"], "OPENAI_API_KEY")

            gw_claude = resolve_gateway("claude-code", env)
            self.assertEqual(gw_claude["base_url"], "https://example.invalid")
            self.assertEqual(gw_claude["api_format"], "anthropic_messages")
            self.assertEqual(gw_claude["api_key_name"], "ANTHROPIC_AUTH_TOKEN")

            # Codex must NOT receive max_turns / max_budget_usd
            cmd_codex = build_harbor_command(
                task_dir=Path("/tmp/t"),
                jobs_out_dir=Path("/tmp/j"),
                agent="codex",
                model="gpt-6.1-terra",
                gateway=gw_codex,
                env_type="podman",
                mounts_json="[]",
                gcp_project="proj",
                agent_cpus=6,
                reasoning_effort="medium",
                max_turns=20,
                max_budget_usd=0.5,
            )
            self.assertNotIn("max_turns=20", cmd_codex)
            self.assertNotIn("max_budget_usd=0.5", cmd_codex)

            cmd_claude = build_harbor_command(
                task_dir=Path("/tmp/t"),
                jobs_out_dir=Path("/tmp/j"),
                agent="claude-code",
                model="claude-haiku-5-5",
                gateway=gw_claude,
                env_type="podman",
                mounts_json="[]",
                gcp_project="proj",
                agent_cpus=6,
                reasoning_effort="medium",
                max_turns=20,
                max_budget_usd=0.5,
            )
            self.assertIn("max_turns=20", cmd_claude)
            self.assertIn("max_budget_usd=0.5", cmd_claude)

    def test_build_task_strict_allowlist_and_leak_pass(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            out_root = Path(td) / "tasks"
            tdir = build_task("unit-check", focus_lens="open", timeout_override=600, output_root=out_root)
            env_dir = tdir / "environment"
            self.assertTrue((env_dir / "data" / "L_labelled.csv").exists())
            self.assertTrue((env_dir / "data" / "unlabelled.csv").exists())
            self.assertFalse((env_dir / "data" / "V_unlabelled.csv").exists())
            self.assertEqual(len(list((env_dir / "rasters").glob("*.tif"))), 19)
            self.assertTrue((env_dir / "prior" / "MainChecklist.md").exists())
            self.assertTrue((env_dir / "prior" / "Plan-GEECatalogSurvey.md").exists())
            self.assertTrue((tdir / "tests" / "incumbent" / "submission" / "pipeline.py").exists())
            self.assertTrue((tdir / "tests" / "incumbent" / "datasets").is_dir())
            self.assertFalse((env_dir / "prior" / "Plan-AgentRSI.md").exists())
            self.assertFalse((env_dir / "data" / "split_assignment.csv").exists())
            self.assertFalse((env_dir / "data" / "labels_master.csv").exists())
            raw_instr = (tdir / "instruction.md").read_text()
            instr = raw_instr.lower()
            self.assertNotIn("gate_passed", instr)
            self.assertNotIn("certification", instr)
            self.assertNotRegex(raw_instr, r"\{\{[A-Z0-9_]+\}\}")
            self.assertIn("You have 600 seconds to complete this task.", raw_instr)
            self.assertIn("`Pd` (double-bounce), 2: `Pv` (volume), 3: `Ps` (surface), 4: `Pc` (helix)", raw_instr)
            self.assertIn("`xgboost`, `lightgbm`, `geopandas`, or `rasterio`", raw_instr)
            self.assertIn("4 CPUs, 4096 MB RAM, 300 s per estate", raw_instr)
            self.assertFalse((env_dir / "prior" / "prev_submission").exists())

    def test_build_task_carries_forward_prev_submission_from_parent(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            out_root = Path(td) / "tasks"
            tdir = build_task(
                "unit-parent-check",
                focus_lens="open",
                parent_session="261127-s011-ablation",
                timeout_override=600,
                output_root=out_root,
            )
            prev_sub = tdir / "environment" / "prior" / "prev_submission"
            self.assertTrue((prev_sub / "pipeline.py").is_file())
            self.assertTrue((prev_sub / "requirements.txt").is_file())
            self.assertFalse((prev_sub / "__pycache__").exists())
            self.assertIn("/app/prior/prev_submission/", (tdir / "instruction.md").read_text())

    def test_yymmdd_session_naming_lens_inference_and_seq(self) -> None:
        lenses_dir = AGENT_ROOT / "templates" / "lenses"
        self.assertEqual(infer_lens_from_session_id("261123-s001-data-expansion", lenses_dir), "data-expansion")
        self.assertEqual(infer_lens_from_session_id("261123-s002-ablation", lenses_dir), "ablation")
        self.assertIsNone(infer_lens_from_session_id("s000-dry", lenses_dir))

        lin = default_lineage(self.split_cfg, self.verifier_cfg)
        lin["sessions"]["s000-dry"] = {"epoch": 1}
        self.assertEqual(next_session_seq(lin), 1)
        sid1 = make_session_id(next_session_seq(lin), "open", date_str="261123")
        self.assertEqual(sid1, "261123-s001-open")

        lin["sessions"][sid1] = {"epoch": 1}
        self.assertEqual(next_session_seq(lin), 2)
        sid2 = make_session_id(next_session_seq(lin), "ablation", date_str="261123")
        self.assertEqual(sid2, "261123-s002-ablation")

        with tempfile.TemporaryDirectory() as td:
            out_root = Path(td) / "tasks"
            tdir = build_task("261123-s001-data-expansion", timeout_override=600, output_root=out_root)
            toml_txt = (tdir / "task.toml").read_text()
            self.assertIn('name = "agentrsi/261123-s001-data-expansion"', toml_txt)
            self.assertNotIn("data-expansion-data-expansion", toml_txt)
            self.assertIn("Focus Lens: Data Expansion", (tdir / "instruction.md").read_text())

    def test_render_handoff_flags_disagreement_and_omits_t(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            sdir = Path(td) / "s001"
            sdir.mkdir()
            (sdir / "reward.json").write_text(
                json.dumps({
                    "score": 0.055,
                    "v_lift_AirHitam": 0.047,
                    "v_lift_Palong": 0.043,
                    "v_lift_Serting": 0.075,
                    "v_delta_vs_incumbent": 0.001,
                    "v_delta_vs_coords": -0.01,
                    "submission_valid": 1,
                })
            )
            (sdir / "v_report.json").write_text(json.dumps({"status": "ok"}))
            (sdir / "certification.json").write_text(json.dumps({"gate_passed": False, "t_secret": 0.999}))
            (sdir / "HANDOFF.md").write_text(
                "## State of play\nClaimed V score: 0.2500\n"
                "## What I tried\nx\n## Dead ends\ny\n## Open leads, ranked\nz\n## How to resume\na\n## Artifact index\nb\n"
            )
            lin = default_lineage(self.split_cfg, self.verifier_cfg)
            lin["sessions"]["s001"] = {"promoted": False}
            rendered = render_handoff_section(sdir, lin)
            self.assertIn("0.055000", rendered)
            self.assertIn("WARNING (verifier vs agent HANDOFF disagreement)", rendered)
            self.assertNotIn("0.999", rendered)
            self.assertNotIn("certification", rendered.lower())
            self.assertEqual(validate_handoff((sdir / "HANDOFF.md").read_text()), [])

            # Ancestor fallback when immediate parent crashed without writing HANDOFF.md
            s2dir = Path(td) / "s002"
            s2dir.mkdir()
            (s2dir / "reward.json").write_text(
                json.dumps({
                    "score": 0.02,
                    "v_lift_AirHitam": 0.0,
                    "v_lift_Palong": 0.03,
                    "v_lift_Serting": 0.03,
                    "v_delta_vs_incumbent": -0.035,
                    "v_delta_vs_coords": -0.02,
                    "submission_valid": 1,
                })
            )
            (s2dir / "v_report.json").write_text(
                json.dumps({
                    "status": "ok",
                    "datasets_audit": {"datasets": {"alos_winstats": {"file": "features.parquet", "rows": 7220, "cols": 403}}},
                })
            )
            (s2dir / "meta.json").write_text(
                json.dumps({"exception_info": {"exception_type": "NonZeroAgentExitCodeError"}})
            )
            lin["sessions"]["s002"] = {"parent_id": "s001", "promoted": False}
            fb = render_handoff_section(s2dir, lin)
            self.assertIn("NonZeroAgentExitCodeError", fb)
            self.assertIn("/app/datasets/alos_winstats/features.parquet", fb)
            self.assertIn("Last Available HANDOFF.md (from `s001`)", fb)
            self.assertIn("## Open leads, ranked", fb)
            self.assertNotIn("0.999", fb)

    def test_collect_secret_scan_smells_idempotency_and_promotion(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            jobs_root = root / "jobs"
            sessions_root = root / "sessions"
            lp = root / "state" / "lineage.json"
            ep = root / "state" / "events.jsonl"
            envp = root / ".env"
            envp.write_text("API_KEY=sk-fake-secret-999999\nDAILY_SPEND_CAP_USD=5\n")

            # 1. Secret leak aborts collection before sessions/ is written
            bad_job = _make_mock_job(
                jobs_root,
                "s-leak",
                "2026-11-23__13-00-00",
                extra_log_text="-----BEGIN PRIVATE KEY-----\nABCDEF\n",
            )
            with self.assertRaises(RuntimeError):
                collect_session(
                    "s-leak",
                    job_dir=bad_job,
                    jobs_root=jobs_root,
                    sessions_root=sessions_root,
                    lineage_path=lp,
                    events_path=ep,
                    env_path=envp,
                )
            self.assertFalse((sessions_root / "s-leak").exists())

            # 2. Smell detection + idempotent re-collection + promotion
            good_job = _make_mock_job(
                jobs_root,
                "s001",
                "2026-11-23__13-10-00",
                gate_passed=True,
                v_score=0.095,
                comp_delta=0.041,
                cost_usd=0.42,
                pipeline_code="import requests\n# https://example.com\ndef fit_predict(l, u, ctx): return [0.0]*len(u)\n",
            )
            collect_session(
                "s001",
                job_dir=good_job,
                jobs_root=jobs_root,
                sessions_root=sessions_root,
                lineage_path=lp,
                events_path=ep,
                env_path=envp,
            )
            events_first = ep.read_text()
            lin_first = json.loads(lp.read_text())
            self.assertEqual(lin_first["incumbent"]["id"], "s001")
            self.assertEqual(lin_first["t_attempts_used"], 1)
            self.assertAlmostEqual(lin_first["daily_spend_usd"]["2026-11-23"], 0.42, places=6)
            types_first = [e["type"] for e in load_events(ep)]
            self.assertIn("promoted", types_first)
            self.assertIn("review_flag", types_first)

            # Re-collect same job: zero duplicate events, zero double counting
            collect_session(
                "s001",
                job_dir=good_job,
                jobs_root=jobs_root,
                sessions_root=sessions_root,
                lineage_path=lp,
                events_path=ep,
                env_path=envp,
            )
            self.assertEqual(ep.read_text(), events_first)
            lin_second = json.loads(lp.read_text())
            self.assertEqual(lin_second["t_attempts_used"], 1)
            self.assertAlmostEqual(lin_second["daily_spend_usd"]["2026-11-23"], 0.42, places=6)

    def test_agentrsi_selfcheck_import_pin_audit(self) -> None:
        import importlib.util

        cli_path = AGENT_ROOT / "templates" / "agentrsi_cli.py"
        spec = importlib.util.spec_from_file_location("agentrsi_cli_mod", cli_path)
        assert spec and spec.loader
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)

        with tempfile.TemporaryDirectory() as td:
            sub = Path(td) / "submission"
            sub.mkdir()
            (sub / "helper.py").write_text("import math\nimport numpy as np\nimport sklearn\n")
            (sub / "pipeline.py").write_text(
                "from typing import TYPE_CHECKING\n"
                "import helper\n"
                "import pandas as pd\n"
                "import yaml\n"
                "if TYPE_CHECKING:\n"
                "    import geopandas\n"
                "try:\n"
                "    import xgboost\n"
                "except ImportError:\n"
                "    xgboost = None\n"
                "def fit_predict(l, u, ctx): return [0.0]*len(u)\n"
            )
            # Base + stdlib + local + guarded optional imports pass with empty requirements
            self.assertEqual(mod._audit_submission_imports(sub, []), [])

            # Unpinned third-party import fails with actionable pin hint
            (sub / "pipeline.py").write_text(
                "import rasterio\n"
                "import lightgbm as lgb\n"
                "def fit_predict(l, u, ctx): return [0.0]*len(u)\n"
            )
            errs = mod._audit_submission_imports(sub, [])
            self.assertEqual(len(errs), 2)
            self.assertTrue(any("rasterio" in e and "rasterio==" in e for e in errs))
            self.assertTrue(any("lightgbm" in e for e in errs))

            # Pinning both packages clears the audit
            self.assertEqual(
                mod._audit_submission_imports(sub, ["rasterio==1.5.2", "lightgbm==4.6.0"]),
                [],
            )

            # Symlinks inside submission/ are rejected
            link = sub / "bad_link.py"
            link.symlink_to(sub / "helper.py")
            sym_errs = mod._audit_submission_imports(sub, ["rasterio==1.5.2", "lightgbm==4.6.0"])
            self.assertTrue(any("Symlink" in e for e in sym_errs))

    def test_smell_ignores_prompt_echo_but_catches_real_repo_access(self) -> None:
        from libbuilder.collect import scan_smells

        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            log_ok = root / "rollout.jsonl"
            log_ok.write_text(
                "Never access, query, or clone `https://github.com/Leejames2305/SAR-OilPalm-Detection`.\n"
                "never access https://github.com/Leejames2305/SAR-OilPalm-Detection, touch /tests\n"
            )
            self.assertEqual(scan_smells(root / "no_sub", [], [log_ok]), [])

            log_bad = root / "bad.jsonl"
            log_bad.write_text("git clone https://github.com/Leejames2305/SAR-OilPalm-Detection\n")
            smells = scan_smells(root / "no_sub", [], [log_bad])
            self.assertEqual(len(smells), 1)
            self.assertIn("blocked research repo", smells[0])

    def test_spend_cap_blocks_run_session(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            tdir = root / "tasks" / "s-cap"
            tdir.mkdir(parents=True)
            lp = root / "lineage.json"
            ep = root / "events.jsonl"
            envp = root / ".env"
            envp.write_text(
                "API_KEY=sk-test-1234567890\nBASE_URL=https://api.gpt.ge/v1\n"
                "GCP_PROJECT=p\nGCP_SA_KEY_PATH=.secrets/Keys_SAEarthEngine.json\nDAILY_SPEND_CAP_USD=1.0\n"
            )
            lin = default_lineage(self.split_cfg, self.verifier_cfg)
            from datetime import datetime, timezone
            today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
            lin["daily_spend_usd"][today] = 1.25
            save_lineage(lp, lin)

            rc = run_session(
                "s-cap",
                tasks_root=root / "tasks",
                jobs_root=root / "jobs",
                env_path=envp,
                lineage_path=lp,
                events_path=ep,
            )
            self.assertEqual(rc, 2)
            self.assertEqual(load_events(ep)[-1]["type"], "spend_cap")


if __name__ == "__main__":
    unittest.main()
