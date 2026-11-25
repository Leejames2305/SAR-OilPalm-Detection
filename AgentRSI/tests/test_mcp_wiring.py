"""Host unit tests for M7/M8 Harbor wiring (Interface Contract C)."""
from __future__ import annotations

import json
import socket
import subprocess
import sys
import tempfile
import tomllib
import unittest
from pathlib import Path
from unittest import mock

import yaml
from pydantic import TypeAdapter

AGENT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = AGENT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from harbor.agents.mcp_headers import resolve_mcp_header_env  # noqa: E402
from harbor.lint.models import COMPOSE_RULES, DEFAULT_RULES  # noqa: E402
from harbor.lint.task import lint_task  # noqa: E402
from harbor.models.task.config import ArtifactConfig, TaskConfig  # noqa: E402
from harbor.models.trial.config import ServiceVolumeConfig  # noqa: E402
from libbuilder.build_task import build_task  # noqa: E402
from libbuilder.collect import collect_session  # noqa: E402
from libbuilder.events import VALID_TYPES, append_event, load_events  # noqa: E402
from libbuilder.run_session import run_session  # noqa: E402
from libbuilder.stage_secrets import (  # noqa: E402
    CONTAINER_COLAB_IO_DIR,
    CONTAINER_COLAB_SOCKET_PATH,
    CONTAINER_SECRET_PATH,
    build_mounts_json,
    stage_gcp_sa,
)


def _run_leak_check(target: Path) -> subprocess.CompletedProcess[str]:
    leak_script = AGENT_ROOT / "src" / "DataPipeline" / "002-CheckLeaks.py"
    return subprocess.run(
        [sys.executable, str(leak_script), "--no-view", "--scan", str(target)],
        capture_output=True,
        text=True,
        check=False,
    )


def _write_mock_job_with_sidecar(
    jobs_root: Path,
    session_id: str,
    job_id: str,
    *,
    queries_lines: list[dict[str, object]] | None = None,
    extra_log_text: str = "clean run\n",
) -> Path:
    job_dir = jobs_root / session_id / job_id
    trial_dir = job_dir / f"{session_id}__trial1"
    (trial_dir / "verifier").mkdir(parents=True)
    (trial_dir / "artifacts" / "app" / "submission").mkdir(parents=True)
    (trial_dir / "artifacts" / "app" / "report").mkdir(parents=True)
    (trial_dir / "agent").mkdir(parents=True)

    (job_dir / "result.json").write_text(json.dumps({"stats": {"cost_usd": 0.1}}))
    (trial_dir / "result.json").write_text(
        json.dumps({
            "started_at": "2026-11-24T05:00:00Z",
            "finished_at": "2026-11-24T05:05:00Z",
            "agent_info": {"name": "codex", "model_info": {"name": "gpt-6.1-terra"}},
            "agent_result": {"cost_usd": 0.1, "n_input_tokens": 10, "n_output_tokens": 5},
            "exception_info": None,
        })
    )
    (trial_dir / "agent" / "codex.log").write_text(extra_log_text)
    (trial_dir / "artifacts" / "app" / "submission" / "pipeline.py").write_text(
        "def fit_predict(l, u, ctx): return [0.0]*len(u)\n"
    )
    (trial_dir / "artifacts" / "app" / "submission" / "requirements.txt").write_text("")
    (trial_dir / "artifacts" / "app" / "report" / "REPORT.md").write_text("# Report\n")
    (trial_dir / "artifacts" / "app" / "report" / "HANDOFF.md").write_text(
        "## State of play\nx\n## What I tried\nx\n## Dead ends\nx\n"
        "## Open leads, ranked\nx\n## How to resume\nx\n## Artifact index\nx\n"
    )
    if queries_lines is not None:
        qpath = trial_dir / "artifacts" / "sidecar" / "queries.jsonl"
        qpath.parent.mkdir(parents=True, exist_ok=True)
        qpath.write_text("".join(json.dumps(row) + "\n" for row in queries_lines))

    (trial_dir / "verifier" / "reward.json").write_text(
        json.dumps({
            "score": 0.06,
            "v_lift_AirHitam": 0.06,
            "v_lift_Palong": 0.06,
            "v_lift_Serting": 0.06,
            "v_delta_vs_incumbent": 0.005,
            "v_delta_vs_coords": 0.01,
            "submission_valid": 1,
            "runtime_sec": 2.0,
        })
    )
    (trial_dir / "verifier" / "v_report.json").write_text(
        json.dumps({"status": "ok", "submission_valid": True, "error": None, "composite": {"score": 0.06}})
    )
    (trial_dir / "verifier" / "certification.json").write_text(
        json.dumps({"gate_passed": False, "composite": {"comp_delta": 0.0, "ci_low": -0.01}})
    )
    return job_dir


class McpWiringTests(unittest.TestCase):
    def test_defaults_enable_m7_m8_and_4h_timeout(self) -> None:
        session_cfg = yaml.safe_load((AGENT_ROOT / "config" / "session.yaml").read_text())
        self.assertEqual(int(session_cfg["agent_timeout_sec"]), 14400)
        self.assertAlmostEqual(float(session_cfg["max_budget_usd"]), 10.0)
        self.assertTrue(bool(session_cfg["inrun_verifier"]))
        self.assertTrue(bool(session_cfg["colab_broker"]["enabled"]))
        self.assertEqual(list(session_cfg["colab_broker"]["allowed_gpus"]), ["T4", "L4"])
        self.assertFalse(bool(session_cfg["colab_broker"]["allow_a100"]))

        with tempfile.TemporaryDirectory() as td:
            tdir = build_task("261125-default-check", output_root=Path(td))
            self.assertTrue((tdir / "environment" / "docker-compose.yaml").is_file())
            self.assertTrue((tdir / "environment" / "sidecar" / "Dockerfile").is_file())

            raw = tomllib.loads((tdir / "task.toml").read_text())
            cfg = TaskConfig.model_validate(raw)
            self.assertEqual(cfg.agent.timeout_sec, 14400.0)
            self.assertEqual(
                [s.name for s in cfg.environment.mcp_servers],
                ["agentrsi-verifier", "agentrsi-colab"],
            )
            self.assertIn(
                ArtifactConfig(
                    source="/var/log/agentrsi-verifier/queries.jsonl",
                    destination="sidecar/queries.jsonl",
                    service="verifier-sidecar",
                ),
                cfg.artifacts,
            )

            instr = (tdir / "instruction.md").read_text()
            self.assertIn("### MCP Tools", instr)
            self.assertIn("agentrsi-verifier", instr)
            self.assertIn("agentrsi-colab", instr)
            self.assertIn("`T4`, `L4`", instr)
            self.assertNotIn("A100", instr)
            self.assertIn("You have 14400 seconds to complete this task.", instr)
            self.assertEqual(_run_leak_check(tdir).returncode, 0)

    def test_explicit_disable_m7_m8_backward_compatible(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            tdir = build_task(
                "261124-m0-compat",
                timeout_override=300,
                enable_inrun_verifier=False,
                enable_colab=False,
                output_root=Path(td),
            )
            self.assertFalse((tdir / "environment" / "docker-compose.yaml").exists())
            self.assertFalse((tdir / "environment" / "sidecar").exists())

            raw = tomllib.loads((tdir / "task.toml").read_text())
            cfg = TaskConfig.model_validate(raw)
            self.assertEqual(cfg.environment.mcp_servers, [])
            self.assertEqual(cfg.artifacts, ["/app/submission", "/app/report", "/app/datasets"])

            instr = (tdir / "instruction.md").read_text()
            self.assertNotIn("agentrsi-verifier", instr)
            self.assertNotIn("agentrsi-colab", instr)
            self.assertNotIn("### MCP Tools", instr)
            self.assertEqual(_run_leak_check(tdir).returncode, 0)

    def test_m7_only_task_build_compose_and_lint(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            tdir = build_task(
                "261124-m7-only",
                timeout_override=300,
                enable_inrun_verifier=True,
                enable_colab=False,
                output_root=Path(td),
            )
            env_dir = tdir / "environment"
            compose_path = env_dir / "docker-compose.yaml"
            sidecar_df = env_dir / "sidecar" / "Dockerfile"
            self.assertTrue(compose_path.is_file())
            self.assertTrue(sidecar_df.is_file())
            self.assertTrue((env_dir / "sidecar" / "incumbent" / "submission" / "pipeline.py").is_file())
            self.assertTrue((env_dir / "sidecar" / "incumbent" / "datasets").is_dir())

            df_text = sidecar_df.read_text()
            self.assertIn("FROM localhost/agentrsi-sealed:epoch1", df_text)
            self.assertIn("mkdir -p /var/log/agentrsi-verifier && touch /var/log/agentrsi-verifier/queries.jsonl", df_text)
            self.assertIn("COPY incumbent/ /opt/agentrsi/incumbent/", df_text)

            compose_data = yaml.safe_load(compose_path.read_text())
            sidecar = compose_data["services"]["verifier-sidecar"]
            self.assertEqual(sidecar["build"]["context"], "./sidecar")
            self.assertEqual(sidecar["cpus"], 2)
            self.assertEqual(sidecar["mem_limit"], "2048m")
            self.assertEqual(sidecar["environment"]["AGENTRSI_VERIFIER_MCP_TOKEN"], "${AGENTRSI_VERIFIER_MCP_TOKEN}")
            self.assertEqual(sidecar["volumes"], ["agentrsi_share:/shared:ro"])
            self.assertIn("/health", " ".join(sidecar["healthcheck"]["test"]))
            self.assertEqual(compose_data["services"]["main"]["volumes"], ["agentrsi_share:/shared:rw"])
            self.assertEqual(
                compose_data["services"]["main"]["depends_on"]["verifier-sidecar"]["condition"],
                "service_healthy",
            )
            self.assertIn("agentrsi_share", compose_data["volumes"])

            # Verify docker-compose merge with Harbor's docker-compose-build.yaml succeeds
            build_compose = (
                AGENT_ROOT
                / ".venv"
                / "lib"
                / "python3.12"
                / "site-packages"
                / "harbor"
                / "environments"
                / "docker"
                / "docker-compose-build.yaml"
            )
            dc_bin = AGENT_ROOT / ".tools" / "bin" / "docker-compose"
            if dc_bin.exists():
                dc_res = subprocess.run(
                    [
                        str(dc_bin),
                        "--project-directory",
                        str(env_dir),
                        "-f",
                        str(build_compose),
                        "-f",
                        str(compose_path),
                        "config",
                    ],
                    env={"CONTEXT_DIR": str(env_dir), "AGENTRSI_VERIFIER_MCP_TOKEN": "testtoken"},
                    capture_output=True,
                    text=True,
                    check=False,
                )
                self.assertEqual(dc_res.returncode, 0, dc_res.stderr)

            raw = tomllib.loads((tdir / "task.toml").read_text())
            cfg = TaskConfig.model_validate(raw)
            self.assertEqual([s.name for s in cfg.environment.mcp_servers], ["agentrsi-verifier"])
            srv = cfg.environment.mcp_servers[0]
            self.assertEqual(srv.transport, "streamable-http")
            self.assertEqual(srv.url, "http://verifier-sidecar:8765/mcp")
            self.assertEqual(srv.headers, {"Authorization": "Bearer ${AGENTRSI_VERIFIER_MCP_TOKEN}"})
            self.assertIn(
                ArtifactConfig(
                    source="/var/log/agentrsi-verifier/queries.jsonl",
                    destination="sidecar/queries.jsonl",
                    service="verifier-sidecar",
                ),
                cfg.artifacts,
            )

            # Harbor non-compose lint rules must return zero findings
            findings = lint_task(tdir, selected_rules=DEFAULT_RULES - COMPOSE_RULES)
            self.assertEqual(findings, [], [f.model_dump() for f in findings])

            instr = (tdir / "instruction.md").read_text()
            self.assertIn("score_v", instr)
            self.assertIn("verifier_budget", instr)
            self.assertIn("verifier_health", instr)
            self.assertIn("/shared", instr)
            self.assertNotIn("gate_passed", instr.lower())
            self.assertNotIn("certification", instr.lower())
            self.assertEqual(_run_leak_check(tdir).returncode, 0)

    def test_m8_only_task_build_and_combined_mounts(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            tdir = build_task(
                "261124-m8-only",
                timeout_override=300,
                enable_inrun_verifier=False,
                enable_colab=True,
                colab_allow_a100=True,
                output_root=root / "tasks",
            )
            self.assertFalse((tdir / "environment" / "docker-compose.yaml").exists())
            raw = tomllib.loads((tdir / "task.toml").read_text())
            cfg = TaskConfig.model_validate(raw)
            self.assertEqual([s.name for s in cfg.environment.mcp_servers], ["agentrsi-colab"])
            srv = cfg.environment.mcp_servers[0]
            self.assertEqual(srv.transport, "stdio")
            self.assertEqual(srv.command, "python3")
            self.assertEqual(
                srv.args,
                ["/usr/local/bin/agentrsi_mcp_relay.py", "--socket", "/run/agentrsi/colab-mcp.sock"],
            )
            instr = (tdir / "instruction.md").read_text()
            self.assertIn("colab_run", instr)
            self.assertIn("colab_status", instr)
            self.assertIn("colab_fetch", instr)
            self.assertIn("colab_stop", instr)
            self.assertIn("A100", instr)
            self.assertEqual(_run_leak_check(tdir).returncode, 0)

            # Combined M7 + M8 task.toml & header env resolution
            tboth = build_task(
                "261124-m78-both",
                timeout_override=300,
                enable_inrun_verifier=True,
                enable_colab=True,
                output_root=root / "tasks",
            )
            cfg_both = TaskConfig.model_validate(tomllib.loads((tboth / "task.toml").read_text()))
            self.assertEqual(
                [s.name for s in cfg_both.environment.mcp_servers],
                ["agentrsi-verifier", "agentrsi-colab"],
            )
            header_env = resolve_mcp_header_env(
                cfg_both.environment.mcp_servers,
                lambda k: "deadbeef" * 8 if k == "AGENTRSI_VERIFIER_MCP_TOKEN" else None,
            )
            self.assertIn("agentrsi-verifier", header_env.refs)
            self.assertIn("Authorization", header_env.refs["agentrsi-verifier"])

            # Stage GCP SA compatibility + build_mounts_json validation
            sa_src = root / "sa.json"
            sa_src.write_text('{"type": "service_account"}\n')
            stage_dir = root / "stage"
            dst, base_mounts_json = stage_gcp_sa(sa_src, stage_dir=stage_dir)
            ta = TypeAdapter(list[ServiceVolumeConfig])
            base_mounts = ta.validate_python(json.loads(base_mounts_json))
            self.assertEqual(len(base_mounts), 1)
            self.assertEqual(base_mounts[0]["target"], CONTAINER_SECRET_PATH)
            self.assertTrue(base_mounts[0]["read_only"])

            full_mounts_json = build_mounts_json(
                dst,
                colab_socket_path=stage_dir / "colab-s1.sock",
                colab_io_dir=stage_dir / "colab-io-s1",
            )
            full_mounts = ta.validate_python(json.loads(full_mounts_json))
            targets = {m["target"]: m for m in full_mounts}
            self.assertEqual(
                set(targets),
                {CONTAINER_SECRET_PATH, CONTAINER_COLAB_IO_DIR, CONTAINER_COLAB_SOCKET_PATH},
            )
            self.assertNotIn("read_only", targets[CONTAINER_COLAB_IO_DIR])
            self.assertNotIn("read_only", targets[CONTAINER_COLAB_SOCKET_PATH])

    def test_run_session_ephemeral_token_and_mock_broker_teardown(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            tasks_root = root / "tasks"
            jobs_root = root / "jobs"
            stage_dir = root / "stage"
            lp = root / "state" / "lineage.json"
            ep = root / "state" / "events.jsonl"
            sa_src = root / "sa.json"
            sa_src.write_text('{"type": "service_account"}\n')
            envp = root / ".env"
            envp.write_text(
                f"API_KEY=sk-test-1234567890\n"
                f"BASE_URL=https://example.invalid/v1\n"
                f"GCP_PROJECT=proj\n"
                f"GCP_SA_KEY_PATH={sa_src}\n"
                f"DAILY_SPEND_CAP_USD=10.0\n"
            )
            sid = "261124-s999-open"
            tdir = build_task(
                sid,
                timeout_override=300,
                enable_inrun_verifier=True,
                enable_colab=True,
                output_root=tasks_root,
            )

            observed: dict[str, object] = {}

            def fake_run(cmd, env=None, check=False):  # noqa: ANN001, ANN202
                observed["cmd"] = list(cmd)
                observed["token"] = (env or {}).get("AGENTRSI_VERIFIER_MCP_TOKEN", "")
                sock_path = stage_dir / f"colab-{sid}.sock"
                io_dir = stage_dir / f"colab-io-{sid}"
                observed["sock_exists_during_run"] = sock_path.exists()
                observed["io_exists_during_run"] = (io_dir / "in").is_dir() and (io_dir / "out").is_dir()
                with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as s:
                    s.settimeout(3.0)
                    s.connect(str(sock_path))
                    s.sendall(b'{"jsonrpc":"2.0","id":1,"method":"initialize","params":{}}\n')
                    resp = s.recv(4096).decode()
                    observed["init_resp"] = json.loads(resp)
                return subprocess.CompletedProcess(cmd, 0)

            with mock.patch("libbuilder.run_session.subprocess.run", side_effect=fake_run):
                rc = run_session(
                    sid,
                    colab_mock=True,
                    tasks_root=tasks_root,
                    jobs_root=jobs_root,
                    env_path=envp,
                    lineage_path=lp,
                    events_path=ep,
                    stage_dir=stage_dir,
                )

            self.assertEqual(rc, 0)
            tok = str(observed["token"])
            self.assertRegex(tok, r"^[0-9a-f]{64}$")
            self.assertNotIn(tok, (tdir / "task.toml").read_text())
            self.assertNotIn(tok, (tdir / "instruction.md").read_text())
            self.assertNotIn(tok, ep.read_text())
            self.assertTrue(observed["sock_exists_during_run"])
            self.assertTrue(observed["io_exists_during_run"])
            self.assertEqual(observed["init_resp"]["result"]["serverInfo"]["name"], "agentrsi-colab")
            # Teardown happened in finally:
            self.assertFalse((stage_dir / f"colab-{sid}.sock").exists())
            self.assertFalse((stage_dir / f"colab-io-{sid}").exists())

            # Also verify teardown happens even if Harbor raises an exception
            def failing_run(*args, **kwargs):  # noqa: ANN002, ANN003, ANN202
                raise RuntimeError("harbor crashed")

            with mock.patch("libbuilder.run_session.subprocess.run", side_effect=failing_run):
                with self.assertRaises(RuntimeError):
                    run_session(
                        sid,
                        colab_mock=True,
                        tasks_root=tasks_root,
                        jobs_root=jobs_root,
                        env_path=envp,
                        lineage_path=lp,
                        events_path=ep,
                        stage_dir=stage_dir,
                    )
            self.assertFalse((stage_dir / f"colab-{sid}.sock").exists())
            self.assertFalse((stage_dir / f"colab-io-{sid}").exists())

            # Auth check blocks before launching Harbor when colab_mock=False
            with mock.patch(
                "colab_broker.backend.CliColabBackend.check_auth_ready",
                return_value=(False, "Run colab auth"),
            ):
                with self.assertRaisesRegex(RuntimeError, "Run colab auth"):
                    run_session(
                        sid,
                        colab_mock=False,
                        tasks_root=tasks_root,
                        jobs_root=jobs_root,
                        env_path=envp,
                        lineage_path=lp,
                        events_path=ep,
                        stage_dir=stage_dir,
                    )

    def test_collect_preserves_sidecar_queries_and_detects_token_leak(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            jobs_root = root / "jobs"
            sessions_root = root / "sessions"
            lp = root / "state" / "lineage.json"
            ep = root / "state" / "events.jsonl"
            envp = root / ".env"
            envp.write_text("API_KEY=sk-fake-secret-999999\nDAILY_SPEND_CAP_USD=5\n")

            job_ok = _write_mock_job_with_sidecar(
                jobs_root,
                "261124-s010-open",
                "2026-11-24__10-00-00",
                queries_lines=[{"q": 1}, {"q": 2}, {"q": 3}],
            )
            res = collect_session(
                "261124-s010-open",
                job_dir=job_ok,
                jobs_root=jobs_root,
                sessions_root=sessions_root,
                lineage_path=lp,
                events_path=ep,
                env_path=envp,
                update_dashboard=False,
            )
            sdir = Path(res["session_dir"])
            self.assertTrue((sdir / "sidecar_queries.jsonl").is_file())
            self.assertEqual(res["meta"]["inrun_queries_used"], 3)

            # Token leak in log aborts collection
            job_leak = _write_mock_job_with_sidecar(
                jobs_root,
                "261124-s011-open",
                "2026-11-24__10-05-00",
                extra_log_text="AGENTRSI_VERIFIER_MCP_TOKEN=" + ("a1b2c3d4" * 8) + "\n",
            )
            with self.assertRaises(RuntimeError):
                collect_session(
                    "261124-s011-open",
                    job_dir=job_leak,
                    jobs_root=jobs_root,
                    sessions_root=sessions_root,
                    lineage_path=lp,
                    events_path=ep,
                    env_path=envp,
                    update_dashboard=False,
                )
            self.assertFalse((sessions_root / "261124-s011-open").exists())

            # Colab event types are valid
            for ev_type in ("colab_vm_start", "colab_vm_stop", "colab_job_run", "colab_blocked"):
                self.assertIn(ev_type, VALID_TYPES)
                self.assertTrue(
                    append_event(ep, session="261124-s010-open", epoch=1, event_type=ev_type, summary=ev_type)
                )
            self.assertTrue(len(load_events(ep)) >= 4)


if __name__ == "__main__":
    unittest.main()
