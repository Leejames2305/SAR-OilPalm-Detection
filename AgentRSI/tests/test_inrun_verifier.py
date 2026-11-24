"""Host unit and ASGI tests for AgentRSI In-Run Verifier v2 sidecar MCP server."""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest import mock

import pandas as pd
import yaml
from starlette.testclient import TestClient

AGENT_ROOT = Path(__file__).resolve().parents[1]
if str(AGENT_ROOT) not in sys.path:
    sys.path.insert(0, str(AGENT_ROOT))

from scripts.build_verifier_base import BASE_DOCKERFILE  # noqa: E402
from verifier.inrun_server import (  # noqa: E402
    InRunVerifierState,
    _load_v_only_bundle,
    _resolve_candidate_base_python,
    build_asgi_app,
)


def _mcp_post(client: TestClient, method: str, params: dict, req_id: int = 1, headers: dict | None = None):
    hdrs = {"Accept": "application/json, text/event-stream"}
    if headers:
        hdrs.update(headers)
    return client.post(
        "/mcp",
        json={"jsonrpc": "2.0", "id": req_id, "method": method, "params": params},
        headers=hdrs,
    )


def _parse_mcp_tool_json(resp) -> dict:
    payload = resp.json()
    text = payload["result"]["content"][0]["text"]
    return json.loads(text)


class InRunVerifierUnitTests(unittest.TestCase):
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
        cls.starter_code = (AGENT_ROOT / "starter" / "pipeline.py").read_text()

    @classmethod
    def tearDownClass(cls) -> None:
        cls.tmp_sealed_obj.cleanup()

    def test_base_dockerfile_sidecar_venv_isolation(self) -> None:
        self.assertIn("/opt/agentrsi/sidecar-venv", BASE_DOCKERFILE)
        self.assertIn("mcp==2.3.0", BASE_DOCKERFILE)
        self.assertIn("uvicorn==0.54.0", BASE_DOCKERFILE)
        self.assertIn("useradd --uid 10001", BASE_DOCKERFILE)
        self.assertIn("chmod 700 /sealed && chmod 600 /sealed/*", BASE_DOCKERFILE)
        # Ensure mcp/uvicorn are not installed into global /usr/local/bin/python pip block
        first_pip_block = BASE_DOCKERFILE.split("RUN python3 -m venv")[0]
        self.assertNotIn("mcp", first_pip_block)
        self.assertNotIn("uvicorn", first_pip_block)

    def test_resolve_candidate_base_python_prefers_container_python(self) -> None:
        # When /usr/local/bin/python exists, it must be preferred over sys.executable
        with mock.patch.object(Path, "exists", autospec=True) as mock_exists:
            mock_exists.side_effect = lambda p: str(p) == "/usr/local/bin/python"
            self.assertEqual(_resolve_candidate_base_python(), Path("/usr/local/bin/python"))

        # Inside sidecar-venv without /usr/local/bin/python, falls back to base_prefix python3
        with (
            mock.patch("verifier.inrun_server.sys.prefix", "/opt/agentrsi/sidecar-venv"),
            mock.patch("verifier.inrun_server.sys.base_prefix", "/opt/basepy"),
            mock.patch.object(Path, "exists", autospec=True) as mock_exists,
        ):
            mock_exists.side_effect = lambda p: str(p) == "/opt/basepy/bin/python3"
            self.assertEqual(_resolve_candidate_base_python(), Path("/opt/basepy/bin/python3"))

        # On host without /usr/local/bin/python and outside sidecar-venv, falls back to sys.executable
        with mock.patch.object(Path, "exists", autospec=True) as mock_exists:
            mock_exists.side_effect = lambda p: str(p) != "/usr/local/bin/python"
            self.assertEqual(_resolve_candidate_base_python(), Path(sys.executable))

    def test_v_only_bundle_strips_all_t_labels(self) -> None:
        bundle = _load_v_only_bundle(self.sealed_dir, self.verifier_cfg)
        self.assertEqual(bundle["estates"], ["AirHitam", "Palong", "Serting"])

        master = pd.read_csv(self.sealed_dir / "labels_master.csv")
        split_df = pd.read_csv(self.sealed_dir / "split_assignment.csv")
        merged = master.merge(split_df[["estate", "id", "part"]], on=["estate", "id"], validate="one_to_one")

        for e in bundle["estates"]:
            ed = bundle["by_estate"][e]
            self.assertEqual(set(ed.keys()), {"labeled", "unlabeled", "v_mask", "v_y"})
            expected_v = merged[(merged["estate"] == e) & (merged["part"] == "V") & (merged["class"] != "Middle")]
            self.assertEqual(int(ed["v_mask"].sum()), len(expected_v))
            self.assertEqual(len(ed["v_y"]), len(expected_v))
            self.assertEqual(int(ed["v_y"].sum()), int((expected_v["class"] == "Unhealthy").sum()))

    def test_config_inrun_defaults_and_overrides(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            cfg = yaml.safe_load(json.dumps(self.verifier_cfg))
            cfg["inrun"] = {
                "max_queries": 7,
                "noise_sigma": 0.01,
                "per_estate_timeout_sec": 90,
                "run_user": "runner",
            }
            state = InRunVerifierState(
                sealed_dir=self.sealed_dir,
                shared_dir=root / "shared",
                log_path=root / "logs" / "queries.jsonl",
                verifier_cfg=cfg,
            )
            self.assertEqual(state.max_queries, 7)
            self.assertAlmostEqual(state.noise_sigma, 0.01)
            self.assertEqual(state.per_estate_timeout_sec, 90.0)
            self.assertEqual(state.run_user, "runner")

    def test_zero_noise_starter_and_leak_check(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            shared = root / "shared"
            shared.mkdir()
            log_path = root / "logs" / "queries.jsonl"
            state = InRunVerifierState(
                sealed_dir=self.sealed_dir,
                shared_dir=shared,
                log_path=log_path,
                max_queries=5,
                noise_sigma=0.0,
                verifier_cfg=self.verifier_cfg,
            )
            res = state.score_v(pipeline_code=self.starter_code, note="starter-zero-noise")
            self.assertTrue(res["ok"])
            self.assertEqual(res["status"], "ok")
            self.assertFalse(res["cached"])
            self.assertEqual(res["queries_used"], 1)
            self.assertEqual(res["queries_remaining"], 4)
            self.assertAlmostEqual(res["composite"]["noisy_delta_vs_incumbent"], 0.0, places=12)
            for e in state.estates:
                self.assertAlmostEqual(res["estates"][e]["noisy_delta_vs_incumbent"], 0.0, places=12)
                self.assertEqual(
                    set(res["estates"][e].keys()),
                    {"noisy_pr_auc_lift", "noisy_delta_vs_incumbent", "noisy_delta_vs_coords"},
                )

            # Ensure zero raw scores, T scores, gate info, or bootstrap arrays appear
            serialized = json.dumps(res)
            for forbidden in ("gate_passed", "certification", "bootstrap", "recall_at_25", "roc_auc", "pr_auc\""):
                self.assertNotIn(forbidden, serialized)

            # Leak checker on audit log & response JSON
            scan_dir = root / "scan"
            scan_dir.mkdir()
            (scan_dir / "score_v.json").write_text(serialized)
            (scan_dir / "budget.json").write_text(json.dumps(state.budget()))
            (scan_dir / "queries.jsonl").write_text(log_path.read_text())
            proc = subprocess.run(
                [
                    sys.executable,
                    str(AGENT_ROOT / "src" / "DataPipeline" / "002-CheckLeaks.py"),
                    "--no-view",
                    "--scan",
                    str(scan_dir),
                ],
                capture_output=True,
                text=True,
            )
            self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)

    def test_noise_caching_and_budget_exhaustion(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            shared = root / "shared"
            sub_dir = shared / "sub1"
            sub_dir.mkdir(parents=True)
            (sub_dir / "pipeline.py").write_text(self.starter_code)

            state = InRunVerifierState(
                sealed_dir=self.sealed_dir,
                shared_dir=shared,
                log_path=root / "queries.jsonl",
                max_queries=3,
                noise_sigma=0.02,
                verifier_cfg=self.verifier_cfg,
            )
            r1 = state.score_v(pipeline_code=self.starter_code, note="inline")
            r2 = state.score_v(shared_submission_dir="sub1", note="shared-identical")
            self.assertTrue(r1["ok"] and r2["ok"])
            self.assertFalse(r1["cached"])
            self.assertTrue(r2["cached"])
            self.assertEqual(r1["content_sha256"], r2["content_sha256"])
            # Identical resubmission must return identical noisy values (no noise averaging)
            self.assertEqual(r1["composite"], r2["composite"])
            self.assertEqual(r1["estates"], r2["estates"])
            # Still charges budget on each executed call
            self.assertEqual(r1["queries_used"], 1)
            self.assertEqual(r2["queries_used"], 2)

            # Different submission gets fresh noise
            alt_code = self.starter_code + "\n# slightly modified\n"
            r3 = state.score_v(pipeline_code=alt_code, note="alt")
            self.assertTrue(r3["ok"])
            self.assertFalse(r3["cached"])
            self.assertNotEqual(r1["content_sha256"], r3["content_sha256"])
            self.assertEqual(r3["queries_used"], 3)
            self.assertEqual(r3["queries_remaining"], 0)

            # 4th query must be rejected without running candidate code
            r4 = state.score_v(pipeline_code=alt_code + "# extra\n")
            self.assertFalse(r4["ok"])
            self.assertEqual(r4["error"], "BUDGET_EXHAUSTED")
            self.assertEqual(r4["queries_used"], 3)
            self.assertEqual(r4["queries_remaining"], 0)

            # verifier_budget does not consume budget
            bud = state.budget()
            self.assertEqual(bud["queries_used"], 3)
            self.assertEqual(len(bud["history"]), 3)

    def test_concurrent_budget_lock(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            shared = root / "shared"
            shared.mkdir()
            state = InRunVerifierState(
                sealed_dir=self.sealed_dir,
                shared_dir=shared,
                log_path=root / "queries.jsonl",
                max_queries=2,
                noise_sigma=0.005,
                verifier_cfg=self.verifier_cfg,
            )

            def _call(i: int) -> dict:
                return state.score_v(pipeline_code=self.starter_code + f"\n# run {i}\n")

            with ThreadPoolExecutor(max_workers=5) as pool:
                results = list(pool.map(_call, range(5)))

            ok_count = sum(1 for r in results if r["ok"])
            exhausted_count = sum(1 for r in results if r.get("error") == "BUDGET_EXHAUSTED")
            self.assertEqual(ok_count, 2)
            self.assertEqual(exhausted_count, 3)
            self.assertEqual(state.health()["queries_used"], 2)

    def test_path_jailing_symlinks_datasets_and_adversarial_read(self) -> None:
        l_df = pd.read_csv(self.sealed_dir / "L_labelled.csv", usecols=["estate", "Long", "Lat"])
        u_df = pd.read_csv(self.sealed_dir / "unlabelled.csv", usecols=["estate", "Long", "Lat"])
        all_coords = pd.concat([l_df, u_df], ignore_index=True)

        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            shared = root / "shared"
            shared.mkdir()
            sub_ok = shared / "sub_ok"
            sub_ok.mkdir()
            (sub_ok / "pipeline.py").write_text(self.starter_code)

            outside_dir = root / "outside"
            outside_dir.mkdir()
            (outside_dir / "pipeline.py").write_text(self.starter_code)

            # Symlinked dir and symlinked file inside shared/
            (shared / "link_dir").symlink_to(outside_dir)
            sub_sym_file = shared / "sub_sym_file"
            sub_sym_file.mkdir()
            (sub_sym_file / "pipeline.py").write_text(self.starter_code)
            (sub_sym_file / "leak.csv").symlink_to(self.sealed_dir / "labels_master.csv")

            # Valid dataset and forbidden-column dataset
            ds_ok = shared / "ds_ok" / "extra"
            ds_ok.mkdir(parents=True)
            ok_df = all_coords.copy()
            ok_df["extra_feat"] = 1.0
            ok_df.to_csv(ds_ok / "features.csv", index=False)

            ds_bad = shared / "ds_bad" / "extra"
            ds_bad.mkdir(parents=True)
            bad_df = ok_df.copy()
            bad_df["label"] = "Healthy"
            bad_df.to_csv(ds_bad / "features.csv", index=False)

            state = InRunVerifierState(
                sealed_dir=self.sealed_dir,
                shared_dir=shared,
                log_path=root / "queries.jsonl",
                max_queries=5,
                noise_sigma=0.0,
                verifier_cfg=self.verifier_cfg,
            )

            # Traversal / escape / symlink rejections do not consume budget
            for bad_sub in ("../outside", "/etc", str(outside_dir), "link_dir", "sub_sym_file"):
                r = state.score_v(shared_submission_dir=bad_sub)
                self.assertFalse(r["ok"], bad_sub)
                self.assertEqual(r["error"], "INVALID_PATH", bad_sub)

            r_bad_ds = state.score_v(shared_submission_dir="sub_ok", shared_datasets_dir="ds_bad")
            self.assertFalse(r_bad_ds["ok"])
            self.assertEqual(r_bad_ds["error"], "INVALID_DATASETS")

            r_big = state.score_v(pipeline_code="#" * (2 * 1024 * 1024 + 10))
            self.assertFalse(r_big["ok"])
            self.assertEqual(r_big["error"], "PAYLOAD_TOO_LARGE")

            # Budget still untouched after validation rejections
            self.assertEqual(state.health()["queries_used"], 0)

            # Valid shared submission + shared datasets succeeds
            r_good = state.score_v(
                shared_submission_dir=str(sub_ok),
                shared_datasets_dir="ds_ok",
                note="with-datasets",
            )
            self.assertTrue(r_good["ok"])
            self.assertTrue(r_good["datasets_audit"]["valid"])
            self.assertIn("extra", r_good["datasets_audit"]["datasets"])
            self.assertEqual(r_good["queries_used"], 1)

            # Adversarial candidate trying to read a protected path fails safely
            locked_secret = root / "secret_unreadable.txt"
            locked_secret.write_text("SECRET")
            os.chmod(locked_secret, 0o000)
            try:
                adv_code = (
                    "def fit_predict(l, u, ctx):\n"
                    f"    open({str(locked_secret)!r}).read()\n"
                )
                r_adv = state.score_v(pipeline_code=adv_code)
                self.assertFalse(r_adv["ok"])
                self.assertEqual(r_adv["status"], "failed")
                self.assertFalse(r_adv["submission_valid"])
                self.assertIsNone(r_adv["composite"])
                self.assertEqual(r_adv["queries_used"], 2)
            finally:
                os.chmod(locked_secret, 0o600)

    def test_custom_incumbent_dir_precomputation(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            inc_dir = root / "incumbent"
            inc_dir.mkdir()
            inc_code = (
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
            (inc_dir / "pipeline.py").write_text(inc_code)
            (inc_dir / "requirements.txt").write_text("")

            state = InRunVerifierState(
                sealed_dir=self.sealed_dir,
                shared_dir=root / "shared",
                log_path=root / "queries.jsonl",
                incumbent_dir=inc_dir,
                max_queries=3,
                noise_sigma=0.0,
                verifier_cfg=self.verifier_cfg,
            )
            res = state.score_v(pipeline_code=inc_code)
            self.assertTrue(res["ok"])
            self.assertAlmostEqual(res["composite"]["noisy_delta_vs_incumbent"], 0.0, places=12)

    def test_asgi_auth_health_and_mcp_streamable_http(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            shared = root / "shared"
            shared.mkdir()
            state = InRunVerifierState(
                sealed_dir=self.sealed_dir,
                shared_dir=shared,
                log_path=root / "queries.jsonl",
                max_queries=5,
                noise_sigma=0.0,
                verifier_cfg=self.verifier_cfg,
            )

            token_env = "TEST_AGENTRSI_VERIFIER_TOKEN"
            with mock.patch.dict(os.environ, {token_env: ""}, clear=False):
                with self.assertRaises(RuntimeError):
                    build_asgi_app(state, token_env=token_env, allow_no_auth=False)

            secret_token = "tok-super-secret-123"
            with mock.patch.dict(os.environ, {token_env: secret_token}, clear=False):
                app = build_asgi_app(state, token_env=token_env, allow_no_auth=False)
                with TestClient(app) as client:
                    # Unauthenticated /health succeeds with zero scores
                    hr = client.get("/health")
                    self.assertEqual(hr.status_code, 200)
                    hjson = hr.json()
                    self.assertTrue(hjson["ok"])
                    self.assertEqual(hjson["queries_used"], 0)
                    self.assertEqual(hjson["max_queries"], 5)

                    # Non-GET /health rejected
                    self.assertEqual(client.post("/health").status_code, 405)

                    # Missing or invalid Bearer token -> 401
                    r_no_auth = _mcp_post(client, "tools/list", {})
                    self.assertEqual(r_no_auth.status_code, 401)
                    r_bad_auth = _mcp_post(
                        client, "tools/list", {}, headers={"Authorization": "Bearer wrong-token"}
                    )
                    self.assertEqual(r_bad_auth.status_code, 401)
                    r_bad_scheme = _mcp_post(
                        client, "tools/list", {}, headers={"Authorization": f"Basic {secret_token}"}
                    )
                    self.assertEqual(r_bad_scheme.status_code, 401)

                    auth_hdr = {"Authorization": f"Bearer {secret_token}"}

                    # Initialize + tools/list
                    r_init = _mcp_post(
                        client,
                        "initialize",
                        {
                            "protocolVersion": "2025-03-26",
                            "capabilities": {},
                            "clientInfo": {"name": "test-client", "version": "1.0"},
                        },
                        req_id=1,
                        headers=auth_hdr,
                    )
                    self.assertEqual(r_init.status_code, 200)

                    r_list = _mcp_post(client, "tools/list", {}, req_id=2, headers=auth_hdr)
                    self.assertEqual(r_list.status_code, 200)
                    tool_names = {t["name"] for t in r_list.json()["result"]["tools"]}
                    self.assertEqual(tool_names, {"verifier_budget", "verifier_health", "score_v"})

                    # Call verifier_health via MCP
                    r_vh = _mcp_post(
                        client,
                        "tools/call",
                        {"name": "verifier_health", "arguments": {}},
                        req_id=3,
                        headers=auth_hdr,
                    )
                    self.assertEqual(r_vh.status_code, 200)
                    self.assertTrue(_parse_mcp_tool_json(r_vh)["ok"])

                    # Call score_v via MCP
                    r_sv = _mcp_post(
                        client,
                        "tools/call",
                        {
                            "name": "score_v",
                            "arguments": {"pipeline_code": self.starter_code, "note": "mcp-e2e"},
                        },
                        req_id=4,
                        headers=auth_hdr,
                    )
                    self.assertEqual(r_sv.status_code, 200)
                    sv_data = _parse_mcp_tool_json(r_sv)
                    self.assertTrue(sv_data["ok"])
                    self.assertEqual(sv_data["queries_used"], 1)
                    self.assertAlmostEqual(sv_data["composite"]["noisy_delta_vs_incumbent"], 0.0, places=12)

                    # Call verifier_budget via MCP
                    r_vb = _mcp_post(
                        client,
                        "tools/call",
                        {"name": "verifier_budget", "arguments": {}},
                        req_id=5,
                        headers=auth_hdr,
                    )
                    self.assertEqual(r_vb.status_code, 200)
                    vb_data = _parse_mcp_tool_json(r_vb)
                    self.assertEqual(vb_data["queries_used"], 1)
                    self.assertEqual(vb_data["queries_remaining"], 4)
                    self.assertEqual(len(vb_data["history"]), 1)

            # --allow-no-auth mode permits /mcp without token
            app_noauth = build_asgi_app(state, token_env=token_env, allow_no_auth=True)
            with TestClient(app_noauth) as client_na:
                r_na = _mcp_post(client_na, "tools/list", {}, req_id=1)
                self.assertEqual(r_na.status_code, 200)


if __name__ == "__main__":
    unittest.main()
