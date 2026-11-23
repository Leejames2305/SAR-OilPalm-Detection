"""Host-only static HTML dashboard generator and exact-match verifier for AgentRSI."""
from __future__ import annotations

import html
import json
import math
import os
import re
from datetime import datetime, timezone
from html.parser import HTMLParser
from pathlib import Path
from typing import Any

import yaml

from libbuilder.env_utils import AGENT_ROOT
from libbuilder.events import load_events
from libbuilder.lineage import default_lineage, load_lineage

ESTATES = ("AirHitam", "Palong", "Serting")
ESTATE_COLORS = {
    "AirHitam": "#0f766e",
    "Palong": "#b45309",
    "Serting": "#7c3aed",
}
EXTERNAL_ASSET_RE = re.compile(
    r"""(?:<script[^>]+src\s*=|<link[^>]+href\s*=|@import\s+|url\s*\()\s*['"]?\s*https?://""",
    re.IGNORECASE,
)


def _finite_float(val: Any) -> float | None:
    if val is None or isinstance(val, bool):
        return None
    try:
        f = float(val)
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None


def _fmt_signed(val: float | None, digits: int = 6) -> str:
    if val is None or not math.isfinite(val):
        return "—"
    return f"{val:+.{digits}f}"


def _fmt_plain(val: float | None, digits: int = 6) -> str:
    if val is None or not math.isfinite(val):
        return "—"
    return f"{val:.{digits}f}"


def _load_configs(agent_root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    split_p = agent_root / "config" / "split.yaml"
    ver_p = agent_root / "config" / "verifier.yaml"
    split_cfg = yaml.safe_load(split_p.read_text()) if split_p.exists() else {
        "epoch": 1,
        "sha256_of_assignment": "",
    }
    ver_cfg = yaml.safe_load(ver_p.read_text()) if ver_p.exists() else {
        "gate": {
            "t_attempt_budget": 20,
            "effect_floor": 0.03,
            "non_degradation_floor": -0.03,
        }
    }
    return split_cfg, ver_cfg


def _rel_link(target: Path, base_dir: Path) -> str:
    return Path(os.path.relpath(target.resolve(), base_dir.resolve())).as_posix()


def collect_dashboard_data(
    agent_root: Path = AGENT_ROOT,
    *,
    sessions_root: Path | None = None,
    lineage_path: Path | None = None,
    events_path: Path | None = None,
    dashboard_dir: Path | None = None,
) -> dict[str, Any]:
    """Read lineage, events, and verifier/collector session files into a canonical payload."""
    split_cfg, ver_cfg = _load_configs(agent_root)
    lp = lineage_path or (agent_root / "state" / "lineage.json")
    ep = events_path or (agent_root / "state" / "events.jsonl")
    sr = sessions_root or (agent_root / "sessions")
    ddir = dashboard_dir or (agent_root / "dashboard")

    lineage = load_lineage(lp, split_cfg, ver_cfg)
    events = load_events(ep)
    lineage_sessions = lineage.get("sessions") or {}

    gate_cfg = ver_cfg.get("gate") or {}
    default_floor = float(gate_cfg.get("effect_floor", 0.03))
    default_nondeg = float(gate_cfg.get("non_degradation_floor", -0.03))

    sessions: list[dict[str, Any]] = []
    if sr.exists():
        for sdir in sorted(p for p in sr.iterdir() if p.is_dir()):
            reward_p = sdir / "reward.json"
            if not reward_p.exists():
                continue
            sid = sdir.name
            reward = json.loads(reward_p.read_text())
            cert_p = sdir / "certification.json"
            cert = json.loads(cert_p.read_text()) if cert_p.exists() else {}
            vrep_p = sdir / "v_report.json"
            vrep = json.loads(vrep_p.read_text()) if vrep_p.exists() else {}
            meta_p = sdir / "meta.json"
            meta = json.loads(meta_p.read_text()) if meta_p.exists() else {}
            lin_s = lineage_sessions.get(sid) or {}

            epoch = int(
                lin_s.get("epoch")
                or cert.get("epoch")
                or meta.get("epoch")
                or lineage.get("epoch")
                or 1
            )
            dry_session = bool(
                meta.get("dry_session") if "dry_session" in meta else lin_s.get("dry_session", False)
            )
            promoted = bool(lin_s.get("promoted", False))
            submission_valid = bool(int(reward.get("submission_valid", 0)))
            gate_passed = bool(cert.get("gate_passed", False))

            comp_cert = cert.get("composite") or {}
            gate_cert = cert.get("gate") or {}
            t_comp_delta = _finite_float(
                comp_cert.get("comp_delta")
                if "comp_delta" in comp_cert
                else gate_cert.get("comp_delta")
            )
            t_ci_low = _finite_float(
                comp_cert.get("ci_low") if "ci_low" in comp_cert else gate_cert.get("comp_ci_low")
            )
            t_ci_high = _finite_float(comp_cert.get("ci_high"))
            effect_floor = _finite_float(gate_cert.get("effect_floor")) or default_floor
            nondeg_floor = _finite_float(gate_cert.get("non_degradation_floor"))
            if nondeg_floor is None:
                nondeg_floor = default_nondeg
            worst_estate = gate_cert.get("worst_estate")
            worst_estate_delta = _finite_float(gate_cert.get("worst_estate_delta"))
            failed_reasons = list(gate_cert.get("failed_reasons") or [])

            per_estate_v: dict[str, dict[str, Any]] = {}
            v_estates = vrep.get("estates") or {}
            t_estates = cert.get("estates") or {}
            per_estate_t: dict[str, dict[str, Any]] = {}
            for est in ESTATES:
                ve = v_estates.get(est) or {}
                d_inc = ve.get("delta_vs_incumbent") or {}
                per_estate_v[est] = {
                    "v_lift": _finite_float(reward.get(f"v_lift_{est}")),
                    "pr_auc": _finite_float(ve.get("pr_auc")),
                    "prevalence": _finite_float(ve.get("prevalence")),
                    "recall_at_25": _finite_float(ve.get("recall_at_25")),
                    "enrichment_top10": _finite_float(ve.get("enrichment_top10")),
                    "delta_vs_incumbent": _finite_float(d_inc.get("delta")),
                    "ci_low": _finite_float(d_inc.get("ci_low")),
                    "ci_high": _finite_float(d_inc.get("ci_high")),
                }
                te = t_estates.get(est) or {}
                td_inc = te.get("delta_vs_incumbent") or {}
                cand_m = te.get("cand_metrics") or {}
                per_estate_t[est] = {
                    "pr_auc_lift": _finite_float(cand_m.get("pr_auc_lift")),
                    "delta_vs_incumbent": _finite_float(td_inc.get("delta")),
                    "ci_low": _finite_float(td_inc.get("ci_low")),
                    "ci_high": _finite_float(td_inc.get("ci_high")),
                    "recall_at_25": _finite_float(cand_m.get("recall_at_25")),
                }

            report_p = sdir / "REPORT.md"
            handoff_p = sdir / "HANDOFF.md"
            pipeline_p = sdir / "submission" / "pipeline.py"
            links: dict[str, str] = {}
            if report_p.exists():
                links["report"] = _rel_link(report_p, ddir)
            if handoff_p.exists():
                links["handoff"] = _rel_link(handoff_p, ddir)
            if vrep_p.exists():
                links["v_report"] = _rel_link(vrep_p, ddir)
            if cert_p.exists():
                links["certification"] = _rel_link(cert_p, ddir)
            if pipeline_p.exists():
                links["pipeline"] = _rel_link(pipeline_p, ddir)

            if not submission_valid:
                verdict = "INVALID"
            elif promoted:
                verdict = "PROMOTED"
            elif dry_session:
                verdict = "DRY"
            elif gate_passed:
                verdict = "PASS"
            else:
                verdict = "FAIL"

            sort_ts = str(meta.get("started_at") or lin_s.get("collected_at") or meta.get("job_id") or "")
            sessions.append({
                "session_id": sid,
                "epoch": epoch,
                "sort_ts": sort_ts,
                "job_id": meta.get("job_id") or lin_s.get("job_id") or "",
                "parent_id": meta.get("parent_id") or lin_s.get("parent_id") or "",
                "agent": meta.get("agent") or "",
                "model": meta.get("model") or "",
                "started_at": meta.get("started_at") or "",
                "finished_at": meta.get("finished_at") or "",
                "dry_session": dry_session,
                "submission_valid": submission_valid,
                "status": str(vrep.get("status") or cert.get("status") or ("ok" if submission_valid else "invalid")),
                "error": vrep.get("error") or cert.get("error"),
                "gate_passed": gate_passed,
                "promoted": promoted,
                "verdict": verdict,
                "v_score": _finite_float(reward.get("score")),
                "v_lift_AirHitam": _finite_float(reward.get("v_lift_AirHitam")),
                "v_lift_Palong": _finite_float(reward.get("v_lift_Palong")),
                "v_lift_Serting": _finite_float(reward.get("v_lift_Serting")),
                "v_delta_vs_incumbent": _finite_float(reward.get("v_delta_vs_incumbent")),
                "v_delta_vs_coords": _finite_float(reward.get("v_delta_vs_coords")),
                "runtime_sec": _finite_float(reward.get("runtime_sec")),
                "t_comp_delta": t_comp_delta,
                "t_ci_low": t_ci_low,
                "t_ci_high": t_ci_high,
                "effect_floor": effect_floor,
                "non_degradation_floor": nondeg_floor,
                "worst_estate": worst_estate,
                "worst_estate_delta": worst_estate_delta,
                "failed_reasons": failed_reasons,
                "cost_usd": _finite_float(meta.get("cost_usd") if "cost_usd" in meta else lin_s.get("cost_usd")),
                "n_input_tokens": meta.get("n_input_tokens"),
                "n_cache_tokens": meta.get("n_cache_tokens"),
                "n_output_tokens": meta.get("n_output_tokens"),
                "smells": list(meta.get("smells") or []),
                "handoff_missing_sections": list(meta.get("handoff_missing_sections") or []),
                "per_estate_v": per_estate_v,
                "per_estate_t": per_estate_t,
                "links": links,
                # Strictly read-only context text (never parsed for numbers).
                "report_md": report_p.read_text(errors="replace") if report_p.exists() else "",
                "handoff_md": handoff_p.read_text(errors="replace") if handoff_p.exists() else "",
            })

    sessions.sort(key=lambda s: (int(s["epoch"]), str(s["sort_ts"]), str(s["session_id"])))

    inc = lineage.get("incumbent") or {}
    daily_spend = {str(k): float(v) for k, v in sorted((lineage.get("daily_spend_usd") or {}).items())}
    latest_day = max(daily_spend.keys()) if daily_spend else ""
    total_spend = round(sum(daily_spend.values()), 6)

    seen_epochs = {int(e["epoch"]) for e in (lineage.get("epochs") or []) if "epoch" in e}
    seen_epochs.update(int(s["epoch"]) for s in sessions)
    seen_epochs.update(int(ev["epoch"]) for ev in events if "epoch" in ev)

    header = {
        "epoch": int(lineage.get("epoch", 1)),
        "split_sha256": str(lineage.get("split_sha256", "")),
        "t_attempts_used": int(lineage.get("t_attempts_used", 0)),
        "t_attempt_budget": int(lineage.get("t_attempt_budget", 20)),
        "kill_switch": bool(lineage.get("kill_switch", False)),
        "incumbent": {
            "id": str(inc.get("id", "starter")),
            "source_dir": str(inc.get("source_dir", "starter")),
            "epoch": int(inc.get("epoch", lineage.get("epoch", 1))),
            "v_score": _finite_float(inc.get("v_score")),
            "per_estate_v_lift": {
                est: _finite_float((inc.get("per_estate_v_lift") or {}).get(est))
                for est in ESTATES
            },
        },
        "daily_spend_usd": daily_spend,
        "latest_spend_day": latest_day,
        "latest_spend_usd": daily_spend.get(latest_day, 0.0) if latest_day else 0.0,
        "total_spend_usd": total_spend,
        "session_count": len(sessions),
        "epochs": sorted(seen_epochs) or [int(lineage.get("epoch", 1))],
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }

    indexed_events = []
    for idx, ev in enumerate(events):
        copy = dict(ev)
        copy["event_idx"] = idx
        indexed_events.append(copy)

    return {
        "header": header,
        "sessions": sessions,
        "events": indexed_events,
    }


def _compute_bounds(values: list[float], defaults: tuple[float, float]) -> tuple[float, float]:
    lo, hi = defaults
    finite = [v for v in values if v is not None and math.isfinite(v)]
    if finite:
        lo = min(lo, min(finite))
        hi = max(hi, max(finite))
    span = hi - lo
    if span < 1e-6:
        lo -= 0.01
        hi += 0.01
        span = hi - lo
    pad = span * 0.14
    return lo - pad, hi + pad


def _epoch_spans(sessions: list[dict[str, Any]]) -> list[tuple[int, int, int]]:
    """Return contiguous `(epoch, start_idx, end_idx)` runs in session order."""
    if not sessions:
        return []
    spans: list[tuple[int, int, int]] = []
    cur_ep = int(sessions[0]["epoch"])
    start = 0
    for i, s in enumerate(sessions[1:], start=1):
        ep = int(s["epoch"])
        if ep != cur_ep:
            spans.append((cur_ep, start, i - 1))
            cur_ep = ep
            start = i
    spans.append((cur_ep, start, len(sessions) - 1))
    return spans


def _build_chart_geometry(sessions: list[dict[str, Any]]) -> dict[str, Any]:
    w, h = 760, 260
    pl, pr, pt, pb = 62, 18, 28, 42
    pw = w - pl - pr
    ph = h - pt - pb
    n = len(sessions)
    xs: list[float] = []
    if n == 1:
        xs = [pl + pw / 2.0]
    elif n > 1:
        step = pw / max(1, n - 1)
        margin = min(36.0, pw * 0.12)
        if n == 2:
            xs = [pl + pw * 0.25, pl + pw * 0.75]
        else:
            usable = pw - 2 * margin
            step = usable / (n - 1)
            xs = [pl + margin + i * step for i in range(n)]
    spans = _epoch_spans(sessions)
    return {"w": w, "h": h, "pl": pl, "pr": pr, "pt": pt, "pb": pb, "pw": pw, "ph": ph, "xs": xs, "spans": spans}


def _render_epoch_bands_and_axes(
    geom: dict[str, Any],
    sessions: list[dict[str, Any]],
    y_min: float,
    y_max: float,
    zero_ref: bool = True,
) -> tuple[list[str], Any]:
    pl, pt, pw, ph = geom["pl"], geom["pt"], geom["pw"], geom["ph"]
    xs = geom["xs"]
    spans = geom["spans"]

    def y_px(val: float) -> float:
        frac = (val - y_min) / (y_max - y_min)
        return pt + ph * (1.0 - frac)

    parts: list[str] = []
    # Epoch bands + vertical dividers
    for span_idx, (ep, s_idx, e_idx) in enumerate(spans):
        if s_idx == 0:
            x0 = float(pl)
        else:
            x0 = (xs[s_idx - 1] + xs[s_idx]) / 2.0
        if e_idx == len(xs) - 1:
            x1 = float(pl + pw)
        else:
            x1 = (xs[e_idx] + xs[e_idx + 1]) / 2.0
        cls = "epoch-band epoch-even" if span_idx % 2 == 0 else "epoch-band epoch-odd"
        parts.append(
            f'<rect class="{cls}" data-epoch="{ep}" x="{x0:.2f}" y="{pt}" '
            f'width="{max(1.0, x1 - x0):.2f}" height="{ph}" />'
        )
        parts.append(
            f'<text class="epoch-label" x="{(x0 + x1) / 2.0:.2f}" y="{pt - 8}" '
            f'text-anchor="middle">Epoch {ep}</text>'
        )
        if span_idx > 0:
            prev_ep = spans[span_idx - 1][0]
            parts.append(
                f'<line class="epoch-boundary" data-from-epoch="{prev_ep}" data-to-epoch="{ep}" '
                f'x1="{x0:.2f}" y1="{pt}" x2="{x0:.2f}" y2="{pt + ph}" />'
            )

    # Grid ticks (4 ticks)
    for k in range(5):
        val = y_min + (y_max - y_min) * (k / 4.0)
        yp = y_px(val)
        parts.append(f'<line class="grid-line" x1="{pl}" y1="{yp:.2f}" x2="{pl + pw}" y2="{yp:.2f}" />')
        parts.append(
            f'<text class="axis-text" x="{pl - 8}" y="{yp + 3:.2f}" text-anchor="end">{val:+.3f}</text>'
        )

    if zero_ref and y_min <= 0.0 <= y_max:
        yz = y_px(0.0)
        parts.append(
            f'<line class="ref-zero" data-ref="zero" x1="{pl}" y1="{yz:.2f}" x2="{pl + pw}" y2="{yz:.2f}" />'
        )

    # X axis labels
    for i, sess in enumerate(sessions):
        sid = html.escape(str(sess["session_id"]))
        xp = xs[i]
        parts.append(
            f'<text class="axis-text x-tick" x="{xp:.2f}" y="{pt + ph + 16}" text-anchor="middle">{sid}</text>'
        )

    parts.append(
        f'<rect class="plot-border" x="{pl}" y="{pt}" width="{pw}" height="{ph}" fill="none" />'
    )
    return parts, y_px


def _render_v_chart_svg(sessions: list[dict[str, Any]], header: dict[str, Any]) -> str:
    geom = _build_chart_geometry(sessions)
    w, h = geom["w"], geom["h"]
    pl, pt, pw, ph = geom["pl"], geom["pt"], geom["pw"], geom["ph"]
    xs = geom["xs"]
    spans = geom["spans"]

    if not sessions:
        return (
            f'<svg id="chart-v" viewBox="0 0 {w} {h}" role="img" aria-label="Validation V search signal chart">'
            f'<rect class="plot-border" x="{pl}" y="{pt}" width="{pw}" height="{ph}" fill="#fafafa" />'
            f'<text class="empty-msg" x="{w / 2:.1f}" y="{h / 2:.1f}" text-anchor="middle">'
            f'No collected sessions yet (V search signal)</text></svg>'
        )

    vals: list[float] = []
    inc_v = _finite_float((header.get("incumbent") or {}).get("v_score"))
    if inc_v is not None:
        vals.append(inc_v)
    for s in sessions:
        for key in ("v_score", "v_lift_AirHitam", "v_lift_Palong", "v_lift_Serting"):
            v = _finite_float(s.get(key))
            if v is not None:
                vals.append(v)
    y_min, y_max = _compute_bounds(vals, (0.0, 0.08))
    parts, y_px = _render_epoch_bands_and_axes(geom, sessions, y_min, y_max, zero_ref=True)

    # Incumbent horizontal reference line
    if inc_v is not None and y_min <= inc_v <= y_max:
        yi = y_px(inc_v)
        inc_id = html.escape(str((header.get("incumbent") or {}).get("id", "incumbent")))
        parts.append(
            f'<line class="ref-incumbent" data-ref="incumbent-v" x1="{pl}" y1="{yi:.2f}" '
            f'x2="{pl + pw}" y2="{yi:.2f}" />'
        )
        parts.append(
            f'<text class="ref-label-blue" x="{pl + pw - 4}" y="{yi - 5:.2f}" text-anchor="end">'
            f'Incumbent ({inc_id}): {inc_v:+.4f}</text>'
        )

    # Per-estate series (polylines per epoch so lines never cross epoch boundaries)
    for est in ESTATES:
        key = f"v_lift_{est}"
        color = ESTATE_COLORS[est]
        for ep, s_idx, e_idx in spans:
            pts_ep = []
            for i in range(s_idx, e_idx + 1):
                v = _finite_float(sessions[i].get(key))
                if v is not None:
                    pts_ep.append((i, xs[i], y_px(v), v))
            if len(pts_ep) >= 2:
                coord_str = " ".join(f"{x:.2f},{y:.2f}" for _, x, y, _ in pts_ep)
                parts.append(
                    f'<polyline class="series-line estate-series" data-series="{key}" data-epoch="{ep}" '
                    f'fill="none" stroke="{color}" stroke-width="1.5" stroke-opacity="0.82" points="{coord_str}" />'
                )
            for _, x, y, v in pts_ep:
                sid = html.escape(str(sessions[_]["session_id"]))
                parts.append(
                    f'<circle class="pt pt-v estate-series" data-series="{key}" data-session="{sid}" data-epoch="{ep}" '
                    f'cx="{x:.2f}" cy="{y:.2f}" r="3" fill="{color}" fill-opacity="0.85">'
                    f'<title>{sid} · {est} V lift: {v:+.6f}</title></circle>'
                )

    # Composite V series (polylines broken cleanly per epoch)
    for ep, s_idx, e_idx in spans:
        pts_comp = []
        for i in range(s_idx, e_idx + 1):
            v = _finite_float(sessions[i].get("v_score"))
            if v is not None:
                pts_comp.append((i, xs[i], y_px(v), v))
        if len(pts_comp) >= 2:
            coord_str = " ".join(f"{x:.2f},{y:.2f}" for _, x, y, _ in pts_comp)
            parts.append(
                f'<polyline class="series-line" data-series="v_score" data-epoch="{ep}" '
                f'fill="none" stroke="#1d4ed8" stroke-width="2.4" points="{coord_str}" />'
            )
        for i, x, y, v in pts_comp:
            s = sessions[i]
            sid = html.escape(str(s["session_id"]))
            is_dry = bool(s.get("dry_session"))
            fill = "#ffffff" if is_dry else "#1d4ed8"
            tip = f"{sid} (Epoch {ep}{', dry' if is_dry else ''}): V composite = {v:+.6f}"
            parts.append(
                f'<circle class="pt pt-v clickable" data-chart="v" data-series="v_score" '
                f'data-session="{sid}" data-epoch="{ep}" cx="{x:.2f}" cy="{y:.2f}" r="5" '
                f'fill="{fill}" stroke="#1d4ed8" stroke-width="2"><title>{tip}</title></circle>'
            )

    return (
        f'<svg id="chart-v" viewBox="0 0 {w} {h}" role="img" '
        f'aria-label="Validation V search signal optimistic">' + "".join(parts) + "</svg>"
    )


def _render_t_chart_svg(sessions: list[dict[str, Any]], header: dict[str, Any]) -> str:
    geom = _build_chart_geometry(sessions)
    w, h = geom["w"], geom["h"]
    pl, pt, pw, ph = geom["pl"], geom["pt"], geom["pw"], geom["ph"]
    xs = geom["xs"]
    spans = geom["spans"]

    if not sessions:
        return (
            f'<svg id="chart-t" viewBox="0 0 {w} {h}" role="img" aria-label="Certified T delta PR-AUC chart">'
            f'<rect class="plot-border" x="{pl}" y="{pt}" width="{pw}" height="{ph}" fill="#fafafa" />'
            f'<text class="empty-msg" x="{w / 2:.1f}" y="{h / 2:.1f}" text-anchor="middle">'
            f'No collected sessions yet (certified T delta PR-AUC)</text></svg>'
        )

    effect_floor = float(sessions[-1].get("effect_floor") or 0.03)
    nondeg_floor = float(sessions[-1].get("non_degradation_floor") or -0.03)

    vals: list[float] = [effect_floor, 0.0, nondeg_floor]
    for s in sessions:
        for k in ("t_comp_delta", "t_ci_low", "t_ci_high"):
            v = _finite_float(s.get(k))
            if v is not None:
                vals.append(v)
    y_min, y_max = _compute_bounds(vals, (nondeg_floor - 0.01, effect_floor + 0.015))
    parts, y_px = _render_epoch_bands_and_axes(geom, sessions, y_min, y_max, zero_ref=True)

    # +0.03 certification effect floor & -0.03 non-degradation guide
    yf = y_px(effect_floor)
    parts.append(
        f'<line class="ref-floor" data-ref="effect-floor" x1="{pl}" y1="{yf:.2f}" '
        f'x2="{pl + pw}" y2="{yf:.2f}" />'
    )
    parts.append(
        f'<text class="ref-label-green" x="{pl + pw - 4}" y="{yf - 5:.2f}" text-anchor="end">'
        f'+{effect_floor:.2f} effect floor (certify)</text>'
    )

    yn = y_px(nondeg_floor)
    parts.append(
        f'<line class="ref-nondeg" data-ref="non-degradation" x1="{pl}" y1="{yn:.2f}" '
        f'x2="{pl + pw}" y2="{yn:.2f}" />'
    )
    parts.append(
        f'<text class="ref-label-red" x="{pl + pw - 4}" y="{yn + 12:.2f}" text-anchor="end">'
        f'{nondeg_floor:+.2f} estate floor</text>'
    )

    # T series broken across epochs + CI bars
    for ep, s_idx, e_idx in spans:
        pts_t = []
        for i in range(s_idx, e_idx + 1):
            v = _finite_float(sessions[i].get("t_comp_delta"))
            if v is not None:
                pts_t.append((i, xs[i], y_px(v), v))
        if len(pts_t) >= 2:
            coord_str = " ".join(f"{x:.2f},{y:.2f}" for _, x, y, _ in pts_t)
            parts.append(
                f'<polyline class="series-line" data-series="t_comp_delta" data-epoch="{ep}" '
                f'fill="none" stroke="#334155" stroke-width="2" points="{coord_str}" />'
            )

        for i, x, y, v in pts_t:
            s = sessions[i]
            sid = html.escape(str(s["session_id"]))
            ci_lo = _finite_float(s.get("t_ci_low"))
            ci_hi = _finite_float(s.get("t_ci_high"))
            is_dry = bool(s.get("dry_session"))
            stroke_col = "#64748b" if is_dry else ("#15803d" if s.get("gate_passed") else "#b91c1c")
            if ci_lo is not None and ci_hi is not None:
                ylo, yhi = y_px(ci_lo), y_px(ci_hi)
                parts.append(
                    f'<line class="ci-bar" data-session="{sid}" data-epoch="{ep}" '
                    f'x1="{x:.2f}" y1="{yhi:.2f}" x2="{x:.2f}" y2="{ylo:.2f}" '
                    f'stroke="{stroke_col}" stroke-width="2" />'
                )
                parts.append(
                    f'<line class="ci-cap" data-session="{sid}" x1="{x - 5:.2f}" y1="{yhi:.2f}" '
                    f'x2="{x + 5:.2f}" y2="{yhi:.2f}" stroke="{stroke_col}" stroke-width="1.5" />'
                )
                parts.append(
                    f'<line class="ci-cap" data-session="{sid}" x1="{x - 5:.2f}" y1="{ylo:.2f}" '
                    f'x2="{x + 5:.2f}" y2="{ylo:.2f}" stroke="{stroke_col}" stroke-width="1.5" />'
                )
            fill = "#ffffff" if is_dry else stroke_col
            ci_desc = (
                f" [{ci_lo:+.6f}, {ci_hi:+.6f}]"
                if (ci_lo is not None and ci_hi is not None)
                else (f" [ci_low {ci_lo:+.6f}]" if ci_lo is not None else "")
            )
            tip = f"{sid} ({s['verdict']}): T dPR-AUC = {v:+.6f}{ci_desc}"
            parts.append(
                f'<circle class="pt pt-t clickable" data-chart="t" data-series="t_comp_delta" '
                f'data-session="{sid}" data-epoch="{ep}" cx="{x:.2f}" cy="{y:.2f}" r="5" '
                f'fill="{fill}" stroke="{stroke_col}" stroke-width="2"><title>{tip}</title></circle>'
            )

    return (
        f'<svg id="chart-t" viewBox="0 0 {w} {h}" role="img" '
        f'aria-label="Certified T delta PR-AUC vs incumbent host-only gate">' + "".join(parts) + "</svg>"
    )


def _render_sessions_rows(sessions: list[dict[str, Any]]) -> str:
    if not sessions:
        return (
            '<tr class="empty-row"><td colspan="15">No collected sessions found under '
            '<code>AgentRSI/sessions/</code>.</td></tr>'
        )
    rows: list[str] = []
    for s in sessions:
        sid = html.escape(str(s["session_id"]))
        verdict = html.escape(str(s["verdict"]))
        badge_cls = {
            "PROMOTED": "badge badge-promoted",
            "PASS": "badge badge-pass",
            "FAIL": "badge badge-fail",
            "DRY": "badge badge-dry",
            "INVALID": "badge badge-invalid",
        }.get(str(s["verdict"]), "badge")
        ci_txt = (
            f"[{_fmt_signed(s['t_ci_low'])}, {_fmt_signed(s['t_ci_high'])}]"
            if s.get("t_ci_low") is not None and s.get("t_ci_high") is not None
            else _fmt_signed(s.get("t_ci_low"))
        )
        worst_txt = (
            f"{html.escape(str(s['worst_estate']))}: {_fmt_signed(s['worst_estate_delta'])}"
            if s.get("worst_estate") and s.get("worst_estate_delta") is not None
            else _fmt_signed(s.get("worst_estate_delta"))
        )
        flags: list[str] = []
        if s.get("smells"):
            flags.append(f"{len(s['smells'])} smell(s)")
        if s.get("handoff_missing_sections"):
            flags.append(f"{len(s['handoff_missing_sections'])} handoff gap(s)")
        flags_txt = ", ".join(flags) if flags else "ok"

        link_bits: list[str] = []
        links = s.get("links") or {}
        if "report" in links:
            link_bits.append(f'<a href="{html.escape(links["report"])}">REPORT</a>')
        if "handoff" in links:
            link_bits.append(f'<a href="{html.escape(links["handoff"])}">HANDOFF</a>')
        links_html = " · ".join(link_bits) if link_bits else "—"

        agent_model = html.escape(
            " / ".join(x for x in (s.get("agent"), s.get("model")) if x) or "—"
        )
        cost_txt = f"${s['cost_usd']:.4f}" if s.get("cost_usd") is not None else "—"
        rt_txt = f"{s['runtime_sec']:.3f}s" if s.get("runtime_sec") is not None else "—"

        rows.append(
            f'<tr class="session-row" data-session="{sid}" data-epoch="{int(s["epoch"])}">'
            f'<td><strong>{sid}</strong></td>'
            f'<td class="num">{int(s["epoch"])}</td>'
            f'<td>{html.escape(str(s.get("parent_id") or "—"))}</td>'
            f'<td>{agent_model}</td>'
            f'<td><span class="{badge_cls}">{verdict}</span></td>'
            f'<td class="num" data-metric="v_score">{_fmt_signed(s["v_score"])}</td>'
            f'<td class="num" data-metric="v_lift_AirHitam">{_fmt_signed(s["v_lift_AirHitam"])}</td>'
            f'<td class="num" data-metric="v_lift_Palong">{_fmt_signed(s["v_lift_Palong"])}</td>'
            f'<td class="num" data-metric="v_lift_Serting">{_fmt_signed(s["v_lift_Serting"])}</td>'
            f'<td class="num" data-metric="v_delta_vs_incumbent">{_fmt_signed(s["v_delta_vs_incumbent"])}</td>'
            f'<td class="num" data-metric="t_comp_delta">{_fmt_signed(s["t_comp_delta"])}</td>'
            f'<td class="num" data-metric="t_ci">{ci_txt}</td>'
            f'<td class="num" data-metric="worst_estate">{worst_txt}</td>'
            f'<td class="num">{cost_txt}</td>'
            f'<td class="num" data-metric="runtime_sec">{rt_txt}</td>'
            f'<td>{html.escape(flags_txt)}</td>'
            f'<td>{links_html}</td>'
            f'</tr>'
        )
    return "\n".join(rows)


def _render_event_rows(events: list[dict[str, Any]]) -> str:
    if not events:
        return '<tr><td colspan="5">No events logged in <code>state/events.jsonl</code>.</td></tr>'
    rows: list[str] = []
    for ev in reversed(events):
        idx = int(ev.get("event_idx", 0))
        sid = html.escape(str(ev.get("session", "")))
        ep = int(ev.get("epoch", 1))
        etype = html.escape(str(ev.get("type", "")))
        ts = html.escape(str(ev.get("ts", "")))
        summary = html.escape(str(ev.get("summary", "")))
        vals = ev.get("values")
        tip = summary + (f" | {json.dumps(vals)}" if vals else "")
        rows.append(
            f'<tr class="event-row" data-event-idx="{idx}" data-session="{sid}" '
            f'data-epoch="{ep}" title="{html.escape(tip)}">'
            f'<td class="mono">{ts}</td>'
            f'<td><span class="ev-type ev-{etype}">{etype}</span></td>'
            f'<td class="num">{ep}</td>'
            f'<td><strong>{sid}</strong></td>'
            f'<td>{summary}</td>'
            f'</tr>'
        )
    return "\n".join(rows)


CSS_STYLES = """
:root {
  --bg: #f8fafc;
  --card: #ffffff;
  --border: #cbd5e1;
  --text: #0f172a;
  --muted: #475569;
  --blue: #1d4ed8;
  --green: #15803d;
  --red: #b91c1c;
  --amber: #b45309;
}
* { box-sizing: border-box; }
body {
  margin: 0;
  padding: 18px 22px 32px;
  font-family: system-ui, -apple-system, Segoe UI, Roboto, sans-serif;
  font-size: 13px;
  line-height: 1.45;
  color: var(--text);
  background: var(--bg);
}
h1, h2, h3 { margin: 0 0 8px; font-weight: 600; }
h1 { font-size: 18px; }
h2 { font-size: 14px; }
h3 { font-size: 13px; }
a { color: var(--blue); text-decoration: none; }
a:hover { text-decoration: underline; }
code, pre, .mono, .num { font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; }
.topbar {
  display: flex;
  flex-wrap: wrap;
  justify-content: space-between;
  align-items: baseline;
  gap: 8px;
  margin-bottom: 12px;
}
.sub { color: var(--muted); font-size: 12px; }
.summary-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
  gap: 10px;
  margin-bottom: 14px;
}
.stat-card {
  background: var(--card);
  border: 1px solid var(--border);
  padding: 9px 11px;
}
.stat-label { font-size: 11px; text-transform: uppercase; color: var(--muted); letter-spacing: 0.03em; }
.stat-value { font-size: 15px; font-weight: 600; margin-top: 2px; }
.stat-sub { font-size: 11px; color: var(--muted); margin-top: 2px; }
.warn { color: var(--amber); }
.danger { color: var(--red); }
.ok { color: var(--green); }
.charts-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(460px, 1fr));
  gap: 14px;
  margin-bottom: 14px;
}
.card {
  background: var(--card);
  border: 1px solid var(--border);
  padding: 12px 14px;
  margin-bottom: 14px;
}
.charts-grid .card { margin-bottom: 0; }
.card-head {
  display: flex;
  flex-wrap: wrap;
  justify-content: space-between;
  align-items: baseline;
  gap: 8px;
  margin-bottom: 8px;
}
.tag-v { font-size: 11px; background: #dbeafe; color: #1e40af; padding: 1px 6px; font-weight: 600; }
.tag-t { font-size: 11px; background: #dcfce7; color: #166534; padding: 1px 6px; font-weight: 600; }
.controls { font-size: 12px; color: var(--muted); display: flex; flex-wrap: wrap; gap: 10px; align-items: center; }
.controls label { cursor: pointer; display: inline-flex; gap: 4px; align-items: center; }
svg { width: 100%; height: auto; display: block; }
.epoch-even { fill: #f8fafc; }
.epoch-odd { fill: #f1f5f9; }
.epoch-boundary { stroke: #64748b; stroke-width: 1.2; stroke-dasharray: 4 3; }
.epoch-label { font-size: 11px; fill: #334155; font-weight: 600; }
.grid-line { stroke: #e2e8f0; stroke-width: 1; }
.plot-border { stroke: #94a3b8; stroke-width: 1; }
.axis-text { font-size: 10px; fill: #475569; font-family: ui-monospace, monospace; }
.ref-zero { stroke: #94a3b8; stroke-width: 1.1; }
.ref-incumbent { stroke: #1d4ed8; stroke-width: 1.2; stroke-dasharray: 5 3; }
.ref-floor { stroke: #15803d; stroke-width: 1.4; stroke-dasharray: 5 3; }
.ref-nondeg { stroke: #b91c1c; stroke-width: 1.1; stroke-dasharray: 3 3; }
.ref-label-blue { font-size: 10px; fill: #1d4ed8; }
.ref-label-green { font-size: 10px; fill: #15803d; font-weight: 600; }
.ref-label-red { font-size: 10px; fill: #b91c1c; }
.empty-msg { font-size: 12px; fill: #64748b; }
.clickable { cursor: pointer; }
.table-wrap { overflow-x: auto; }
table { width: 100%; border-collapse: collapse; font-size: 12px; }
th, td { border-bottom: 1px solid #e2e8f0; padding: 6px 8px; text-align: left; white-space: nowrap; }
th { background: #f1f5f9; font-weight: 600; color: #334155; border-top: 1px solid var(--border); }
td.num { text-align: right; }
tr.session-row, tr.event-row { cursor: pointer; }
tr.session-row:hover, tr.event-row:hover { background: #eff6ff; }
tr.selected { background: #dbeafe !important; }
.badge, .ev-type {
  display: inline-block;
  padding: 1px 5px;
  font-size: 11px;
  font-weight: 600;
  border: 1px solid #cbd5e1;
  background: #f1f5f9;
}
.badge-promoted, .ev-promoted, .ev-certification_pass { background: #dcfce7; border-color: #86efac; color: #166534; }
.badge-pass { background: #e0f2fe; border-color: #7dd3fc; color: #075985; }
.badge-fail, .badge-invalid, .ev-certification_fail, .ev-run_error, .ev-spend_cap { background: #fee2e2; border-color: #fca5a5; color: #991b1b; }
.badge-dry { background: #f1f5f9; color: #334155; }
.ev-review_flag, .ev-budget_alert { background: #fef3c7; border-color: #fcd34d; color: #92400e; }
.bottom-grid {
  display: grid;
  grid-template-columns: 1fr 1.25fr;
  gap: 14px;
  align-items: start;
}
@media (max-width: 980px) { .bottom-grid { grid-template-columns: 1fr; } }
.events-wrap { max-height: 460px; overflow: auto; }
.events-wrap td:last-child { white-space: normal; min-width: 220px; }
.tab-bar { display: flex; gap: 6px; margin: 8px 0; }
.tab-btn {
  border: 1px solid var(--border);
  background: #f1f5f9;
  padding: 4px 10px;
  cursor: pointer;
  font-size: 12px;
}
.tab-btn.active { background: var(--blue); color: #fff; border-color: var(--blue); }
pre.md-view {
  background: #f8fafc;
  border: 1px solid #e2e8f0;
  padding: 10px;
  max-height: 360px;
  overflow: auto;
  white-space: pre-wrap;
  word-break: break-word;
  margin: 0;
  font-size: 12px;
}
.event-detail {
  background: #f8fafc;
  border: 1px solid var(--border);
  padding: 8px 10px;
  margin-bottom: 10px;
  font-size: 12px;
}
.tooltip {
  position: fixed;
  display: none;
  pointer-events: none;
  background: #0f172a;
  color: #f8fafc;
  padding: 5px 8px;
  font-size: 11px;
  max-width: 360px;
  z-index: 50;
  border: 1px solid #334155;
}
"""

INLINE_JS = """
(function () {
  var raw = document.getElementById("dashboard-data").textContent;
  var DATA = JSON.parse(raw);
  var sessionsById = {};
  DATA.sessions.forEach(function (s) { sessionsById[s.session_id] = s; });
  var eventsByIdx = {};
  DATA.events.forEach(function (e) { eventsByIdx[String(e.event_idx)] = e; });

  var activeTab = "handoff";
  var currentSessionId = DATA.sessions.length ? DATA.sessions[DATA.sessions.length - 1].session_id : null;
  var currentEventIdx = null;
  var tooltip = document.getElementById("tooltip");

  function fmt(val, signed) {
    if (val === null || val === undefined || isNaN(val)) return "—";
    var s = Number(val).toFixed(6);
    return (signed && Number(val) >= 0) ? ("+" + s) : s;
  }

  function esc(str) {
    return String(str || "")
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;");
  }

  function showTooltip(ev, text) {
    if (!text) return;
    tooltip.textContent = text;
    tooltip.style.display = "block";
    tooltip.style.left = Math.min(window.innerWidth - 380, ev.clientX + 12) + "px";
    tooltip.style.top = Math.max(8, ev.clientY + 12) + "px";
  }

  function hideTooltip() { tooltip.style.display = "none"; }

  function renderInspector() {
    var box = document.getElementById("inspector-body");
    var evObj = currentEventIdx !== null ? eventsByIdx[String(currentEventIdx)] : null;
    var sid = currentSessionId || (evObj && sessionsById[evObj.session] ? evObj.session : null);
    var s = sid ? sessionsById[sid] : null;

    var parts = [];
    if (evObj) {
      parts.push(
        '<div class="event-detail">' +
        '<strong>Event:</strong> <span class="mono">' + esc(evObj.type) + '</span> · ' +
        esc(evObj.ts) + ' · Epoch ' + esc(evObj.epoch) + ' · Session <strong>' + esc(evObj.session) + '</strong><br>' +
        '<div>' + esc(evObj.summary) + '</div>' +
        (evObj.ref ? '<div class="sub">ref: <code>' + esc(evObj.ref) + '</code></div>' : '') +
        (evObj.values ? '<div class="sub mono">values: ' + esc(JSON.stringify(evObj.values)) + '</div>' : '') +
        '</div>'
      );
    }

    if (!s) {
      parts.push('<div class="sub">No session selected or available for inspection.</div>');
      box.innerHTML = parts.join("");
      return;
    }

    var linkItems = [];
    if (s.links) {
      ["report", "handoff", "v_report", "certification", "pipeline"].forEach(function (k) {
        if (s.links[k]) {
          linkItems.push('<a href="' + esc(s.links[k]) + '">' + esc(k) + '</a>');
        }
      });
    }

    var headerHtml =
      '<div style="display:flex;justify-content:space-between;flex-wrap:wrap;gap:8px;margin-bottom:8px;">' +
      '<div><strong>' + esc(s.session_id) + '</strong> (Epoch ' + s.epoch + ', ' + esc(s.verdict) + ') ' +
      '<span class="sub">parent: ' + esc(s.parent_id || "—") + ' · ' + esc(s.agent || "") + ' ' + esc(s.model || "") + '</span></div>' +
      '<div class="sub">' + linkItems.join(" · ") + '</div>' +
      '</div>';

    var estRows = ["AirHitam", "Palong", "Serting"].map(function (est) {
      var ev = (s.per_estate_v && s.per_estate_v[est]) || {};
      var et = (s.per_estate_t && s.per_estate_t[est]) || {};
      var vLift = ev.v_lift !== null && ev.v_lift !== undefined ? ev.v_lift : s["v_lift_" + est];
      return (
        '<tr>' +
        '<td><strong>' + est + '</strong></td>' +
        '<td class="num">' + fmt(vLift, true) + '</td>' +
        '<td class="num">' + fmt(ev.delta_vs_incumbent, true) + '</td>' +
        '<td class="num">' + (ev.recall_at_25 !== null && ev.recall_at_25 !== undefined ? Number(ev.recall_at_25).toFixed(3) : "—") + '</td>' +
        '<td class="num">' + fmt(et.pr_auc_lift, true) + '</td>' +
        '<td class="num">' + fmt(et.delta_vs_incumbent, true) + '</td>' +
        '<td class="num">[' + fmt(et.ci_low, true) + ', ' + fmt(et.ci_high, true) + ']</td>' +
        '</tr>'
      );
    }).join("");

    var breakdown =
      '<div class="table-wrap" style="margin-bottom:8px;"><table>' +
      '<thead><tr><th>Estate</th><th>V Lift</th><th>V Δ Inc</th><th>V Rec@25%</th><th>T Lift</th><th>T Δ Inc</th><th>T 95% CI</th></tr></thead>' +
      '<tbody>' + estRows + '</tbody></table></div>';

    var gateHtml = "";
    if (s.failed_reasons && s.failed_reasons.length) {
      gateHtml += '<div class="sub"><strong>Gate reasons:</strong> ' + s.failed_reasons.map(esc).join("; ") + '</div>';
    }
    if ((s.smells && s.smells.length) || (s.handoff_missing_sections && s.handoff_missing_sections.length)) {
      gateHtml += '<div class="sub warn"><strong>Review flags:</strong> ' +
        esc(JSON.stringify({smells: s.smells, handoff_missing: s.handoff_missing_sections})) + '</div>';
    }

    var docText = activeTab === "report"
      ? (s.report_md || "(No REPORT.md collected for this session)")
      : (s.handoff_md || "(No HANDOFF.md collected for this session)");

    var tabs =
      '<div class="tab-bar">' +
      '<button type="button" class="tab-btn' + (activeTab === "handoff" ? " active" : "") + '" data-tab="handoff">HANDOFF.md</button>' +
      '<button type="button" class="tab-btn' + (activeTab === "report" ? " active" : "") + '" data-tab="report">REPORT.md</button>' +
      '<span class="sub" style="align-self:center;margin-left:6px;">Read-only agent context (chart/table numbers come from verifier JSONs only)</span>' +
      '</div>' +
      '<pre class="md-view">' + esc(docText) + '</pre>';

    parts.push(headerHtml + breakdown + gateHtml + tabs);
    box.innerHTML = parts.join("");

    box.querySelectorAll(".tab-btn").forEach(function (btn) {
      btn.addEventListener("click", function () {
        activeTab = btn.getAttribute("data-tab");
        renderInspector();
      });
    });
  }

  function highlightSelection() {
    document.querySelectorAll("tr.selected").forEach(function (el) { el.classList.remove("selected"); });
    if (currentSessionId) {
      var sr = document.querySelector('tr.session-row[data-session="' + CSS.escape(currentSessionId) + '"]');
      if (sr) sr.classList.add("selected");
    }
    if (currentEventIdx !== null) {
      var er = document.querySelector('tr.event-row[data-event-idx="' + CSS.escape(String(currentEventIdx)) + '"]');
      if (er) er.classList.add("selected");
    }
  }

  document.querySelectorAll(".session-row, .clickable").forEach(function (el) {
    el.addEventListener("click", function () {
      var sid = el.getAttribute("data-session");
      if (sid && sessionsById[sid]) {
        currentSessionId = sid;
        currentEventIdx = null;
        highlightSelection();
        renderInspector();
      }
    });
  });

  document.querySelectorAll(".event-row").forEach(function (el) {
    el.addEventListener("click", function () {
      currentEventIdx = el.getAttribute("data-event-idx");
      var sid = el.getAttribute("data-session");
      if (sid && sessionsById[sid]) currentSessionId = sid;
      highlightSelection();
      renderInspector();
    });
    el.addEventListener("mousemove", function (ev) { showTooltip(ev, el.getAttribute("title")); });
    el.addEventListener("mouseleave", hideTooltip);
  });

  document.querySelectorAll("svg .pt").forEach(function (pt) {
    var titleEl = pt.querySelector("title");
    var txt = titleEl ? titleEl.textContent : "";
    pt.addEventListener("mousemove", function (ev) { showTooltip(ev, txt); });
    pt.addEventListener("mouseleave", hideTooltip);
  });

  document.querySelectorAll("input[data-toggle-series]").forEach(function (cb) {
    cb.addEventListener("change", function () {
      var key = cb.getAttribute("data-toggle-series");
      var show = cb.checked ? "" : "none";
      document.querySelectorAll('#chart-v [data-series="' + key + '"]').forEach(function (node) {
        node.style.display = show;
      });
    });
  });

  highlightSelection();
  renderInspector();
})();
"""


def render_dashboard_html(data: dict[str, Any]) -> str:
    """Render a zero-dependency offline HTML document from canonical dashboard data."""
    header = data["header"]
    sessions = data["sessions"]
    events = data["events"]
    inc = header["incumbent"]

    used = int(header["t_attempts_used"])
    budget = int(header["t_attempt_budget"])
    budget_cls = "danger" if used >= budget else ("warn" if budget > 0 and used >= int(0.8 * budget) else "")
    kill_cls = "danger" if header["kill_switch"] else "ok"
    kill_txt = "ACTIVE" if header["kill_switch"] else "OFF"
    inc_v = _fmt_signed(_finite_float(inc.get("v_score")))
    sha_short = html.escape(str(header.get("split_sha256", ""))[:12] or "—")
    latest_day = html.escape(str(header.get("latest_spend_day") or "none"))
    latest_usd = float(header.get("latest_spend_usd") or 0.0)
    total_usd = float(header.get("total_spend_usd") or 0.0)

    chart_v = _render_v_chart_svg(sessions, header)
    chart_t = _render_t_chart_svg(sessions, header)
    sessions_rows = _render_sessions_rows(sessions)
    events_rows = _render_event_rows(events)

    embedded_json = json.dumps(data, separators=(",", ":")).replace("</", "<\\/")

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>AgentRSI Dashboard — Epoch {int(header["epoch"])}</title>
<style>{CSS_STYLES}</style>
</head>
<body>
<div id="tooltip" class="tooltip"></div>

<header class="topbar">
  <div>
    <h1>AgentRSI Dashboard</h1>
    <div class="sub">Oil-palm SAR triage · 80 m spatial split · Epoch {int(header["epoch"])} (<code>{sha_short}</code>)</div>
  </div>
  <div class="sub">Generated <span class="mono">{html.escape(str(header["generated_at"]))}</span> · Offline HTML (verifier &amp; collector JSONs only)</div>
</header>

<section class="summary-grid" id="header-summary">
  <div class="stat-card">
    <div class="stat-label">Epoch</div>
    <div class="stat-value mono" data-header="epoch">{int(header["epoch"])}</div>
    <div class="stat-sub">Seen epochs: {", ".join(str(e) for e in header["epochs"])}</div>
  </div>
  <div class="stat-card">
    <div class="stat-label">Incumbent</div>
    <div class="stat-value mono" data-header="incumbent">{html.escape(str(inc["id"]))}</div>
    <div class="stat-sub">V composite lift: <span class="mono">{inc_v}</span></div>
  </div>
  <div class="stat-card">
    <div class="stat-label">T Attempts</div>
    <div class="stat-value mono {budget_cls}" data-header="t_attempts">{used} / {budget}</div>
    <div class="stat-sub">Certification budget per epoch</div>
  </div>
  <div class="stat-card">
    <div class="stat-label">Kill Switch</div>
    <div class="stat-value mono {kill_cls}" data-header="kill_switch">{kill_txt}</div>
    <div class="stat-sub">Controlled via <code>state/lineage.json</code></div>
  </div>
  <div class="stat-card">
    <div class="stat-label">Spend (USD)</div>
    <div class="stat-value mono">${total_usd:.4f}</div>
    <div class="stat-sub">Latest ({latest_day}): ${latest_usd:.4f}</div>
  </div>
  <div class="stat-card">
    <div class="stat-label">Sessions</div>
    <div class="stat-value mono">{int(header["session_count"])}</div>
    <div class="stat-sub">Events logged: {len(events)}</div>
  </div>
</section>

<section class="charts-grid">
  <div class="card">
    <div class="card-head">
      <div>
        <h2>Validation V <span class="tag-v">search signal — optimistic</span></h2>
        <div class="sub">Composite &amp; per-estate V lift (never merged with T; lines break at epoch boundaries)</div>
      </div>
      <div class="controls">
        <label><input type="checkbox" checked data-toggle-series="v_lift_AirHitam"><span style="color:#0f766e">AirHitam</span></label>
        <label><input type="checkbox" checked data-toggle-series="v_lift_Palong"><span style="color:#b45309">Palong</span></label>
        <label><input type="checkbox" checked data-toggle-series="v_lift_Serting"><span style="color:#7c3aed">Serting</span></label>
      </div>
    </div>
    {chart_v}
  </div>

  <div class="card">
    <div class="card-head">
      <div>
        <h2>Certification T <span class="tag-t">certified — host-only gate</span></h2>
        <div class="sub">Paired composite T &Delta;PR-AUC vs incumbent with bootstrap CI &amp; <strong>+0.03</strong> effect floor</div>
      </div>
      <div class="sub">Hollow circle = dry session</div>
    </div>
    {chart_t}
  </div>
</section>

<section class="card">
  <div class="card-head">
    <h2>Sessions</h2>
    <div class="sub">Click any row or chart point to inspect per-estate metrics, gate verdict, <code>HANDOFF.md</code>, and <code>REPORT.md</code></div>
  </div>
  <div class="table-wrap">
    <table id="sessions-table">
      <thead>
        <tr>
          <th>Session</th>
          <th>Epoch</th>
          <th>Parent</th>
          <th>Agent / Model</th>
          <th>Verdict</th>
          <th>V Comp</th>
          <th>V AirHitam</th>
          <th>V Palong</th>
          <th>V Serting</th>
          <th>V &Delta; Inc</th>
          <th>T &Delta;PR-AUC</th>
          <th>T CI</th>
          <th>Worst Estate T &Delta;</th>
          <th>Cost</th>
          <th>Runtime</th>
          <th>Flags</th>
          <th>Context</th>
        </tr>
      </thead>
      <tbody>
        {sessions_rows}
      </tbody>
    </table>
  </div>
</section>

<section class="bottom-grid">
  <div class="card" id="events-section">
    <div class="card-head">
      <h2>Events (<code>state/events.jsonl</code>)</h2>
      <div class="sub">Hover for values · click for details</div>
    </div>
    <div class="events-wrap">
      <table>
        <thead>
          <tr><th>Time (UTC)</th><th>Type</th><th>Epoch</th><th>Session</th><th>Summary</th></tr>
        </thead>
        <tbody>
          {events_rows}
        </tbody>
      </table>
    </div>
  </div>

  <div class="card" id="inspector">
    <div class="card-head">
      <h2>Session &amp; Event Inspector</h2>
      <div class="sub">Numeric fields from verifier/collector only</div>
    </div>
    <div id="inspector-body"></div>
  </div>
</section>

<script id="dashboard-data" type="application/json">{embedded_json}</script>
<script>{INLINE_JS}</script>
</body>
</html>
"""


def generate_dashboard(
    agent_root: Path = AGENT_ROOT,
    *,
    output_path: Path | None = None,
    sessions_root: Path | None = None,
    lineage_path: Path | None = None,
    events_path: Path | None = None,
) -> Path:
    """Collect host files and write the self-contained dashboard HTML."""
    out = output_path or (agent_root / "dashboard" / "index.html")
    out.parent.mkdir(parents=True, exist_ok=True)
    data = collect_dashboard_data(
        agent_root,
        sessions_root=sessions_root,
        lineage_path=lineage_path,
        events_path=events_path,
        dashboard_dir=out.parent,
    )
    out.write_text(render_dashboard_html(data), encoding="utf-8")
    return out


class _DashboardHTMLInspector(HTMLParser):
    """Lightweight HTML/SVG inspector for exact-match verification without extra deps."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.in_data_script = False
        self.data_script_buf: list[str] = []
        self.cur_svg: str | None = None
        self.svgs: dict[str, list[tuple[str, dict[str, str]]]] = {}
        self.cur_session_row: str | None = None
        self.cur_metric: str | None = None
        self.metric_buf: list[str] = []
        self.table_metrics: dict[str, dict[str, str]] = {}
        self.in_events = False
        self.events_depth = 0
        self.rendered_events: list[tuple[str, str]] = []
        self.cur_event_idx: str | None = None
        self.event_row_buf: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        amap = {k: (v or "") for k, v in attrs}
        if tag == "script" and amap.get("id") == "dashboard-data":
            self.in_data_script = True
            self.data_script_buf = []
        if tag == "svg":
            sid = amap.get("id", "")
            self.cur_svg = sid
            if sid:
                self.svgs.setdefault(sid, [])
        elif self.cur_svg and tag in {"rect", "line", "polyline", "circle"}:
            self.svgs[self.cur_svg].append((tag, amap))

        if tag == "tr" and "session-row" in amap.get("class", "").split():
            self.cur_session_row = amap.get("data-session")
            if self.cur_session_row:
                self.table_metrics.setdefault(self.cur_session_row, {})
        elif self.cur_session_row and tag == "td" and "data-metric" in amap:
            self.cur_metric = amap["data-metric"]
            self.metric_buf = []

        if amap.get("id") == "events-section":
            self.in_events = True
            self.events_depth = 1
        elif self.in_events:
            self.events_depth += 1
            if tag == "tr" and "event-row" in amap.get("class", "").split():
                self.cur_event_idx = amap.get("data-event-idx")
                self.event_row_buf = []

    def handle_endtag(self, tag: str) -> None:
        if tag == "script" and self.in_data_script:
            self.in_data_script = False
        if tag == "svg":
            self.cur_svg = None
        if tag == "td" and self.cur_metric and self.cur_session_row:
            self.table_metrics[self.cur_session_row][self.cur_metric] = "".join(self.metric_buf).strip()
            self.cur_metric = None
        elif tag == "tr" and self.cur_session_row:
            self.cur_session_row = None

        if self.in_events:
            if tag == "tr" and self.cur_event_idx is not None:
                self.rendered_events.append((self.cur_event_idx, "".join(self.event_row_buf)))
                self.cur_event_idx = None
            self.events_depth -= 1
            if self.events_depth <= 0:
                self.in_events = False

    def handle_data(self, data: str) -> None:
        if self.in_data_script:
            self.data_script_buf.append(data)
        if self.cur_metric is not None:
            self.metric_buf.append(data)
        if self.cur_event_idx is not None:
            self.event_row_buf.append(data)


def verify_dashboard(
    html_path: Path,
    *,
    sessions_root: Path,
    lineage_path: Path,
    events_path: Path,
) -> list[str]:
    """Verify exact equality between rendered dashboard HTML and host JSON/JSONL files."""
    errors: list[str] = []
    if not html_path.exists():
        return [f"Dashboard HTML missing: {html_path}"]

    text = html_path.read_text(encoding="utf-8")
    if EXTERNAL_ASSET_RE.search(text):
        errors.append("External HTTP(S) asset reference detected in dashboard HTML")

    parser = _DashboardHTMLInspector()
    parser.feed(text)

    if not parser.data_script_buf:
        return errors + ["Missing embedded <script id='dashboard-data'> payload"]

    payload_raw = "".join(parser.data_script_buf).replace("<\\/", "</")
    payload = json.loads(payload_raw)

    # 1. Header vs lineage.json
    lineage = json.loads(lineage_path.read_text()) if lineage_path.exists() else None
    header = payload.get("header") or {}
    if lineage is not None:
        for k in ("epoch", "split_sha256", "t_attempts_used", "t_attempt_budget", "kill_switch"):
            if header.get(k) != lineage.get(k):
                errors.append(f"Header mismatch on {k}: {header.get(k)!r} != {lineage.get(k)!r}")
        inc_disk = lineage.get("incumbent") or {}
        inc_html = header.get("incumbent") or {}
        if inc_html.get("id") != inc_disk.get("id"):
            errors.append(f"Incumbent id mismatch: {inc_html.get('id')!r} != {inc_disk.get('id')!r}")
        if _finite_float(inc_html.get("v_score")) != _finite_float(inc_disk.get("v_score")):
            errors.append(f"Incumbent v_score mismatch: {inc_html.get('v_score')} != {inc_disk.get('v_score')}")

    # 2. Sessions vs disk reward.json + certification.json + static table cells
    sessions_list = payload.get("sessions") or []
    sessions_by_id = {s["session_id"]: s for s in sessions_list}
    disk_session_dirs = (
        sorted(p for p in sessions_root.iterdir() if p.is_dir() and (p / "reward.json").exists())
        if sessions_root.exists()
        else []
    )
    if set(sessions_by_id.keys()) != {p.name for p in disk_session_dirs}:
        errors.append(
            f"Session set mismatch: embedded={sorted(sessions_by_id)} vs disk={[p.name for p in disk_session_dirs]}"
        )

    for sdir in disk_session_dirs:
        sid = sdir.name
        s = sessions_by_id.get(sid)
        if not s:
            continue
        rew = json.loads((sdir / "reward.json").read_text())
        cert = json.loads((sdir / "certification.json").read_text()) if (sdir / "certification.json").exists() else {}
        t_cells = parser.table_metrics.get(sid)
        if t_cells is None:
            errors.append(f"{sid}: missing <tr class='session-row'> in #sessions-table")
            continue

        reward_map = [
            ("v_score", "score"),
            ("v_lift_AirHitam", "v_lift_AirHitam"),
            ("v_lift_Palong", "v_lift_Palong"),
            ("v_lift_Serting", "v_lift_Serting"),
            ("v_delta_vs_incumbent", "v_delta_vs_incumbent"),
            ("v_delta_vs_coords", "v_delta_vs_coords"),
            ("runtime_sec", "runtime_sec"),
        ]
        for html_k, rew_k in reward_map:
            disk_val = _finite_float(rew.get(rew_k))
            emb_val = _finite_float(s.get(html_k))
            if emb_val != disk_val:
                errors.append(f"{sid}: embedded {html_k}={emb_val} != reward.json {rew_k}={disk_val}")
            if html_k in t_cells:
                expected_txt = (
                    f"{disk_val:.3f}s" if (html_k == "runtime_sec" and disk_val is not None)
                    else _fmt_signed(disk_val)
                )
                if t_cells[html_k] != expected_txt:
                    errors.append(f"{sid}: table cell {html_k}='{t_cells[html_k]}' != '{expected_txt}'")

        comp = cert.get("composite") or {}
        gate = cert.get("gate") or {}
        if cert:
            if bool(s.get("gate_passed")) != bool(cert.get("gate_passed", False)):
                errors.append(f"{sid}: gate_passed mismatch")
            disk_t = _finite_float(comp.get("comp_delta") if "comp_delta" in comp else gate.get("comp_delta"))
            if _finite_float(s.get("t_comp_delta")) != disk_t:
                errors.append(f"{sid}: t_comp_delta={s.get('t_comp_delta')} != {disk_t}")
            if t_cells.get("t_comp_delta") != _fmt_signed(disk_t):
                errors.append(f"{sid}: table t_comp_delta='{t_cells.get('t_comp_delta')}' != '{_fmt_signed(disk_t)}'")
            for emb_k, comp_k in (("t_ci_low", "ci_low"), ("t_ci_high", "ci_high")):
                d_ci = _finite_float(comp.get(comp_k))
                if _finite_float(s.get(emb_k)) != d_ci:
                    errors.append(f"{sid}: {emb_k}={s.get(emb_k)} != {d_ci}")
            d_w = _finite_float(gate.get("worst_estate_delta"))
            if _finite_float(s.get("worst_estate_delta")) != d_w:
                errors.append(f"{sid}: worst_estate_delta={s.get('worst_estate_delta')} != {d_w}")

    # 3. Events vs state/events.jsonl
    disk_events = load_events(events_path)
    emb_events = payload.get("events") or []
    if len(emb_events) != len(disk_events):
        errors.append(f"Event count mismatch: {len(emb_events)} != {len(disk_events)}")
    else:
        for i, (ee, de) in enumerate(zip(emb_events, disk_events)):
            if ee.get("summary") != de.get("summary") or ee.get("type") != de.get("type") or ee.get("session") != de.get("session"):
                errors.append(f"Event {i} mismatch")
        rendered_by_idx = {idx: row_txt for idx, row_txt in parser.rendered_events}
        for i, de in enumerate(disk_events):
            row_txt = rendered_by_idx.get(str(i))
            if row_txt is None or str(de.get("summary", "")) not in row_txt:
                errors.append(f"Event {i} summary missing in #events-section HTML")

    # 4. Chart & SVG invariants (V != T, +0.03 floor, CI bars, epoch breaks)
    if "chart-v" not in parser.svgs or "chart-t" not in parser.svgs:
        errors.append("Separate #chart-v and #chart-t SVGs are both required")
        return errors
    if "search signal" not in text.lower() or "certified" not in text.lower():
        errors.append("Missing 'search signal' (V) or 'certified' (T) chart labels")

    v_elems = parser.svgs["chart-v"]
    t_elems = parser.svgs["chart-t"]
    if sessions_list:
        t_refs = {a.get("data-ref") for tag, a in t_elems if tag == "line" and a.get("data-ref")}
        if "effect-floor" not in t_refs:
            errors.append("Chart T missing +0.03 effect-floor reference line")

        ci_bar_sessions = {
            a.get("data-session")
            for tag, a in t_elems
            if tag == "line" and "ci-bar" in a.get("class", "").split()
        }
        for s in sessions_list:
            if s.get("t_ci_low") is not None and s.get("t_ci_high") is not None:
                if s["session_id"] not in ci_bar_sessions:
                    errors.append(f"{s['session_id']}: missing CI error bar in #chart-t")

        epochs_present = sorted({int(s["epoch"]) for s in sessions_list})
        if len(epochs_present) > 1:
            for chart_name, elems in (("chart-v", v_elems), ("chart-t", t_elems)):
                bands = {a.get("data-epoch") for tag, a in elems if tag == "rect" and "epoch-band" in a.get("class", "").split()}
                boundaries = [a for tag, a in elems if tag == "line" and "epoch-boundary" in a.get("class", "").split()]
                if not boundaries or len(bands) < len(epochs_present):
                    errors.append(f"{chart_name}: missing epoch bands or epoch boundaries for multi-epoch data")
                polylines = [a for tag, a in elems if tag == "polyline"]
                epoch_counts: dict[str, int] = {}
                for s in sessions_list:
                    epoch_counts[str(s["epoch"])] = epoch_counts.get(str(s["epoch"]), 0) + 1
                for pa in polylines:
                    ep_attr = pa.get("data-epoch", "")
                    n_pts = len([p for p in pa.get("points", "").strip().split() if p])
                    if ep_attr not in epoch_counts or n_pts > epoch_counts[ep_attr]:
                        errors.append(f"{chart_name}: polyline data-epoch={ep_attr!r} crosses epoch boundary ({n_pts} pts)")

    return errors
