"""Gate B runner — per-mint QC report + the dirty-clip-never-ships rule (T7).

Combines the deterministic instruments (drift scan, motion index,
first-frame fidelity) with the 12-sample consistency scan into one
per-shot report. A mint with ANY fail is excluded from the master cut
and routed to the re-roll loop (reroll.py).

Report JSON shape::

    {
      "beat": "D2_bootmacro", "clip": ".../D2_...mp4", "seed": 1901,
      "verdict": "PASS" | "FAIL",
      "fails": ["drift: gradual (swing 40.2)", ...],
      "drift": {...}, "motion": {...}, "consistency": {...},
      "usage": {...}, "gated_at": "..."
    }
"""

from __future__ import annotations

import datetime as _dt
import json
import os

from continuity.gate_b.consistency import consistency_scan
from continuity.gate_b.drift_scan import drift_scan
from continuity.gate_b.motion import motion_scan
from continuity.gate_a.gate import GateSpec

GATE_B_VERSION = "gate-b/1"


def _utcnow() -> str:
    return _dt.datetime.now(_dt.timezone.utc).replace(microsecond=0).isoformat()


def gate_mint(clip_path: str, beat_id: str, spec: GateSpec, seed: int | None = None,
              keyframe_path: str | None = None, provider=None,
              workdir: str = ".", samples: int = 12) -> dict:
    """Full Gate B battery on one mint. provider=None skips the vision
    consistency scan (deterministic-only mode — CI, quick re-checks)."""
    os.makedirs(workdir, exist_ok=True)
    fails: list[str] = []

    if getattr(spec, "tod_scan", True):
        drift = drift_scan(clip_path, beat_id=beat_id,
                           workdir=os.path.join(workdir, beat_id, "drift"))
        if not drift.passed:
            fails.append(f"drift: {drift.fail_class} "
                         f"(swing {drift.swing} maxstep {drift.max_step})")
    else:
        drift = None

    motion = motion_scan(clip_path, beat_id=beat_id, samples=6,
                         keyframe_path=keyframe_path,
                         workdir=os.path.join(workdir, beat_id, "motion"))
    if motion.frozen:
        fails.append(f"motion: FROZEN (index {motion.motion_index})")
    if motion.ffd_flag:
        fails.append(f"first-frame drift {motion.first_frame_drift} vs keyframe")

    consistency = None
    if provider is not None:
        consistency = consistency_scan(
            clip_path, beat_id, provider,
            facing_subjects=spec.facing_subjects, cast=spec.cast,
            workdir=os.path.join(workdir, beat_id, "seq"),
            samples=samples,
            out_path=os.path.join(workdir, beat_id, "consistency.json"))
        if consistency.verdict != "PASS":
            fails.extend(f"consistency: {f}" for f in consistency.fails)

    report = {
        "beat": beat_id, "clip": clip_path, "seed": seed,
        "verdict": "PASS" if not fails else "FAIL",
        "fails": fails,
        "drift": ({"luminance": drift.luminance, "swing": drift.swing,
                   "max_step": drift.max_step, "fail_class": drift.fail_class,
                   "direction_flips": drift.direction_flips}
                  if drift else "skipped: no sky band (macro/CU comp)"),
        "motion": {"motion_index": motion.motion_index, "frozen": motion.frozen,
                   "first_frame_drift": motion.first_frame_drift,
                   "ffd_flag": motion.ffd_flag},
        "consistency": ({"verdict": consistency.verdict, "fails": consistency.fails,
                         "series": consistency.series}
                        if consistency else "skipped: no provider"),
        "usage": (consistency.usage if consistency
                  else {"calls": 0, "cost_usd": 0.0}),
        "gate_b_version": GATE_B_VERSION,
        "gated_at": _utcnow(),
    }
    with open(os.path.join(workdir, beat_id, "gate_b.json"), "w",
              encoding="utf-8") as fh:
        json.dump(report, fh, indent=1)
    return report


def resolve_beat(clip_filename: str, spec_ids) -> str:
    """Filename -> beat id: stem before the ``_s<seed>`` marker, matched
    against spec ids (longest prefix wins). A wrong spec gates against
    the wrong cast/facing contract (live-run lesson: 'D2' matched nothing
    and an env macro got gated as a pair beat)."""
    import os as _os
    import re as _re
    base = _re.split(r"_s\d+", _os.path.basename(clip_filename))[0]
    if base in spec_ids:
        return base
    cand = sorted((k for k in spec_ids if base.startswith(k)), key=len)
    return cand[-1] if cand else _os.path.basename(clip_filename)


def summarize(reports: list[dict]) -> dict:
    """Batch summary — the input to master assembly and re-roll routing."""
    failed = [r["beat"] for r in reports if r["verdict"] != "PASS"]
    return {
        "gated": len(reports), "passed": len(reports) - len(failed),
        "failed": failed, "all_pass": not failed,
        "total_cost_usd": round(sum(r.get("usage", {}).get("cost_usd", 0) or 0
                                    for r in reports), 5),
        "gate_b_version": GATE_B_VERSION, "gated_at": _utcnow(),
    }


def report_markdown(reports: list[dict], summary: dict) -> str:
    """Per-shot QC report table (the human-readable deliverable)."""
    lines = [
        "# Gate B — per-shot QC report",
        "",
        f"gated {summary['gated']} · passed {summary['passed']} · "
        f"failed {summary['failed'] or 'none'} · "
        f"vision cost ${summary['total_cost_usd']}",
        "",
        "| beat | verdict | drift (swing/maxstep) | motion (ffd) | consistency fails |",
        "|---|---|---|---|---|",
    ]
    for r in reports:
        d, m = r["drift"], r["motion"]
        drift_cell = ("skip (macro/CU)" if isinstance(d, str)
                      else f"{d['swing']}/{d['max_step']}"
                      + ("" if d["fail_class"] == "" else " " + d["fail_class"]))
        cons = r.get("consistency")
        cons_fails = "; ".join(cons["fails"][:2]) if isinstance(cons, dict) else "skipped"
        lines.append(
            f"| {r['beat']} | {r['verdict']} "
            f"| {drift_cell} "
            f"| {m['motion_index']} ({m.get('first_frame_drift')}) "
            f"| {cons_fails} |")
    lines.append("")
    return "\n".join(lines)
