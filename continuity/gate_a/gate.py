"""Gate A runner — two-pass solo reads, code-adjudicated JSON verdicts.

Consumes the T4 keyframe manifest (``{beat_id: {file, seed, ...}}``) plus the
deck's per-beat ``qc`` specs and emits one verdict JSON per frame. A frame
with ANY fail never reaches the mint stage (T6 reads the gate summary and
skips dirty beats — a dirty frame never mints).

Verdict JSON shape::

    {
      "beat": "D1_depart", "frame": ".../D1_depart_s7101_00001_.png",
      "seed": 7101, "verdict": "PASS" | "FAIL",
      "fails": ["rope: free-hanging segment (verdict YES)"],
      "reads": {"rope": [{"free_hanging": false}, {"free_hanging": false}],
                 "facing_man": ["RIGHT", "RIGHT"], ...},
      "passes": 2, "agreed": true,
      "provider": "openai-compat:anthropic/claude-haiku-5.5",
      "instrument_version": "gate-a/1",
      "usage": {"calls": 8, "prompt_tokens": 1900, "cost_usd": 0.0002},
      "gated_at": "2026-10-09T..."
    }
"""

from __future__ import annotations

import datetime as _dt
import json
import os

from continuity.gate_a.instruments import (
    CAST_PROMPT,
    FACING_PROMPT,
    GAZE_PROMPT,
    SCALE_PROMPT,
    SEGMENT_INVENTORY_PROMPT,
    INSTRUMENT_VERSION,
    against_expected,
    facing_consistency,
    FacingCompass,
    parse_cast,
    parse_facing,
    parse_gaze,
    parse_scale,
    parse_segment_inventory,
)
from continuity.gate_a.providers import VisionProvider

MODES = ("walk", "still", "solo_man", "solo_camel", "env")

_DEFAULT_CAST = {
    "walk": {"man": 1, "camel": 1},
    "still": {"man": 1, "camel": 1},
    "solo_man": {"man": 1, "camel": 0},
    "solo_camel": {"man": 0, "camel": 1},
    "env": {"man": 0, "camel": 0},
}


class GateSpec:
    """Per-beat QC contract, from the deck's ``qc`` block.

    mode      walk | still | solo_man | solo_camel | env
    expect    expected travel direction "L"|"R"|"TOWARD"|"AWAY"|null
    cast      expected {man, camel} counts (mode defaults when absent)
    rope_check, scale_check   instrument toggles (default on when a camel
              is expected to be visible)
    gaze_atcam_ok  allow gaze-at-camera on this beat (default False —
              walking wides get zero camera gaze)
    """

    def __init__(self, d: dict | None):
        d = d or {}
        self.mode = d.get("mode", "still")
        if self.mode not in MODES:
            raise ValueError(f"qc.mode must be one of {MODES}, got {self.mode!r}")
        self.expect = d.get("expect")
        # Gate B drift scan applicability: the sky-band instrument requires
        # the top 22% of frame to BE sky. Env macros are tight comps by
        # definition (background is terrain, not sky) -> default off there;
        # explicit qc.tod_scan overrides either way (live-run lesson: a boot
        # macro swung 94.7 on background texture, not time-of-day).
        self.tod_scan = bool(d.get("tod_scan", self.mode != "env"))
        self.cast = {**_DEFAULT_CAST[self.mode], **(d.get("cast") or {})}
        camel_visible = self.cast.get("camel", 0) > 0
        self.rope_check = bool(d.get("rope_check", camel_visible))
        self.scale_check = bool(d.get("scale_check", camel_visible))
        self.gaze_atcam_ok = bool(d.get("gaze_atcam_ok", False))
        self.charsheet = d.get("charsheet")  # optional path for scale probe

    @property
    def facing_subjects(self) -> list[str]:
        return {
            "walk": ["man", "camel"],
            "still": ["man"],
            "solo_man": ["man"],
            "solo_camel": ["camel"],
            "env": [],
        }[self.mode]


def _utcnow() -> str:
    return _dt.datetime.now(_dt.timezone.utc).replace(microsecond=0).isoformat()


def gate_frame(frame_path: str, spec: GateSpec, provider: VisionProvider,
               beat: str = "", seed: int | None = None,
               passes: int = 2) -> dict:
    """Run the full instrument battery on one frame, solo reads only."""
    reads: dict[str, list] = {}
    fails: list[str] = []
    usage = {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0,
             "cost_usd": 0.0}

    def ask(image: str, prompt: str, instrument: str, parser, label: str):
        """Two independent passes; None parse or disagreement = fail."""
        answers = []
        for _ in range(passes):
            text, u = provider.ask(image, prompt, key=f"{beat}|{instrument}")
            usage["calls"] += 1
            for k in ("prompt_tokens", "completion_tokens"):
                if u.get(k):
                    usage[k] += u[k]
            if u.get("cost_usd"):
                usage["cost_usd"] += u["cost_usd"]
            parsed = parser(text)
            answers.append(parsed)
        reads[label] = answers
        if any(a is None for a in answers):
            fails.append(f"{label}: unusable read {answers}")
        elif len(set(map(repr, answers))) > 1:
            fails.append(f"{label}: passes disagree {answers}")
        return answers[0] if answers and answers[0] is not None else None

    # 1. rope / strap segment inventory (camel in frame)
    if spec.rope_check:
        res = ask(frame_path, SEGMENT_INVENTORY_PROMPT,
                  "rope", parse_segment_inventory, "rope")
        if res is not None and res["free_hanging"]:
            why = (f"verdict {res['verdict_line']}" if res["verdict_line"] == "YES"
                   else f"{res['segments_b']} free segment(s) listed vs verdict NO")
            fails.append(f"rope: free-hanging segment ({why})")

    # 2. per-subject facing compass (atomic; cross-subject computed below)
    facings: dict[str, str] = {}
    for subj in spec.facing_subjects:
        d = ask(frame_path, FACING_PROMPT.format(subject=subj),
                f"facing_{subj}", parse_facing, f"facing_{subj}")
        if d is not None:
            facings[subj] = d

    # 3. cast counts — asked for BOTH kinds even when expected is 0:
    # phantom subjects (second camel fading in) are exactly what this catches
    for kind, expected in spec.cast.items():
        n = ask(frame_path, CAST_PROMPT.format(kind=kind),
                f"cast_{kind}", parse_cast, f"cast_{kind}")
        if n is not None and n != expected:
            fails.append(f"cast_{kind}: counted {n}, expected {expected}")

    # 4. scale vs charsheet (composite probe image) — fail CLOSED (T14):
    # a required scale check without a usable reference is a fail, never a
    # silent skip; decks that legitimately have no scale contract set
    # scale_check:false in the validated spec.
    if spec.scale_check and spec.charsheet and os.path.exists(spec.charsheet):
        probe = _scale_probe_image(frame_path, spec.charsheet)
        s = ask(probe, SCALE_PROMPT, "scale", parse_scale, "scale")
        if s is not None and s in ("SMALL", "PONY"):
            fails.append(f"scale: camel reads {s} vs charsheet")
    elif spec.scale_check:
        why = ("no charsheet configured" if not spec.charsheet
               else f"charsheet not found on disk ({spec.charsheet})")
        reads["scale"] = [f"UNVERIFIABLE: {why}"]
        fails.append(f"scale: required but unverifiable — {why} (fail closed)")

    # 5. gaze (walk wides: zero camera gaze unless explicitly allowed)
    if spec.facing_subjects:
        g = ask(frame_path, GAZE_PROMPT.format(subject=spec.facing_subjects[0]),
                "gaze", parse_gaze, "gaze")
        if g == "CAMERA" and not spec.gaze_atcam_ok:
            fails.append("gaze: subject looks at camera (budget 0 on this beat)")

    # computed pair verdicts (never asked, always derived)
    if "man" in facings and "camel" in facings:
        ok = facing_consistency(
            FacingCompass("man", facings["man"], ""),
            FacingCompass("camel", facings["camel"], ""),
        )
        if not ok:
            fails.append(
                f"facing: cross-subject mismatch "
                f"(man {facings['man']} vs camel {facings['camel']})")
    for subj, d in facings.items():
        if spec.expect and against_expected(d, spec.expect):
            fails.append(f"facing_{subj}: {d} walks against expected {spec.expect}")

    verdict = "PASS" if not fails else "FAIL"
    return {
        "beat": beat, "frame": frame_path, "seed": seed,
        "verdict": verdict, "fails": fails, "reads": reads,
        "passes": passes, "agreed": not any("disagree" in f for f in fails),
        "mode": spec.mode, "expect": spec.expect,
        "provider": provider.name, "instrument_version": INSTRUMENT_VERSION,
        "usage": usage, "gated_at": _utcnow(),
    }


def _scale_probe_image(frame_path: str, charsheet_path: str) -> str:
    """Charsheet + frame side by side in ONE image — the only scale
    instrument that works (relative judgment; absolute size reads don't)."""
    from PIL import Image

    side = "_gate_a_scale_probe.jpg"

    def load(p):
        im = Image.open(p).convert("RGB")
        im.thumbnail((1024, 1024))
        return im

    a, b = load(charsheet_path), load(frame_path)
    h = max(a.height, b.height)
    canvas = Image.new("RGB", (a.width + b.width + 16, h), (24, 24, 28))
    canvas.paste(a, (0, 0))
    canvas.paste(b, (a.width + 16, 0))
    out = os.path.join(os.path.dirname(os.path.abspath(frame_path)) or ".",
                       os.path.basename(frame_path).rsplit(".", 1)[0] + side)
    canvas.save(out, quality=88)
    return out


def gate_manifest(manifest_path: str, deck, provider: VisionProvider,
                  out_dir: str, passes: int = 2) -> dict:
    """Gate every rendered frame in a T4 manifest. Writes one verdict JSON
    per beat + a summary; returns the summary (all_pass gates the mint)."""
    with open(manifest_path, "r", encoding="utf-8") as fh:
        manifest = json.load(fh)
    specs = {b.id: GateSpec(getattr(b, "qc", None)) for b in deck.beats}
    os.makedirs(out_dir, exist_ok=True)

    def _resolve_frame(entry: dict) -> str | None:
        """Frame path from a manifest entry. Relative paths resolve against
        the manifest's own directory first (shipped fixture bundles), then
        against the CWD (render manifests written next to their frames)."""
        f = entry.get("file")
        if not f or os.path.exists(f):
            return f
        beside = os.path.join(os.path.dirname(os.path.abspath(manifest_path)), f)
        return beside if os.path.exists(beside) else f

    reports = {}
    for beat_id, entry in manifest.items():
        frame = _resolve_frame(entry)
        if not frame or not os.path.exists(frame):
            reports[beat_id] = {"verdict": "FAIL",
                                "fails": [f"frame missing: {frame}"]}
            continue
        reports[beat_id] = gate_frame(
            frame, specs.get(beat_id, GateSpec(None)), provider,
            beat=beat_id, seed=entry.get("seed"), passes=passes)
        path = os.path.join(out_dir, f"{beat_id}.gate_a.json")
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(reports[beat_id], fh, indent=1)

    failed = [b for b, r in reports.items() if r["verdict"] != "PASS"]
    summary = {
        "manifest": manifest_path, "gated": len(reports),
        "passed": len(reports) - len(failed), "failed": failed,
        "all_pass": not failed,
        "instrument_version": INSTRUMENT_VERSION,
        "provider": provider.name,
        "gated_at": _utcnow(),
        "reports": {b: os.path.join(out_dir, f"{b}.gate_a.json")
                    for b in reports},
    }
    with open(os.path.join(out_dir, "gate_a_summary.json"), "w",
              encoding="utf-8") as fh:
        json.dump(summary, fh, indent=1)
    return summary
