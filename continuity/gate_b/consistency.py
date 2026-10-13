"""12-sample sequence consistency — the mid-clip-turn catcher (T7).

Samples 12 frames @1fps from a mint and runs ATOMIC per-frame reads
(facing compass per subject, cast counts per kind) through the same
provider protocol as Gate A. Sequence verdicts are COMPUTED from the
token series, never asked:

- ANY facing flip anywhere in the series = auto-fail (a walking pair that
  turns around mid-clip reads as consistent in any single frame; only the
  series catches it).
- Cast deviation in ANY frame = fail (phantom second camel fading in).
- Unusable / missing read = fail, not skip.

Only atomic tokens are stable across vision passes (Gate A doctrine);
derived questions ("advancing together?") flip-flop and are banned here.
"""

from __future__ import annotations

import datetime as _dt
import json
import os
import subprocess
from dataclasses import dataclass, field

from continuity.gate_a.instruments import (
    CAST_PROMPT,
    FACING_PROMPT,
    parse_cast,
    parse_facing,
)

SAMPLES = 12


@dataclass
class ConsistencyReport:
    beat_id: str
    verdict: str                     # PASS | FAIL
    fails: list[str] = field(default_factory=list)
    series: dict[str, list] = field(default_factory=dict)   # subject -> token series
    usage: dict = field(default_factory=dict)
    frames_dir: str = ""

    def summary(self) -> str:
        return (f"{self.beat_id}: {'PASS' if self.verdict == 'PASS' else 'FAIL'} "
                + ("; ".join(self.fails) if self.fails else ""))


def extract_series_frames(clip_path: str, n: int, workdir: str) -> list[str]:
    os.makedirs(workdir, exist_ok=True)
    procs = []
    for i in range(n):
        p = os.path.join(workdir, f"s{i:02d}.png")
        if os.path.exists(p):
            continue
        procs.append(subprocess.Popen(
            ["ffmpeg", "-y", "-loglevel", "error", "-ss", str(0.5 + i),
             "-i", clip_path, "-frames:v", "1", p],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL))
    for pr in procs:
        pr.wait()
    return [os.path.join(workdir, f"s{i:02d}.png") for i in range(n)
            if os.path.exists(os.path.join(workdir, f"s{i:02d}.png"))]


def adjudicate_series(beat_id: str, facings: dict[str, list[str | None]],
                      casts: dict[str, tuple[int, dict[str, int]]]) -> ConsistencyReport:
    """Pure adjudication from token series.

    facings: subject -> list of per-frame facing tokens (LEFT/RIGHT/TOWARD/
             AWAY/CAMERA/UNKNOWN or None when unusable)
    casts:   kind -> (expected_count, {frame_index: counted})
    """
    fails: list[str] = []

    for subj, series in facings.items():
        if any(t is None for t in series):
            bad = [i for i, t in enumerate(series) if t is None]
            fails.append(f"facing_{subj}: unusable read at frames {bad}")
            continue
        uniq = set(series)
        if len(uniq) > 1:
            fails.append(
                f"facing_{subj}: FLIP mid-clip — series {series[:4]}... "
                f"uniq {sorted(uniq)}")
        elif uniq == {"CAMERA"}:
            fails.append(f"facing_{subj}: faces camera for the whole clip")

    for kind, (expected, counts) in casts.items():
        unusable = [i for i, n in counts.items() if n == -1]
        if unusable:
            fails.append(f"cast_{kind}: unusable read at frames "
                         f"{unusable[:4]}")
            continue
        wrong = {i: n for i, n in counts.items() if n != expected}
        if wrong:
            fails.append(f"cast_{kind}: deviation at frames "
                         f"{dict(list(wrong.items())[:4])}, expected {expected}")

    return ConsistencyReport(
        beat_id=beat_id,
        verdict="PASS" if not fails else "FAIL",
        fails=fails,
        series={k: list(v) for k, v in facings.items()},
    )


def consistency_scan(clip_path: str, beat_id: str, provider,
                     facing_subjects: list[str], cast: dict[str, int],
                     workdir: str, samples: int = SAMPLES,
                     out_path: str | None = None) -> ConsistencyReport:
    """Live scan: 12 frames, one atomic read per instrument per frame.

    Single pass per frame — the SEQUENCE is the second instrument (a flip
    must appear in the series to matter; single-frame verdicts are already
    two-pass gated at Gate A on the keyframe).
    """
    frames = extract_series_frames(clip_path, samples, workdir)
    if len(frames) < samples:
        return ConsistencyReport(
            beat_id=beat_id, verdict="FAIL",
            fails=[f"only extracted {len(frames)}/{samples} frames"],
            frames_dir=workdir)

    usage = {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0, "cost_usd": 0.0}
    facings: dict[str, list] = {s: [] for s in facing_subjects}
    casts: dict[str, tuple[int, dict[int, int]]] = {}

    for i, fp in enumerate(frames):
        for subj in facing_subjects:
            text, u = provider.ask(fp, FACING_PROMPT.format(subject=subj),
                                   key=f"{beat_id}|seq{i:02d}|facing_{subj}")
            usage["calls"] += 1
            for k in ("prompt_tokens", "completion_tokens"):
                usage[k] += u.get(k) or 0
            usage["cost_usd"] += u.get("cost_usd") or 0.0
            facings[subj].append(parse_facing(text))
        for kind in cast:
            text, u = provider.ask(fp, CAST_PROMPT.format(kind=kind),
                                   key=f"{beat_id}|seq{i:02d}|cast_{kind}")
            usage["calls"] += 1
            for k in ("prompt_tokens", "completion_tokens"):
                usage[k] += u.get(k) or 0
            usage["cost_usd"] += u.get("cost_usd") or 0.0
            if kind not in casts:
                casts[kind] = (cast[kind], {})
            n = parse_cast(text)
            casts[kind][1][i] = -1 if n is None else n

    report = adjudicate_series(beat_id, facings, casts)
    report.usage = usage
    report.frames_dir = workdir
    if out_path:
        with open(out_path, "w", encoding="utf-8") as fh:
            json.dump({
                "beat": beat_id, "verdict": report.verdict,
                "fails": report.fails, "series": report.series,
                "usage": usage, "samples": samples,
                "scanned_at": _dt.datetime.now(_dt.timezone.utc)
                                  .replace(microsecond=0).isoformat(),
            }, fh, indent=1)
    return report
