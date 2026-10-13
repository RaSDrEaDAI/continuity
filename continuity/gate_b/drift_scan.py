"""Sky-band drift scan — deterministic day-drift detection (T7).

Port of the proven SN lane instrument (tmp_work/sn_qc/sn_drift_scan.py, roll v4+):
12 frames @1fps, mean luminance of the sky band (top 22%, center 60% width).

    FAIL = total swing > 25  OR  max step > 12

Vision grids OVER-call oscillation ("pulsing sky" that pixel-scans show as a
smooth dim) and UNDER-call monotonic slides — the scan settles both. Golden
hour beats are a known drift basin: they go to the operator as taste calls,
not auto re-mints.
"""

from __future__ import annotations

import colorsys
import os
import subprocess
import tempfile
from dataclasses import dataclass, field

SWING_LIMIT = 25.0
STEP_LIMIT = 12.0
SAMPLES = 12
SKY_BAND = (0.0, 0.22)   # top 22% of frame
SKY_WIDTH = (0.2, 0.8)   # center 60% of width


@dataclass
class DriftReport:
    beat_id: str
    luminance: list[float]
    swing: float
    max_step: float
    passed: bool
    fail_class: str = ""           # "gradual_drift" | "sudden_jump" | ""
    direction_flips: int = 0       # oscillation hint (taste context, not verdict)
    hue: list[float] = field(default_factory=list)

    def summary(self) -> str:
        why = (
            "gradual drift (re-mint)" if self.fail_class == "gradual_drift"
            else "sudden jump (inspect step frame before re-minting)"
            if self.fail_class == "sudden_jump" else "clean"
        )
        return (f"{self.beat_id}: swing {self.swing:.1f} maxstep {self.max_step:.1f} "
                f"-> {'PASS' if self.passed else 'FAIL'} ({why})")


def _band_stats(frame) -> tuple[float, float]:
    """(mean luminance, mean hue) of the sky band of one PIL RGB frame."""
    w, h = frame.size
    box = frame.crop((int(w * SKY_WIDTH[0]), int(h * SKY_BAND[0]),
                      int(w * SKY_WIDTH[1]), int(h * SKY_BAND[1])))
    px = list(box.getdata())
    n = len(px)
    r = sum(p[0] for p in px) / n
    g = sum(p[1] for p in px) / n
    b = sum(p[2] for p in px) / n
    lum = 0.2126 * r + 0.7152 * g + 0.0722 * b
    hue = colorsys.rgb_to_hsv(r / 255, g / 255, b / 255)[0] * 360
    return lum, hue


def _extract_frames(clip_path: str, n: int, workdir: str,
                    ffmpeg: str = "ffmpeg") -> list[str]:
    """Pull n frames ~1fps into workdir; returns paths (missing files skipped)."""
    os.makedirs(workdir, exist_ok=True)
    paths = []
    procs = []
    for i in range(n):
        p = os.path.join(workdir, f"f{i}.png")
        paths.append(p)
        if os.path.exists(p):
            continue
        procs.append(subprocess.Popen(
            [ffmpeg, "-y", "-loglevel", "error", "-ss", str(0.5 + i),
             "-i", clip_path, "-frames:v", "1", p],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL))
    for pr in procs:
        pr.wait()
    return [p for p in paths if os.path.exists(p)]


def drift_scan(clip_path: str, beat_id: str = "?", samples: int = SAMPLES,
               workdir: str | None = None, cleanup: bool = True) -> DriftReport:
    """Extract `samples` frames at ~1fps and scan the sky band.

    Frame files land in workdir (temp dir when None); pass workdir + cleanup=False
    to keep frames for strip QC / step-frame inspection.
    """
    from PIL import Image

    tmp = workdir or tempfile.mkdtemp(prefix="continuity_drift_")
    frames = _extract_frames(clip_path, samples, tmp)
    if len(frames) < 2:
        raise RuntimeError(f"drift_scan: extracted {len(frames)} frames from {clip_path}")
    lums, hues = [], []
    for p in frames:
        lum, hue = _band_stats(Image.open(p).convert("RGB"))
        lums.append(round(lum, 1))
        hues.append(round(hue, 1))
    report = adjudicate(beat_id, lums, hue=hues)
    if cleanup and workdir is None:
        for p in frames:
            try:
                os.remove(p)
            except OSError:
                pass
        try:
            os.rmdir(tmp)
        except OSError:
            pass
    return report


def adjudicate(beat_id: str, luminance: list[float],
               hue: list[float] | None = None) -> DriftReport:
    """Compute swing/max-step verdict from a luminance series (pure, testable).

    Doctrine: swing > 25 = gradual drift (re-mint). max_step > 12 with swing
    <= 25 = sudden jump — inspect the step frame BEFORE re-minting; it may be
    a scene-motivated lighting event (taste flag). Both still read FAIL; the
    fail_class drives the repair route.
    """
    swing = max(luminance) - min(luminance)
    steps = [abs(b - a) for a, b in zip(luminance, luminance[1:])]
    max_step = max(steps) if steps else 0.0
    passed = swing <= SWING_LIMIT and max_step <= STEP_LIMIT
    fail_class = ""
    if swing > SWING_LIMIT:
        fail_class = "gradual_drift"
    elif max_step > STEP_LIMIT:
        fail_class = "sudden_jump"
    diffs = [1 if b > a else (-1 if b < a else 0) for a, b in zip(luminance, luminance[1:])]
    nz = [x for x in diffs if x]
    flips = sum(1 for a, b in zip(nz, nz[1:]) if a != b)
    return DriftReport(
        beat_id=beat_id,
        luminance=luminance,
        swing=round(swing, 1),
        max_step=round(max_step, 1),
        passed=passed,
        fail_class=fail_class,
        direction_flips=flips,
        hue=hue or [],
    )
