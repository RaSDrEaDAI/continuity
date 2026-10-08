"""Sky-band drift scan — deterministic day-drift detection (T7).

Port target (proven): tmp_work/sn_qc/sn_drift_scan.py. 12 frames @1fps,
mean luminance of the sky band (top 22%, center 60% width).

    FAIL = total swing > 25  OR  max step > 12

Vision grids OVER-call oscillation ("pulsing sky" that pixel-scans show as a
smooth dim) and UNDER-call monotonic slides — the scan settles both. Golden
hour beats are a known drift basin: they go to the operator as taste calls,
not auto re-mints.
"""

from __future__ import annotations

from dataclasses import dataclass

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

    def summary(self) -> str:
        why = (
            "gradual drift (re-mint)" if self.swing > SWING_LIMIT
            else "sudden jump (inspect step frame before re-minting)"
            if self.max_step > STEP_LIMIT else "clean"
        )
        return (f"{self.beat_id}: swing {self.swing:.1f} maxstep {self.max_step:.1f} "
                f"-> {'PASS' if self.passed else 'FAIL'} ({why})")


def _band_mean(frame) -> float:
    """Mean luminance of the sky band of one PIL frame (T7 port)."""
    w, h = frame.size
    box = frame.crop((int(w * SKY_WIDTH[0]), int(h * SKY_BAND[0]),
                      int(w * SKY_WIDTH[1]), int(h * SKY_BAND[1])))
    return sum(box.convert("L").getdata()) / (box.size[0] * box.size[1])


def drift_scan(clip_path: str, beat_id: str = "?") -> DriftReport:
    """Extract SAMPLES frames at 1fps and scan the sky band (T7 port)."""
    raise NotImplementedError(
        "T7: port ffmpeg frame extraction from tmp_work/sn_qc/sn_drift_scan.py"
    )


def adjudicate(beat_id: str, luminance: list[float]) -> DriftReport:
    """Compute swing/max-step verdict from a luminance series (pure, testable)."""
    swing = max(luminance) - min(luminance)
    steps = [abs(b - a) for a, b in zip(luminance, luminance[1:])]
    max_step = max(steps) if steps else 0.0
    return DriftReport(
        beat_id=beat_id,
        luminance=luminance,
        swing=round(swing, 1),
        max_step=round(max_step, 1),
        passed=swing <= SWING_LIMIT and max_step <= STEP_LIMIT,
    )
