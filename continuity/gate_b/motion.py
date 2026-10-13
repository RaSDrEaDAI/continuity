"""Motion index + first-frame fidelity — deterministic, no vision (T7).

- Motion index: mean absolute luminance diff between consecutive sampled
  frames. Near-zero motion on a beat that specs movement = frozen mint
  (the fal lipsync failure class: animation silently never applied).
- First-frame drift: mean abs diff between the mint's frame 0 and the
  approved keyframe it was minted from. The first-frame i2v contract is
  "same picture, now it moves" — a high diff means the mint departed from
  the approved frame immediately.
"""

from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass

FROZEN_LIMIT = 1.5      # mean abs 8-bit diff below this = frozen (measured:
                        # live mints read 4-9; a still-image mux reads <1)
FFD_SOFT_LIMIT = 12.0   # VAE+h264 roundtrip alone costs ~6; >12 = real drift


@dataclass
class MotionReport:
    beat_id: str
    motion_index: float
    frozen: bool
    first_frame_drift: float | None = None
    ffd_flag: bool = False
    frames_dir: str = ""

    def summary(self) -> str:
        s = (f"{self.beat_id}: motion {self.motion_index:.2f}"
             f"{' FROZEN' if self.frozen else ''}")
        if self.first_frame_drift is not None:
            s += (f", ffd {self.first_frame_drift:.2f}"
                  f"{' DRIFT-FLAG' if self.ffd_flag else ''}")
        return s


def _mean_abs_diff(a, b) -> float:
    """Mean absolute 8-bit diff between two same-size PIL images (0-255)."""
    pa, pb = a.convert("L").getdata(), b.convert("L").getdata()
    n = len(pa)
    return sum(abs(x - y) for x, y in zip(pa, pb)) / n


def _extract(clip: str, ts: float, out_path: str) -> bool:
    r = subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-ss", str(ts),
         "-i", clip, "-frames:v", "1", out_path],
        capture_output=True)
    return os.path.exists(out_path) and r.returncode == 0


def motion_scan(clip_path: str, beat_id: str = "?", samples: int = 6,
                keyframe_path: str | None = None,
                workdir: str | None = None,
                frozen_limit: float = FROZEN_LIMIT,
                ffd_limit: float = FFD_SOFT_LIMIT) -> MotionReport:
    """Sample `samples` frames spread over the clip; diff consecutive pairs.

    When keyframe_path is given, also diffs frame 0 against it (first-frame
    fidelity). Frame files are kept when workdir is given (reuse for strips).
    """
    from PIL import Image

    tmp = workdir or ""
    if tmp:
        os.makedirs(tmp, exist_ok=True)
    import tempfile
    keep = bool(tmp)
    if not keep:
        tmp = tempfile.mkdtemp(prefix="continuity_motion_")

    paths = []
    # spread over duration: skip t=0 (black frame risk) — start at 0.5s
    for i in range(samples):
        p = os.path.join(tmp, f"m{i}.png")
        if not os.path.exists(p):
            _extract(clip_path, 0.5 + i * 2.0, p)
        if os.path.exists(p):
            paths.append(p)

    if len(paths) < 2:
        raise RuntimeError(f"motion_scan: extracted {len(paths)} frames from {clip_path}")

    diffs = []
    for a, b in zip(paths, paths[1:]):
        ia, ib = Image.open(a), Image.open(b)
        if ia.size != ib.size:
            ib = ib.resize(ia.size)
        diffs.append(_mean_abs_diff(ia, ib))
    motion = sum(diffs) / len(diffs)

    ffd = None
    flag = False
    if keyframe_path and os.path.exists(keyframe_path):
        f0 = os.path.join(tmp, "f0.png")
        if not os.path.exists(f0):
            _extract(clip_path, 0.04, f0)  # first rendered frame, past container start
        if os.path.exists(f0):
            kf = Image.open(keyframe_path)
            im = Image.open(f0)
            if kf.size != im.size:
                im = im.resize(kf.size)
            ffd = round(_mean_abs_diff(kf, im), 2)
            flag = ffd > ffd_limit

    if not keep:
        for p in paths + ([os.path.join(tmp, "f0.png")] if ffd is not None else []):
            try:
                os.remove(p)
            except OSError:
                pass
        try:
            os.rmdir(tmp)
        except OSError:
            pass

    return MotionReport(
        beat_id=beat_id,
        motion_index=round(motion, 2),
        frozen=motion < frozen_limit,
        first_frame_drift=ffd,
        ffd_flag=flag,
        frames_dir=tmp if keep else "",
    )
