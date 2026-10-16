"""Ordered master assembly — PASS clips only (T7).

Concat via the ffmpeg concat demuxer. Every input must already be
same-codec/same-resolution (the mint lane guarantees this: one graph,
one resolution). A beat whose Gate B verdict is not PASS never enters
the concat — a cut missing a shot is a wrong deliverable, and so is a
cut containing an ungated shot.
"""

from __future__ import annotations

import os
import subprocess


def master_concat(clips: list[str], out_path: str, workdir: str = ".") -> str:
    """Concat PASS clips in beat order -> out_path.

    clips must be absolute/relative file paths that exist. Uses stream
    copy when every clip shares the codec family (they do within a lane);
    falls back to re-encode on failure.
    """
    if not clips:
        raise RuntimeError("master_concat: no clips to assemble")
    missing = [c for c in clips if not os.path.exists(c)]
    if missing:
        raise RuntimeError(f"master_concat: missing clips {missing}")
    os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)
    os.makedirs(workdir, exist_ok=True)
    lst = os.path.join(workdir, "concat_list.txt")
    with open(lst, "w", encoding="utf-8") as fh:
        for c in clips:
            fh.write(f"file '{os.path.abspath(c).replace(chr(39), chr(39)*2)}'\n")

    r = subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-f", "concat", "-safe", "0",
         "-i", lst, "-c", "copy", out_path], capture_output=True)
    if r.returncode != 0 or not os.path.exists(out_path):
        # re-encode fallback (mixed containers)
        r = subprocess.run(
            ["ffmpeg", "-y", "-loglevel", "error", "-f", "concat", "-safe", "0",
             "-i", lst, "-c:v", "libx264", "-crf", "18", "-pix_fmt", "yuv420p",
             "-r", "24", "-an", out_path], capture_output=True)
        if r.returncode != 0:
            raise RuntimeError(f"master_concat failed: {r.stderr.decode()[-400:]}")
    return out_path


def assemble_master(reports: list[dict], out_path: str, workdir: str = ".",
                    required_beats: list[str] | None = None,
                    expected_clip_sha1: dict[str, str] | None = None,
                    strict: bool = False) -> dict:
    """Gate-B-PASS clips in report order -> master + manifest sidecar.

    T14 admission validation — a report enters the concat only when:
      * verdict is exactly PASS (NEEDS_REVIEW/FAIL excluded), AND
      * its clip exists on disk, AND
      * its clip sha1 (when the caller recorded one) still matches the file
        (a changed artifact invalidates the approval behind the verdict).

    Production (strict=True) REFUSES to assemble when any required beat is
    missing; partial previews (strict=False) assemble and carry a prominent
    missing_beats list plus is_complete=False.
    """
    def _sha1(path: str) -> str | None:
        try:
            import hashlib
            h = hashlib.sha1()
            with open(path, "rb") as fh:
                for chunk in iter(lambda: fh.read(1 << 16), b""):
                    h.update(chunk)
            return h.hexdigest()
        except OSError:
            return None

    admitted: list[dict] = []
    excluded: list[dict] = []
    for r in reports:
        if r["verdict"] != "PASS":
            excluded.append({"beat": r["beat"],
                             "why": f"verdict {r['verdict']}: "
                                    f"{'; '.join(r.get('fails') or r.get('coverage_gaps') or ['not PASS'])}"})
            continue
        clip = r.get("clip")
        exp = (expected_clip_sha1 or {}).get(r["beat"])
        if not clip or not os.path.exists(clip):
            excluded.append({"beat": r["beat"],
                             "why": "PASS but clip missing on disk"})
            continue
        if exp:
            actual = _sha1(clip)
            if actual != exp:
                excluded.append({"beat": r["beat"],
                                 "why": "clip sha1 changed since gate approval "
                                        f"(expected {exp[:12]}, found {(actual or '?')[:12]})"})
                continue
        admitted.append(r)

    order_ids = [r["beat"] for r in admitted]
    missing = [b for b in (required_beats or []) if b not in order_ids]
    if strict and missing:
        raise RuntimeError(
            f"assemble_master(strict): required beats missing — {missing}; "
            "partial previews must be assembled with strict=False and listed "
            "as previews, never shipped as production")

    clips = [r["clip"] for r in admitted]
    master_concat(clips, out_path, workdir)
    return {
        "master": out_path,
        "included": order_ids,
        "excluded": excluded,
        "is_complete": not missing,
        "missing_beats": missing,
        "duration_note": "concat stream-copy; inputs are lane-uniform "
                         "containers (1248x704 h264)",
    }
