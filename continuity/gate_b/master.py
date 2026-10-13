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


def assemble_master(reports: list[dict], out_path: str, workdir: str = ".") -> dict:
    """Gate-B-PASS clips in report order -> master + manifest sidecar."""
    order = [r for r in reports if r["verdict"] == "PASS"]
    clips = [r["clip"] for r in order]
    master_concat(clips, out_path, workdir)
    excluded = [
        {"beat": r["beat"], "why": "; ".join(r["fails"]) or "not PASS"}
        for r in reports if r["verdict"] != "PASS"]
    return {
        "master": out_path,
        "included": [r["beat"] for r in order],
        "excluded": excluded,
        "duration_note": "concat stream-copy; inputs are lane-uniform "
                         "containers (1248x704 h264)",
    }
