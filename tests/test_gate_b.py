"""Gate B tests — pure adjudication, reroll routing, consistency, master."""
import json
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from continuity.gate_b.drift_scan import adjudicate, DriftReport
from continuity.gate_b.consistency import adjudicate_series
from continuity.gate_b.reroll import classify_fails, plan_reroll, reroll_loop, HARDENED
from continuity.gate_b.master import assemble_master, master_concat
from continuity.gate_b.gate import summarize, report_markdown
from continuity.gate_a.providers import ReplayProvider


# ---------- drift adjudication ----------

def test_drift_clean_pass():
    r = adjudicate("B1", [100.0, 101.0, 100.5, 102.0, 101.5, 100.0, 99.5, 100.5,
                         101.0, 100.0, 100.5, 101.0])
    assert r.passed and r.fail_class == ""
    assert "PASS" in r.summary()


def test_drift_gradual_fail():
    lums = [95 - i * 4 for i in range(12)]  # slide of 44
    r = adjudicate("B2", lums)
    assert not r.passed and r.fail_class == "gradual_drift"
    assert r.swing > 25


def test_drift_sudden_jump_class():
    lums2 = [100.0] * 5 + [86.0] * 7   # one 14-step, swing 14 <= 25
    r2 = adjudicate("B4", lums2)
    assert not r2.passed and r2.fail_class == "sudden_jump"


def test_drift_report_roundtrip():
    r = adjudicate("B", [10.0, 12.0, 10.0], hue=[30.0, 31.0])
    assert r.hue == [30.0, 31.0]
    assert r.direction_flips == 1


# ---------- consistency adjudication ----------

def test_consistency_pass_on_stable_series():
    r = adjudicate_series("D1", {"man": ["RIGHT"] * 12, "camel": ["RIGHT"] * 12},
                          {"man": (1, {i: 1 for i in range(12)}),
                           "camel": (1, {i: 1 for i in range(12)})})
    assert r.verdict == "PASS"


def test_consistency_catches_midclip_flip():
    series = ["RIGHT"] * 6 + ["LEFT"] * 6
    r = adjudicate_series("D2", {"man": series}, {})
    assert r.verdict == "FAIL"
    assert any("FLIP" in f for f in r.fails)


def test_consistency_catches_phantom_cast():
    counts = {i: 1 for i in range(12)}
    counts[7] = 2   # second camel fades in at frame 7
    r = adjudicate_series("D3", {}, {"camel": (1, counts)})
    assert r.verdict == "FAIL"
    assert any("cast_camel" in f for f in r.fails)


def test_consistency_unusable_reads_fail():
    series = ["RIGHT", None] + ["RIGHT"] * 10
    r = adjudicate_series("D4", {"man": series}, {})
    assert r.verdict == "FAIL" and any("unusable" in f for f in r.fails)
    r2 = adjudicate_series("D5", {}, {"camel": (1, {**{i: 1 for i in range(12)}, 3: -1})})
    assert r2.verdict == "FAIL" and any("unusable" in f for f in r2.fails)


def test_consistency_all_camera_fails():
    r = adjudicate_series("D6", {"man": ["CAMERA"] * 12}, {})
    assert r.verdict == "FAIL" and any("camera" in f for f in r.fails)


# ---------- reroll routing ----------

def _fail_report(fails, drift_class="gradual_drift"):
    return {"beat": "X", "verdict": "FAIL", "fails": fails,
            "drift": {"fail_class": drift_class}}


def test_classify_drift_grammar():
    classes = classify_fails(_fail_report(["drift: gradual_drift (swing 40.2)"]))
    assert classes == ["gradual_drift"]
    plan = plan_reroll(_fail_report(["drift: gradual_drift (swing 40.2)"]), 1901)
    assert plan.to_seed == 2001
    assert "does not sink" in plan.suffix()
    assert plan.allowed


def test_sudden_jump_needs_inspection_first():
    plan = plan_reroll(_fail_report(["drift: sudden_jump (maxstep 14.7)"],
                                    drift_class="sudden_jump"), 1901)
    assert not plan.allowed
    assert "inspect" in plan.note


def test_golden_hour_one_shot_only():
    rep = _fail_report(["drift: gradual_drift (swing 122.6)"])
    p1 = plan_reroll(rep, 1901, attempt=1, tod_class="golden_hour")
    assert p1.allowed
    p2 = plan_reroll(rep, 2001, attempt=2, tod_class="golden_hour")
    assert not p2.allowed and "basin" in p2.note


def test_facing_flip_gets_heading_lock():
    plan = plan_reroll(_fail_report(["consistency: facing_man: FLIP mid-clip"]), 1821)
    assert "heading" in plan.suffix()


def test_three_strikes_escalate():
    plan = plan_reroll(_fail_report(["drift: gradual_drift (swing 30)"]), 2101,
                       attempt=3)
    assert not plan.allowed and "escalate" in plan.note


def test_reroll_loop_repairs_then_stops():
    seed_state = {"A": 1000}
    calls = []

    def mint_fn(beat, seed, suffix):
        calls.append((beat, seed, suffix))
        return f"/tmp/{beat}_s{seed}.mp4"

    def gate_fn(beat, clip):
        # first re-mint still fails, second passes
        ok = len(calls) >= 2
        return {"beat": beat, "clip": clip, "verdict": "PASS" if ok else "FAIL",
                "fails": [] if ok else ["drift: gradual_drift (swing 40)"],
                "drift": {"fail_class": "gradual_drift"},
                "usage": {}}

    reports = {"A": _fail_report(["drift: gradual_drift (swing 40)"])}
    out = reroll_loop(mint_fn, gate_fn, reports, seed_state, max_rounds=2)
    assert out["A"]["verdict"] == "PASS"
    assert seed_state["A"] == 1200
    assert calls[0][1] == 1100 and calls[1][1] == 1200


def test_reroll_loop_respects_disallowed():
    def mint_fn(beat, seed, suffix):
        raise AssertionError("should not mint")

    def gate_fn(beat, clip):
        raise AssertionError("should not gate")

    rep = _fail_report(["drift: sudden_jump (maxstep 14.7)"], drift_class="sudden_jump")
    out = reroll_loop(mint_fn, gate_fn, {"A": rep}, {"A": 1000}, max_rounds=1)
    assert "reroll" in out["A"] and not out["A"]["reroll"]["allowed"]


# ---------- master assembly ----------

def test_master_excludes_failed(tmp_path):
    import subprocess
    # build two tiny valid mp4s (same codec/size)
    clips = []
    for i, name in enumerate(("a", "b")):
        p = str(tmp_path / f"{name}.mp4")
        subprocess.run(
            ["ffmpeg", "-y", "-loglevel", "error", "-f", "lavfi",
             "-i", f"color=c=gray:s=320x240:d=1:r=24", "-c:v", "libx264",
             "-pix_fmt", "yuv420p", p], check=True, capture_output=True)
        clips.append(p)
    reports = [
        {"beat": "A", "clip": clips[0], "verdict": "PASS", "fails": [], "usage": {}},
        {"beat": "B", "clip": clips[1], "verdict": "FAIL",
         "fails": ["drift: gradual_drift (swing 40)"], "usage": {}},
    ]
    out = assemble_master(reports, str(tmp_path / "master.mp4"), str(tmp_path))
    assert out["included"] == ["A"]
    assert out["excluded"][0]["beat"] == "B"
    assert os.path.exists(out["master"])


def test_master_missing_clip_raises(tmp_path):
    reports = [{"beat": "A", "clip": str(tmp_path / "nope.mp4"),
                "verdict": "PASS", "fails": [], "usage": {}}]
    with pytest.raises(RuntimeError):
        assemble_master(reports, str(tmp_path / "m.mp4"), str(tmp_path))


# ---------- beat resolution (live-run regression) ----------

def test_tod_scan_applicability():
    from continuity.gate_a.gate import GateSpec
    assert GateSpec({"mode": "env"}).tod_scan is False        # macro: no sky band
    assert GateSpec({"mode": "walk"}).tod_scan is True
    assert GateSpec({"mode": "env", "tod_scan": True}).tod_scan is True
    assert GateSpec({"mode": "walk", "tod_scan": False}).tod_scan is False


def test_resolve_beat_matches_seed_stem():
    from continuity.gate_b.gate import resolve_beat
    specs = {"D2_bootmacro", "D4_signal", "D1_depart"}
    # env macro must resolve to D2_bootmacro, NOT bare "D2" (the live bug)
    assert resolve_beat("D2_bootmacro_s1901_00001_.mp4", specs) == "D2_bootmacro"
    assert resolve_beat("D4_signal_s1902_00001_.mp4", specs) == "D4_signal"
    assert resolve_beat("D:/x/D1_depart_s7101_00001_.png", specs) == "D1_depart"
    assert resolve_beat("unknown_clip.mp4", specs) == "unknown_clip.mp4"


# ---------- gate summary/markdown ----------

def test_summarize_and_markdown():
    reports = [
        {"beat": "A", "verdict": "PASS", "fails": [],
         "drift": {"swing": 5.0, "max_step": 2.0, "fail_class": ""},
         "motion": {"motion_index": 6.0, "first_frame_drift": 5.9},
         "consistency": {"verdict": "PASS", "fails": []},
         "usage": {"cost_usd": 0.001}},
        {"beat": "B", "verdict": "FAIL", "fails": ["drift: gradual_drift (swing 40)"],
         "drift": {"swing": 40.0, "max_step": 6.0, "fail_class": "gradual_drift"},
         "motion": {"motion_index": 5.0, "first_frame_drift": 6.0},
         "consistency": {"verdict": "PASS", "fails": []},
         "usage": {"cost_usd": 0.001}},
    ]
    s = summarize(reports)
    assert s["passed"] == 1 and s["failed"] == ["B"] and not s["all_pass"]
    md = report_markdown(reports, s)
    assert "| A | PASS |" in md and "gradual_drift" in md


# ---------- replay provider integration (consistency scan offline) ----------

def test_consistency_scan_with_replay(tmp_path):
    # 12-frame series; frame 5 flips facing -> FAIL via ReplayProvider answers
    answers = {"default": ["RIGHT"]}
    for i in range(12):
        answers[f"D9|seq{i:02d}|facing_man"] = ["LEFT"] if i == 5 else ["RIGHT"]
    provider = ReplayProvider(answers)
    from continuity.gate_b.consistency import consistency_scan
    # fake clip: reuse a tiny real mp4
    import subprocess
    clip = str(tmp_path / "c.mp4")
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-f", "lavfi",
         "-i", "color=c=gray:s=320x240:d=13:r=24", "-c:v", "libx264",
         "-pix_fmt", "yuv420p", clip], check=True, capture_output=True)
    r = consistency_scan(clip, "D9", provider, facing_subjects=["man"],
                         cast={}, workdir=str(tmp_path / "seq"), samples=12)
    assert r.verdict == "FAIL"
    assert any("FLIP" in f for f in r.fails)
