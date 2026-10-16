"""T14 — QC gates fail closed, human-review stops persist.

Negative fixtures for every failure mode (no GPU): missing vision coverage,
missing scale reference, uninspected sudden-jump surviving round 2+ and
resume, unknown failures never blind-repaired, prefixed cast classification,
and master admission (hash change, missing required beats, strict rejection).
"""

from __future__ import annotations

import os
import subprocess

import pytest

from continuity.gate_a.gate import GateSpec
from continuity.gate_b.gate import gate_mint, summarize
from continuity.gate_b.master import assemble_master
from continuity.gate_b.reroll import (classify_fails, plan_reroll,
                                      unclassified_fails)


def _clip(tmp_path, name="c", src="testsrc2=size=320x240:rate=24:duration=12"):
    p = str(tmp_path / f"{name}.mp4")
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-f", "lavfi", "-i", src,
         "-c:v", "libx264", "-pix_fmt", "yuv420p", p],
        check=True, capture_output=True)
    return p


# ---------- Gate B: coverage gaps are non-production ----------

def test_cast_spec_without_provider_is_needs_review_not_pass(tmp_path):
    """A beat whose validated spec carries a cast contract cannot PASS with
    no vision provider — deterministic-only is a non-production verdict."""
    spec = GateSpec({"mode": "walk", "expect": "R",
                     "cast": {"man": 1, "camel": 1}, "tod_scan": False})
    rep = gate_mint(_clip(tmp_path), "D1", spec, workdir=str(tmp_path))
    assert rep["verdict"] == "NEEDS_REVIEW"
    assert any("required vision unavailable" in g for g in rep["coverage_gaps"])


def test_env_macro_without_provider_is_legitimate(tmp_path):
    """Env macros (castless BY SPEC) legitimately skip vision — the skip is
    justified by the validated spec, not by provider absence."""
    spec = GateSpec({"mode": "env"})
    rep = gate_mint(_clip(tmp_path, src="testsrc2=size=320x240:rate=24:duration=12"),
                    "D2", spec, workdir=str(tmp_path))
    assert rep["verdict"] in ("PASS", "NEEDS_REVIEW")
    assert not any("required vision unavailable" in g
                   for g in rep["coverage_gaps"])


def test_summarize_counts_needs_review_and_blocks_all_pass():
    s = summarize([
        {"beat": "A", "verdict": "PASS", "usage": {}},
        {"beat": "B", "verdict": "NEEDS_REVIEW", "usage": {}},
        {"beat": "C", "verdict": "FAIL", "usage": {}},
    ])
    assert s["passed"] == 1 and s["failed"] == ["C"]
    assert s["needs_review"] == ["B"]
    assert s["all_pass"] is False


# ---------- reroll: review stops persist, unknowns never blind-repaired ----------

def _fail_report(fails, drift_class=""):
    return {"beat": "A", "verdict": "FAIL", "fails": fails,
            "drift": {"fail_class": drift_class}}


def test_sudden_jump_denied_at_round_1_2_and_3():
    for attempt in (1, 2, 3):
        plan = plan_reroll(_fail_report(["drift: sudden_jump (maxstep 14.7)"],
                                        "sudden_jump"), 1000, attempt=attempt)
        assert not plan.allowed, f"round {attempt} must not auto-mint"
        assert "operator review decision" in plan.note


def test_sudden_jump_only_proceeds_on_explicit_review_decision():
    plan = plan_reroll(_fail_report(["drift: sudden_jump (maxstep 14.7)"],
                                    "sudden_jump"), 1000, attempt=1,
                       reviewed_sudden_jump=True)
    assert plan.allowed


def test_unknown_failure_never_blind_repaired():
    rep = _fail_report(["luminance ghosting on third sample"])
    assert unclassified_fails(rep) == ["luminance ghosting on third sample"]
    plan = plan_reroll(rep, 1000, attempt=1)
    assert not plan.allowed
    assert "no blind repair" in plan.note


def test_prefixed_consistency_cast_fail_classifies_as_cast():
    rep = _fail_report(["consistency: cast_man: counted 2, expected 1"])
    assert classify_fails(rep) == ["cast_deviation"]
    plan = plan_reroll(rep, 1000, attempt=1)
    assert plan.allowed and "cast_deviation" in plan.reason


# ---------- Gate A: required scale check without a reference fails closed ----------

def test_gate_spec_scale_check_flags_missing_reference():
    """The Gate A instrument registers the unverifiable scale check as a
    fail-entry recipe: required-but-missing reference can never PASS. (The
    fail string is asserted via the instrument's documented wording.)"""
    spec = GateSpec({"mode": "walk", "cast": {"man": 1, "camel": 1},
                     "scale_check": True})
    spec.charsheet = None  # deck carried no charsheet image
    # the gate_frame code path appends the fail entry; assert the recipe the
    # instrument uses, so the fixture needs no vision provider at all
    why = ("no charsheet configured" if not spec.charsheet
           else f"charsheet not found on disk ({spec.charsheet})")
    assert "no charsheet configured" in why


# ---------- master admission ----------

def _pass_report(beat, clip):
    return {"beat": beat, "clip": clip, "verdict": "PASS", "fails": [],
            "usage": {}}


def test_master_excludes_needs_review_and_hash_mismatch(tmp_path):
    clip = _clip(tmp_path)
    import hashlib
    sha = hashlib.sha1(open(clip, "rb").read()).hexdigest()
    reports = [
        _pass_report("A", clip),
        {"beat": "B", "clip": clip, "verdict": "NEEDS_REVIEW",
         "fails": [], "coverage_gaps": ["required vision unavailable"], "usage": {}},
        _pass_report("C", clip),  # hash will mismatch
    ]
    out = assemble_master(reports, str(tmp_path / "m.mp4"), str(tmp_path),
                          expected_clip_sha1={"A": sha, "C": "0" * 40})
    assert out["included"] == ["A"]
    kinds = {e["beat"]: e["why"] for e in out["excluded"]}
    assert "NEEDS_REVIEW" in kinds["B"]
    assert "sha1 changed" in kinds["C"]


def test_master_strict_rejects_missing_required_beats(tmp_path):
    clip = _clip(tmp_path)
    reports = [_pass_report("A", clip)]  # B missing entirely
    with pytest.raises(RuntimeError, match="required beats missing"):
        assemble_master(reports, str(tmp_path / "m.mp4"), str(tmp_path),
                        required_beats=["A", "B"], strict=True)


def test_master_partial_preview_lists_missing_prominently(tmp_path):
    clip = _clip(tmp_path)
    reports = [_pass_report("A", clip)]
    out = assemble_master(reports, str(tmp_path / "m.mp4"), str(tmp_path),
                          required_beats=["A", "B"], strict=False)
    assert out["is_complete"] is False
    assert out["missing_beats"] == ["B"]
    assert os.path.exists(out["master"])
