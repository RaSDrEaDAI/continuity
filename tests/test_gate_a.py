"""Gate A offline tests — parsers, adjudicators, two-pass gate via ReplayProvider."""

import json
import os

import pytest

from continuity.gate_a.gate import GateSpec, gate_frame
from continuity.gate_a.instruments import (
    FacingCompass,
    against_expected,
    facing_consistency,
    parse_cast,
    parse_facing,
    parse_gaze,
    parse_scale,
    parse_segment_inventory,
)
from continuity.gate_a.providers import ReplayProvider

CLEAN_ROPE = (
    "(a) cheek strap, flat against face\n"
    "(a) nose band, flat against face\n"
    "(a) throat strap, flat against neck\n"
    "FREE-HANGING SEGMENT: NO"
)
DIRTY_ROPE = (
    "(a) cheek strap, flat against face\n"
    "(b) lead rope hanging free from halter, unattached end dangling\n"
    "FREE-HANGING SEGMENT: NO"  # self-contradiction -> still dirty
)
DIRTY_ROPE_HONEST = (
    "(b) lead rope hanging free, unattached end\nFREE-HANGING SEGMENT: YES"
)


# ---------------- parsers ----------------

def test_parse_segment_inventory_clean():
    r = parse_segment_inventory(CLEAN_ROPE)
    assert r["free_hanging"] is False
    assert r["verdict_line"] == "NO"
    assert r["segments_b"] == 0


def test_parse_segment_inventory_dirty_list_beats_verdict():
    """List says (b) segment but verdict line says NO -> free_hanging True.

    This is the self-contradiction rule: a read that contradicts itself is
    treated as dirty until proven clean."""
    r = parse_segment_inventory(DIRTY_ROPE)
    assert r["free_hanging"] is True


def test_parse_segment_inventory_dirty_honest():
    r = parse_segment_inventory(DIRTY_ROPE_HONEST)
    assert r["free_hanging"] is True


def test_parse_segment_inventory_no_verdict_line_is_unusable():
    assert parse_segment_inventory("(a) strap only") is None
    assert parse_segment_inventory("") is None


def test_parse_facing_atomic():
    assert parse_facing("RIGHT") == "RIGHT"
    assert parse_facing("  toward \n") == "TOWARD"
    assert parse_facing("NV") == "NV"


def test_parse_facing_ambiguous_is_none():
    assert parse_facing("LEFT and also RIGHT") is None
    assert parse_facing("the man walks") is None


def test_parse_cast():
    assert parse_cast("COUNT: 1") == 1
    assert parse_cast("count: 2") == 2
    assert parse_cast("I see one camel") is None


def test_parse_scale_and_gaze_vocab():
    assert parse_scale("PONY") == "PONY"
    assert parse_scale("OK") == "OK"
    assert parse_scale("SMALL PONY") is None
    assert parse_gaze("CAMERA") == "CAMERA"
    assert parse_gaze("looks at ground") == "GROUND"


# ---------------- adjudicators (pure) ----------------

def test_facing_consistency_same_direction():
    a, b = FacingCompass("man", "RIGHT", ""), FacingCompass("camel", "RIGHT", "")
    assert facing_consistency(a, b) is True


def test_facing_consistency_opposite_fails():
    a, b = FacingCompass("man", "LEFT", ""), FacingCompass("camel", "RIGHT", "")
    assert facing_consistency(a, b) is False


def test_facing_consistency_oblique_pairs_ok():
    for x, y in (("AWAY", "RIGHT"), ("RIGHT", "AWAY"), ("TOWARD", "LEFT")):
        a = FacingCompass("man", x, "")
        b = FacingCompass("camel", y, "")
        assert facing_consistency(a, b) is True, (x, y)


def test_against_expected_clear_opposite_only():
    assert against_expected("LEFT", "R") is True
    assert against_expected("RIGHT", "R") is False
    assert against_expected("AWAY", "R") is False   # oblique -> taste call
    assert against_expected("NV", "R") is False


def test_against_expected_toward_away_axis():
    # regression (Oct 9 live run): opp map wrongly mapped TOWARD->TOWARD,
    # condemning D8 for walking TOWARD when TOWARD was expected
    assert against_expected("TOWARD", "TOWARD") is False
    assert against_expected("AWAY", "TOWARD") is True
    assert against_expected("TOWARD", "AWAY") is True
    assert against_expected("AWAY", "AWAY") is False


# ---------------- GateSpec defaults ----------------

def test_gate_spec_mode_defaults():
    s = GateSpec({"mode": "solo_camel"})
    assert s.cast == {"man": 0, "camel": 1}
    assert s.rope_check is True
    assert s.facing_subjects == ["camel"]


def test_gate_spec_env_mode_disables_camel_instruments():
    s = GateSpec({"mode": "env"})
    assert s.rope_check is False
    assert s.scale_check is False
    assert s.facing_subjects == []


# ---------------- two-pass gate via replay ----------------

def _frame(tmp_path):
    # a tiny real image so the provider/replay path runs end to end
    from PIL import Image
    p = os.path.join(tmp_path, "f.png")
    Image.new("RGB", (32, 32), (10, 10, 10)).save(p)
    return p


def _replay(answers):
    return ReplayProvider(answers)


def test_gate_frame_clean_pass(tmp_path):
    spec = GateSpec({"mode": "walk", "expect": "R"})
    answers = {
        "B1|rope": [CLEAN_ROPE, CLEAN_ROPE],
        "B1|facing_man": ["RIGHT", "RIGHT"],
        "B1|facing_camel": ["RIGHT", "RIGHT"],
        "B1|cast_man": ["COUNT: 1", "COUNT: 1"],
        "B1|cast_camel": ["COUNT: 1", "COUNT: 1"],
        "B1|gaze": ["FORWARD", "FORWARD"],
    }
    r = gate_frame(_frame(tmp_path), spec, _replay(answers),
                   beat="B1", seed=1, passes=2)
    assert r["verdict"] == "PASS"
    assert r["fails"] == []
    assert r["agreed"] is True
    assert json.dumps(r)  # verdict JSON must serialize


def test_gate_frame_dirty_rope_condemned(tmp_path):
    spec = GateSpec({"mode": "walk"})
    answers = {
        "B1|rope": [DIRTY_ROPE_HONEST, DIRTY_ROPE_HONEST],
        "B1|facing_man": ["RIGHT", "RIGHT"],
        "B1|facing_camel": ["RIGHT", "RIGHT"],
        "B1|cast_man": ["COUNT: 1", "COUNT: 1"],
        "B1|cast_camel": ["COUNT: 1", "COUNT: 1"],
        "B1|gaze": ["FORWARD", "FORWARD"],
    }
    r = gate_frame(_frame(tmp_path), spec, _replay(answers), beat="B1")
    assert r["verdict"] == "FAIL"
    assert any("rope" in f for f in r["fails"])


def test_gate_frame_phantom_camel_condemned(tmp_path):
    spec = GateSpec({"mode": "solo_man"})
    answers = {
        "B1|cast_man": ["COUNT: 1", "COUNT: 1"],
        "B1|cast_camel": ["COUNT: 2", "COUNT: 2"],  # phantom second camel
        "B1|facing_man": ["TOWARD", "TOWARD"],
        "B1|gaze": ["FORWARD", "FORWARD"],
    }
    r = gate_frame(_frame(tmp_path), spec, _replay(answers), beat="B1")
    assert r["verdict"] == "FAIL"
    assert any("cast_camel" in f for f in r["fails"])


def test_gate_frame_pass_disagreement_fails(tmp_path):
    spec = GateSpec({"mode": "walk"})
    answers = {
        "B1|rope": [CLEAN_ROPE, CLEAN_ROPE],
        "B1|facing_man": ["RIGHT", "LEFT"],  # flip-flop across passes
        "B1|facing_camel": ["RIGHT", "RIGHT"],
        "B1|cast_man": ["COUNT: 1", "COUNT: 1"],
        "B1|cast_camel": ["COUNT: 1", "COUNT: 1"],
        "B1|gaze": ["FORWARD", "SIDE"],
    }
    r = gate_frame(_frame(tmp_path), spec, _replay(answers), beat="B1")
    assert r["verdict"] == "FAIL"
    assert any("disagree" in f for f in r["fails"])
    assert r["agreed"] is False


def test_gate_frame_cross_subject_mismatch_computed(tmp_path):
    spec = GateSpec({"mode": "walk"})
    answers = {
        "B1|rope": [CLEAN_ROPE, CLEAN_ROPE],
        "B1|facing_man": ["LEFT", "LEFT"],
        "B1|facing_camel": ["RIGHT", "RIGHT"],
        "B1|cast_man": ["COUNT: 1", "COUNT: 1"],
        "B1|cast_camel": ["COUNT: 1", "COUNT: 1"],
        "B1|gaze": ["FORWARD", "FORWARD"],
    }
    r = gate_frame(_frame(tmp_path), spec, _replay(answers), beat="B1")
    assert r["verdict"] == "FAIL"
    assert any("cross-subject" in f for f in r["fails"])


def test_gate_frame_gaze_at_camera_fails_walking_wide(tmp_path):
    spec = GateSpec({"mode": "walk"})
    answers = {
        "B1|rope": [CLEAN_ROPE, CLEAN_ROPE],
        "B1|facing_man": ["RIGHT", "RIGHT"],
        "B1|facing_camel": ["RIGHT", "RIGHT"],
        "B1|cast_man": ["COUNT: 1", "COUNT: 1"],
        "B1|cast_camel": ["COUNT: 1", "COUNT: 1"],
        "B1|gaze": ["CAMERA", "CAMERA"],
    }
    r = gate_frame(_frame(tmp_path), spec, _replay(answers), beat="B1")
    assert any("gaze" in f for f in r["fails"])


def test_gate_frame_unusable_read_fails(tmp_path):
    spec = GateSpec({"mode": "walk"})
    answers = {"default": ["the image shows travelers"]}  # no vocab tokens
    r = gate_frame(_frame(tmp_path), spec, _replay(answers), beat="B1")
    assert r["verdict"] == "FAIL"
    assert any("unusable" in f for f in r["fails"])
