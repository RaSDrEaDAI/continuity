"""Skeleton tests — package imports, CLI surface, example schema, pure logic."""

import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def test_import_every_stage():
    import continuity  # noqa: F401
    import continuity.api.endpoint  # noqa: F401
    import continuity.auditor.prompt_lint  # noqa: F401
    import continuity.gate_a.instruments  # noqa: F401
    import continuity.gate_b.drift_scan  # noqa: F401
    import continuity.keyframe.lane  # noqa: F401
    import continuity.mint.lane  # noqa: F401
    import continuity.orchestrator.pipeline  # noqa: F401


def test_cli_help():
    out = subprocess.run(
        [sys.executable, "-m", "continuity.cli", "--help"], capture_output=True
    )
    assert out.returncode == 0
    assert b"audit" in out.stdout


def test_beats_example_schema():
    data = json.loads((REPO / "examples" / "beats.example.json").read_text())
    assert data["charsheet"]
    for beat in data["beats"]:
        for key in ("id", "locks", "scene", "tod", "camera", "seed"):
            assert key in beat, f"{beat.get('id')} missing {key}"
        assert beat["locks"], "locks must not be empty (locks-first doctrine)"


def test_audit_estimate_locks_first():
    from continuity.auditor.prompt_lint import audit_prompt_text
    prompt = (
        "exactly one camel and absolutely no people visible anywhere; "
        "skinny nomad in off-white galabia; wide salt flat, flat dry horizon, "
        "no water; low golden dawn light"
    )
    audit = audit_prompt_text(prompt)
    assert audit.total_tokens > 0
    assert audit.locks, "lock prefixes should register"
    assert audit.locks[0].start_token == 0, "first lock must open the prompt"


def test_drift_adjudicate_pure_logic():
    from continuity.gate_b.drift_scan import adjudicate
    clean = adjudicate("b1", [90.0, 91.0, 90.5, 91.5, 90.0, 91.0, 90.5, 91.0])
    drift = adjudicate("b2", [95.0, 90.0, 85.0, 75.0, 60.0, 52.0, 52.0, 51.0])
    jump = adjudicate("b3", [80.0, 81.0, 96.0, 81.0, 80.0, 81.0, 80.0, 80.5])
    assert clean.passed and not drift.passed and not jump.passed


def test_mint_prompt_carries_camera():
    from continuity.mint.lane import CAMERA_GRAMMAR, MintSpec, build_mint_prompt
    spec = MintSpec(beat_id="b1", keyframe_path="kf.png", camera="lateral_track")
    text = build_mint_prompt(spec, "two figures cross the salt flat")
    assert CAMERA_GRAMMAR["lateral_track"].split(",")[0] in text
