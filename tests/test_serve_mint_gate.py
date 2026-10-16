"""T15 — serve API Gate A enforcement on POST /runs/mint (fail-closed).

A keyframe path alone must NEVER be enough to mint: the request must carry a
Gate A verdict showing PASS for that beat, and the PASS must name THIS exact
frame (a PASS for a different frame is a stale report, not an approval).
The queued job's Comfy endpoint is pointed at an unreachable port so no GPU
is touched — these tests exercise the gate, not the render.
"""

import json
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

fastapi_testclient = pytest.importorskip("fastapi.testclient")


@pytest.fixture()
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("CONTINUITY_SECRET", "test-secret")
    monkeypatch.setenv("CONTINUITY_COMFY_URL", "http://127.0.0.1:9")
    monkeypatch.setenv("CONTINUITY_OUT_ROOT", str(tmp_path))
    import importlib
    import continuity.api.serve as serve
    importlib.reload(serve)  # module reads env at import
    return fastapi_testclient.TestClient(serve.app), tmp_path


DECK = {
    "charsheet": "CHARBLOCK", "style": "S4", "model": {},
    "beats": [{"id": "B1_walk", "seed": 7101, "locks": ["both face RIGHT"],
               "scene": "man walks the salt flat", "tod": "midday",
               "camera": "lateral_track", "qc": {"mode": "walk"}}],
}


def _setup(tmp_path):
    from PIL import Image
    frame = tmp_path / "B1_walk_s7101_00001_.png"
    Image.new("RGB", (64, 64)).save(str(frame))
    return str(frame)


def _report(tmp_path, verdict, frame):
    p = tmp_path / f"gate_{verdict}.json"
    p.write_text(json.dumps({"beat": "B1_walk", "verdict": verdict,
                             "frame": frame, "fails": []}),
                 encoding="utf-8")
    return str(p)


def _body(tmp_path, frame, report):
    return {"beat_id": "B1_walk", "deck": DECK, "keyframe_path": frame,
            "gate_a_report": report, "camera": "lateral_track",
            "vseed": 3001, "duration_s": 5.04}


def test_mint_without_gate_a_report_is_rejected(client):
    tc, tmp = client
    frame = _setup(tmp)
    r = tc.post("/runs/mint",
                headers={"x-agent-secret": "test-secret"},
                json={"beat_id": "B1_walk", "deck": DECK,
                      "keyframe_path": frame})
    assert r.status_code == 422  # required field missing (fail-closed)


def test_mint_with_fail_verdict_is_rejected(client):
    tc, tmp = client
    frame = _setup(tmp)
    r = tc.post("/runs/mint", headers={"x-agent-secret": "test-secret"},
                json=_body(tmp, frame, _report(tmp, "FAIL", frame)))
    assert r.status_code == 400
    assert "never mint" in r.json()["detail"]


def test_mint_with_stale_report_for_another_frame_is_rejected(client):
    tc, tmp = client
    frame = _setup(tmp)
    other = tmp / "other.png"
    other.write_bytes(b"not a frame")
    r = tc.post("/runs/mint", headers={"x-agent-secret": "test-secret"},
                json=_body(tmp, frame, _report(tmp, "PASS", str(other))))
    assert r.status_code == 400
    assert "stale report" in r.json()["detail"]


def test_mint_with_matching_pass_queues_the_run(client):
    tc, tmp = client
    frame = _setup(tmp)
    r = tc.post("/runs/mint", headers={"x-agent-secret": "test-secret"},
                json=_body(tmp, frame, _report(tmp, "PASS", frame)))
    assert r.status_code == 200, r.text
    assert "run_id" in r.json()
