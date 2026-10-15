"""T12 dispatch route — offline mint_one tests (no GPU, no network).

Covers the bounded single-shot mint contract the RasOrchestrator Comfy runner
invokes: usage refusals, Gate A refusal, GPU-busy refusal, and the happy path
with injected upload/submit. Exit codes are the stable dispatcher contract.
"""

import json
import os

import pytest

from continuity.mint.lane import (
    EXIT_GATE_REFUSED, EXIT_GPU_BUSY, EXIT_OK, EXIT_USAGE, mint_one,
)

DECK = {
    "charsheet": "CHARBLOCK: one man, one camel",
    "style": "S4 CAICO retro anime",
    "model": {},
    "beats": [
        {
            "id": "D4_signal",
            "seed": 7204,
            "locks": ["exactly one man, zero animals in frame"],
            "scene": "SCENE: man kneeling at a brass radio, salt flat, dry horizon",
            "tod": "late afternoon light",
            "camera": "push_in",
            "qc": {"mode": "solo_man"},
        },
        {
            "id": "D9_noframe",
            "seed": 7209,
            "locks": ["exactly one camel"],
            "scene": "SCENE: camel at distance",
            "tod": "dusk",
            "camera": "static_drift",
            "qc": {"mode": "camel"},
        },
    ],
}


@pytest.fixture()
def workdir(tmp_path, monkeypatch):
    """Deck + manifest with one real keyframe file on disk."""
    kf = tmp_path / "D4_signal_s7204_00001_.png"
    kf.write_bytes(b"\x89PNG fake keyframe bytes")
    deck_p = tmp_path / "deck.json"
    deck_p.write_text(json.dumps(DECK), encoding="utf-8")
    manifest_p = tmp_path / "manifest.json"
    manifest_p.write_text(json.dumps({
        "D4_signal": {"file": str(kf), "rendered": True, "tokens": 210},
        "D9_noframe": {"file": None, "rendered": False, "tokens": 180},
    }), encoding="utf-8")
    return tmp_path


def _patch_vram(monkeypatch, gb):
    import continuity.mint.lane as lane
    monkeypatch.setattr(lane, "vram_free_gb", lambda endpoint: gb)


def test_beat_not_in_deck_refused(workdir, monkeypatch):
    _patch_vram(monkeypatch, 30.0)
    r = mint_one(manifest_path=str(workdir / "manifest.json"),
                 beats_path=str(workdir / "deck.json"),
                 beat_id="NOPE", seed=1, out_dir=str(workdir / "o"))
    assert r["ok"] is False and r["exit_code"] == EXIT_USAGE
    assert "not in deck" in r["reason"]


def test_missing_keyframe_refused_no_gpu(workdir, monkeypatch):
    # D9 has a deck entry but manifest file=null -> usage refusal before any GPU call
    called = {"submit": 0}

    def _boom(endpoint, graph):
        called["submit"] += 1
        raise AssertionError("submit must never run")

    r = mint_one(manifest_path=str(workdir / "manifest.json"),
                 beats_path=str(workdir / "deck.json"),
                 beat_id="D9_noframe", seed=1, out_dir=str(workdir / "o"),
                 _submit=_boom)
    assert r["ok"] is False and r["exit_code"] == EXIT_USAGE
    assert called["submit"] == 0


def test_gate_a_fail_never_mints(workdir, monkeypatch):
    gate_p = workdir / "gate_a"
    gate_p.mkdir()
    (gate_p / "D4_signal.json").write_text(json.dumps({
        "beat": "D4_signal", "verdict": "FAIL",
        "fails": ["rope: free-hanging segment"]}), encoding="utf-8")
    submitted = {"n": 0}

    def _submit(endpoint, graph):
        submitted["n"] += 1
        raise AssertionError("submit must never run on a FAIL verdict")

    r = mint_one(manifest_path=str(workdir / "manifest.json"),
                 beats_path=str(workdir / "deck.json"),
                 beat_id="D4_signal", seed=7, out_dir=str(workdir / "o"),
                 gate_a_report=str(gate_p), _submit=_submit)
    assert r["ok"] is False and r["exit_code"] == EXIT_GATE_REFUSED
    assert "dirty frames never mint" in r["reason"]
    assert submitted["n"] == 0


def test_gpu_busy_refusal(workdir, monkeypatch):
    _patch_vram(monkeypatch, 3.2)
    r = mint_one(manifest_path=str(workdir / "manifest.json"),
                 beats_path=str(workdir / "deck.json"),
                 beat_id="D4_signal", seed=7, out_dir=str(workdir / "o"),
                 vram_floor_gb=8.0)
    assert r["ok"] is False and r["exit_code"] == EXIT_GPU_BUSY
    assert "vram_free" in r["reason"]


def test_happy_path_mints_and_copies_clip(workdir, monkeypatch):
    _patch_vram(monkeypatch, 30.0)
    # Fake ComfyUI: submit returns a video output written under the fake root
    comfy_out = workdir / "comfy_output"
    comfy_out.mkdir()
    monkeypatch.setenv("CONTINUITY_COMFY_OUTPUT", str(comfy_out))

    def _upload(endpoint, path):
        assert os.path.isfile(path)
        return "uploaded_name.png"

    def _submit(endpoint, graph):
        clip = comfy_out / "cont_mint_D4_signal_00001.mp4"
        clip.write_bytes(b"00 fake mp4")
        return {"wall_s": 42.0, "vram_end_gb": 21.0, "error": None,
                "outputs": [("20", clip.name, "", "output")]}

    out = workdir / "mint_out"
    r = mint_one(manifest_path=str(workdir / "manifest.json"),
                 beats_path=str(workdir / "deck.json"),
                 beat_id="D4_signal", seed=2107, out_dir=str(out),
                 duration_s=5.0, width=960, height=544,
                 _upload=_upload, _submit=_submit)
    assert r["ok"] is True and r["exit_code"] == EXIT_OK
    assert os.path.isfile(r["clip"])
    assert os.path.basename(r["clip"]) == "cont_mint_D4_signal_s2107.mp4"
    assert r["width"] == 960 and r["height"] == 544
    assert r["frames"] == 113  # 5.0s @24fps -> 120 -> rounds DOWN to 8n+1 = 113
    assert r["camera"] == "push_in"
    assert "pushes in slowly" in r["prompt"]  # camera grammar applied
    assert r["gate_a"] == "not_enforced"


def test_cli_mint_writes_result_json(workdir, monkeypatch):
    _patch_vram(monkeypatch, 30.0)
    from continuity.cli import main
    comfy_out = workdir / "comfy_output"
    comfy_out.mkdir()
    monkeypatch.setenv("CONTINUITY_COMFY_OUTPUT", str(comfy_out))

    def _upload(endpoint, path):
        return "u.png"

    def _submit(endpoint, graph):
        clip = comfy_out / "cont_mint_D4_signal_00001.mp4"
        clip.write_bytes(b"fake")
        return {"wall_s": 10.0, "vram_end_gb": 20.0, "error": None,
                "outputs": [("20", clip.name, "", "output")]}

    import continuity.mint.lane as lane
    monkeypatch.setattr(lane, "upload_keyframe", _upload)
    monkeypatch.setattr(lane, "submit_and_wait", _submit)

    rj = workdir / "res" / "result.json"
    rc = main(["mint", "--manifest", str(workdir / "manifest.json"),
               "--beats", str(workdir / "deck.json"),
               "--beat", "D4_signal", "--seed", "55",
               "--duration", "5", "--width", "960", "--height", "544",
               "--out", str(workdir / "cli_out"),
               "--result-json", str(rj)])
    assert rc == 0
    written = json.loads(rj.read_text(encoding="utf-8"))
    assert written["ok"] is True and written["beat"] == "D4_signal"
    assert written["seed"] == 55
