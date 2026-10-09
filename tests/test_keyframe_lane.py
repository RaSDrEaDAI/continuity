"""T4 keyframe lane tests — offline (no GPU, no network)."""

import json
import os

import pytest

from continuity.auditor.prompt_lint import PromptAudit
from continuity.keyframe.lane import (
    BeatSpec,
    Deck,
    ShipGateError,
    build_graph,
    build_prompt,
    find_existing,
    gate_deck,
    load_deck,
    render_deck,
)

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _deck(tmp_path, beats, charsheet="a lean nomad in an off-white galabia", style="retro anime cel"):
    path = tmp_path / "deck.json"
    path.write_text(json.dumps(
        {"charsheet": charsheet, "style": style,
         "model": {"loras": [{"name": "flux/style.safetensors", "strength": 0.9}]},
         "beats": beats}), encoding="utf-8")
    return load_deck(str(path))


def _beat(**over):
    base = dict(id="M1_01", seed=7101,
                locks=["exactly one man and one camel, no other people anywhere"],
                scene="wide salt flat, two figures small in mid-distance",
                tod="low golden dawn light", camera="lateral_track")
    base.update(over)
    return base


def test_load_deck_and_prompt_order():
    # real load from the shipped example
    deck = load_deck(os.path.join(REPO, "examples", "beats.example.json"))
    assert deck.beats and deck.beats[0].id == "M1_01"
    prompt = build_prompt(deck.beats[0], deck)
    # blocks appear in doctrine order regardless of internal clause separators
    assert prompt.index("one man and one camel") < prompt.index("galabia") < prompt.index("salt flat")


def test_ship_gate_blocks_overlong_prompt():
    d = Deck(charsheet="lean nomad", style="cel",
             beats=[BeatSpec(id="B", locks=["one man"], scene=" ".join(["word"] * 400), seed=1)])
    with pytest.raises(ShipGateError):
        gate_deck(d)


def test_gate_deck_passes_clean_deck():
    d = Deck(charsheet="lean nomad", style="cel",
             beats=[BeatSpec(id="B1", locks=["one man"], scene="wide salt flat", seed=1)])
    audits = gate_deck(d)
    assert len(audits) == 1 and audits[0].ship_ok


def test_graph_shape_matches_proven_lane():
    d = Deck(charsheet="lean nomad", style="cel",
             beats=[BeatSpec(id="B1", locks=["one man"], scene="wide salt flat", seed=7101)])
    d.model.loras = [type("L", (), {"name": "l1.safetensors", "strength": 0.9})(),
                     type("L", (), {"name": "l2.safetensors", "strength": 0.8})()]
    beat = d.beats[0]
    graph = build_graph(beat, "prompt text", d, "cont4/B1_s7101")
    types = {k: v["class_type"] for k, v in graph.items()}
    for node in ("UNETLoader", "DualCLIPLoader", "VAELoader", "EmptySD3LatentImage",
                 "CLIPTextEncode", "FluxGuidance", "ConditioningZeroOut", "KSampler",
                 "VAEDecode", "SaveImage", "LoraLoaderModelOnly"):
        assert node in types.values(), f"missing {node}"
    ks = graph["ksampler"]["inputs"]
    assert ks["seed"] == 7101 and ks["cfg"] == 1.0 and ks["steps"] == 22
    assert ks["positive"] == ["guidance", 0] and ks["negative"] == ["zeros", 0]
    assert ks["model"] == ["lora1", 0]                       # lora chain: unet->l0->l1
    assert graph["lora0"]["inputs"]["model"] == ["unet", 0]
    assert graph["lora1"]["inputs"]["model"] == ["lora0", 0]
    assert graph["save"]["inputs"]["filename_prefix"] == "cont4/B1_s7101"
    assert graph["latent"]["inputs"]["width"] == 1536 and graph["latent"]["inputs"]["height"] == 864


def test_resume_and_dry_run_never_touch_network(tmp_path):
    d = Deck(charsheet="lean nomad", style="cel",
             beats=[BeatSpec(id="B1", locks=["one man"], scene="wide salt flat", seed=7101)])
    out = tmp_path / "out"
    (out / "cont4").mkdir(parents=True)
    existing = out / "cont4" / "B1_s7101_00001_.png"
    existing.write_bytes(b"\x89PNG fake")
    # endpoint points at a dead port — resume + dry-run must never dial it
    res = render_deck(d, endpoint="http://127.0.0.1:1", out_dir=str(out),
                      prefix="cont4", dry_run=False)
    assert res["manifest"]["B1"]["file"].endswith(".png")
    assert res["manifest"]["B1"]["note"] == "resumed from disk"
    res2 = render_deck(d, endpoint="http://127.0.0.1:1", out_dir=str(tmp_path / "dry"),
                       prefix="cont4", dry_run=True)
    assert res2["manifest"]["B1"]["note"] == "dry run — not queued"
    assert os.path.exists(res["seeds_path"])


def test_find_existing_empty(tmp_path):
    d = Deck(charsheet="c", style="s", beats=[BeatSpec(id="X", locks=["l"], scene="s", seed=9)])
    assert find_existing(str(tmp_path), "cont4", d.beats[0]) is None
