"""T15 pipeline tests — end-to-end wiring with fakes; zero GPU, zero network.

The pipeline's own resume/strike/reroll logic is the product here, so the
lanes it drives (_render_one, mint_one, gate_mint, assemble_master) are
monkeypatched with recording fakes; Gate A runs REAL (ReplayProvider) so the
verdict plumbing — the dirty-frame-never-mints rule — is exercised for real.
"""

import json
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import continuity.orchestrator.pipeline as pl
from continuity.orchestrator.pipeline import PipelineConfig, UsageError, run
from continuity.mint.lane import EXIT_GPU_BUSY, EXIT_OK

def _detect_tokenizer():
    """Box-local T5 tokenizer when available (exact audit coverage on the dev
    machine); None elsewhere so the pipeline tests run in estimate mode on a
    clean install (no auditor extra, no machine-local paths)."""
    p = "C:/Users/Owner/ComfyUI-Installs/ComfyUI/ComfyUI/comfy/text_encoders/t5_tokenizer"
    if not os.path.isdir(p):
        return None
    try:
        import transformers  # noqa: F401  (auditor extra)
    except ImportError:
        return None
    return p


TOK = _detect_tokenizer()

# Production-run tests exercise the real pipeline (exact-T5 audit is a
# fail-closed contract for non-dry runs). On a clean machine without the
# tokenizer + auditor extra they skip; everything else (validation, dry-run,
# offline gates) runs everywhere.
needs_tok = pytest.mark.skipif(
    TOK is None,
    reason="production-run test: needs T5 tokenizer + auditor extra "
           "(box-local path or HF-cached t5-small)")


# ---------- fixtures ----------

def _deck_dict(scene_a="man walks the salt flat, dry horizon",
               scene_b="camel rests by the pack"):
    return {
        "charsheet": "CHARBLOCK: one man one camel, teal tagelmust",
        "style": "S4 CAICO retro anime",
        "model": {"loras": [{"name": "flux/sn_caico_retro_anime.safetensors",
                             "strength": 0.9}]},
        "beats": [
            {"id": "B1_walk", "seed": 7101, "locks": ["both face RIGHT"],
             "scene": scene_a, "tod": "flat midday light",
             "camera": "lateral_track",
             # T14: no charsheet reference image exists for this synthetic
             # deck, so the scale contract is declared absent (validated
             # spec justification) instead of silently skipped.
             "qc": {"mode": "walk", "expect": "R", "scale_check": False}},
            {"id": "B2_rest", "seed": 7102, "locks": ["zero people"],
             "scene": scene_b, "tod": "afternoon",
             "camera": "static_drift",
             "qc": {"mode": "solo_camel", "scale_check": False}},
        ],
    }


@pytest.fixture
def env(tmp_path, monkeypatch):
    """Isolated CONTINUITY_COMFY_OUTPUT + out root; token deck on disk."""
    comfy_out = tmp_path / "comfy_out"
    comfy_out.mkdir()
    monkeypatch.setenv("CONTINUITY_COMFY_OUTPUT", str(comfy_out))
    return {"tmp": tmp_path, "comfy_out": comfy_out}


def _write_deck(tmp_path, deck=None, name="deck.json"):
    p = tmp_path / name
    p.write_text(json.dumps(deck or _deck_dict()), encoding="utf-8")
    return str(p)


def _replay_answers_pass():
    return {
        "B1_walk|rope": ["FREE-HANGING SEGMENT: NO"],
        "B1_walk|facing_man": ["RIGHT"],
        "B1_walk|facing_camel": ["RIGHT"],
        "B1_walk|cast_man": ["COUNT: 1"],
        "B1_walk|cast_camel": ["COUNT: 1"],
        "B1_walk|gaze": ["FORWARD"],
        "B2_rest|rope": ["FREE-HANGING SEGMENT: NO"],
        "B2_rest|facing_camel": ["AWAY"],
        "B2_rest|cast_man": ["COUNT: 0"],
        "B2_rest|cast_camel": ["COUNT: 1"],
        "B2_rest|gaze": ["FORWARD"],
    }


def _replay_file(tmp_path, answers, name="replay.json"):
    p = tmp_path / name
    p.write_text(json.dumps(answers), encoding="utf-8")
    return str(p)


class Lanes:
    """Recording fakes for the four GPU/IO lanes."""

    def __init__(self, tmp, gate_b_fail_first=False):
        self.tmp = tmp
        self.renders = []       # (beat, seed)
        self.mints = []         # kwargs dicts
        self.master = None
        self.gate_b_fail_first = gate_b_fail_first
        self._gate_b_calls = {}

    def render(self, config, deck, beat, seed, run_dir, prompt):
        self.renders.append((beat.id, seed))
        from PIL import Image
        d = os.environ["CONTINUITY_COMFY_OUTPUT"]
        sub = os.path.join(d, config.prefix)
        os.makedirs(sub, exist_ok=True)
        p = os.path.join(sub, f"{beat.id}_s{seed}_00001_.png")
        Image.new("RGB", (64, 64), (200, 160, 120)).save(p)
        return p

    def mint(self, **kw):
        self.mints.append(kw)
        clip_dir = os.path.join(kw["out_dir"])
        os.makedirs(clip_dir, exist_ok=True)
        seed = kw["seed"]
        p = os.path.join(clip_dir, f"clip_{kw['beat_id']}_s{seed}.mp4")
        with open(p, "wb") as fh:
            fh.write(b"00" * 64)  # content irrelevant: gate_b is faked
        return {"ok": True, "exit_code": EXIT_OK, "beat": kw["beat_id"],
                "seed": seed, "clip": p, "wall_s": 1.5, "frames": 121,
                "width": kw.get("width"), "height": kw.get("height"),
                "gate_a": "enforced-PASS", "reason": None}

    def gate_b(self, clip_path, beat_id, spec, seed=None, keyframe_path=None,
               provider=None, workdir=".", samples=12):
        n = self._gate_b_calls.get(beat_id, 0)
        self._gate_b_calls[beat_id] = n + 1
        verdict = "FAIL" if (self.gate_b_fail_first and n == 0) else "PASS"
        report = {"beat": beat_id, "clip": clip_path, "seed": seed,
                  "verdict": verdict,
                  "fails": ["drift: gradual_drift (swing 40.2 maxstep 9.0)"]
                  if verdict == "FAIL" else [],
                  "drift": {"swing": 8.0, "max_step": 3.0, "fail_class": ""},
                  "motion": {"motion_index": 5.0, "frozen": False,
                             "first_frame_drift": 6.0, "ffd_flag": False},
                  "consistency": "skipped: no provider",
                  "usage": {"calls": 0, "cost_usd": 0.0}}
        path = os.path.join(workdir, beat_id, "gate_b.json")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(report, fh)
        return report

    def assemble(self, reports, out_path, workdir=".", **kw):
        self.master = out_path
        order = [r for r in reports if r["verdict"] == "PASS"]
        with open(out_path, "wb") as fh:
            fh.write(b"master")
        return {"master": out_path, "included": [r["beat"] for r in order],
                "excluded": [{"beat": r["beat"], "why": "not PASS"}
                             for r in reports if r["verdict"] != "PASS"]}


@pytest.fixture
def lanes(env, monkeypatch):
    def _make(gate_b_fail_first=False):
        l = Lanes(env["tmp"], gate_b_fail_first)
        monkeypatch.setattr(pl, "_render_one", l.render)
        monkeypatch.setattr(pl, "mint_one", l.mint)
        monkeypatch.setattr(pl, "gate_mint", l.gate_b)
        monkeypatch.setattr(pl, "assemble_master", l.assemble)
        return l
    return _make


def _cfg(deck_path, tmp, **kw):
    base = dict(
        beats_path=deck_path, out_root=str(tmp / "runs"), tokenizer_path=TOK,
        gate_a_replay=_replay_file(tmp, _replay_answers_pass()),
        run_id="t1", prefix="cont15",
    )
    base.update(kw)
    return PipelineConfig(**base)


# ---------- validation ----------

def test_production_requires_exact_tokenizer(env):
    deck = _write_deck(env["tmp"])
    with pytest.raises(UsageError, match="tokenizer"):
        run(PipelineConfig(beats_path=deck, out_root=str(env["tmp"] / "r")))


def test_unknown_model_key_is_an_error_not_a_silent_drop(env):
    bad = _deck_dict()
    bad["model"]["lora"] = "flux/sn_caico.safetensors"  # the historical footgun
    deck = _write_deck(env["tmp"], bad)
    with pytest.raises(UsageError, match="unknown model keys.*lora"):
        run(_cfg(deck, env["tmp"], dry_run=True))


def test_unknown_beat_key_rejected(env):
    bad = _deck_dict()
    bad["beats"][0]["checkpoint"] = "x.safetensors"
    deck = _write_deck(env["tmp"], bad)
    with pytest.raises(UsageError, match="unknown keys.*checkpoint"):
        run(_cfg(deck, env["tmp"], dry_run=True))


def test_duplicate_beat_ids_rejected(env):
    bad = _deck_dict()
    bad["beats"][1]["id"] = bad["beats"][0]["id"]
    deck = _write_deck(env["tmp"], bad)
    with pytest.raises(UsageError, match="duplicate beat ids"):
        run(_cfg(deck, env["tmp"], dry_run=True))


# ---------- dry run ----------

def test_dry_run_touches_no_gpu_no_network(env, monkeypatch, lanes):
    def _boom(*a, **k):
        raise AssertionError("dry run must not reach the endpoint")
    monkeypatch.setattr(pl, "queue_prompt", _boom)
    monkeypatch.setattr(pl, "provider_from_env", _boom)
    deck = _write_deck(env["tmp"])
    summary = run(_cfg(deck, env["tmp"], dry_run=True, tokenizer_path=None))
    assert summary["ok"] is True
    assert all(s == "planned:dry_run" for s in summary["statuses"].values())
    state = json.load(open(summary["state_path"], encoding="utf-8"))
    assert "DRY RUN" in state["tokenizer_mode"]
    assert state["beats"]["B1_walk"]["gate_a"]["verdict"] == "skipped:dry_run"


# ---------- happy path ----------

@needs_tok
def test_full_run_passes_both_beats_and_assembles_master(env, lanes):
    l = lanes()
    deck = _write_deck(env["tmp"])
    summary = run(_cfg(deck, env["tmp"]))
    assert summary["ok"] is True, summary["statuses"]
    assert summary["statuses"] == {"B1_walk": "gate_b_pass",
                                   "B2_rest": "gate_b_pass"}
    # both beats rendered once, minted once, master from both, deck order
    assert l.renders == [("B1_walk", 7101), ("B2_rest", 7102)]
    assert [m["beat_id"] for m in l.mints] == ["B1_walk", "B2_rest"]
    assert summary["master"]["included"] == ["B1_walk", "B2_rest"]
    assert summary["master"]["declared_incomplete"] is False
    # every mint carried the Gate A report dir — dirty frames never mint
    assert all(m["gate_a_report"] for m in l.mints)
    # state persisted with hashes + checksums
    state = json.load(open(summary["state_path"], encoding="utf-8"))
    b1 = state["beats"]["B1_walk"]
    assert b1["frame"]["sha1"] and b1["clip"]["sha1"]
    assert state["fingerprint"]["model_hash"]
    assert "NOT image-identity conditioning" in state["charsheet_definition"]
    assert state["usage"]["gpu_seconds"] == 3.0  # 2 mints x 1.5s fake


@needs_tok
def test_gate_a_fail_rolls_seed_ladder_then_escalates(env, lanes):
    l = lanes()
    answers = _replay_answers_pass()
    answers["B1_walk|rope"] = [  # free-hanging rope on EVERY strike
        "(b) cheek strap hanging free with unattached end\n"
        "FREE-HANGING SEGMENT: YES"]
    deck = _write_deck(env["tmp"])
    cfg = _cfg(deck, env["tmp"],
               gate_a_replay=_replay_file(env["tmp"], answers, "bad.json"))
    summary = run(cfg)
    assert summary["statuses"]["B1_walk"] == "escalated"
    assert summary["statuses"]["B2_rest"] == "gate_b_pass"
    # 2 strikes max: base + 2 strikes = 3 renders, seeds 7101/7102/7103
    assert [("B1_walk", s) for s in (7101, 7102, 7103)] == \
        [r for r in l.renders if r[0] == "B1_walk"]
    # the dirty beat NEVER minted
    assert "B1_walk" not in [m["beat_id"] for m in l.mints]
    # escalation is a recorded human intervention + declared incomplete master
    assert any("comp wall" in n
               for n in summary["human_interventions"])
    assert summary["master"]["declared_incomplete"] is True
    assert summary["master"]["missing_beats"][0]["beat"] == "B1_walk"
    assert summary["ok"] is False


@needs_tok
def test_gate_b_fail_routes_hardened_remint_at_seed_plus_100(env, lanes):
    l = lanes(gate_b_fail_first=True)  # B1's first gate B fails (gradual drift)
    deck = _write_deck(env["tmp"])
    summary = run(_cfg(deck, env["tmp"]))
    assert summary["statuses"]["B1_walk"] == "gate_b_pass"
    seeds = [m["seed"] for m in l.mints if m["beat_id"] == "B1_walk"]
    assert seeds == [3001, 3101]  # seed+100 hardened re-mint
    kw2 = l.mints[1]
    assert "sun does not sink" in kw2["prompt_suffix"]  # drift grammar
    assert l.renders == [("B1_walk", 7101), ("B2_rest", 7102)]  # no kf re-roll
    state = json.load(open(summary["state_path"], encoding="utf-8"))
    rr = state["beats"]["B1_walk"]["rerolls"]
    assert rr and rr[0]["from_seed"] == 3001 and rr[0]["to_seed"] == 3101


@needs_tok
def test_gpu_busy_pauses_the_run_resume_safe(env, monkeypatch, lanes):
    l = lanes()
    calls = {"n": 0}
    real_mint = l.mint

    def mint_pausing(**kw):
        calls["n"] += 1
        if calls["n"] == 1:
            return {"ok": False, "exit_code": EXIT_GPU_BUSY,
                    "beat": kw["beat_id"], "seed": kw["seed"],
                    "reason": "vram_free 6.2GB < floor 8.0GB"}
        return real_mint(**kw)

    monkeypatch.setattr(pl, "mint_one", mint_pausing)
    deck = _write_deck(env["tmp"])
    summary = run(_cfg(deck, env["tmp"]))
    assert summary["paused"] is True
    assert summary["ok"] is False
    assert summary["statuses"]["B1_walk"] == "paused_gpu_busy"
    state = json.load(open(summary["state_path"], encoding="utf-8"))
    assert state["status"] == "paused_gpu_busy"


# ---------- resume ----------

@needs_tok
def test_resume_repeats_nothing_approved(env, lanes):
    l = lanes()
    deck = _write_deck(env["tmp"])
    cfg = _cfg(deck, env["tmp"])
    first = run(cfg)
    n_renders, n_mints = len(l.renders), len(l.mints)
    assert first["ok"] is True

    l2 = lanes()
    second = run(PipelineConfig(**{**cfg.__dict__,
                                   "resume_run_dir": first["run_dir"]}))
    assert second["ok"] is True
    assert second["statuses"] == first["statuses"]
    assert len(l2.renders) == 0 and len(l2.mints) == 0  # zero repeated work
    assert second["master"]["included"] == ["B1_walk", "B2_rest"]


@needs_tok
def test_resume_invalidates_on_changed_prompt(env, lanes):
    l = lanes()
    deck = _write_deck(env["tmp"])
    cfg = _cfg(deck, env["tmp"])
    first = run(cfg)

    l2 = lanes()
    changed = _deck_dict(scene_a="man walks the DUNES, red sand")  # canon change
    deck2 = _write_deck(env["tmp"], changed, name="deck2.json")
    run(PipelineConfig(**{**cfg.__dict__, "beats_path": deck2,
                          "resume_run_dir": first["run_dir"]}))
    assert len(l2.renders) == 2  # everything re-rendered: inputs changed
    state = json.load(open(first["state_path"], encoding="utf-8"))
    b1 = state["beats"]["B1_walk"]
    assert "invalidated" in b1                      # invalidation recorded
    assert b1["history"].get("status") == "gate_b_pass"  # prior run preserved


# ---------- mint_one prompt_suffix (dispatcher contract) ----------

def test_mint_one_appends_hardened_suffix(env, tmp_path, monkeypatch):
    from PIL import Image
    import continuity.mint.lane as ml
    from continuity.mint.lane import mint_one

    frame = tmp_path / "B1_walk_s7101_00001_.png"
    Image.new("RGB", (64, 64)).save(str(frame))
    comfy_root = tmp_path / "comfyroot"
    comfy_root.mkdir()
    deck = _write_deck(tmp_path, _deck_dict())
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({
        "B1_walk": {"beat": "B1_walk", "seed": 7101,
                    "file": str(frame), "prompt": "x"}}), encoding="utf-8")

    def fake_upload(endpoint, path):
        return "uploaded.png"

    def fake_submit(endpoint, graph):
        # write the file ComfyUI would have produced, under our fake root
        save = graph["20"]["inputs"]["filename_prefix"]
        out = comfy_root / f"{save}.mp4"
        out.write_bytes(b"x" * 32)
        return {"wall_s": 1.0, "vram_end_gb": 20.0, "error": None,
                "outputs": [("20", out.name, "", "output")]}

    monkeypatch.setattr(ml, "vram_free_gb", lambda e: 30.0)
    monkeypatch.setattr(ml, "_comfy_output_root", lambda: str(comfy_root))

    res = mint_one(manifest_path=str(manifest), beats_path=deck,
                   beat_id="B1_walk", seed=3001,
                   endpoint="http://x", out_dir=str(tmp_path / "clips"),
                   prompt_suffix="the sun does not sink, does not set",
                   _upload=fake_upload, _submit=fake_submit)
    assert res["ok"] is True, res.get("reason")
    assert res["prompt"].endswith("the sun does not sink, does not set")
