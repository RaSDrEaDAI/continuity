"""T6 mint lane — graph construction tests (no GPU)."""

from continuity.mint.lane import (
    CAMERA_GRAMMAR, DEFAULT_MODELS, MintSpec, build_i2v_graph, build_mint_prompt,
)


def _spec(**kw):
    base = dict(beat_id="D2_bootmacro", keyframe_path="x.png", camera="static_drift")
    base.update(kw)
    return MintSpec(**base)


def test_length_rounds_to_8n_plus_1():
    assert _spec(duration_s=12.25, fps=24).length % 8 == 1   # 294 -> 297? just check form
    n = _spec(duration_s=5.0, fps=24).length
    assert n % 8 == 1 and n >= 9


def test_graph_swaps_empty_latent_for_img2video():
    g = build_i2v_graph(_spec(), "scene text", "upload_name.png")
    classes = {v["class_type"] for v in g.values()}
    assert "LTXVImgToVideo" in classes
    assert "EmptyLTXVLatentVideo" not in classes
    assert "LoadImage" in classes
    # i2v node pulls the conditioning pair and the video VAE
    i2v = [v for v in g.values() if v["class_type"] == "LTXVImgToVideo"][0]["inputs"]
    assert i2v["image"] == ["21", 0]
    assert i2v["vae"] == ["3", 0]
    assert i2v["strength"] == 1.0
    # LTXVImgToVideo returns [pos COND, neg COND, latent] — guider must use the
    # image-conditioned pair (slot 0/1), concat uses the latent (slot 2).
    guider = [v for v in g.values() if v["class_type"] == "CFGGuider"][0]["inputs"]
    assert guider["positive"] == ["8", 0] and guider["negative"] == ["8", 1]
    concat = [v for v in g.values() if v["class_type"] == "LTXVConcatAVLatent"][0]["inputs"]
    assert concat["video_latent"] == ["8", 2]
    # LoadImage must not carry UI-only keys (400s the API)
    load = g["21"]["inputs"]
    assert set(load.keys()) == {"image"}


def test_graph_sampler_envelope_matches_proven_smoke():
    g = build_i2v_graph(_spec(), "s", "u.png")
    ks = [v for v in g.values() if v["class_type"] == "KSamplerSelect"][0]["inputs"]
    assert ks["sampler_name"] == "euler_ancestral"
    cfg = [v for v in g.values() if v["class_type"] == "CFGGuider"][0]["inputs"]
    assert cfg["cfg"] == 1.0


def test_audio_latent_frame_count_follows_spec():
    spec = _spec(duration_s=5.0, fps=24)
    g = build_i2v_graph(spec, "s", "u.png")
    audio = [v for v in g.values() if v["class_type"] == "LTXVEmptyLatentAudio"][0]["inputs"]
    i2v = [v for v in g.values() if v["class_type"] == "LTXVImgToVideo"][0]["inputs"]
    assert audio["frames_number"] == i2v["length"] == spec.length


def test_default_models_are_local_gguf_parts():
    assert DEFAULT_MODELS["unet"].endswith("Q6_K.gguf")
