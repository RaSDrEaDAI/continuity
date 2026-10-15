"""Mint lane — LTX-2.5 first-frame i2v, camera grammar per beat (T6).

Proven route: single-gen clips from a Gate-A-passed keyframe via
LTXVImgToVideo (strength 1.0), local GGUF (Q6_K distilled) on the 5090.
Static-only prompts freeze — every beat gets a camera sentence.

Graph shape proven live Oct 10 (t6_i2v_smoke): same sampler envelope as the
known-good T2V smoke (Q6_K, euler_ancestral, 9-step ManualSigmas, cfg 1.0),
with EmptyLTXVLatentVideo swapped for LoadImage -> LTXVImgToVideo and the
audio latent concat retained (silent audio track).
"""

from __future__ import annotations

import json
import os
import time
import urllib.request
import uuid
from dataclasses import dataclass, field

CAMERA_GRAMMAR = {
    "retreating_follow": "camera retreats slowly ahead of the subjects, keeping distance constant",
    "lateral_track": "camera tracks laterally at walking pace, parallel to the subjects",
    "push_in": "camera pushes in slowly, staying medium — never reaching close-up",
    "static_drift": "camera holds static with a slow breathing drift, subjects move through frame",
}

# Local GGUF parts on the 5090 box (ComfyUI 8189). All OSS-weighted.
DEFAULT_MODELS = {
    "unet": "LTX-2.5-Distilled-Q6_K.gguf",
    "clip": "gemma4-12b-with-proj-ltx-2.5-comfy-int8-convrot.safetensors",
    "video_vae": "ltx-2.5-video-vae-bf16.safetensors",
    "audio_vae": "ltx-2.5-audio-vae-bf16.safetensors",
}

# 9-step distilled schedule — identical to the proven T2V smoke.
DEFAULT_SIGMAS = "1.0, 0.99375, 0.9875, 0.98125, 0.975, 0.909375, 0.725, 0.421875, 0.0"

DEFAULT_NEGATIVE = (
    "worst quality, inconsistent motion, blurry, jittery, watermark, text overlay, "
    "deformed, low resolution"
)


@dataclass
class MintSpec:
    beat_id: str
    keyframe_path: str
    camera: str                 # key into CAMERA_GRAMMAR
    duration_s: float = 12.25
    width: int = 1248
    height: int = 704
    fps: int = 24
    seed: int = 0
    audio_path: str | None = None   # optional ambient bed, built-to-mute
    models: dict = field(default_factory=lambda: dict(DEFAULT_MODELS))

    @property
    def length(self) -> int:
        """Frame count, rounded to 8n+1 (LTX latent constraint)."""
        n = int(round(self.duration_s * self.fps))
        return max(9, ((n - 1) // 8) * 8 + 1)


def build_mint_prompt(spec: MintSpec, scene_text: str) -> str:
    """Mint prompt = scene text + camera sentence + motion locks.

    H3/LTX video prompts TOLERATE negation locks ("never stands, never
    walks") — unlike FLUX keyframe prompts, which are positive-only.
    """
    camera = CAMERA_GRAMMAR.get(spec.camera, spec.camera)
    return f"{scene_text}; {camera}; one continuous unbroken motion"


def build_i2v_graph(spec: MintSpec, prompt_text: str, upload_name: str,
                    negative_text: str = DEFAULT_NEGATIVE,
                    filename_prefix: str = "cont6_mint",
                    sigmas: str = DEFAULT_SIGMAS) -> dict:
    """API-format graph: keyframe (already uploaded) -> first-frame i2v clip.

    `upload_name` is the ComfyUI-side name returned by upload_keyframe().
    """
    m = spec.models
    length = spec.length
    return {
        "1": {"class_type": "UnetLoaderGGUF", "inputs": {"unet_name": m["unet"]}},
        "2": {"class_type": "CLIPLoader", "inputs": {"clip_name": m["clip"], "type": "ltxv", "device": "default"}},
        "3": {"class_type": "VAELoader", "inputs": {"vae_name": m["video_vae"]}},
        "4": {"class_type": "VAELoader", "inputs": {"vae_name": m["audio_vae"]}},
        "5": {"class_type": "CLIPTextEncode", "inputs": {"text": prompt_text, "clip": ["2", 0]}},
        "6": {"class_type": "CLIPTextEncode", "inputs": {"text": negative_text, "clip": ["2", 0]}},
        "7": {"class_type": "LTXVConditioning", "inputs": {"positive": ["5", 0], "negative": ["6", 0], "frame_rate": spec.fps}},
        "21": {"class_type": "LoadImage", "inputs": {"image": upload_name}},
        # LTXVImgToVideo outputs [positive COND, negative COND, latent] — the
        # guider must consume the image-conditioned pair, NOT the raw node-7 pair.
        "8": {"class_type": "LTXVImgToVideo", "inputs": {
            "positive": ["7", 0], "negative": ["7", 1], "vae": ["3", 0], "image": ["21", 0],
            "width": spec.width, "height": spec.height, "length": length,
            "batch_size": 1, "strength": 1.0}},
        "9": {"class_type": "LTXVEmptyLatentAudio", "inputs": {"frames_number": length, "frame_rate": 25, "batch_size": 1, "audio_vae": ["4", 0]}},
        "10": {"class_type": "LTXVConcatAVLatent", "inputs": {"video_latent": ["8", 2], "audio_latent": ["9", 0]}},
        "11": {"class_type": "RandomNoise", "inputs": {"noise_seed": spec.seed}},
        "12": {"class_type": "CFGGuider", "inputs": {"model": ["1", 0], "positive": ["8", 0], "negative": ["8", 1], "cfg": 1.0}},
        "13": {"class_type": "KSamplerSelect", "inputs": {"sampler_name": "euler_ancestral"}},
        "14": {"class_type": "ManualSigmas", "inputs": {"sigmas": sigmas}},
        "15": {"class_type": "SamplerCustomAdvanced", "inputs": {"noise": ["11", 0], "guider": ["12", 0], "sampler": ["13", 0], "sigmas": ["14", 0], "latent_image": ["10", 0]}},
        "16": {"class_type": "LTXVSeparateAVLatent", "inputs": {"av_latent": ["15", 0]}},
        "17": {"class_type": "VAEDecodeTiled", "inputs": {"samples": ["16", 0], "vae": ["3", 0], "tile_size": 512, "overlap": 64, "temporal_size": 128, "temporal_overlap": 32}},
        "18": {"class_type": "LTXVAudioVAEDecode", "inputs": {"samples": ["16", 1], "audio_vae": ["4", 0]}},
        "19": {"class_type": "CreateVideo", "inputs": {"images": ["17", 0], "fps": spec.fps, "audio": ["18", 0], "bit_depth": "auto"}},
        "20": {"class_type": "SaveVideo", "inputs": {"video": ["19", 0], "filename_prefix": filename_prefix, "format": "auto"}},
    }


def upload_keyframe(endpoint: str, path: str) -> str:
    """POST a keyframe PNG to ComfyUI /upload/image; returns the server-side name."""
    boundary = uuid.uuid4().hex
    with open(path, "rb") as f:
        img = f.read()
    body = (f"--{boundary}\r\nContent-Disposition: form-data; name=\"image\"; "
            f"filename=\"{os.path.basename(path)}\"\r\nContent-Type: image/png\r\n\r\n").encode() + img + \
        f"\r\n--{boundary}\r\nContent-Disposition: form-data; name=\"overwrite\"\r\n\r\ntrue\r\n--{boundary}--\r\n".encode()
    req = urllib.request.Request(
        endpoint.rstrip("/") + "/upload/image", data=body,
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"})
    with urllib.request.urlopen(req, timeout=120) as r:
        return json.loads(r.read())["name"]


def vram_free_gb(endpoint: str) -> float:
    with urllib.request.urlopen(endpoint.rstrip("/") + "/system_stats", timeout=10) as r:
        d = json.loads(r.read())
    return round(d["devices"][0]["vram_free"] / 1e9, 2)


def submit_and_wait(endpoint: str, graph: dict, client_id: str = "continuity-mint",
                    poll_s: float = 5.0, timeout_s: int = 1800) -> dict:
    """Queue one graph and block until it finishes.

    Returns {"wall_s", "vram_end_gb", "outputs": [(node, filename, subfolder, type)],
    "error": str|None}.
    """
    endpoint = endpoint.rstrip("/")
    data = json.dumps({"prompt": graph, "client_id": client_id}).encode()
    req = urllib.request.Request(endpoint + "/prompt", data=data,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=60) as r:
        prompt_id = json.loads(r.read())["prompt_id"]

    t0 = time.time()
    while time.time() - t0 < timeout_s:
        time.sleep(poll_s)
        with urllib.request.urlopen(endpoint + f"/history/{prompt_id}", timeout=30) as r:
            hist = json.loads(r.read())
        if prompt_id in hist:
            h = hist[prompt_id]
            outputs = []
            for node, out in h.get("outputs", {}).items():
                for vid in (out.get("videos") or []) + (out.get("gifs") or []) + (out.get("images") or []):
                    outputs.append((node, vid.get("filename"), vid.get("subfolder", ""), vid.get("type")))
            err = None
            if h.get("status", {}).get("status_str") == "error":
                err = json.dumps(h["status"].get("messages", []))[:1500]
            return {"wall_s": round(time.time() - t0, 1),
                    "vram_end_gb": vram_free_gb(endpoint),
                    "outputs": outputs, "error": err, "prompt_id": prompt_id}
    return {"wall_s": round(time.time() - t0, 1), "vram_end_gb": vram_free_gb(endpoint),
            "outputs": [], "error": "timeout", "prompt_id": prompt_id}


# ── Single-shot bounded mint (T12 dispatch route) ────────────────────────────
#
# One beat, one keyframe, one clip — a bounded execution unit the RasOrchestrator
# Comfy runner can invoke as `python -m continuity.cli mint ...` with a fixed
# argv and a machine-readable result file. No orchestration, no repair loops
# (that is the T15 pipeline); this is the smallest validated execution route.

# Exit codes (stable contract for the dispatcher)
EXIT_OK = 0
EXIT_USAGE = 2          # bad args / missing files — no GPU time spent
EXIT_GATE_REFUSED = 3   # Gate A says this beat's keyframe is dirty — never mint
EXIT_GPU_BUSY = 4       # VRAM floor not met — dispatcher should retry later
EXIT_RENDER_FAILED = 5  # ComfyUI error / timeout / no output

VRAM_FLOOR_GB = 8.0


def _comfy_output_root() -> str:
    """Where THIS ComfyUI writes outputs (config via env; two known roots)."""
    for env in ("CONTINUITY_COMFY_OUTPUT", "COMFY_OUTPUT"):
        p = os.environ.get(env)
        if p and os.path.isdir(p):
            return p
    for p in (r"D:\1.AI\Comfy\output", r"D:\1.AI\Comfy\ComfyUI\output"):
        if os.path.isdir(p):
            return p
    return r"D:\1.AI\Comfy\output"


def _resolve_keyframe(manifest_path: str, beat_id: str) -> tuple[str, str]:
    """Return (abs keyframe path, manifest prompt) for a beat, or raise ValueError."""
    with open(manifest_path, "r", encoding="utf-8") as fh:
        manifest = json.load(fh)
    entry = manifest.get(beat_id)
    if not isinstance(entry, dict):
        raise ValueError(
            f"beat '{beat_id}' not in manifest {manifest_path} "
            f"(have: {', '.join(list(manifest)[:12])})")
    rel = entry.get("file")
    if not rel:
        raise ValueError(f"beat '{beat_id}' has no rendered keyframe (file=null — "
                         f"render the deck first)")
    candidates = [rel] if os.path.isabs(rel) else [
        os.path.join(os.path.dirname(os.path.abspath(manifest_path)), rel),
        os.path.join(_comfy_output_root(), rel),
    ]
    for c in candidates:
        if os.path.isfile(c):
            return os.path.abspath(c), entry.get("prompt") or ""
    raise ValueError(f"keyframe for '{beat_id}' not found on disk: {rel}")


def _gate_verdict(gate_report_path: str, beat_id: str) -> str:
    """Read a Gate A report dir/JSON; return 'PASS'|'FAIL'|'NO_REPORT'.

    Accepts either the summary JSON (has 'reports'/per-beat verdicts) or a
    directory containing per-beat verdict files.
    """
    p = gate_report_path
    if os.path.isdir(p):
        import glob as _glob
        hits = _glob.glob(os.path.join(p, f"*{beat_id}*"))
        if not hits:
            return "NO_REPORT"
        p = hits[0]
    if not os.path.isfile(p):
        return "NO_REPORT"
    try:
        with open(p, "r", encoding="utf-8") as fh:
            d = json.load(fh)
    except Exception:
        return "NO_REPORT"
    if isinstance(d, dict):
        # summary form: {"verdicts": {beat: {...verdict}}} or per-beat file {verdict}
        for holder in (d.get("verdicts"), d.get("reports"), d):
            if isinstance(holder, dict) and beat_id in holder:
                v = holder[beat_id]
                v = v.get("verdict") if isinstance(v, dict) else v
                return str(v).upper() if v else "NO_REPORT"
        if d.get("beat") == beat_id and d.get("verdict"):
            return str(d["verdict"]).upper()
    return "NO_REPORT"


def mint_one(manifest_path: str, beats_path: str, beat_id: str, seed: int,
             endpoint: str = "http://127.0.0.1:8189", out_dir: str = ".",
             duration_s: float | None = None, width: int | None = None,
             height: int | None = None, gate_a_report: str | None = None,
             vram_floor_gb: float = VRAM_FLOOR_GB,
             _upload=None, _submit=None) -> dict:
    """Mint ONE clip from a manifest keyframe. Bounded, single-shot, honest.

    Returns a result dict (also the shape written by the CLI to --result-json):
      {ok, exit_code, beat, seed, reason?, clip?, comfy_clip?, wall_s?,
       vram_end_gb?, frames?, width?, height?, camera?, prompt?, gate_a?}
    Never raises for expected failures — the caller (CLI/dispatcher) checks
    result["ok"] and result["exit_code"].
    """
    from continuity.keyframe.lane import load_deck

    def _res(ok: bool, code: int, **kw) -> dict:
        return {"ok": ok, "exit_code": code, "beat": beat_id, "seed": seed, **kw}

    # 1. Resolve inputs (no GPU touched before this succeeds)
    try:
        deck = load_deck(beats_path)
    except Exception as e:  # noqa: BLE001
        return _res(False, EXIT_USAGE, reason=f"deck load failed: {e}")
    beat = next((b for b in deck.beats if b.id == beat_id), None)
    if beat is None:
        return _res(False, EXIT_USAGE,
                    reason=f"beat '{beat_id}' not in deck {beats_path}")
    try:
        keyframe, _kf_prompt = _resolve_keyframe(manifest_path, beat_id)
    except ValueError as e:
        return _res(False, EXIT_USAGE, reason=str(e))

    # 2. Gate A enforcement (opt-in via --gate-a-report; FAIL never mints)
    gate_state = "not_enforced"
    if gate_a_report:
        verdict = _gate_verdict(gate_a_report, beat_id)
        if verdict == "FAIL":
            return _res(False, EXIT_GATE_REFUSED,
                        reason=f"Gate A verdict FAIL for '{beat_id}' — dirty frames never mint",
                        gate_a="enforced-FAIL")
        gate_state = f"enforced-{verdict}"

    # 3. Spec + prompt (camera grammar mandatory — static-only prompts freeze)
    camera = beat.camera or "static_drift"
    scene_text = "; ".join([*(beat.locks or []), beat.scene or "",
                            beat.tod or ""]).strip(" ;")
    spec = MintSpec(beat_id=beat_id, keyframe_path=keyframe, camera=camera,
                    duration_s=duration_s if duration_s else 12.25,
                    width=width or 1248, height=height or 704,
                    seed=int(seed))
    prompt = build_mint_prompt(spec, scene_text)

    upload = _upload or upload_keyframe
    submit = _submit or submit_and_wait
    endpoint = endpoint.rstrip("/")

    # 4. VRAM floor — a busy GPU returns GPU_BUSY, not a 30-min hang
    try:
        free = vram_free_gb(endpoint)
    except Exception as e:  # noqa: BLE001
        return _res(False, EXIT_RENDER_FAILED,
                    reason=f"ComfyUI unreachable at {endpoint}: {e}")
    if free < vram_floor_gb:
        return _res(False, EXIT_GPU_BUSY,
                    reason=f"vram_free {free}GB < floor {vram_floor_gb}GB at {endpoint}",
                    vram_free_gb=free)

    # 5. Upload keyframe -> build graph -> submit
    try:
        upload_name = upload(endpoint, keyframe)
    except Exception as e:  # noqa: BLE001
        return _res(False, EXIT_RENDER_FAILED, reason=f"keyframe upload failed: {e}")
    prefix = f"cont_mint_{beat_id}"
    graph = build_i2v_graph(spec, prompt, upload_name, filename_prefix=prefix)
    r = submit(endpoint, graph)
    if r.get("error"):
        err_txt = str(r["error"])[:400]
        return _res(False, EXIT_RENDER_FAILED,
                    reason=f"render error: {err_txt}",
                    wall_s=r.get("wall_s"), vram_end_gb=r.get("vram_end_gb"))

    # 6. Locate the clip (ComfyUI writes under its own output root)
    clip_rel = None
    for _node, fname, sub, _t in r.get("outputs", []):
        if fname and str(fname).lower().endswith((".mp4", ".webm", ".mkv")):
            clip_rel = os.path.join(sub, fname) if sub else fname
            break
    if not clip_rel:
        return _res(False, EXIT_RENDER_FAILED,
                    reason="render finished but no video output found in history",
                    wall_s=r.get("wall_s"))
    root = _comfy_output_root()
    comfy_clip = clip_rel if os.path.isabs(clip_rel) else os.path.join(root, clip_rel)
    if not os.path.isfile(comfy_clip):
        # ComfyUI output-root mismatch — recorded, surfaced, not guessed
        comfy_clip = os.path.join(r"D:\1.AI\Comfy\ComfyUI\output", clip_rel)
        if not os.path.isfile(comfy_clip):
            return _res(False, EXIT_RENDER_FAILED,
                        reason=f"clip not on disk under output roots: {clip_rel}",
                        wall_s=r.get("wall_s"))

    os.makedirs(out_dir, exist_ok=True)
    clip_out = os.path.abspath(
        os.path.join(out_dir, f"{prefix}_s{seed}.mp4"))
    import shutil
    shutil.copyfile(comfy_clip, clip_out)
    return _res(True, EXIT_OK, clip=clip_out, comfy_clip=os.path.abspath(comfy_clip),
                wall_s=r.get("wall_s"), vram_end_gb=r.get("vram_end_gb"),
                frames=spec.length, width=spec.width, height=spec.height,
                camera=camera, prompt=prompt, gate_a=gate_state,
                endpoint=endpoint, vram_start_gb=free)
