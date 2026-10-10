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
