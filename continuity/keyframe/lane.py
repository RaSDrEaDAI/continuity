"""Keyframe lane — beats.json -> gated prompts -> API-format FLUX graphs -> frames.

Ported from the proven Signal Nomad production driver (``sn_o2_keys.py``, roll v5,
Oct 2026 — 50+ shipped frames): locks front-loaded, <250 T5 tokens, FLUX.1-dev fp8
UNET + optional style LoRAs, euler/simple 22, cfg 1.0, FluxGuidance 3.5.

The self-QC loop starts here: every assembled prompt passes the T3 auditor
(:mod:`continuity.auditor.prompt_lint`) BEFORE any GPU time is spent — a dirty
prompt raises :class:`ShipGateError` and the whole run aborts.

Endpoint-agnostic by design: submits to any Comfy API v2 surface — the local
GPU instances (8188/8189), a ``comfy-api-proxy`` (8190), or a managed Comfy API
deployment. Same wire format everywhere.
"""

from __future__ import annotations

import glob
import hashlib
import json
import os
import sys
import time
import urllib.request
import uuid
from dataclasses import dataclass, field

from continuity.auditor.prompt_lint import (
    PromptAudit,
    audit_prompt,
    audit_prompt_text,
)

DEFAULT_ENDPOINT = "http://127.0.0.1:8189"


class ShipGateError(RuntimeError):
    """A prompt failed the T3 auditor — dirty prompts never reach the GPU."""


@dataclass
class LoraSpec:
    name: str
    strength: float = 0.9

    @staticmethod
    def from_dict(d: dict) -> "LoraSpec":
        return LoraSpec(name=d["name"], strength=float(d.get("strength", 0.9)))


@dataclass
class ModelSpec:
    """FLUX lane defaults measured on RTX 5090 (~20 s/frame warm, 1536x864)."""

    unet: str = "flux1-krea-dev_fp8_scaled.safetensors"
    unet_dtype: str = "default"  # NEVER fp8_e4m3fn: loads, then crashes the sampler
    clip_l: str = "clip_l.safetensors"
    t5: str = "t5xxl_fp16.safetensors"
    vae: str = "ae.safetensors"
    loras: list[LoraSpec] = field(default_factory=list)
    steps: int = 22
    cfg: float = 1.0
    sampler: str = "euler"
    scheduler: str = "simple"
    guidance: float = 3.5
    width: int = 1536
    height: int = 864

    @staticmethod
    def from_dict(d: dict | None) -> "ModelSpec":
        m = ModelSpec()
        if not d:
            return m
        for k in ("unet", "unet_dtype", "clip_l", "t5", "vae", "sampler", "scheduler"):
            if d.get(k):
                setattr(m, k, d[k])
        for k in ("steps", "width", "height"):
            if d.get(k) is not None:
                setattr(m, k, int(d[k]))
        for k in ("cfg", "guidance"):
            if d.get(k) is not None:
                setattr(m, k, float(d[k]))
        m.loras = [LoraSpec.from_dict(x) for x in d.get("loras", [])]
        return m


@dataclass
class BeatSpec:
    """One storyboard beat — the atom of the pipeline.

    ``locks`` must land inside tokens 1-40 (auditor-enforced); ``charsheet``
    overrides the deck-level frozen character block (e.g. pack-off variant);
    ``scene`` is compressed prose; ``camera`` is carried for the mint stage
    (T6), which pairs camera grammar with the approved keyframe.
    """

    id: str
    locks: list[str]
    scene: str
    seed: int
    charsheet: str = ""
    tod: str = ""
    camera: str = ""
    width: int | None = None
    height: int | None = None
    qc: dict = field(default_factory=dict)  # Gate A spec (mode/expect/cast)

    @staticmethod
    def from_dict(d: dict) -> "BeatSpec":
        return BeatSpec(
            id=d["id"],
            locks=[str(x) for x in d.get("locks", [])],
            scene=d.get("scene", ""),
            seed=int(d["seed"]),
            charsheet=d.get("charsheet", ""),
            tod=d.get("tod", ""),
            camera=d.get("camera", ""),
            width=d.get("width"),
            height=d.get("height"),
            qc=d.get("qc", {}) or {},
        )


@dataclass
class Deck:
    charsheet: str
    style: str
    beats: list[BeatSpec]
    model: ModelSpec = field(default_factory=ModelSpec)


def load_deck(path: str) -> Deck:
    with open(path, "r", encoding="utf-8") as fh:
        raw = json.load(fh)
    beats = [BeatSpec.from_dict(b) for b in raw.get("beats", [])]
    if not beats:
        raise ValueError(f"deck {path} has no beats")
    return Deck(
        charsheet=raw.get("charsheet", ""),
        style=raw.get("style", ""),
        beats=beats,
        model=ModelSpec.from_dict(raw.get("model")),
    )


def build_prompt(beat: BeatSpec, deck: Deck) -> str:
    """Assemble blocks locks-first; scene/tod/style tail (first to fall off window)."""
    locks = "; ".join(beat.locks)
    chars = beat.charsheet or deck.charsheet
    parts = [p for p in (locks, chars, beat.scene, beat.tod, deck.style) if p]
    return "; ".join(parts)


def audit_beat(beat: BeatSpec, deck: Deck, tokenizer_path: str | None = None) -> PromptAudit:
    prompt = build_prompt(beat, deck)
    if tokenizer_path:
        return audit_prompt(prompt, tokenizer_path)
    return audit_prompt_text(prompt)


def gate_deck(deck: Deck, tokenizer_path: str | None = None) -> list[PromptAudit]:
    """Ship-gate every prompt in the deck. Raises ShipGateError on any failure."""
    audits: list[PromptAudit] = []
    for beat in deck.beats:
        audit = audit_beat(beat, deck, tokenizer_path)
        audits.append(audit)
        if not audit.ship_ok:
            raise ShipGateError(
                f"beat {beat.id} failed the ship gate "
                f"({audit.total_tokens} tokens, dead_text={bool(audit.dead_text)}):\n"
                + audit.summary()
            )
    return audits


def build_graph(beat: BeatSpec, prompt: str, deck: Deck, filename_prefix: str) -> dict:
    """Full API-format FLUX graph (11 nodes) matching the proven production lane.

    UNETLoader -> [LoraLoaderModelOnly chain] -> KSampler; DualCLIPLoader ->
    CLIPTextEncode -> FluxGuidance (positive) + ConditioningZeroOut (negative);
    EmptySD3LatentImage; VAEDecode; SaveImage with a resumable prefix.
    """
    m = deck.model
    width = beat.width or m.width
    height = beat.height or m.height
    graph: dict = {
        "unet": {"class_type": "UNETLoader",
                 "inputs": {"unet_name": m.unet, "weight_dtype": m.unet_dtype}},
        "clip": {"class_type": "DualCLIPLoader",
                 "inputs": {"clip_name1": m.clip_l, "clip_name2": m.t5, "type": "flux"}},
        "vae": {"class_type": "VAELoader", "inputs": {"vae_name": m.vae}},
        "latent": {"class_type": "EmptySD3LatentImage",
                   "inputs": {"width": width, "height": height, "batch_size": 1}},
        "pos": {"class_type": "CLIPTextEncode",
                "inputs": {"text": prompt, "clip": ["clip", 0]}},
        "guidance": {"class_type": "FluxGuidance",
                     "inputs": {"conditioning": ["pos", 0], "guidance": m.guidance}},
        "zeros": {"class_type": "ConditioningZeroOut",
                  "inputs": {"conditioning": ["pos", 0]}},
        "ksampler": {"class_type": "KSampler",
                     "inputs": {"seed": beat.seed, "steps": m.steps, "cfg": m.cfg,
                                "sampler_name": m.sampler, "scheduler": m.scheduler,
                                "denoise": 1.0, "model": ["unet", 0],
                                "positive": ["guidance", 0], "negative": ["zeros", 0],
                                "latent_image": ["latent", 0]}},
        "decode": {"class_type": "VAEDecode",
                   "inputs": {"samples": ["ksampler", 0], "vae": ["vae", 0]}},
        "save": {"class_type": "SaveImage",
                 "inputs": {"images": ["decode", 0], "filename_prefix": filename_prefix}},
    }
    prev = "unet"
    for i, lora in enumerate(m.loras):
        key = f"lora{i}"
        graph[key] = {"class_type": "LoraLoaderModelOnly",
                      "inputs": {"lora_name": lora.name, "strength_model": lora.strength,
                                 "model": [prev, 0]}}
        prev = key
    graph["ksampler"]["inputs"]["model"] = [prev, 0]
    return graph


def frame_prefix(prefix: str, beat: BeatSpec) -> str:
    return f"{prefix}/{beat.id}_s{beat.seed}"


def find_existing(out_dir: str, prefix: str, beat: BeatSpec) -> str | None:
    """Resume-safe: an already-rendered frame for this beat+seed short-circuits."""
    hits = sorted(glob.glob(os.path.join(out_dir, frame_prefix(prefix, beat) + "*.png")),
                  key=os.path.getmtime)
    return hits[-1] if hits else None


def _http_json(url: str, data=None, timeout: int = 30) -> dict:
    req = urllib.request.Request(
        url, data=json.dumps(data).encode() if data is not None else None,
        headers={"Content-Type": "application/json"} if data else {})
    return json.loads(urllib.request.urlopen(req, timeout=timeout).read())


def queue_prompt(endpoint: str, graph: dict) -> str:
    res = _http_json(f"{endpoint}/prompt",
                     {"prompt": graph, "client_id": str(uuid.uuid4())}, timeout=60)
    return res["prompt_id"]


def wait_for_prompt(endpoint: str, prompt_id: str, out_dir: str,
                    timeout: int = 900, poll_s: int = 5) -> dict:
    """Poll /history until the job lands; resolve saved images against out_dir."""
    t0 = time.time()
    while time.time() - t0 < timeout:
        time.sleep(poll_s)
        try:
            hist = _http_json(f"{endpoint}/history/{prompt_id}", timeout=30)
        except Exception:
            hist = {}
        entry = hist.get(prompt_id)
        if not entry:
            continue
        images = []
        for node_out in (entry.get("outputs") or {}).values():
            images.extend(node_out.get("images", []))
        if not images:
            return {"prompt_id": prompt_id, "files": [],
                    "status": entry.get("status", {}), "error": "no images in outputs"}
        files = []
        for img in images:
            rel = os.path.join(img.get("subfolder", ""), img["filename"])
            local = os.path.join(out_dir, rel)
            files.append({"filename": img["filename"], "local": local if os.path.exists(local) else None})
        return {"prompt_id": prompt_id, "files": files, "status": entry.get("status", {})}
    return {"prompt_id": prompt_id, "files": [], "status": {}, "error": "timeout"}


def sha1_of(path: str) -> str:
    h = hashlib.sha1()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def render_deck(deck: Deck, endpoint: str = DEFAULT_ENDPOINT, out_dir: str = ".",
                tokenizer_path: str | None = None, prefix: str = "cont4",
                limit: int | None = None, dry_run: bool = False,
                verbose: bool = True) -> dict:
    """Gate -> render (resume-safe) -> manifest + seed log. One command, whole deck.

    Returns ``{"manifest": {beat_id: {...}}, "contact_sheet": path|None,
    "manifest_path": path}``. Every prompt is audited first; any failure aborts
    the whole run before a single GPU-second is spent.
    """
    beats = deck.beats[:limit] if limit else deck.beats
    audits = {b.id: audit_beat(b, deck, tokenizer_path) for b in beats}
    bad = [b.id for b in beats if not audits[b.id].ship_ok]
    if bad:
        raise ShipGateError(f"ship gate failed for beats: {bad} — no GPU time spent")

    manifest: dict = {}
    os.makedirs(out_dir, exist_ok=True)
    for beat in beats:
        audit = audits[beat.id]
        prompt = build_prompt(beat, deck)
        prefix_full = frame_prefix(prefix, beat)
        existing = find_existing(out_dir, prefix, beat)
        entry = {
            "beat": beat.id, "seed": beat.seed, "camera": beat.camera,
            "tokens": audit.total_tokens,
            "audit_mode": audit.notes[0] if audit.notes else "",
            "prompt": prompt, "file": None, "sha1": None, "rendered": False,
        }
        if existing:
            entry.update(file=existing, sha1=sha1_of(existing), rendered=False,
                         note="resumed from disk")
        elif dry_run:
            entry.update(note="dry run — not queued")
        else:
            graph = build_graph(beat, prompt, deck, prefix_full)
            pid = queue_prompt(endpoint, graph)
            if verbose:
                print(f"[{beat.id}] queued {pid} ({audit.total_tokens} tok)", flush=True)
            res = wait_for_prompt(endpoint, pid, out_dir)
            local = next((f["local"] for f in res["files"] if f["local"]), None)
            if not local:
                raise RuntimeError(f"[{beat.id}] render failed: {json.dumps(res)[:300]}")
            entry.update(file=local, sha1=sha1_of(local), rendered=True,
                         prompt_id=pid)
            if verbose:
                print(f"[{beat.id}] -> {local}", flush=True)
        manifest[beat.id] = entry

    manifest_path = os.path.join(out_dir, f"{prefix}_manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=1)
    seeds_path = os.path.join(out_dir, f"{prefix}_seeds.log")
    with open(seeds_path, "w", encoding="utf-8") as fh:
        for beat in beats:
            fh.write(f"{beat.id}\t{beat.seed}\t{manifest[beat.id]['sha1'] or '-'}\n")

    sheet = build_contact_sheet(manifest, os.path.join(out_dir, f"{prefix}_contact.jpg"))
    return {"manifest": manifest, "manifest_path": manifest_path,
            "seeds_path": seeds_path, "contact_sheet": sheet}


def build_contact_sheet(manifest: dict, out_path: str, thumb_w: int = 512) -> str | None:
    """Labeled contact sheet (needs Pillow — optional extra ``continuity[keyframe]``)."""
    try:
        from PIL import Image, ImageDraw
    except ImportError:
        print("contact sheet skipped: Pillow not installed", file=sys.stderr)
        return None
    cells = []
    for beat_id in sorted(manifest):
        path = manifest[beat_id].get("file")
        if path and os.path.exists(path):
            try:
                cells.append((beat_id, Image.open(path).convert("RGB")))
            except Exception as exc:  # truncated/corrupt frame must not
                print(f"contact sheet: unreadable frame {path}: {exc}",  # kill the run
                      file=sys.stderr)
    if not cells:
        return None
    cols = 4 if len(cells) > 4 else len(cells)
    rows = (len(cells) + cols - 1) // cols
    th = round(thumb_w * cells[0][1].height / cells[0][1].width)
    label_h = 28
    sheet = Image.new("RGB", (cols * thumb_w, rows * (th + label_h)), (16, 16, 16))
    draw = ImageDraw.Draw(sheet)
    for i, (beat_id, img) in enumerate(cells):
        x = (i % cols) * thumb_w
        y = (i // cols) * (th + label_h)
        sheet.paste(img.resize((thumb_w, th)), (x, y))
        draw.text((x + 6, y + th + 6),
                  f"{beat_id}  s{manifest[beat_id]['seed']}", fill=(240, 240, 240))
    sheet.save(out_path, quality=88)
    return out_path
