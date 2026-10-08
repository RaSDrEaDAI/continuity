"""Keyframe lane — prompt assembly + API-format graph (T4).

Port target (proven driver): tmp_work/sn_oasis5/sn_o2_keys.py —
locks front-loaded, <250 T5 tokens, probe-gated, FLUX.1-dev fp8 with
CAICO-style frozen character blocks. Endpoint-agnostic: submits to whatever
Comfy API v2 surface continuity.api is pointed at.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class BeatSpec:
    """One storyboard beat — the atom of the pipeline.

    prompt_blocks carries ordered text: ``locks`` first (must land inside
    tokens 1-40), then ``charsheet`` (frozen character block), then ``scene``.
    The auditor gates the assembled prompt before it ever reaches the GPU.
    """

    id: str
    locks: list[str]
    charsheet: str
    scene: str
    camera: str          # camera grammar sentence (mint stage reuses it)
    tod: str             # time-of-day lock
    seed: int
    width: int = 1536
    height: int = 864


def build_prompt(beat: BeatSpec) -> str:
    """Assemble blocks locks-first; scene text last (first to fall off window)."""
    locks = "; ".join(beat.locks)
    return "; ".join(part for part in (locks, beat.charsheet, beat.scene, beat.tod) if part)


def build_graph(beat: BeatSpec, prompt: str, checkpoint: str = "flux1-dev-fp8.safetensors") -> dict:
    """API-format txt2img graph skeleton (T4 fills the full node graph).

    Shape mirrors the proven lane: LoadCheckpoint -> CLIPTextEncode ->
    EmptyLatentImage -> KSampler -> VAEDecode -> SaveImage, fp8 weights,
    euler/simple 22, cfg 1.0, FluxGuidance 3.5.
    """
    return {
        "3": {"class_type": "KSampler", "inputs": {"seed": beat.seed}},
        "6": {"class_type": "CLIPTextEncode", "inputs": {"text": prompt}},
        "checkpoint": checkpoint,
    }
