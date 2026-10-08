"""Mint lane — LTX-2.5 first-frame i2v, camera grammar per beat (T6).

Proven route: single-gen 12.25s mints @1248x704 from the approved keyframe
(Ref2VA conditioning), local GGUF on the 5090. Static-only prompts freeze —
every beat gets a camera sentence.
"""

from __future__ import annotations

from dataclasses import dataclass

CAMERA_GRAMMAR = {
    "retreating_follow": "camera retreats slowly ahead of the subjects, keeping distance constant",
    "lateral_track": "camera tracks laterally at walking pace, parallel to the subjects",
    "push_in": "camera pushes in slowly, staying medium — never reaching close-up",
    "static_drift": "camera holds static with a slow breathing drift, subjects move through frame",
}


@dataclass
class MintSpec:
    beat_id: str
    keyframe_path: str
    camera: str                 # key into CAMERA_GRAMMAR
    duration_s: float = 12.25
    width: int = 1248
    height: int = 704
    seed: int = 0
    audio_path: str | None = None   # optional ambient bed, built-to-mute


def build_mint_prompt(spec: MintSpec, scene_text: str) -> str:
    """Mint prompt = scene text + camera sentence + motion locks.

    H3/LTX video prompts TOLERATE negation locks ("never stands, never
    walks") — unlike FLUX keyframe prompts, which are positive-only.
    """
    camera = CAMERA_GRAMMAR.get(spec.camera, spec.camera)
    return f"{scene_text}; {camera}; one continuous unbroken motion"
