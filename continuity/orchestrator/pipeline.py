"""Pipeline orchestration (T-stage wiring).

    load beats + charsheet
      -> for each beat: audit (ship gate) -> keyframe -> gate A
         (fail: seed ladder re-roll) -> mint -> gate B
         (fail: hardened re-mint seed+100)
      -> ordered master concat + per-shot QC report

Re-roll discipline: keyframe repairs cap at 2 strikes, then the COMP changes
(operator escalation), never a third text patch. Mint failures re-roll at
seed+100 with hardened prompt text.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class PipelineConfig:
    beats_path: str
    charsheet_path: str | None = None
    dry_run: bool = False
    max_keyframe_strikes: int = 2
    mint_seed_step: int = 100


def run(config: PipelineConfig) -> dict:
    """Full pipeline run; returns the per-shot QC report (wired with T4-T7)."""
    raise NotImplementedError("lands with T4-T7 modules")
