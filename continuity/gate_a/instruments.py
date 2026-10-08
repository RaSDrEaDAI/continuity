"""Gate A instruments — solo vision reads with atomic, code-adjudicated verdicts.

Doctrine (burned in across the Signal Nomad lanes):

- SOLO reads only: grids and stacked pair-sheets systematically under-call
  close-scale defects (a whole wave passed grid QC with real lead ropes).
- SEGMENT-INVENTORY prompt for rope/strap reads: list every segment, mark
  (a) flat-attached vs (b) hanging FREE with unattached end; forced
  YES/NO verdict line.
- FACING COMPASS per subject ("nose/chin pointing: left/right/toward/away?");
  cross-subject consistency is COMPUTED in code from the two answers —
  never asked as one vague question.
- Scale judged against the charsheet COMPOSITED IN THE SAME IMAGE.
- Two independent passes must agree; disagreement = fail.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class Direction(str, Enum):
    LEFT = "left"
    RIGHT = "right"
    TOWARD = "toward camera"
    AWAY = "away from camera"


@dataclass
class FacingCompass:
    subject: str
    direction: Direction
    raw_answer: str


SEGMENT_INVENTORY_PROMPT = (
    "List every strap/rope/cord segment at her head, one line each, marked "
    "(a) flat-attached against face/neck or (b) hanging FREE with an "
    "unattached end. End with the verdict line: FREE-HANGING SEGMENT: YES/NO"
)


def segment_inventory(image_path: str) -> dict:
    """Rope/strap read for a keyframe (T5 wires the vision backend).

    Returns a JSON verdict: segments[], free_hanging (bool), pass (bool).
    A frame with any free-hanging segment FAILS and never reaches the mint.
    """
    raise NotImplementedError("T5: wire vision backend; prompt is fixed above")


def facing_consistency(a: FacingCompass, b: FacingCompass) -> bool:
    """Cross-subject facing check, computed in code.

    Walking-together beats require both subjects to face the same travel
    direction (LEFT==LEFT, RIGHT==RIGHT) or the composed intent (e.g. both
    TOWARD). Only atomic per-subject answers go to the model; the pair
    verdict is computed here.
    """
    return a.direction == b.direction
