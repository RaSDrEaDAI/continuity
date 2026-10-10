"""Gate A instruments — solo vision reads with atomic, code-adjudicated verdicts.

Doctrine (burned in across the Signal Nomad lanes, Sep-Oct 2026):

- SOLO reads only: grids and stacked pair-sheets systematically under-call
  close-scale defects (a whole 8-frame wave passed grid QC with real lead
  ropes hanging off every halter — roll v5, Oct 7).
- SEGMENT-INVENTORY prompt for rope/strap reads: list every segment, mark
  (a) flat-attached vs (b) hanging FREE with unattached end; forced
  YES/NO verdict line. Verdict line and segment list must agree — a read
  that contradicts itself is an instrument failure, not a pass.
- FACING COMPASS per subject ("nose/chin pointing: left/right/toward/away?");
  cross-subject consistency is COMPUTED in code from the two atomic answers —
  never asked as one vague question ("advancing together?" flip-flops).
- Scale judged against the charsheet COMPOSITED IN THE SAME IMAGE (relative
  judgment catches pony-camels; absolute size questions don't).
- Two independent passes must AGREE; disagreement = fail (unreliable read).
- Only ATOMIC tokens are stable across vision passes; derived questions
  flip-flop — adjudicate in code from tokens only.

Prompt versions are frozen in-repo (challenge requirement: instrument
prompts versioned). Bump INSTRUMENT_VERSION when a prompt text changes.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum

INSTRUMENT_VERSION = "gate-a/1"

# ---------------------------------------------------------------------------
# Prompts (frozen — versioned above)
# ---------------------------------------------------------------------------

SEGMENT_INVENTORY_PROMPT = (
    "List every strap/rope/cord segment at the camel's head, one line each, "
    "marked (a) flat-attached against face/neck or (b) hanging FREE with an "
    "unattached end. Also list any rope, cord or tether anywhere else in the "
    "frame connecting the man to the camel. End with the verdict line: "
    "FREE-HANGING SEGMENT: YES/NO"
)

FACING_PROMPT = (
    "Look at the {subject}. Which way does the {subject}'s nose and face "
    "point relative to the camera? Answer with EXACTLY one token from this "
    "list, nothing else: LEFT, RIGHT, TOWARD, AWAY, NV "
    "(NV = subject not visible)."
)

CAST_PROMPT = (
    "Count every distinct {kind} fully or partially visible in this image. "
    "Answer with EXACTLY one line in the form: COUNT: <integer>"
)

SCALE_PROMPT = (
    "The LEFT image is a character reference sheet showing the camel's true "
    "proportions. The RIGHT image is a scene frame from the same world. "
    "Judge the camel in the scene against the reference: does it have correct "
    "dromedary proportions (single hump, long legs, correct head-to-body "
    "ratio)? Answer with EXACTLY one token from: OK, SMALL, PONY, NV "
    "(SMALL = undersized but camel-shaped; PONY = pony-sized wrong anatomy; "
    "NV = no camel visible)."
)

GAZE_PROMPT = (
    "Look at the {subject}'s eyes. Where is the gaze directed? Answer with "
    "EXACTLY one token from this list: FORWARD, CAMERA, GROUND, SKY, SIDE, NV "
    "(NV = eyes not visible)."
)

FACING_VOCAB = {"LEFT", "RIGHT", "TOWARD", "AWAY", "NV"}
GAZE_VOCAB = {"FORWARD", "CAMERA", "GROUND", "SKY", "SIDE", "NV"}
SCALE_VOCAB = {"OK", "SMALL", "PONY", "NV"}

# Oblique pairs that still read as "walking together" (from the proven SN
# stage-B adjudicator): AWAY+R etc. are oblique-away walking, not a split.
_OBLIQUE_OK = {
    ("AWAY", "RIGHT"), ("RIGHT", "AWAY"),
    ("AWAY", "LEFT"), ("LEFT", "AWAY"),
    ("TOWARD", "RIGHT"), ("RIGHT", "TOWARD"),
    ("TOWARD", "LEFT"), ("LEFT", "TOWARD"),
}


class Direction(str, Enum):
    LEFT = "LEFT"
    RIGHT = "RIGHT"
    TOWARD = "TOWARD"
    AWAY = "AWAY"
    NV = "NV"


@dataclass
class FacingCompass:
    subject: str
    direction: Direction
    raw_answer: str


# ---------------------------------------------------------------------------
# Parsers — free functions, pure, unit-tested. A parse that yields nothing
# usable returns None; the gate treats None as an instrument failure.
# ---------------------------------------------------------------------------

_VERDICT_RE = re.compile(r"FREE-HANGING\s+SEGMENT\s*:\s*(YES|NO)", re.IGNORECASE)
_SEGMENT_RE = re.compile(r"\(b\)|hanging\s+free|free[- ]hanging|unattached", re.IGNORECASE)


def parse_segment_inventory(text: str) -> dict | None:
    """Parse a segment-inventory answer.

    Returns ``{"segments_a": int, "segments_b": int, "free_hanging": bool,
    "verdict_line": str}`` or None when the answer carries no verdict line
    (unusable read). The adjudicated ``free_hanging`` is True when EITHER the
    verdict line says YES OR any (b) segment is listed — a self-contradictory
    read is treated as dirty until proven clean.
    """
    if not text:
        return None
    m = _VERDICT_RE.search(text)
    if not m:
        return None
    verdict_yes = m.group(1).upper() == "YES"
    lines = [ln for ln in text.splitlines()
             if ln.strip() and not _VERDICT_RE.search(ln)]
    seg_b = sum(1 for ln in lines if _SEGMENT_RE.search(ln))
    seg_a = sum(
        1 for ln in lines
        if re.search(r"\(a\)|flat[- ]attached", ln, re.IGNORECASE)
    )
    return {
        "segments_a": seg_a,
        "segments_b": seg_b,
        "verdict_line": m.group(1).upper(),
        "free_hanging": verdict_yes or seg_b > 0,
    }


def parse_token(text: str, vocab: set[str]) -> str | None:
    """Extract exactly one vocab token from a short atomic answer.

    Multiple different tokens = ambiguous -> None (fail). Case-insensitive;
    whitespace-padded answers are the norm.
    """
    if not text:
        return None
    found = {tok for tok in vocab if re.search(rf"\b{tok}\b", text, re.IGNORECASE)}
    if len(found) != 1:
        return None
    return found.pop()


def parse_facing(text: str) -> str | None:
    return parse_token(text, FACING_VOCAB)


def parse_gaze(text: str) -> str | None:
    return parse_token(text, GAZE_VOCAB)


def parse_scale(text: str) -> str | None:
    return parse_token(text, SCALE_VOCAB)


_COUNT_RE = re.compile(r"COUNT\s*:\s*(\d+)", re.IGNORECASE)


def parse_cast(text: str) -> int | None:
    if not text:
        return None
    m = _COUNT_RE.search(text)
    if not m:
        return None
    return int(m.group(1))


# ---------------------------------------------------------------------------
# Adjudicators — pure, code-computed pair/group verdicts
# ---------------------------------------------------------------------------

def facing_consistency(a: FacingCompass, b: FacingCompass) -> bool:
    """Cross-subject facing check, computed in code.

    Walking-together beats require both subjects to face the same travel
    direction or an oblique-away/toward pair (both still walking together).
    Only atomic per-subject answers go to the model; the pair verdict is
    computed here.
    """
    da = getattr(a.direction, "value", a.direction)
    db = getattr(b.direction, "value", b.direction)
    pair = {da, db}
    if "NV" in pair:
        return True  # one subject unresolvable -> cross-check not applicable
    if da == db:
        return True
    return (da, db) in _OBLIQUE_OK


def against_expected(direction: str, expect: str) -> bool:
    """True when an atomic facing CLEARLY walks against the expected travel
    direction. Obliques (AWAY/TOWARD vs L/R) are not flagged — exact heading
    is a human taste call; only clear opposites auto-fail (SN doctrine)."""
    if expect not in ("L", "R", "TOWARD", "AWAY") or direction == "NV":
        return False
    wrong_dir = {"L": "RIGHT", "R": "LEFT", "TOWARD": "AWAY", "AWAY": "TOWARD"}[expect]
    return direction == wrong_dir
