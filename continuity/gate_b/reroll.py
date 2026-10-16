"""Unattended re-roll loop — hardened seed+100 re-mints (T7).

Gate B failures route to one hardened re-mint per the SN repair doctrine:
the grammar is chosen from the MEASURED failure class, never a blind
re-roll. Sudden-jump drift gets ONE re-mint only after the step frame is
inspected (taste calls are not re-mintable).

DRIFT BASIN EXCEPTION (binding, roll v5 lesson): low-gold / golden-hour
beats get max ONE hardened re-mint — the sun-lock grammar that fixes
day-locked beats makes golden-hour ones WORSE. After one failed re-mint
they go to the operator as a taste call, never a third mint.
"""

from __future__ import annotations

from dataclasses import dataclass

SEED_STEP = 100

# Hardened grammar per failure class — measured fixes from the SN lanes.
HARDENED = {
    "gradual_drift": (
        "the sun does not sink, does not set, does not move; the sky "
        "brightness stays exactly constant from the first frame to the last"
    ),
    "sudden_jump": (
        "lighting stays continuous, no cuts, no flash, no sudden change in "
        "brightness at any moment"
    ),
    "facing_flip": (
        "both travelers hold their heading for the entire shot: they keep "
        "facing and moving the same direction from the first frame to the "
        "last, heads never turn around"
    ),
    "gaze_camera": (
        "his head never turns toward the camera; eyes stay fixed forward on "
        "the line of travel for the whole shot"
    ),
    "cast_deviation": (
        "only the two travelers exist in the world of this shot — no second "
        "person, no second camel, no other figure ever appears or fades in"
    ),
    "frozen": (
        "steady continuous forward motion through the whole shot, the scene "
        "is never frozen or still"
    ),
    "ffd": (
        "the opening frame is the picture itself; the scene begins exactly "
        "here and evolves smoothly"
    ),
}

# Beats whose TOD is golden-hour: one re-mint max, then operator taste call.
TOD_ONE_SHOT_CLASSES = ("low_gold", "golden_hour")


@dataclass
class RerollPlan:
    beat_id: str
    from_seed: int
    to_seed: int
    grammar: list[str]
    reason: str
    allowed: bool = True
    note: str = ""

    def suffix(self) -> str:
        return "; ".join(self.grammar)


def _strip_gate_prefix(f: str) -> str:
    """Gate B fails arrive prefixed ('consistency: cast_man: ...'); classify on
    the inner text (T14: prefixed cast failures were falling through)."""
    return f.split(": ", 1)[1] if f.startswith("consistency: ") else f


def classify_fails(report: dict) -> list[str]:
    """Map a Gate B report's fail strings to HARDENED grammar classes."""
    classes: list[str] = []
    for f in report.get("fails", []):
        inner = _strip_gate_prefix(f)
        if inner.startswith("drift:"):
            fc = report.get("drift", {}).get("fail_class", "gradual_drift")
            classes.append(fc if fc else "gradual_drift")
        elif "FROZEN" in inner:
            classes.append("frozen")
        elif inner.startswith("first-frame drift"):
            classes.append("ffd")
        elif "FLIP mid-clip" in inner or "unusable" in inner:
            classes.append("facing_flip")
        elif "faces camera" in inner:
            classes.append("gaze_camera")
        elif inner.startswith("cast_") or "cast" in inner.split(":")[0]:
            classes.append("cast_deviation")
    return classes


def unclassified_fails(report: dict) -> list[str]:
    """Fail strings that map to NO known repair grammar (T14: unknown failures
    must never be blind-repaired as gradual_drift)."""
    known = classify_fails(report)
    out: list[str] = []
    for f in report.get("fails", []):
        inner = _strip_gate_prefix(f)
        matched = (
            inner.startswith("drift:") or "FROZEN" in inner
            or inner.startswith("first-frame drift")
            or "FLIP mid-clip" in inner or "unusable" in inner
            or "faces camera" in inner
            or inner.startswith("cast_") or "cast" in inner.split(":")[0])
        if not matched:
            out.append(f)
    # if drift failed but carried NO class, that is unknown too
    if any(_strip_gate_prefix(f).startswith("drift:") for f in report.get("fails", [])) \
            and not any(c in ("gradual_drift", "sudden_jump") for c in known):
        out.append("drift without a classified fail_class")
    return out


def plan_reroll(report: dict, seed: int, attempt: int = 1,
                tod_class: str = "day",
                reviewed_sudden_jump: bool = False) -> RerollPlan:
    """Build the hardened re-mint plan for a failed mint.

    attempt counts prior re-mints of this beat (1 = first repair).

    T14 fail-closed rules:
    * sudden_jump is NEVER auto-repaired — the human-review stop persists
      across every round and every resume until an explicit operator decision
      (``reviewed_sudden_jump=True``) is recorded by the caller.
    * unknown failure classes get no grammar and no repair — operator review,
      never a blind gradual_drift re-mint.
    """
    unknowns = unclassified_fails(report)
    classes = classify_fails(report)
    if unknowns:
        return RerollPlan(
            beat_id=report.get("beat", "?"), from_seed=seed,
            to_seed=seed + SEED_STEP, grammar=[],
            reason="unknown: " + "; ".join(unknowns)[:200],
            allowed=False,
            note=("unknown failure class — operator review required, "
                  "no blind repair (T14)"))
    if not classes:
        return RerollPlan(
            beat_id=report.get("beat", "?"), from_seed=seed,
            to_seed=seed + SEED_STEP, grammar=[],
            reason="no classified failures",
            allowed=False,
            note="no repair class matched — operator review (T14)")
    grammar = [HARDENED[c] for c in dict.fromkeys(classes)]

    allowed = True
    note = ""
    if "sudden_jump" in classes and not reviewed_sudden_jump:
        # taste call at EVERY attempt — round 2+ may not mint uninspected
        allowed = False
        note = ("inspect the step frame first — may be a scene-motivated "
                "lighting event (taste call, not a re-mint); this stop "
                "persists across rounds and resumes until an explicit "
                "operator review decision (T14)")
    elif tod_class in TOD_ONE_SHOT_CLASSES and attempt >= 2:
        allowed = False
        note = ("golden-hour drift basin: one hardened re-mint spent; "
                "operator taste call with v1/v2 comparison sheet")
    elif attempt >= 3:
        allowed = False
        note = "three strikes — escalate comp redesign to the operator"

    return RerollPlan(
        beat_id=report.get("beat", "?"),
        from_seed=seed,
        to_seed=seed + SEED_STEP,
        grammar=grammar,
        reason=", ".join(classes),
        allowed=allowed,
        note=note,
    )


def reroll_loop(mint_fn, gate_fn, reports: dict[str, dict], seeds: dict[str, int],
                tod_class: dict[str, str] | None = None,
                max_rounds: int = 2, on_plan=None) -> dict[str, dict]:
    """Unattended repair driver.

    mint_fn(beat_id, seed, prompt_suffix) -> clip_path
    gate_fn(beat_id, clip_path) -> gate_b report dict
    reports: beat -> latest gate report (FAILs to repair)
    seeds:   beat -> last used video seed
    Returns updated beat -> report map (PASSes included verbatim).
    """
    tod_class = tod_class or {}
    out = dict(reports)
    for rnd in range(1, max_rounds + 1):
        todo = [b for b, r in out.items() if r["verdict"] != "PASS"]
        if not todo:
            break
        for beat in todo:
            attempt = rnd
            plan = plan_reroll(out[beat], seeds.get(beat, 0), attempt=attempt,
                               tod_class=tod_class.get(beat, "day"))
            if on_plan:
                on_plan(beat, plan)
            if not plan.allowed:
                out[beat]["reroll"] = {"allowed": False, "note": plan.note,
                                       "reason": plan.reason}
                continue
            clip = mint_fn(beat, plan.to_seed, plan.suffix())
            out[beat] = gate_fn(beat, clip)
            out[beat]["reroll"] = {
                "allowed": True, "from_seed": plan.from_seed,
                "to_seed": plan.to_seed, "reason": plan.reason,
                "round": rnd}
            seeds[beat] = plan.to_seed
    return out
