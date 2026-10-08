"""PromptLint — per-clause token audit, locks-front ordering, <250 ship gate.

Ships in two modes:

1. ESTIMATE (default, zero deps): whitespace-word heuristic per clause.
   Good enough for structure + ordering checks on a clean clone.
2. T5 EXACT (T3): T5TokenizerFast over each clause; token spans sum to the
   real prompt budget. Requires ``pip install continuity[auditor]`` and a
   tokenizer path (ComfyUI ships one at comfy/text_encoders/t5_tokenizer).
"""

from __future__ import annotations

from dataclasses import dataclass, field

SHIP_GATE_TOKENS = 250      # hard ship gate (window is 256, keep headroom)
WINDOW_TOKENS = 256         # FLUX T5 window — text past this is dead
LOCK_ZONE_TOKENS = 40       # direction/gaze/no-rope locks must land inside this
LOCK_PREFIXES = (
    "exactly one", "zero", "no other", "never", "both face", "both walk",
    "facing", "gaze", "neck straight", "nothing hanging", "only one",
)


@dataclass
class Clause:
    text: str
    est_tokens: int
    start_token: int
    is_lock: bool


@dataclass
class PromptAudit:
    prompt: str
    clauses: list[Clause] = field(default_factory=list)
    total_tokens: int = 0
    dead_text: str = ""
    locks: list[Clause] = field(default_factory=list)
    ship_ok: bool = False
    notes: list[str] = field(default_factory=list)

    def summary(self) -> str:
        lines = [
            f"prompt: {self.total_tokens} est tokens "
            f"(ship gate {SHIP_GATE_TOKENS}, window {WINDOW_TOKENS})",
            f"clauses: {len(self.clauses)}  locks: {len(self.locks)}",
        ]
        if self.dead_text:
            lines.append(f"DEAD TEXT past token {WINDOW_TOKENS}: {self.dead_text[:80]!r}")
        for lock in self.locks:
            flag = "ok" if lock.start_token <= LOCK_ZONE_TOKENS else "LATE"
            lines.append(f"  lock @{lock.start_token:>3} [{flag}]: {lock.text[:60]}")
        lines.append(f"SHIP: {'PASS' if self.ship_ok else 'FAIL'}")
        lines.extend(self.notes)
        return "\n".join(lines)


def _estimate_tokens(text: str) -> int:
    """Word-count heuristic. T3 replaces this with T5TokenizerFast counts."""
    return max(1, round(len(text.split()) * 1.35))


def _split_clauses(prompt: str) -> list[str]:
    parts = [p.strip() for p in prompt.replace("\n", ";").split(";") if p.strip()]
    return parts or ([prompt.strip()] if prompt.strip() else [])


def _is_lock(text: str) -> bool:
    low = text.lower()
    return any(low.startswith(p) or f" {p}" in low for p in LOCK_PREFIXES)


def audit_prompt_text(prompt: str) -> PromptAudit:
    """Zero-dependency audit: clause structure + locks-front ordering."""
    audit = PromptAudit(prompt=prompt)
    cursor = 0
    for raw in _split_clauses(prompt):
        est = _estimate_tokens(raw)
        clause = Clause(
            text=raw, est_tokens=est, start_token=cursor, is_lock=_is_lock(raw)
        )
        audit.clauses.append(clause)
        if clause.is_lock:
            audit.locks.append(clause)
        cursor += est
    audit.total_tokens = cursor
    audit.notes.append("mode: ESTIMATE (word heuristic; T5 exact lands with T3)")
    audit.ship_ok = audit.total_tokens <= SHIP_GATE_TOKENS and not audit.dead_text
    return audit


def audit_prompt(prompt: str, tokenizer_path: str | None = None) -> PromptAudit:
    """T5-exact audit (T3). Token spans per clause via T5TokenizerFast."""
    try:
        from transformers import T5TokenizerFast  # type: ignore
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "T5-exact audit needs the auditor extra: pip install continuity[auditor]"
        ) from exc
    tok = T5TokenizerFast.from_pretrained(tokenizer_path)
    audit = PromptAudit(prompt=prompt)
    audit.notes.append("mode: T5 EXACT")
    cursor = 0
    for raw in _split_clauses(prompt):
        n = len(tok.encode(raw, add_special_tokens=False))
        clause = Clause(
            text=raw, est_tokens=n, start_token=cursor, is_lock=_is_lock(raw)
        )
        audit.clauses.append(clause)
        if clause.is_lock:
            audit.locks.append(clause)
        cursor += n
    audit.total_tokens = len(tok.encode(prompt, add_special_tokens=False))
    if audit.total_tokens > WINDOW_TOKENS:
        ids = tok.encode(prompt, add_special_tokens=False)
        audit.dead_text = tok.decode(ids[WINDOW_TOKENS:])
    audit.ship_ok = audit.total_tokens <= SHIP_GATE_TOKENS and not audit.dead_text
    return audit
