"""Continuity — charsheet + beats in, QC-gated multi-shot video out.

Pipeline: auditor (T3) -> keyframe (T4) -> gate_a (T5) -> mint (T6)
-> gate_b (T7), driven by orchestrator over the api surface (T8).
"""

__version__ = "0.1.0"

STAGES = ["auditor", "keyframe", "gate_a", "mint", "gate_b", "orchestrator", "api"]
