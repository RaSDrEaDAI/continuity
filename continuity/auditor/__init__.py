"""T3 — PromptLint: token-audited prompts.

Doctrine (proven nightly, Signal Nomad lane): FLUX's T5 window is 256 tokens;
ComfyUI passes the full prompt untruncated with NO warning, so anything past
token 256 is silently dead text. Locks (direction/gaze/no-rope) that sit in
the tail never execute. The auditor makes the window visible.
"""

from continuity.auditor.prompt_lint import PromptAudit, audit_prompt, audit_prompt_text

__all__ = ["PromptAudit", "audit_prompt", "audit_prompt_text"]
