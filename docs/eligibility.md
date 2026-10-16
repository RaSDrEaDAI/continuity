# Eligibility & license matrix (T13)

Honest accounting for the Comfy Dev Platform Challenge submission. Every row
names its evidence; unknowns stay unknown — nothing here is inferred.

## Challenge requirements (sourced)

| Requirement | Status | Evidence |
|---|---|---|
| Built with Comfy Developer Platform | **YES, verified** | Official `comfy-sdk==0.5.0` drives `comfy-api-proxy==0.1.6` (Comfy API v2) in front of self-hosted ComfyUI: adapter `continuity/api/sdk_endpoint.py`, contract tests `tests/test_sdk_endpoint.py`, live redacted trace `examples/sdk_proof/trace.json` |
| Open-source license | **PARTIAL — see matrix** | Code is MIT. Model weights keep their own licenses (below); the full stack is not OSS |
| Setup detail (fresh clone, one-command demo) | in progress (T16) | README quickstart; clean-machine proof pending |
| Demo video | done (T9) | v1 62s / v2 2:09 cut — README "Demo video" |
| SFW, own IP only | YES | Signal Nomad IP (own); no third-party likenesses |
| One submission | YES | Single entry, one form (Uros's account — signup complete 2026-10-16) |
| Deadline | **PROVISIONAL Oct 19 2026 9:00 PT** | The announcement post is delisted (404, not in sitemap). Date from the recorded brief; NOT independently re-verified against a live primary source. The submission form URL is held by the operator (email/Discord, Oct 7) |
| Free first-100 credits | N/A | Operator registered; credits status not independently verified |

## Component licenses

| Component | License | Source / provenance |
|---|---|---|
| Continuity code | MIT | `LICENSE` in repo root |
| FLUX.1-dev (keyframes) | **FLUX.1 Dev Non-Commercial License** — open weights, NOT OSS, no commercial use | Black Forest Labs distribution via local fp8 checkpoint on the build box |
| CAICO style LoRA | per-base license inherited; provenance: locally trained style LoRA on this box | `loras/` on the build machine; deck `model.loras` |
| LTX-2.5 (mints, GGUF Q6_K) | Lightricks LTX-2.5 community license terms — verify at submission time | Local GGUF quant on the build box |
| T5 text encoder / tokenizer | Apache-2.0 (standard T5) | local tokenizer path |
| VAE / audio encoders | ship with their model families | local |
| comfy-api-proxy | open source (PyPI 0.1.6) | pip |
| comfy-sdk | official Comfy Python SDK (PyPI 0.5.0) | pip |
| Gate A/B vision (optional) | **proprietary hosted** (OpenRouter `anthropic/claude-haiku-5.5` or equivalent) when enabled; $0.0007–0.02 per gate run | `continuity/gate_a/providers.py`; deterministic instruments (rope/facing/cast parsers, drift scans, motion index) are LOCAL code |

## What we claim (and do not)

- **Claim**: built on the official Comfy Developer Platform surface (SDK +
  proxy, verified live), with 100% LOCAL generation on consumer hardware.
- **Claim**: MIT code; deterministic QC instruments are local code.
- **Do NOT claim**: "all-OSS stack" (FLUX.1-dev and LTX weights are not
  OSS-licensed); "fully local QC" (vision gates call a hosted provider when
  enabled — replay/deterministic modes are local); FLUX bonus eligibility
  (requires FLUX 2/3 as core model — not used).
- **Prize targets**: Most Practical + Grand + Best-OSS *(only to the extent the
  code-MIT + local-generation story qualifies — judges decide, we do not
  claim OSS-only)* + LTX VFX bonus (LTX-2.5 is the core video model).
