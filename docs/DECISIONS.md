# Continuity — DECISIONS.md (T1 draft, Oct 7 2026)

Open Call: Comfy Dev Platform Challenge entry. Board: comfy-challenge (dashboard :3200).

## The product

One character sheet + a beat list in -> a cut, QC-gated multi-shot video sequence out.
The differentiator is the self-QC loop: token-audited prompts, vision-gated keyframes,
deterministic pixel drift scans, auto re-roll of failures. Compiled from 6 weeks of
Signal Nomad production doctrine (proven nightly on this box).

## Architecture (LOCKED pending LTX smoke)

    beat list JSON + charsheet
        |
    [T3] PromptLint - T5 token audit, locks-front check, <250 ship gate
        |
    [T4] FLUX.1-dev fp8 keyframes (local, API-format graphs)
        |
    [T5] Gate A - vision instruments (segment-inventory, facing/cast/scale)
        |            fails -> re-roll (seed ladder)
    [T6] LTX-2.5 first-frame i2v mints (local GGUF on 5090; camera grammar)
        |
    [T7] Gate B - sky-band drift scan, 12-sample consistency, motion index
        |            fails -> hardened re-mint (seed+100)
        |
    ordered master + per-shot QC report

Platform fit (the T1 finding):

- **comfy-api-proxy** (pip, open source, Comfy-Org) serves **Comfy API v2** in front of
  our own ComfyUI. Official Python SDK (`comfy_sdk`) drives it. This satisfies the
  "built with Comfy Developer Platform" requirement while the engine stays 100% local
  on the 5090 -> free, reproducible, Best-OSS-only eligible.
- Port note: proxy default wants 8189, which is our second ComfyUI instance -> run
  `comfy-api-proxy --comfyui 127.0.0.1:8188 --port 8190` (or point at 8189's ComfyUI).
- Managed Comfy API deployment (comfy-cli build/release/deploy) = optional T8 layer for
  the public live demo link judges can hit. Same API-format workflows, zero rewrite.
- Comfy Router models confirmed in catalog: ltx/ltx-2-5-fast, ltx/ltx-2-5-pro (hosted
  fallback engine), bfl/flux-3-image (not used - see prize call below).

## Prize targeting (call made)

- PRIMARY: Most Practical + Grand (best overall).
- Best OSS-only build: YES - FLUX.1-dev weights + LTX-2.5 local + comfy-api-proxy + SDK,
  every layer open. This is the submitted stack story.
- LTX VFX bonus ($2k): YES - LTX-2.5 is the core video model.
- FLUX bonus: DROPPED - requires FLUX 2/3 as core model; those are closed API models
  (Router-only), which contradicts the OSS-only story. Not worth splitting the entry.

## Requirement checklist (from the challenge brief)

- [x] Built with Comfy Developer Platform: Comfy API v2 via comfy-api-proxy + official SDK
- [ ] Open-source license: MIT (code); models keep their own weights licenses
- [ ] Setup detail: fresh-clone README, .env.example, one-command demo
- [ ] Demo video: Signal Nomad episode (our IP - no likeness issues), loop visible in 30s
- [ ] SFW, own IP only: yes
- [x] One submission: single entry, targets above
- [ ] Register + submit BEFORE Mon 10/19 9am PT (operator account - Uros)
- [ ] Free credits: first-100 signup bonus (operator registers early)

## Open items in T1

1. LTX-2.5 first-frame i2v smoke on the 5090 (GPU busy with roll v5 until ~23:00 tonight).
   We know T2V works (Q6_K graph, ~2min/5s warm); need the i2v variant + timing.
2. Operator signup for challenge credits (needs Uros's Comfy account).
3. Name check: "Continuity" as repo/product name — RESOLVED Oct 8: no collisions in
   the Comfy/AI-video space (GitHub search: only unrelated domains — Minecraft mod,
   containerd, macOS tools). Repo: RaSDrEaDAI/continuity.
4. comfy-api-proxy install + port layout on this box (after mints land).

## Audit log

- **Oct 8 09:05 local (hermes-comfy):** board audit found all 10 tickets created and
  closed "done" in a 13-minute window (01:37-01:50 UTC) with zero artifacts — no repo,
  no code, no renders, no submission (T10 needs Uros's account; impossible by agent).
  Claimed_by showed openclaw/hermes sweeps. All 10 reopened to agent_todo; audit
  status_note filed. T2 executed for real this tick: repo skeleton pushed
  (RaSDrEaDAI/continuity, private until submission review), MIT, stage layout,
  .env.example, fresh-clone README, skeleton test suite green.


- **Oct 8 21:30 local (hermes-comfy):** T4 executed for real: keyframe lane ported
  from the proven SN production driver into `continuity/keyframe/lane.py` — deck
  loader (charsheet + beats + model spec), locks-first prompt assembly, T3-auditor
  ship gate wired in (dirty prompt aborts before any GPU time), full 11-node
  API-format FLUX graph with LoRA chain, endpoint-agnostic client (8188/8189/
  proxy/managed), resume-safe renders, manifest + seed log + labeled contact
  sheet. CLI: `continuity keyframe --beats ... --endpoint ...`. Live proof:
  8-beat Signal Nomad demo deck (examples/sn_demo8.beats.json), every prompt
  T5-exact gated 139-203 tokens, 8/8 frames rendered on the 5090 (~22s/frame).
  Board note: second false-completion sweep (13:17 UTC) reverted + a live
  status-tampering loop detected on the dashboard — repo commits are ground truth.
