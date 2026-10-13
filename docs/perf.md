# Continuity — measured performance (5090 / 32GB, Blackwell)

All numbers from live runs on this box. Update after each measured increment.

## LTX-2.5 first-frame i2v (T6, Oct 10 2026)

Graph: Q6_K distilled GGUF + gemma4-12b int8 text encoder + video/audio VAEs,
euler_ancestral, 9-step ManualSigmas, cfg 1.0, `LTXVImgToVideo` strength 1.0.

| run | res | frames | wall | notes |
|---|---|---|---|---|
| i2v smoke (D2 boot-macro keyframe) | 960x544 | 121 (5.04s @24) | 196s COLD | models loaded from disk during run; 15.7GB VRAM free at completion (~16GB resident incl. cached models); output h264+AAC mp4, 701KiB |
| (reference) T2V smoke | 960x544 | 121 | ~120s warm | no image conditioning; same sampler envelope |

First-frame fidelity: mean abs diff 9.5/255 vs source keyframe (VAE + h264
roundtrip only); f120 diff 51.3/255 = real motion. Vision read on f120: style
held (2D anime cel), scene held (boot macro + cracked salt), no unexpected
casts. Cost of solo vision read: $0.0002 (OpenRouter haiku).

Known wiring trap (fixed, regression-tested): `LTXVImgToVideo` returns
`[positive COND, negative COND, latent]` — the CFGGuider must consume slots
0/1 (image-conditioned pair) and `LTXVConcatAVLatent.video_latent` takes slot
2. Wiring the raw pre-image conditioning into the guider 400s at validation
with `return_type_mismatch`. Also: `LoadImage` must not carry the UI-only
`upload` key in API-format graphs.

## T6 overnight run (Oct 12 2026) — deck v2, Gate A, mints @ full spec

Keyframes (FLUX.1-dev krea fp8 + CAICO LoRA @0.9, 1536x864, 22 steps):
**15.0-15.1 s/frame warm**, 8/8 rendered sequential; ship gate 8/8 PASS
(exact T5 tokenizer). Gate A live re-gate (gate-a/1, OpenRouter haiku-5.5,
2-pass): **2/8 PASS** — D2 (env macro) + D4 (solo man); all 6 pair-scale
beats condemned (free-hanging rope on every pair frame; + phantom man D5,
phantom camel D6, gaze-at-camera D7/D8). Solo-man and env classes pass
reliably; pair-scale remains the documented FLUX rope wall (2 prompt
strikes spent).

| run | res | frames | wall | f/s | first-frame diff | notes |
|---|---|---|---|---|---|---|
| D2 boot-macro (static_drift) | 1248x704 | 289 (12.04s @24) | 302.4s | 0.96 | 5.93/255 | warm; vram_free_end 11.4 GB; strip QC CLEAN |
| D4 signal (static_drift) | 1248x704 | 289 | 215.5s | 1.34 | 6.96/255 | warm; same envelope; strip QC CLEAN |

Solo strip QC (haiku, ~$0.0003/read): style held (2D cel), cast exact
(one boot / one man, zero phantoms), motion visible, no fused hands /
corruption. Artifacts: Crux `projects/comfy-challenge/t6v2/` (contact,
8 frames, gate summary, 2 mints + strips). Third DoD mint blocked on
operator A/B/C (rope-as-canon vs man-only beat vs Gemini erase re-credit).

Known repo issue (hit tonight): `render_deck` resolves saved frames against
`out_dir`, but ComfyUI writes to its own output root — manifests record
`file: null` when they differ (the committed T4 manifest shows exactly
this). Fix queued: resolve `/history` subfolders against a configurable
`CONTINUITY_COMFY_OUTPUT` root (default `D:/1.AI/Comfy/output`).

## Next measurements

- third DoD mint once the pair-beat call lands (operator A/B/C)
- Gate B (T7) numbers: drift scan + consistency on the mint set

