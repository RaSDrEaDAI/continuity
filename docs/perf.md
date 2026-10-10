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

## Next measurements

- 3-beat mint run @1248x704 x 294f (12.25s) from Gate-A-passed frames (warm,
  sequential) — fills the mint-row of this table and the T6 DoD.
