# Continuity

**One character sheet + a beat list in → a cut, QC-gated multi-shot video sequence out.**

Built for the Comfy Dev Platform Challenge (Open Call, Oct 2026) on top of the
[Comfy Developer Platform](https://developers.comfy.org) — a local
`comfy-api-proxy` serves Comfy API v2 in front of our own ComfyUI instance, so
every workflow runs 100% locally on an RTX 5090 while staying on the official
API surface.

The differentiator is the **self-QC loop**. Generation pipelines fail silently:
prompts past the T5 token window are dead text, keyframes drift off-canon,
mints change time-of-day mid-shot. Continuity gates every stage and re-rolls
failures automatically — compiled from six weeks of nightly production
doctrine on the Signal Nomad IP.

```
beats.json + charsheet
      |
[T3] auditor ..... T5 token audit, locks-front ordering, <250 ship gate
      |
[T4] keyframe .... FLUX.1-dev fp8 keyframes (API-format graphs)
      |
[T5] gate_a ...... vision instruments: segment-inventory, facing compass, cast/scale
      |                fails -> re-roll (seed ladder)
[T6] mint ........ LTX-2.5 first-frame i2v, camera grammar per beat
      |
[T7] gate_b ...... sky-band drift scan (code, not eyes), 12-sample consistency,
      |                motion index — fails -> hardened re-mint (seed+100)
      |
ordered master + per-shot QC report
```

## Status

| Stage | Module | State |
|---|---|---|
| Token auditor | `continuity/auditor` | done (T3) — T5-exact spans, dead-text flag, ship gate |
| Keyframe lane | `continuity/keyframe` | done (T4) — gated deck renderer, resume-safe, live 8-beat demo |
| Gate A | `continuity/gate_a` | done (T5) — two-pass solo vision instruments, live 10-frame gate |
| Mint lane | `continuity/mint` | done (T6) — LTX-2.5 i2v graphs + submit/wait, live mints + perf table |
| Gate B | `continuity/gate_b` | done (T7) — drift/motion/consistency + re-roll loop + master assembly |
| Orchestrator | `continuity/orchestrator` | skeleton |
| API surface | `continuity/api` | skeleton (T8) |

## Quickstart (fresh clone)

```bash
git clone https://github.com/RaSDrEaDAI/continuity.git
cd continuity
python -m venv .venv && .venv/Scripts/activate    # or source .venv/bin/activate
pip install -e ".[dev]"
pytest                        # skeleton tests: imports, CLI, example beat schema
continuity --help             # stage CLIs wired
```

Optional (token auditor needs the T5 tokenizer):

```bash
pip install -e ".[auditor]"
```

## Configuration

Copy `.env.example` to `.env` and point `CONTINUITY_COMFY_URL` at your
ComfyUI (or a `comfy-api-proxy` speaking Comfy API v2):

```bash
cp .env.example .env
```

Try the pipeline surface without a GPU:

```bash
continuity audit --text "exactly one camel, zero people, wide salt flat, dawn light"
continuity run --beats examples/beats.example.json --dry-run
```

## Beat list format

See `examples/beats.example.json`. Each beat carries its prompt blocks
(locks first — direction/gaze/no-rope clauses must land inside tokens 1-40),
camera grammar, time-of-day lock, and a seed. Locks-front ordering is enforced
by the auditor, not by convention.

## License

MIT — code. Model weights keep their own licenses (FLUX.1-dev, LTX-2.5).
