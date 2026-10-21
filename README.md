# Continuity

**One character sheet + a beat list in → a cut, QC-gated multi-shot video sequence out.**

Built for the Comfy Dev Platform Challenge (Open Call, Oct 2026) on top of the
[Comfy Developer Platform](https://developers.comfy.org) — the official
`comfy-sdk` (pinned 0.5.0) drives `comfy-api-proxy` (pinned 0.1.6, Comfy API
v2) in front of our own ComfyUI instance, so generation runs 100% locally on
an RTX 5090 while submission, polling and output retrieval stay on the
official API surface (`continuity/api/sdk_endpoint.py`; redacted live trace in
`examples/sdk_proof/trace.json`). Native local mode (`/prompt`+`/history`)
remains the default engine path — the two are explicit, separate modes.

Honest scope notes (full matrix in `docs/eligibility.md`): code is MIT; model
weights keep their own licenses (FLUX.1-dev is open weights under a
non-commercial license — this is not an all-OSS stack); deterministic QC
instruments are local code, while the optional vision passes call a hosted
provider.

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
| Orchestrator | `continuity/orchestrator` | done (T15) — end-to-end `continuity run`: audit → keyframe → Gate A → mint → Gate B → bounded repair → master; persisted run state, hash-validated resume, Gate A enforced before every mint (CLI + serve API) |
| API surface | `continuity/api` | done (T8) — FastAPI serve (health/runs/mint/files), live behind gateway proxy |

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

Full pipeline on a live Comfy endpoint (exact T5 tokenizer required; the
dry-run estimate is labeled and never gates production):

```bash
continuity run --beats examples/sn_demo8_v2.beats.json \
  --tokenizer <t5_tokenizer_path> --limit 2 --duration 5.04 \
  --width 960 --height 544 --out-root runs
# → runs/<id>/{state.json, report.md, gate_a/, gate_b/, clips/, master.mp4}
# resume after a pause or a crash: add --resume runs/<id>
```

Every mint carries its Gate A verdict (a FAIL or missing verdict refuses the
mint — the serve API enforces the same rule). The master declares any missing
beat instead of silently dropping it, and the run report records wall time,
GPU seconds, vision cost, retries and every operator escalation.

## Beat list format

See `examples/beats.example.json`. Each beat carries its prompt blocks
(locks first — direction/gaze/no-rope clauses must land inside tokens 1-40),
camera grammar, time-of-day lock, and a seed. Locks-front ordering is enforced
by the auditor, not by convention.

## HTTP API (serve)

`pip install -e ".[api]"`, then serve the lanes over HTTP (one GPU job at a
time, `x-agent-secret` auth on every route):

```bash
CONTINUITY_SECRET=... python -m continuity.api.serve   # binds :8191
```

Deployed live behind the team gateway — public JSON surface:

```bash
API="https://n8n.babcockranchdirectory.com/webhook/continuity"
SECRET="<x-agent-secret value>"

# health (GPU present, VRAM free)
curl -X POST "$API?action=health" -H "x-agent-secret: $SECRET"

# render a keyframe deck (beats.json content in the body; limit trims the deck)
curl -X POST "$API?action=runs/keyframe" -H "x-agent-secret: $SECRET" \
     -H 'content-type: application/json' \
     -d '{"deck": {…beats.json…}, "prefix": "demo", "limit": 1}'
# -> {"run_id":"df206953","status_url":"/runs/df206953"}

# poll until status=done — manifest carries file paths + sha1 per beat
curl -X POST "$API?action=runs/df206953" -H "x-agent-secret: $SECRET"

# mint a clip from a rendered keyframe (same pattern, kind=mint)
curl -X POST "$API?action=runs/mint" -H "x-agent-secret: $SECRET" \
     -H 'content-type: application/json' \
     -d '{"beat_id":"D5_whiteout","deck":{…},"keyframe_path":"…png",
          "camera":"lateral_track","vseed":2301}'
```

Gateway notes: the public proxy speaks JSON (submit + poll + results);
binary artifacts (`/files/<path>`) are served by the box directly on the
private network. Every failed gate aborts before GPU time is spent
(`no_gpu_spent: true` in the run record).

## Demo video

Watch the self-QC loop work — gate condemnations, the prompt respec, and the
gate-passed mints, in 62 seconds:

<video src="docs/demo/t9_demo_v1.mp4" controls muted preload="metadata" width="800"></video>

Mirror (if the player above does not load):
**https://res.cloudinary.com/dg1lb4q9o/video/upload/v1791939296/projects/comfy-challenge/t9_demo_v1.mp4**

## License

MIT — code only. Model weights keep their own licenses: FLUX.1-dev
(non-commercial), LTX-2.5, T5 (Apache-2.0), plus locally trained LoRAs. The
optional hosted vision gates are proprietary services. See
`docs/eligibility.md` for the full matrix and evidence.
