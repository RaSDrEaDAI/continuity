# Gate A demo evidence — live run, Oct 9 2026

Verdict JSONs from a real `continuity gate-a` run over the T4 demo manifest
(`examples/sn_demo8.beats.json` rendered frames) plus two control frames.
Provider: `openai-compat:anthropic/claude-haiku-5.5` via OpenRouter,
2 independent passes per instrument, `gate-a/1`.

| file | what it shows |
|---|---|
| `gate_a_summary.json` | run summary: 10 gated, 1 passed, 9 condemned |
| `D2_bootmacro.gate_a.json` | CLEAN frame PASSES (env macro, cast 0/0, passes agree) |
| `DIRTY_O2_04_s5717.gate_a.json` | deliberately dirty control (known lead strap) CONDEMNED — rope reads disagree, one pass inventories the free-hanging segment; disagreement = fail by doctrine |
| `BORDER_O2_01_s5712.gate_a.json` | borderline tassel-cord frame CONDEMNED — free-hanging segment YES, passes agree |
| `D1_depart.gate_a.json` | a REAL defect caught on a demo beat: trailing lead rope (verdict YES) + man/camel facing split (LEFT vs RIGHT) |

Every camel-bearing demo beat was condemned for free-hanging halter rope —
consistent with the production record behind these instruments (FLUX hangs a
strap off nearly every halter it draws at close scale). That is the gate
working: dirty frames never reach the mint stage.
