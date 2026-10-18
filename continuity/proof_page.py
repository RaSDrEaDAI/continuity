"""T17 — Proof viewer page generator.

Builds the judge-facing proof surface from RECORDED run evidence only:

- ``examples/offline_demo/`` fixture bundle (deck + manifest + replay answers)
- Gate A report dir (output of ``continuity gate-a --replay ...``)
- Gate B report dir (output of ``continuity gate-b --no-vision ...``)

Two renderers, one data source (no drift between them):

- ``render_markdown`` -> ``PROOF.md`` (GitHub renders it; repo-relative images)
- ``render_html``     -> standalone ``proof_page.html`` (absolute public URLs
  only; safe to serve from any static host)

Design rules (T17 ticket):

- Only public Cloudinary URLs from the PUBLIC_ASSETS allowlist — never local
  box paths, never ZeroTier IPs, never fleet secrets.
- Replay is labeled replay; the live lane is linked, not claimed.
- Verdict text is quoted verbatim from the recorded gate reports — this module
  never re-adjudicates.
"""

from __future__ import annotations

import datetime as _dt
import json
import os

# Public asset allowlist (Crux cloud, signed uploads, verified live).
# Keys are fixture-relative references; values are the only absolute URLs the
# HTML renderer is allowed to emit.
PUBLIC_ASSETS = {
    "R1_envmacro": {
        "frame": "https://res.cloudinary.com/dg1lb4q9o/image/upload/v1792328916/projects/comfy-challenge/viewer/R1_envmacro_s9101.jpg",
        "clip": "https://res.cloudinary.com/dg1lb4q9o/video/upload/v1792328919/projects/comfy-challenge/viewer/R1_envmacro_s9101.mp4",
    },
    "R2_pair": {
        "frame": "https://res.cloudinary.com/dg1lb4q9o/image/upload/v1792328917/projects/comfy-challenge/viewer/R2_pair_s9102.jpg",
        "clip": None,  # condemned at Gate A — never minted
    },
    "R3_driftdusk": {
        "frame": "https://res.cloudinary.com/dg1lb4q9o/image/upload/v1792328918/projects/comfy-challenge/viewer/R3_driftdusk_s9103.jpg",
        "clip": "https://res.cloudinary.com/dg1lb4q9o/video/upload/v1792328921/projects/comfy-challenge/viewer/R3_driftdusk_s9103.mp4",
    },
}

DEMO_VIDEO = ("https://res.cloudinary.com/dg1lb4q9o/video/upload/v1791939296/"
              "projects/comfy-challenge/t9_demo_v1.mp4")

REPO_URL = "https://github.com/RaSDrEaDAI/continuity"

# Live-lane measured numbers (source of truth: docs/perf.md — do not edit here
# without updating that file).
LIVE_LANE = {
    "keyframe_s_per_frame": "15.0 s/frame warm (FLUX.1-dev fp8 + style LoRA, 1536x864)",
    "deck_gate_a": "8/8 rendered, ship gate 8/8 exact-T5 PASS; Gate A 2/8 PASS "
                   "(pair-scale rope wall — documented)",
    "mint": "LTX-2.5 i2v 1248x704x289f: 302.4 s and 215.5 s warm (two beats)",
    "vision_cost": "104 live vision calls for a full 10-frame gate: $0.021 "
                   "(solo-read instruments, hosted vision API)",
    "offline_replay": "the whole 3-beat proof below replays in ~12 s for $0",
}


def build_data(fixtures_dir, gate_a_dir=None, gate_b_dir=None):
    """Aggregate recorded evidence into one render-ready dict."""
    fixtures_dir = os.path.abspath(fixtures_dir)
    with open(os.path.join(fixtures_dir, "offline_demo.beats.json")) as f:
        deck = json.load(f)
    with open(os.path.join(fixtures_dir, "offline_demo.manifest.json")) as f:
        manifest = json.load(f)
    replay_path = os.path.join(fixtures_dir, "replay", "gate_a_answers.json")
    replay_answers = {}
    if os.path.exists(replay_path):
        with open(replay_path) as f:
            replay_answers = json.load(f)

    beats = []
    for b in deck["beats"]:
        bid = b["id"]
        m = manifest.get(bid, {})
        beats.append({
            "id": bid,
            "seed": b.get("seed"),
            "locks": b.get("locks", ""),
            "scene": b.get("scene", ""),
            "camera": b.get("camera", ""),
            "tod": b.get("tod", ""),
            "tokens": m.get("tokens"),
            "audit_mode": m.get("audit_mode", ""),
            "frame_file": m.get("file", ""),
            "frame_url": (PUBLIC_ASSETS.get(bid) or {}).get("frame"),
            "clip_url": (PUBLIC_ASSETS.get(bid) or {}).get("clip"),
        })

    data = {
        "charsheet": deck.get("charsheet", ""),
        "beats": beats,
        "gate_a": _load_gate_a(gate_a_dir) if gate_a_dir else None,
        "gate_b": _load_gate_b(gate_b_dir) if gate_b_dir else None,
        "replay_answers": replay_answers,
        "live_lane": LIVE_LANE,
        "generated_at": _dt.datetime.now().astimezone().isoformat(timespec="seconds"),
    }
    return data


def _load_gate_a(gate_a_dir):
    out = {"summary": None, "reports": {}}
    spath = os.path.join(gate_a_dir, "gate_a_summary.json")
    if os.path.exists(spath):
        with open(spath) as f:
            out["summary"] = json.load(f)
    for bid in PUBLIC_ASSETS:
        rpath = os.path.join(gate_a_dir, f"{bid}.gate_a.json")
        if os.path.exists(rpath):
            with open(rpath) as f:
                out["reports"][bid] = json.load(f)
    return out


def _load_gate_b(gate_b_dir):
    out = {"summary": None, "reports": {}}
    spath = os.path.join(gate_b_dir, "gate_b_summary.json")
    if os.path.exists(spath):
        with open(spath) as f:
            out["summary"] = json.load(f)
    for name in os.listdir(gate_b_dir):
        # flat files (beat.gate_b.json) and per-beat dirs (beat/gate_b.json)
        flat = os.path.join(gate_b_dir, name)
        nested = os.path.join(flat, "gate_b.json")
        for cand in ([nested] if os.path.isdir(flat) else
                     [flat] if name.endswith(".gate_b.json") else []):
            with open(cand) as f:
                rep = json.load(f)
            out["reports"][rep.get("beat", name)] = rep
    return out


def _gate_a_verdict(data, bid):
    ga = data.get("gate_a") or {}
    rep = (ga.get("reports") or {}).get(bid)
    if not rep:
        return None
    return {
        "verdict": rep.get("verdict"),
        "fails": rep.get("fails", []),
        "agreed": rep.get("agreed"),
        "provider": rep.get("provider"),
        "instrument": rep.get("instrument_version"),
        "usage": rep.get("usage", {}),
    }


def _rope_verdict_line(data, bid):
    """Verbatim two-pass rope inventory from the recorded replay answers.

    The gate report stores the PARSED verdict (segments_a/b, YES/NO); the
    verbatim instrument text lives in the shipped replay answers. Both passes
    must agree, else the gate would have condemned on disagreement.
    """
    answers = data.get("replay_answers") or {}
    reads = answers.get(f"{bid}|rope")
    if isinstance(reads, list) and reads and isinstance(reads[0], str):
        return reads[0]
    return ""


def _gate_b_verdict(data, bid):
    gb = data.get("gate_b") or {}
    rep = (gb.get("reports") or {}).get(bid)
    if not rep:
        return None
    out = {"verdict": rep.get("verdict"), "fails": rep.get("fails", []),
           "coverage_gaps": rep.get("coverage_gaps", [])}
    drift = rep.get("drift")
    if isinstance(drift, dict):
        out["drift"] = {
            "swing": drift.get("swing"),
            "max_step": drift.get("max_step"),
            "fail_class": drift.get("fail_class"),
            "luminance": drift.get("luminance") or [],
        }
    return out


def render_markdown(data):
    """PROOF.md — GitHub-rendered proof viewer (repo-relative images)."""
    lines = []
    a = lines.append
    a("# Continuity — proof viewer (recorded run, replays offline)")
    a("")
    a("> **This page is a replay of a recorded run**, generated from evidence "
      "committed in this repo (`examples/offline_demo/` + recorded gate "
      "reports). It is not a live generation service. Judge-facing surfaces:")
    a("> the [repo](" + REPO_URL + "), the [62-second demo video](" + DEMO_VIDEO + "), "
      "and the offline replay commands at the bottom (seconds, $0).")
    a("")
    a("**The claim under test:** a character sheet + a beat list goes in; "
      "keyframes pass vision gates *before* any video is minted; mints are "
      "drift-scanned in code; failures never reach the cut. This page shows "
      "one clean loop and two real rejections.")
    a("")
    a("---")
    a("")
    a("## Input — charsheet + 3-beat deck (verbatim)")
    a("")
    a("```")
    a(data["charsheet"])
    a("```")
    a("")
    a("| beat | seed | locks (front-loaded) | scene | camera | T5 tokens |")
    a("|---|---|---|---|---|---|")
    for b in data["beats"]:
        scene = b["scene"] if len(b["scene"]) <= 68 else b["scene"][:65] + "..."
        locks = b["locks"] if len(b["locks"]) <= 58 else b["locks"][:55] + "..."
        a(f"| `{b['id']}` | {b['seed']} | {locks} | {scene} | {b['camera']} | "
          f"{b['tokens']} |")
    a("")
    a(f"Ship gate: all beats under 250 T5 tokens, locks front-loaded "
      f"(audit mode: {data['beats'][0]['audit_mode']}).")
    a("")
    a("## Stage 1 — Gate A: keyframe vision instruments (before any mint)")
    a("")

    for b in data["beats"]:
        bid = b["id"]
        v = _gate_a_verdict(data, bid)
        badge = ("**PASS**" if v and v["verdict"] == "PASS"
                 else "**CONDEMNED**" if v else "*not recorded*")
        a(f"### {bid} — {badge}")
        a("")
        img = f"examples/offline_demo/{b['frame_file']}" if b["frame_file"] else ""
        if img:
            a(f"![{bid} keyframe]({img})")
            a("")
        if v:
            a(f"- verdict: `{v['verdict']}` — instrument `{v['instrument']}`, "
              f"provider `{v['provider']}`, two independent passes "
              f"{'AGREE' if v['agreed'] else 'DISAGREE'}")
            if v["fails"]:
                a("- fails (verbatim from the recorded report):")
                for fl in v["fails"]:
                    a(f"  - {fl}")
            if bid == "R2_pair":
                rope = _rope_verdict_line(data, bid)
                if rope:
                    a("- the rope instrument's recorded segment inventory "
                      "(pass 1 of 2, verbatim):")
                    a("")
                    a("  ```")
                    for ln in rope.splitlines():
                        a(f"  {ln}")
                    a("  ```")
            a("")
        if v and v["verdict"] == "FAIL":
            a(f"**{bid} never reaches the mint stage.** The GPU cost of a bad "
              f"video (about 5 minutes per 12-second mint on our RTX 5090) is "
              f"never spent on it.")
            a("")

    a("## Stage 2 — mint + Gate B: deterministic clip QC")
    a("")
    a("Only Gate-A PASS beats are minted (first-frame image-to-video, "
      "LTX-2.5). Two were recorded for this deck. Replay mode runs the "
      "deterministic instruments only (drift scan, motion index) — the "
      "12-sample vision consistency pass needs a vision provider and is "
      "skipped here, and the reports say so explicitly (`coverage_gaps`).")
    a("")
    for b in data["beats"]:
        bid = b["id"]
        if not b["clip_url"]:
            continue
        v = _gate_b_verdict(data, bid)
        badge = ("**PASS**" if v and v["verdict"] == "PASS"
                 else "**CONDEMNED**" if v else "*not recorded*")
        a(f"### {bid} clip — {badge}")
        a("")
        a(f"[12-second mint clip ({bid})]({b['clip_url']})")
        a("")
        if v:
            a(f"- verdict: `{v['verdict']}` (gate-b/1, deterministic "
              f"instruments, $0)")
            if v.get("drift"):
                d = v["drift"]
                a(f"- sky-band luminance drift scan: swing **{d['swing']}** "
                  f"(fail threshold 25), max step {d['max_step']}, class "
                  f"`{d['fail_class']}` — the dusk clip's sky keeps sliding; "
                  f"a gate-a-clean keyframe still fails here, exactly as "
                  f"designed.")
                if d.get("luminance"):
                    lums = ", ".join(f"{x:.0f}" for x in d["luminance"])
                    a(f"- measured sky-band luminance @1fps: `{lums}` "
                      f"(code, not vision)")
            if v["fails"]:
                for fl in v["fails"]:
                    a(f"- fail: {fl}")
            if v and v["verdict"] == "FAIL":
                a(f"- **excluded from the master cut**; hardened re-mint "
                  f"(seed ladder) is the recorded repair path.")
            a("")
    a("## What the live lane measured (RTX 5090, 32 GB)")
    a("")
    for k, val in data["live_lane"].items():
        a(f"- {val}")
    a("")
    a("Sources: `docs/perf.md` (raw numbers, wiring notes), "
      "`docs/eligibility.md` (claims matrix).")
    a("")
    a("## Replay this page yourself (offline, $0)")
    a("")
    a("```bash")
    a("git clone https://github.com/RaSDrEaDAI/continuity.git")
    a("cd continuity && pip install -e .[dev]")
    a("continuity gate-a --manifest examples/offline_demo/offline_demo.manifest.json \\")
    a("  --beats examples/offline_demo/offline_demo.beats.json \\")
    a("  --replay examples/offline_demo/replay/gate_a_answers.json --out /tmp/ga")
    a("continuity gate-b --beats examples/offline_demo/offline_demo.beats.json \\")
    a("  --keyframes examples/offline_demo/offline_demo.manifest.json \\")
    a("  --seeds examples/offline_demo/replay/seeds.json --no-vision \\")
    a("  --out /tmp/gb examples/offline_demo/clips/R1_envmacro_s9101.mp4 \\")
    a("  examples/offline_demo/clips/R3_driftdusk_s9103.mp4")
    a("```")
    a("")
    a(f"Regenerate this page: `continuity proof-page --out PROOF.md` "
      f"(generated {data['generated_at']}).")
    a("")
    a("## Claims discipline")
    a("")
    a("- **Implemented and verifiable here:** token audit, gated keyframe "
      "lane, Gate A vision instruments, LTX-2.5 first-frame mints, Gate B "
      "drift scan, PASS-only master assembly, offline replay.")
    a("- **Human-assisted:** prompts/decks authored with operator review; "
      "vision instruments call a hosted vision API (cheap, disclosed); final "
      "taste calls are operator-gated by design.")
    a("- **Roadmap (not claimed):** fully autonomous unattended production, "
      "public hosted viewer, video-project tracking UI.")
    a("")
    _footer(a)
    return "\n".join(lines) + "\n"


def _footer(append):
    append("---")
    append("")
    append("Signal Nomad IP (own IP, no likeness issues). MIT code; model "
           "weights keep their own licenses. Generated by "
           "`continuity proof-page`.")


_HTML_CSS = """
:root { color-scheme: dark; }
* { box-sizing: border-box; }
body { font-family: -apple-system, 'Segoe UI', Roboto, sans-serif; margin: 0;
       background: #101418; color: #d8dee6; line-height: 1.55; }
.wrap { max-width: 980px; margin: 0 auto; padding: 32px 20px 80px; }
h1 { font-size: 1.7rem; letter-spacing: .3px; }
h2 { margin-top: 2.2rem; border-bottom: 1px solid #2a3138; padding-bottom: .3rem; }
.banner { background: #1b2530; border: 1px solid #31506b; border-radius: 10px;
          padding: 12px 16px; margin: 18px 0; }
.card { background: #161b21; border: 1px solid #2a3138; border-radius: 10px;
        padding: 16px 18px; margin: 14px 0; }
.card img { max-width: 100%; border-radius: 6px; display: block; }
.pass { color: #7dd97a; font-weight: 700; }
.fail { color: #ff7d6b; font-weight: 700; }
.mono { font-family: ui-monospace, 'Cascadia Code', Menlo, Consolas, monospace;
        font-size: .85rem; background: #0d1117; border: 1px solid #2a3138;
        border-radius: 6px; padding: 10px 12px; white-space: pre-wrap; }
table { border-collapse: collapse; width: 100%; font-size: .9rem; }
td, th { border: 1px solid #2a3138; padding: 6px 9px; text-align: left; }
video { max-width: 100%; border-radius: 6px; display: block; margin-top: 8px; }
a { color: #7ab8ff; }
.small { color: #8a94a0; font-size: .85rem; }
"""


def render_html(data):
    """Standalone proof page — absolute PUBLIC_ASSETS URLs only."""
    ga_sum = (data.get("gate_a") or {}).get("summary") or {}
    gb_sum = (data.get("gate_b") or {}).get("summary") or {}
    h = []
    h.append("<!DOCTYPE html><html lang='en'><head><meta charset='utf-8'>")
    h.append("<meta name='viewport' content='width=device-width, initial-scale=1'>")
    h.append("<title>Continuity — proof viewer (recorded run)</title>")
    h.append(f"<style>{_HTML_CSS}</style></head><body><div class='wrap'>")
    h.append("<h1>Continuity — proof viewer</h1>")
    h.append("<div class='banner'><b>REPLAY of a recorded run</b> — built from "
             "evidence committed to the <a href='" + REPO_URL + "'>repo</a> "
             "(fixtures + recorded gate reports). Not a live service. "
             "Demo video: <a href='" + DEMO_VIDEO + "'>62s cut</a>.</div>")
    h.append("<p>One character sheet + a beat list in; keyframes pass vision "
             "gates <i>before</i> any video is minted; mints are drift-scanned "
             "in code; failures never reach the cut. Below: one clean loop and "
             "two real rejections.</p>")

    h.append("<h2>Input — charsheet + 3-beat deck</h2>")
    h.append(f"<div class='mono'>{data['charsheet']}</div>")
    h.append("<table><tr><th>beat</th><th>seed</th><th>locks</th><th>scene"
             "</th><th>camera</th><th>T5</th></tr>")
    for b in data["beats"]:
        h.append(f"<tr><td>{b['id']}</td><td>{b['seed']}</td>"
                 f"<td>{b['locks']}</td><td>{b['scene']}</td>"
                 f"<td>{b['camera']}</td><td>{b['tokens']}</td></tr>")
    h.append("</table>")

    h.append(f"<h2>Gate A — keyframes (gated: {ga_sum.get('gated', '?')}, "
             f"passed: {ga_sum.get('passed', '?')}, instrument "
             f"{ga_sum.get('instrument_version', '?')}, provider "
             f"{ga_sum.get('provider', '?')})</h2>")
    for b in data["beats"]:
        bid = b["id"]
        v = _gate_a_verdict(data, bid)
        cls = "pass" if v and v["verdict"] == "PASS" else "fail"
        word = "PASS" if v and v["verdict"] == "PASS" else "CONDEMNED"
        h.append(f"<div class='card'><h3>{bid} — <span class='{cls}'>{word}"
                 f"</span></h3>")
        if b["frame_url"]:
            h.append(f"<img src='{b['frame_url']}' alt='{bid} keyframe'>")
        if v:
            h.append(f"<p class='small'>two independent passes "
                     f"{'AGREE' if v['agreed'] else 'DISAGREE'} · "
                     f"instrument {v['instrument']}</p>")
            for fl in v["fails"]:
                h.append(f"<p><span class='fail'>fail:</span> {fl}</p>")
            rope = _rope_verdict_line(data, bid)
            if rope:
                h.append("<p class='small'>recorded rope segment inventory "
                         "(pass 1 of 2, verbatim):</p>")
                h.append(f"<div class='mono'>{rope}</div>")
            if v["verdict"] == "FAIL":
                h.append("<p><b>Never minted.</b> The GPU cost of a bad video "
                         "is never spent on it.</p>")
        h.append("</div>")

    h.append(f"<h2>Mint + Gate B (gated: {gb_sum.get('gated', '?')}, passed: "
             f"{gb_sum.get('passed', '?')}, $0 deterministic)</h2>")
    for b in data["beats"]:
        bid = b["id"]
        if not b["clip_url"]:
            continue
        v = _gate_b_verdict(data, bid)
        cls = "pass" if v and v["verdict"] == "PASS" else "fail"
        word = "PASS" if v and v["verdict"] == "PASS" else "CONDEMNED"
        h.append(f"<div class='card'><h3>{bid} clip — <span class='{cls}'>"
                 f"{word}</span></h3>")
        h.append(f"<video controls preload='metadata' src='{b['clip_url']}'></video>")
        if v:
            for fl in v["fails"]:
                h.append(f"<p><span class='fail'>fail:</span> {fl}</p>")
            if v.get("drift"):
                d = v["drift"]
                h.append(f"<p>sky-band drift scan: swing <b>{d['swing']}</b> "
                         f"(threshold 25), max step {d['max_step']}, class "
                         f"{d['fail_class']} — excluded from the master cut.")
        h.append("</div>")

    h.append("<h2>Live-lane measurements (RTX 5090)</h2><ul>")
    for val in data["live_lane"].values():
        h.append(f"<li>{val}</li>")
    h.append("</ul>")
    h.append("<h2>Replay it yourself</h2>")
    h.append("<div class='mono'>git clone " + REPO_URL + "\n"
             "cd continuity &amp;&amp; pip install -e .[dev]\n"
             "continuity gate-a --manifest examples/offline_demo/offline_demo.manifest.json \\\n"
             "  --beats examples/offline_demo/offline_demo.beats.json \\\n"
             "  --replay examples/offline_demo/replay/gate_a_answers.json --out /tmp/ga\n"
             "continuity gate-b --beats examples/offline_demo/offline_demo.beats.json \\\n"
             "  --keyframes examples/offline_demo/offline_demo.manifest.json \\\n"
             "  --seeds examples/offline_demo/replay/seeds.json --no-vision \\\n"
             "  --out /tmp/gb examples/offline_demo/clips/R1_envmacro_s9101.mp4 \\\n"
             "  examples/offline_demo/clips/R3_driftdusk_s9103.mp4</div>")
    h.append("<h2>Claims discipline</h2><ul>"
             "<li><b>Implemented:</b> token audit, gated keyframes, Gate A "
             "instruments, LTX-2.5 first-frame mints, Gate B drift scan, "
             "PASS-only master, offline replay.</li>"
             "<li><b>Human-assisted:</b> decks authored with operator review; "
             "hosted vision API (disclosed); taste calls operator-gated.</li>"
             "<li><b>Roadmap:</b> unattended production, hosted viewer, "
             "project tracking UI.</li></ul>")
    h.append(f"<p class='small'>Signal Nomad IP · MIT code · generated "
             f"{data['generated_at']} by continuity proof-page</p>")
    h.append("</div></body></html>")
    return "\n".join(h)


def _iter_text(obj):
    if isinstance(obj, str):
        yield obj
    elif isinstance(obj, dict):
        for v in obj.values():
            yield from _iter_text(v)
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            yield from _iter_text(v)


def audit_public_safety(html):
    """Fail-closed check: no local paths, private IPs, or secrets in HTML."""
    banned = ("D:/", "D:\\\\", "172.26.", "x-agent-secret", "api_key",
              "127.0.0.1")
    hits = [b for b in banned if b in html]
    return hits  # empty list == safe
