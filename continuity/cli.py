"""Continuity CLI — one subcommand per pipeline stage."""

import argparse
import sys

from continuity import __version__


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="continuity",
        description="charsheet + beats in, QC-gated multi-shot video out",
    )
    p.add_argument("--version", action="version", version=f"continuity {__version__}")
    sub = p.add_subparsers(dest="stage")

    sub.add_parser("audit", help="T3: T5 token audit + locks-front ship gate")

    audit_p = sub.add_parser("audit-text", help="quick audit of one prompt string")
    audit_p.add_argument("--text", required=True)
    audit_p.add_argument(
        "--tokenizer",
        help="T5 tokenizer path for exact token counts (else word estimate)",
    )

    kf_p = sub.add_parser("keyframe", help="T4: render a deck of beats (gated)")
    kf_p.add_argument("--beats", required=True, help="path to beats/deck JSON")
    kf_p.add_argument("--endpoint", default="http://127.0.0.1:8189",
                      help="Comfy API v2 surface (local 8188/8189, proxy 8190, managed)")
    kf_p.add_argument("--out", default="./out", help="output dir for frames + manifest")
    kf_p.add_argument("--tokenizer", default=None,
                      help="T5 tokenizer path for exact token gating (else estimate)")
    kf_p.add_argument("--prefix", default="cont4", help="filename prefix / subfolder")
    kf_p.add_argument("--limit", type=int, default=None, help="render only first N beats")
    kf_p.add_argument("--dry-run", action="store_true",
                      help="audit + plan without rendering")

    ga_p = sub.add_parser("gate-a", help="T5: keyframe vision instruments")
    ga_p.add_argument("--manifest", required=True,
                      help="T4 keyframe manifest to gate")
    ga_p.add_argument("--beats", required=True,
                      help="deck JSON (per-beat qc specs live here)")
    ga_p.add_argument("--out", default="./gate_a_reports",
                      help="output dir for verdict JSONs")
    ga_p.add_argument("--replay", default=None,
                      help="answers JSON for offline ReplayProvider")
    ga_p.add_argument("--passes", type=int, default=2,
                      help="independent passes per instrument (default 2)")

    gb_p = sub.add_parser("gate-b", help="T7: mint QC — drift/motion/consistency")
    gb_p.add_argument("clips", nargs="+", help="mint clip paths (beat order)")
    gb_p.add_argument("--beats", required=True, help="deck JSON (qc specs)")
    gb_p.add_argument("--keyframes", default=None,
                      help="T4 manifest mapping beat -> keyframe file (for ffd)")
    gb_p.add_argument("--seeds", default=None,
                      help="JSON {beat: video_seed} (recorded in reports)")
    gb_p.add_argument("--out", default="./gate_b_reports",
                      help="output dir for per-shot reports")
    gb_p.add_argument("--master", default=None,
                      help="assemble PASS clips into this master mp4")
    gb_p.add_argument("--no-vision", action="store_true",
                      help="deterministic instruments only (skip consistency)")
    gb_p.add_argument("--replay", default=None,
                      help="answers JSON for offline ReplayProvider")
    gb_p.add_argument("--samples", type=int, default=12,
                      help="consistency samples per clip (~1fps)")

    run_p = sub.add_parser("run", help="orchestrate the full pipeline (T15)")
    run_p.add_argument("--beats", required=True, help="path to beats.json")
    run_p.add_argument("--charsheet", default=None,
                       help="charsheet IMAGE path (Gate A scale probe; the "
                            "deck charsheet text is the identity block)")
    run_p.add_argument("--out-root", default="runs", help="run dir root")
    run_p.add_argument("--endpoint", default="http://127.0.0.1:8189")
    run_p.add_argument("--tokenizer", default=None,
                       help="T5 tokenizer path (REQUIRED unless --dry-run)")
    run_p.add_argument("--dry-run", action="store_true",
                       help="audit + plan only: no GPU, no network calls")
    run_p.add_argument("--limit", type=int, default=None,
                       help="only the first N beats")
    run_p.add_argument("--prefix", default="cont15",
                       help="keyframe filename prefix (resume matches on it)")
    run_p.add_argument("--vseed-base", type=int, default=3001)
    run_p.add_argument("--duration", type=float, default=12.25)
    run_p.add_argument("--width", type=int, default=1248)
    run_p.add_argument("--height", type=int, default=704)
    run_p.add_argument("--vram-floor", type=float, default=8.0)
    run_p.add_argument("--max-mint-rounds", type=int, default=2)
    run_p.add_argument("--replay", default=None,
                       help="Gate A replay answers JSON (offline)")
    run_p.add_argument("--no-gate-b-vision", action="store_true",
                       help="Gate B deterministic instruments only")
    run_p.add_argument("--resume", default=None,
                       help="path to an existing run dir to resume")
    run_p.add_argument("--run-id", default=None, help="name the run dir")

    mint_p = sub.add_parser("mint", help="T6: mint ONE clip from a manifest keyframe")
    mint_p.add_argument("--manifest", required=True,
                        help="T4 keyframe manifest (beat -> rendered frame)")
    mint_p.add_argument("--beats", required=True, help="deck JSON (camera/locks/qc)")
    mint_p.add_argument("--beat", required=True, help="beat id to mint")
    mint_p.add_argument("--seed", type=int, required=True, help="video seed")
    mint_p.add_argument("--endpoint", default="http://127.0.0.1:8189",
                        help="ComfyUI API surface (local 8188/8189)")
    mint_p.add_argument("--out", default="./out_mint", help="output dir for the clip")
    mint_p.add_argument("--duration", type=float, default=None,
                        help="clip length seconds (default 12.25)")
    mint_p.add_argument("--width", type=int, default=None, help="default 1248")
    mint_p.add_argument("--height", type=int, default=None, help="default 704")
    mint_p.add_argument("--gate-a-report", default=None,
                        help="Gate A report dir/JSON; a FAIL verdict refuses the mint")
    mint_p.add_argument("--result-json", default=None,
                        help="write the machine-readable result here (dispatcher contract)")
    mint_p.add_argument("--vram-floor", type=float, default=8.0,
                        help="refuse below this vram_free GB (default 8)")

    return p


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not getattr(args, "stage", None):
        parser.print_help()
        return 0

    if args.stage == "audit-text":
        if args.tokenizer:
            from continuity.auditor.prompt_lint import audit_prompt
            result = audit_prompt(args.text, args.tokenizer)
        else:
            from continuity.auditor.prompt_lint import audit_prompt_text
            result = audit_prompt_text(args.text)
        print(result.summary())
        return 0 if result.ship_ok else 1

    if args.stage == "keyframe":
        from continuity.keyframe.lane import ShipGateError, load_deck, render_deck
        deck = load_deck(args.beats)
        try:
            result = render_deck(
                deck, endpoint=args.endpoint, out_dir=args.out,
                tokenizer_path=args.tokenizer, prefix=args.prefix,
                limit=args.limit, dry_run=args.dry_run,
            )
        except ShipGateError as exc:
            print(f"SHIP GATE FAILED — no GPU time spent:\n{exc}", file=sys.stderr)
            return 2
        for beat_id, entry in result["manifest"].items():
            print(f"  {beat_id}: {entry['tokens']} tok -> {entry['file'] or entry.get('note')}")
        print(f"manifest: {result['manifest_path']}")
        if result["contact_sheet"]:
            print(f"contact sheet: {result['contact_sheet']}")
        return 0

    if args.stage == "gate-a":
        from continuity.gate_a.gate import gate_manifest
        from continuity.gate_a.providers import ReplayProvider, provider_from_env
        from continuity.keyframe.lane import load_deck
        deck = load_deck(args.beats)
        if args.replay:
            with open(args.replay, "r", encoding="utf-8") as fh:
                provider = ReplayProvider(json.load(fh))
        else:
            provider = provider_from_env()
        summary = gate_manifest(args.manifest, deck, provider,
                                out_dir=args.out, passes=args.passes)
        for beat, path in summary["reports"].items():
            print(f"  {beat}: {path}")
        print(f"all_pass: {summary['all_pass']} "
              f"({summary['passed']}/{summary['gated']} passed, "
              f"failed: {summary['failed'] or 'none'})")
        return 0 if summary["all_pass"] else 1

    if args.stage == "gate-b":
        import json as _json
        from continuity.gate_a.gate import GateSpec
        from continuity.gate_a.providers import ReplayProvider, provider_from_env
        import os as _os
        from continuity.gate_b.gate import gate_mint, report_markdown, resolve_beat, summarize
        from continuity.keyframe.lane import load_deck

        deck = load_deck(args.beats)
        specs = {b.id: GateSpec(getattr(b, "qc", None)) for b in deck.beats}
        keyframes, seeds = {}, {}
        if args.keyframes:
            with open(args.keyframes, "r", encoding="utf-8") as fh:
                keyframes = _json.load(fh)
        if args.seeds:
            with open(args.seeds, "r", encoding="utf-8") as fh:
                seeds = _json.load(fh)
        provider = None
        if not args.no_vision:
            if args.replay:
                with open(args.replay, "r", encoding="utf-8") as fh:
                    provider = ReplayProvider(_json.load(fh))
            else:
                provider = provider_from_env()

        import os as _os
        reports = []
        for clip in args.clips:
            beat = resolve_beat(clip, specs)
            r = gate_mint(clip, beat, specs.get(beat, GateSpec(None)),
                          seed=seeds.get(beat),
                          keyframe_path=keyframes.get(beat, {}).get("file")
                          if isinstance(keyframes.get(beat), dict) else keyframes.get(beat),
                          provider=provider, workdir=args.out,
                          samples=args.samples)
            reports.append(r)
            print(f"  {r['beat']}: {r['verdict']}"
                  + (f" — {'; '.join(r['fails'][:2])}" if r["fails"] else ""))
        summary = summarize(reports)
        md = report_markdown(reports, summary)
        md_path = _os.path.join(args.out, "gate_b_report.md")
        with open(md_path, "w", encoding="utf-8") as fh:
            fh.write(md)
        with open(_os.path.join(args.out, "gate_b_summary.json"), "w",
                  encoding="utf-8") as fh:
            _json.dump(summary, fh, indent=1)
        print(f"summary: {summary['passed']}/{summary['gated']} PASS, "
              f"cost ${summary['total_cost_usd']} -> {md_path}")
        if args.master:
            from continuity.gate_b.master import assemble_master
            if not any(r["verdict"] == "PASS" for r in reports):
                print("master: skipped — no PASS clips to assemble")
            else:
                m = assemble_master(reports, args.master, args.out)
                print(f"master: {m['master']} "
                      f"(included {len(m['included'])}, excluded {len(m['excluded'])})")
        return 0 if summary["all_pass"] else 1

    if args.stage == "mint":
        from continuity.mint.lane import mint_one
        result = mint_one(
            manifest_path=args.manifest, beats_path=args.beats, beat_id=args.beat,
            seed=args.seed, endpoint=args.endpoint, out_dir=args.out,
            duration_s=args.duration, width=args.width, height=args.height,
            gate_a_report=args.gate_a_report, vram_floor_gb=args.vram_floor,
        )
        if args.result_json:
            import json as _j
            import os as _os
            _os.makedirs(_os.path.dirname(_os.path.abspath(args.result_json)), exist_ok=True)
            with open(args.result_json, "w", encoding="utf-8") as fh:
                _j.dump(result, fh, indent=1)
        if result["ok"]:
            print(f"minted {result['beat']} s{result['seed']}: {result['clip']} "
                  f"({result.get('frames')}f, {result.get('wall_s')}s, "
                  f"vram_end {result.get('vram_end_gb')}GB)")
        else:
            print(f"MINT REFUSED/FAILED [{result['exit_code']}]: {result.get('reason')}",
                  file=sys.stderr)
        return result["exit_code"]

    if args.stage == "run":
        from continuity.orchestrator.pipeline import (
            PipelineConfig, UsageError, run as pipeline_run,
        )
        from continuity.keyframe.lane import ShipGateError as _SGE
        config = PipelineConfig(
            beats_path=args.beats, charsheet_path=args.charsheet,
            out_root=args.out_root, endpoint=args.endpoint,
            tokenizer_path=args.tokenizer, dry_run=args.dry_run,
            limit=args.limit, prefix=args.prefix,
            video_seed_base=args.vseed_base, duration_s=args.duration,
            width=args.width, height=args.height,
            vram_floor_gb=args.vram_floor,
            max_mint_rounds=args.max_mint_rounds,
            gate_a_replay=args.replay,
            gate_b_vision=not args.no_gate_b_vision,
            resume_run_dir=args.resume, run_id=args.run_id)
        try:
            summary = pipeline_run(config)
        except (UsageError, _SGE) as exc:
            print(f"ABORTED (no GPU time spent): {exc}", file=sys.stderr)
            return 2
        statuses = summary["statuses"]
        for bid, st in statuses.items():
            print(f"  {bid}: {st}")
        print(f"state: {summary['state_path']}")
        if summary.get("master"):
            m = summary["master"]
            note = (" — DECLARED INCOMPLETE: "
                    + ", ".join(x["beat"] for x in m.get("missing_beats", []))
                    if m.get("declared_incomplete") else "")
            print(f"master: {m['master']} ({len(m['included'])} shots)"
                  f"{note}")
        u = summary["usage"]
        print(f"wall {u.get('wall_s')}s · gpu {u.get('gpu_seconds')}s · "
              f"vision ${u.get('vision_cost_usd')} · retries {u.get('retries')}"
              + (f" · interventions {len(u.get('human_interventions', []))}"
                 if u.get("human_interventions") else ""))
        if summary["paused"]:
            print("run PAUSED (GPU busy) — resume with --resume <run_dir>",
                  file=sys.stderr)
            return 1
        return 0 if summary["ok"] else 1

    print(f"stage '{args.stage}' is not implemented", file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main())
