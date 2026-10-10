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
    sub.add_parser("mint", help="T6: keyframes -> LTX-2.5 i2v clips")
    sub.add_parser("gate-b", help="T7: drift scan + consistency + motion index")

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

    run_p = sub.add_parser("run", help="orchestrate the full pipeline")
    run_p.add_argument("--beats", required=True, help="path to beats.json")
    run_p.add_argument("--charsheet", help="path to charsheet prompt block")
    run_p.add_argument("--dry-run", action="store_true", help="plan without rendering")

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

    print(f"stage '{args.stage}' wired; module implementation lands with its ticket "
          f"(see README status table)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
