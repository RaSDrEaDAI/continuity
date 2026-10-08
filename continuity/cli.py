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
    sub.add_parser("keyframe", help="T4: beats -> FLUX keyframes")
    sub.add_parser("gate-a", help="T5: keyframe vision instruments")
    sub.add_parser("mint", help="T6: keyframes -> LTX-2.5 i2v clips")
    sub.add_parser("gate-b", help="T7: drift scan + consistency + motion index")

    audit_p = sub.add_parser("audit-text", help="quick audit of one prompt string")
    audit_p.add_argument("--text", required=True)

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
        from continuity.auditor.prompt_lint import audit_prompt_text
        result = audit_prompt_text(args.text)
        print(result.summary())
        return 0 if result.ship_ok else 1

    print(f"stage '{args.stage}' wired; module implementation lands with its ticket "
          f"(see README status table)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
