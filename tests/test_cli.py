"""CLI regression pins — version banner, audit-text happy path, unknown stage."""

import subprocess
import sys


def test_cli_version():
    out = subprocess.run(
        [sys.executable, "-m", "continuity.cli", "--version"], capture_output=True
    )
    assert out.returncode == 0
    assert b"continuity 0.1.0" in out.stdout


def test_cli_audit_text_ships_short_prompt():
    out = subprocess.run(
        [sys.executable, "-m", "continuity.cli", "audit-text",
         "--text", "a short test prompt"],
        capture_output=True,
    )
    assert out.returncode == 0
    assert b"SHIP: PASS" in out.stdout


def test_cli_unknown_subcommand_fails():
    out = subprocess.run(
        [sys.executable, "-m", "continuity.cli", "definitely-not-a-stage"],
        capture_output=True,
    )
    assert out.returncode != 0
