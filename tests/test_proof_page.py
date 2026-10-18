"""T17 proof-page tests — offline, no network, no GPU, no ffmpeg."""
import json
import os
import subprocess
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from continuity import proof_page

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIXTURES = os.path.join(REPO, "examples", "offline_demo")


@pytest.fixture(scope="module")
def recorded_gates(tmp_path_factory):
    """Run the REAL offline gates once ($0, seconds) and reuse reports."""
    tmp = tmp_path_factory.mktemp("gates")
    ga_out = tmp / "ga"
    ga_out.mkdir()
    code = subprocess.call(
        [sys.executable, "-m", "continuity.cli", "gate-a",
         "--manifest", os.path.join(FIXTURES, "offline_demo.manifest.json"),
         "--beats", os.path.join(FIXTURES, "offline_demo.beats.json"),
         "--replay", os.path.join(FIXTURES, "replay", "gate_a_answers.json"),
         "--out", str(ga_out)],
        cwd=REPO)
    assert code == 1, "fixture deck must fail gate-a (R2 is rope-dirty)"
    summary = json.load(open(ga_out / "gate_a_summary.json"))
    assert summary["failed"] == ["R2_pair"]

    # minimal recorded gate-b summary + per-beat report (deterministic scan
    # numbers copied from the recorded run — keeps this test ffmpeg-free)
    gb_out = tmp / "gb"
    gb_out.mkdir()
    (gb_out / "gate_b_summary.json").write_text(json.dumps({
        "gated": 2, "passed": 1, "failed": ["R3_driftdusk"],
        "needs_review": [], "all_pass": False, "total_cost_usd": 0,
        "gate_b_version": "gate-b/1"}))
    (gb_out / "R1_envmacro.gate_b.json").write_text(json.dumps({
        "beat": "R1_envmacro", "verdict": "PASS", "fails": []}))
    (gb_out / "R3_driftdusk.gate_b.json").write_text(json.dumps({
        "beat": "R3_driftdusk", "verdict": "FAIL",
        "fails": ["drift: gradual_drift (swing 30.2 maxstep 6.4)"],
        "drift": {"swing": 30.2, "max_step": 6.4,
                  "fail_class": "gradual_drift"}}))
    return ga_out, gb_out


def test_build_data_reads_fixtures(recorded_gates):
    ga, gb = recorded_gates
    data = proof_page.build_data(FIXTURES, str(ga), str(gb))
    ids = [b["id"] for b in data["beats"]]
    assert ids == ["R1_envmacro", "R2_pair", "R3_driftdusk"]
    assert all(b["tokens"] and b["tokens"] < 250 for b in data["beats"])
    assert "Signal Nomad" in data["charsheet"]


def test_markdown_carries_the_rejection_story(recorded_gates):
    ga, gb = recorded_gates
    data = proof_page.build_data(FIXTURES, str(ga), str(gb))
    md = proof_page.render_markdown(data)
    # the three verdicts
    assert "R1_envmacro — **PASS**" in md
    assert "R2_pair — **CONDEMNED**" in md
    assert "R3_driftdusk — **PASS**" in md
    # verbatim rope instrument line (two passes agree)
    assert "FREE-HANGING SEGMENT: YES" in md
    # cross-subject facing catch, computed not asked
    assert "man RIGHT vs camel LEFT" in md
    # gate-b drift catch with the measured number
    assert "swing **30.2**" in md
    # never-minted economics
    assert "never reaches the mint stage" in md
    # repo-relative images only for markdown (no absolute cloudinary <img>)
    assert "](examples/offline_demo/frames/" in md
    # replay commands present
    assert "gate-a --manifest" in md and "--no-vision" in md


def test_html_public_safety_and_story(recorded_gates):
    ga, gb = recorded_gates
    data = proof_page.build_data(FIXTURES, str(ga), str(gb))
    html = proof_page.render_html(data)
    assert proof_page.audit_public_safety(html) == []
    # every absolute URL emitted is from the allowlist or known-public set
    allowed = set()
    for asset in proof_page.PUBLIC_ASSETS.values():
        for u in asset.values():
            if u:
                allowed.add(u)
    allowed.add(proof_page.DEMO_VIDEO)
    allowed.add(proof_page.REPO_URL)
    import re
    for m in re.finditer(r"(?:src|href)='([^']+)'", html):
        assert m.group(1) in allowed, f"non-allowlisted URL: {m.group(1)}"
    # replay banner + verdicts
    assert "REPLAY of a recorded run" in html
    assert "R2_pair" in html and "CONDEMNED" in html
    assert "swing <b>30.2</b>" in html


def test_audit_public_safety_catches_leaks():
    assert proof_page.audit_public_safety("<a href='http://172.26.1.5/x'>") == ["172.26."]
    assert proof_page.audit_public_safety("path D:/secret") == ["D:/"]
    assert proof_page.audit_public_safety("clean page") == []


def test_cli_proof_page_end_to_end(recorded_gates, tmp_path):
    ga, gb = recorded_gates
    out_md = tmp_path / "PROOF.md"
    out_html = tmp_path / "proof_page.html"
    code = subprocess.call(
        [sys.executable, "-m", "continuity.cli", "proof-page",
         "--fixtures", FIXTURES, "--gate-a-out", str(ga),
         "--gate-b-out", str(gb),
         "--out-md", str(out_md), "--out-html", str(out_html)],
        cwd=REPO)
    assert code == 0
    assert out_md.exists() and out_html.exists()
    assert "FREE-HANGING SEGMENT: YES" in out_md.read_text(encoding="utf-8")
