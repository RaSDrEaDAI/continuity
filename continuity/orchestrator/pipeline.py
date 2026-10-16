"""End-to-end pipeline — beats + charsheet in, QC-gated cut out (T15).

Drives the real lanes beat-by-beat with crash-safe persisted state:

    load deck (strict validation — unknown keys are errors, never silently
    dropped: a shorthand ``lora`` key once rendered a whole deck WITHOUT its
    style LoRA)
      -> T3 audit (exact T5 tokenizer in production; word-estimate is a
         labeled dry-run mode only)
      -> T4 keyframe (resume-safe by frame sha1 + prompt hash)
      -> T5 Gate A   (fail: seed-ladder re-roll, max ``max_keyframe_strikes``;
             then the beat is ESCALATED to the operator — comp change, never
             a third text patch)
      -> T6 mint via :func:`continuity.mint.lane.mint_one` — the same bounded
         single-shot route the RasOrchestrator dispatcher invokes, Gate A
         report passed in so a dirty frame can never mint (enforced twice:
         here and inside mint_one)
      -> T7 Gate B   (fail: hardened re-mint at seed+100 per the reroll
             doctrine; sudden-jump and golden-hour classes stop for operator
             taste calls — bounded, never a third mint)
      -> ordered master from PASS clips, with every missing beat DECLARED
         (a cut missing a beat is an incomplete deliverable, not a success)

State: ``<out_root>/<run_id>/state.json`` — spec/input/model/workflow hashes,
seed + repair history, checked/failed/skipped gates, artifact checksums,
usage (wall time, GPU seconds, vision cost, retries, human interventions).
Resume never repeats approved work and invalidates stale reuse when the
prompt, seed, model spec or canon text changes.

Charsheet honesty: the deck ``charsheet`` is a TEXT prompt block, and the
optional charsheet image is a Gate A scale-probe reference only. There is no
image-identity conditioning in this lane — the run report says so explicitly.
"""

from __future__ import annotations

import datetime as _dt
import hashlib
import json
import os
import time

from continuity.auditor.prompt_lint import audit_prompt, audit_prompt_text
from continuity.keyframe.lane import (
    ShipGateError,
    build_graph,
    build_prompt,
    queue_prompt,
    sha1_of,
    wait_for_prompt,
)
from continuity.gate_a.gate import GateSpec, gate_frame
from continuity.gate_a.providers import ReplayProvider, provider_from_env
from continuity.gate_b.gate import gate_mint
from continuity.gate_b.master import assemble_master
from continuity.gate_b.reroll import plan_reroll
from continuity.mint.lane import EXIT_GPU_BUSY, mint_one

PIPELINE_VERSION = "pipeline/1"
BEAT_KEYS = {"id", "seed", "locks", "scene", "tod", "camera", "qc",
             "charsheet", "width", "height"}


class UsageError(RuntimeError):
    """Bad invocation — no GPU time, no network calls."""


def _utcnow() -> str:
    return _dt.datetime.now(_dt.timezone.utc).replace(microsecond=0).isoformat()


def _sha1_bytes(b: bytes) -> str:
    return hashlib.sha1(b).hexdigest()


def _sha1_obj(obj) -> str:
    return _sha1_bytes(json.dumps(obj, sort_keys=True).encode())


# ── config ───────────────────────────────────────────────────────────────────

class PipelineConfig:
    def __init__(
        self,
        beats_path: str,
        charsheet_path: str | None = None,
        out_root: str = "runs",
        endpoint: str = "http://127.0.0.1:8189",
        tokenizer_path: str | None = None,
        dry_run: bool = False,
        limit: int | None = None,
        prefix: str = "cont15",
        video_seed_base: int = 3001,
        duration_s: float = 12.25,
        width: int = 1248,
        height: int = 704,
        vram_floor_gb: float = 8.0,
        max_keyframe_strikes: int = 2,
        max_mint_rounds: int = 2,
        gate_a_replay: str | None = None,
        gate_b_vision: bool = True,
        resume_run_dir: str | None = None,
        run_id: str | None = None,
    ):
        self.beats_path = beats_path
        self.charsheet_path = charsheet_path
        self.out_root = out_root
        self.endpoint = endpoint.rstrip("/")
        self.tokenizer_path = tokenizer_path
        self.dry_run = dry_run
        self.limit = limit
        self.prefix = prefix
        self.video_seed_base = video_seed_base
        self.duration_s = duration_s
        self.width = width
        self.height = height
        self.vram_floor_gb = vram_floor_gb
        self.max_keyframe_strikes = max_keyframe_strikes
        self.max_mint_rounds = max_mint_rounds
        self.gate_a_replay = gate_a_replay
        self.gate_b_vision = gate_b_vision
        self.resume_run_dir = resume_run_dir
        self.run_id = run_id

    def validate(self) -> None:
        if not self.dry_run and not self.tokenizer_path:
            raise UsageError(
                "production runs require --tokenizer (exact T5 counts); "
                "word-estimate is a labeled dry-run mode only")
        if not self.dry_run and not os.path.isfile(self.beats_path):
            raise UsageError(f"beats file not found: {self.beats_path}")
        if self.charsheet_path and not os.path.isfile(self.charsheet_path):
            raise UsageError(f"charsheet image not found: {self.charsheet_path}")
        if self.max_keyframe_strikes < 1 or self.max_mint_rounds < 0:
            raise UsageError("strike/round bounds must be >= 1 / >= 0")
        if self.duration_s <= 0 or self.width <= 0 or self.height <= 0:
            raise UsageError("duration/width/height must be positive")


# ── state ────────────────────────────────────────────────────────────────────

def _load_deck_strict(path: str):
    """load_deck + strict key validation (unknown keys are errors, not drops)."""
    from continuity.keyframe.lane import load_deck

    with open(path, "r", encoding="utf-8") as fh:
        raw = json.load(fh)
    unknown_deck = set(raw) - {"charsheet", "style", "model", "notes", "beats"}
    if unknown_deck:
        raise UsageError(
            f"deck {path}: unknown top-level keys {sorted(unknown_deck)} "
            f"(valid: charsheet, style, model, notes, beats)")
    model = raw.get("model") or {}
    unknown_model = set(model) - {
        "unet", "unet_dtype", "clip_l", "t5", "vae", "loras", "steps", "cfg",
        "sampler", "scheduler", "guidance", "width", "height"}
    if unknown_model:
        # A shorthand 'lora'/'checkpoint' key here once rendered a whole deck
        # WITHOUT its style LoRA — silently. Fail instead.
        raise UsageError(
            f"deck {path}: unknown model keys {sorted(unknown_model)} "
            f"(valid: unet, unet_dtype, clip_l, t5, vae, loras, steps, cfg, "
            f"sampler, scheduler, guidance, width, height) — LoRAs that "
            f"misspell their key are dropped silently otherwise")
    for i, b in enumerate(raw.get("beats", [])):
        unknown = set(b) - BEAT_KEYS
        if unknown:
            raise UsageError(
                f"deck {path}: beat #{i} ({b.get('id', '?')}): unknown keys "
                f"{sorted(unknown)} (valid: {sorted(BEAT_KEYS)})")
        if not str(b.get("id", "")).strip():
            raise UsageError(f"deck {path}: beat #{i} has an empty id")
    ids = [b["id"] for b in raw.get("beats", [])]
    dupes = {i for i in ids if ids.count(i) > 1}
    if dupes:
        raise UsageError(f"deck {path}: duplicate beat ids {sorted(dupes)}")
    return load_deck(path)


class RunState:
    """Crash-safe run state — written after every stage transition."""

    def __init__(self, run_dir: str):
        self.path = os.path.join(run_dir, "state.json")
        self.doc: dict = {}

    def load(self) -> bool:
        if not os.path.isfile(self.path):
            return False
        with open(self.path, "r", encoding="utf-8") as fh:
            self.doc = json.load(fh)
        return True

    def init(self, config: PipelineConfig, fingerprint: dict,
             tokenizer_mode: str) -> None:
        self.doc = {
            "run_id": os.path.basename(os.path.dirname(self.path)),
            "pipeline_version": PIPELINE_VERSION,
            "created_at": _utcnow(), "updated_at": _utcnow(),
            "status": "running",
            "config": {
                "beats_path": os.path.abspath(config.beats_path),
                "charsheet_path": (os.path.abspath(config.charsheet_path)
                                   if config.charsheet_path else None),
                "endpoint": config.endpoint, "prefix": config.prefix,
                "dry_run": config.dry_run, "limit": config.limit,
                "video_seed_base": config.video_seed_base,
                "duration_s": config.duration_s,
                "width": config.width, "height": config.height,
                "vram_floor_gb": config.vram_floor_gb,
                "max_keyframe_strikes": config.max_keyframe_strikes,
                "max_mint_rounds": config.max_mint_rounds,
            },
            "tokenizer_mode": tokenizer_mode,
            "fingerprint": fingerprint,
            "charsheet_definition": (
                "TEXT prompt block + optional Gate A scale-probe image; "
                "NOT image-identity conditioning — identity is carried by "
                "the frozen text block only"),
            "beats": {},
            "master": None,
            "usage": {"wall_s": None, "gpu_seconds": 0.0,
                      "vision_cost_usd": 0.0, "retries": 0,
                      "human_interventions": []},
        }

    def beat(self, beat_id: str) -> dict:
        return self.doc.setdefault("beats", {}).setdefault(
            beat_id, {"status": "pending"})

    def save(self) -> None:
        self.doc["updated_at"] = _utcnow()
        tmp = self.path + ".tmp"
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(self.doc, fh, indent=1)
        os.replace(tmp, self.path)


# ── the pipeline ─────────────────────────────────────────────────────────────

def _tod_class(beat) -> str:
    tod = (getattr(beat, "tod", "") or "").lower()
    qc = getattr(beat, "qc", None) or {}
    explicit = qc.get("tod_class")
    if explicit:
        return str(explicit)
    if "gold" in tod or "dusk" in tod:
        return "low_gold"
    return "day"


def _render_one(config, deck, beat, seed, run_dir, prompt):
    """Render one keyframe; resume-safe per (beat, seed). Returns abs path."""
    import glob as _glob
    comfy_out = os.environ.get("CONTINUITY_COMFY_OUTPUT",
                               r"D:\1.AI\Comfy\output")
    os.makedirs(comfy_out, exist_ok=True)
    # resume matches on the ACTUAL seed (strike ladders change it — globbing
    # the beat's base seed would re-find the original frame forever)
    hits = sorted(_glob.glob(
        os.path.join(comfy_out, f"{config.prefix}/{beat.id}_s{seed}*.png")),
        key=os.path.getmtime)
    if hits:
        return hits[-1]
    prefix_full = f"{config.prefix}/{beat.id}_s{seed}"
    graph = build_graph(beat, prompt, deck, prefix_full)
    graph["ksampler"]["inputs"]["seed"] = seed
    pid = queue_prompt(config.endpoint, graph)
    res = wait_for_prompt(config.endpoint, pid, comfy_out)
    local = next((f["local"] for f in res["files"] if f["local"]), None)
    if not local:
        raise RuntimeError(
            f"[{beat.id}] keyframe render failed: {json.dumps(res)[:300]}")
    return local


def _write_mint_manifest(run_dir: str, beat_id: str, frame: str, seed: int,
                         prompt: str) -> str:
    path = os.path.join(run_dir, "keyframe_manifest.json")
    manifest: dict = {}
    if os.path.isfile(path):
        with open(path, "r", encoding="utf-8") as fh:
            manifest = json.load(fh)
    manifest[beat_id] = {"beat": beat_id, "seed": seed, "file": frame,
                         "prompt": prompt}
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=1)
    return path


def run(config: PipelineConfig) -> dict:
    """Execute the full pipeline; returns the run state summary.

    Exit semantics for the CLI: ``summary["ok"]`` True → 0; completed with
    failed/escalated beats → ok False (master declared incomplete); usage /
    ship-gate errors raise :class:`UsageError` / :class:`ShipGateError`.
    """
    t0 = time.time()
    config.validate()
    dry = config.dry_run

    # ── inputs + fingerprint (hashes BEFORE any work) ──────────────────────
    deck = _load_deck_strict(config.beats_path)
    with open(config.beats_path, "rb") as fh:
        beats_sha1 = _sha1_bytes(fh.read())
    model_hash = _sha1_obj(
        {k: getattr(deck.model, k) for k in
         ("unet", "unet_dtype", "clip_l", "t5", "vae",
          "steps", "cfg", "sampler", "scheduler", "guidance", "width", "height")}
        | {"loras": [(l.name, l.strength) for l in deck.model.loras]})
    canon_hash = _sha1_obj({"charsheet": deck.charsheet,
                            "style": deck.style})
    fingerprint = {"beats_sha1": beats_sha1, "model_hash": model_hash,
                   "canon_hash": canon_hash}

    beats = deck.beats[:config.limit] if config.limit else deck.beats

    # ── T3 audit: every prompt, before any GPU second ──────────────────────
    tokenizer_mode = ("estimate (DRY RUN — not production-grade counts)"
                      if dry or not config.tokenizer_path else "exact")
    audits = {}
    for beat in beats:
        prompt = build_prompt(beat, deck)
        audits[beat.id] = (
            audit_prompt_text(prompt) if dry or not config.tokenizer_path
            else audit_prompt(prompt, config.tokenizer_path))
    bad = [b for b in beats if not audits[b.id].ship_ok]
    if bad:
        raise ShipGateError(
            f"ship gate failed for {[(b.id, audits[b.id].total_tokens) for b in bad]}"
            f" ({tokenizer_mode}) — no GPU time spent")

    # ── run dir + resume ────────────────────────────────────────────────────
    if config.resume_run_dir:
        run_dir = os.path.abspath(config.resume_run_dir)
    else:
        rid = config.run_id or (
            "t15_" + _dt.datetime.now().strftime("%Y%m%d_%H%M%S"))
        run_dir = os.path.join(config.out_root, rid)
    os.makedirs(run_dir, exist_ok=True)
    gate_a_dir = os.path.join(run_dir, "gate_a")
    gate_b_dir = os.path.join(run_dir, "gate_b")
    os.makedirs(gate_a_dir, exist_ok=True)
    os.makedirs(gate_b_dir, exist_ok=True)

    state = RunState(run_dir)
    resumed = state.load()
    if resumed:
        stale = []
        for k in ("beats_sha1", "model_hash", "canon_hash"):
            if state.doc.get("fingerprint", {}).get(k) != fingerprint[k]:
                stale.append(k)
        if stale:
            # changed prompts/models/canon invalidate ALL reuse — recorded,
            # never silently trusted
            for bid in list(state.doc.get("beats", {})):
                state.doc["beats"][bid] = {
                    "status": "pending",
                    "invalidated": f"inputs changed ({', '.join(stale)})",
                    "history": state.doc["beats"][bid],
                }
            state.doc["fingerprint"] = fingerprint
    if not resumed:
        state.init(config, fingerprint, tokenizer_mode)
    for i, beat in enumerate(beats):
        prompt = build_prompt(beat, deck)
        b = state.beat(beat.id)
        b["status"] = b.get("status", "pending")
        b["keyframe_seed"] = b.get("keyframe_seed", beat.seed)
        b["video_seed"] = b.get("video_seed",
                                config.video_seed_base + i)
        b["prompt_hash"] = _sha1_bytes(prompt.encode())
        b["tokens"] = audits[beat.id].total_tokens
        b["tod_class"] = _tod_class(beat)
        b.setdefault("strikes", {"keyframe": 0, "mint": 0})
        b.setdefault("rerolls", [])
    state.save()

    # ── vision provider (None in dry run: zero network) ────────────────────
    provider = None
    if not dry:
        if config.gate_a_replay:
            with open(config.gate_a_replay, "r", encoding="utf-8") as fh:
                provider = ReplayProvider(json.load(fh))
        else:
            provider = provider_from_env()

    usage = state.doc["usage"]

    def cost(add: float) -> None:
        usage["vision_cost_usd"] = round(
            usage.get("vision_cost_usd", 0.0) + (add or 0.0), 5)

    def intervene(note: str) -> None:
        usage.setdefault("human_interventions", []).append(note)

    specs = {b.id: GateSpec(getattr(b, "qc", None)) for b in beats}
    if config.charsheet_path:
        for spec in specs.values():
            if not spec.charsheet:
                spec.charsheet = os.path.abspath(config.charsheet_path)

    gate_b_reports: list[dict] = []

    for beat in beats:
        b = state.beat(beat.id)
        prompt = build_prompt(beat, deck)
        spec = specs[beat.id]

        # 1 ── keyframe ─────────────────────────────────────────────────────
        frame = b.get("frame", {})
        reusable = (
            frame.get("path") and os.path.isfile(frame["path"])
            and frame.get("sha1") == (sha1_of(frame["path"])
                                      if os.path.isfile(frame["path"]) else None)
            and b.get("prompt_hash") == _sha1_bytes(prompt.encode()))
        if dry:
            b["status"] = "planned:dry_run"
            b["gate_a"] = {"verdict": "skipped:dry_run"}
            b["gate_b"] = {"verdict": "skipped:dry_run"}
            state.save()
            continue
        if reusable:
            b["status"] = "keyframed"
            b.setdefault("notes", []).append("frame resumed (sha1 match)")
        else:
            b["status"] = "keyframing"
            state.save()
            frame_path = _render_one(config, deck, beat,
                                     b["keyframe_seed"], run_dir, prompt)
            b["frame"] = {"path": os.path.abspath(frame_path),
                          "sha1": sha1_of(frame_path)}
            b["status"] = "keyframed"
        state.save()

        # 2 ── Gate A (dirty frames never mint; PASS verdicts are reusable) ─
        ga = b.get("gate_a", {})
        ga_reusable = (
            ga.get("verdict") == "PASS"
            and ga.get("frame_sha1") == b["frame"]["sha1"])
        if not ga_reusable:
            report = gate_frame(b["frame"]["path"], spec, provider,
                                beat=beat.id, seed=b["keyframe_seed"])
            report["frame_sha1"] = b["frame"]["sha1"]
            report["prompt_hash"] = b["prompt_hash"]
            with open(os.path.join(gate_a_dir, f"{beat.id}.json"), "w",
                      encoding="utf-8") as fh:
                json.dump(report, fh, indent=1)
            b["gate_a"] = {"verdict": report["verdict"],
                           "frame_sha1": b["frame"]["sha1"],
                           "fails": report["fails"],
                           "report": os.path.join(gate_a_dir, f"{beat.id}.json")}
            cost(report.get("usage", {}).get("cost_usd", 0))
            state.save()

        strikes = b["strikes"]
        while b["gate_a"]["verdict"] != "PASS":
            if strikes["keyframe"] >= config.max_keyframe_strikes:
                b["status"] = "escalated"
                note = (f"{beat.id}: keyframe comp wall after "
                        f"{strikes['keyframe']} strikes "
                        f"({'; '.join(b['gate_a'].get('fails', []))[:200]}) "
                        f"— operator comp change required, never a third "
                        f"text patch")
                b.setdefault("notes", []).append(note)
                intervene(note)
                state.save()
                break
            strikes["keyframe"] += 1
            b["keyframe_seed"] = beat.seed + strikes["keyframe"]
            usage["retries"] += 1
            b["status"] = "keyframing"
            state.save()
            frame_path = _render_one(config, deck, beat,
                                     b["keyframe_seed"], run_dir, prompt)
            b["frame"] = {"path": os.path.abspath(frame_path),
                          "sha1": sha1_of(frame_path)}
            report = gate_frame(b["frame"]["path"], spec, provider,
                                beat=beat.id, seed=b["keyframe_seed"])
            report["frame_sha1"] = b["frame"]["sha1"]
            with open(os.path.join(gate_a_dir, f"{beat.id}.json"), "w",
                      encoding="utf-8") as fh:
                json.dump(report, fh, indent=1)
            b["gate_a"] = {"verdict": report["verdict"],
                           "frame_sha1": b["frame"]["sha1"],
                           "fails": report["fails"],
                           "report": os.path.join(gate_a_dir, f"{beat.id}.json")}
            cost(report.get("usage", {}).get("cost_usd", 0))
            state.save()

        if b["status"] == "escalated":
            continue

        # 3 ── mint (Gate A report handed to mint_one: enforced twice) ──────
        clip = b.get("clip", {})
        clip_reusable = (
            clip.get("path") and os.path.isfile(clip["path"])
            and clip.get("sha1") == sha1_of(clip["path"])
            and b.get("gate_b", {}).get("verdict") == "PASS"
            and b.get("gate_b", {}).get("clip_sha1") == clip.get("sha1"))
        if clip_reusable:
            b["status"] = "gate_b_pass"  # approved work: reuse, never repeat
            state.save()
            gate_b_reports.append(_reload_gate_b(gate_b_dir, beat.id, b))
            continue

        manifest_path = _write_mint_manifest(
            run_dir, beat.id, b["frame"]["path"], b["keyframe_seed"], prompt)
        b["status"] = "minting"
        state.save()
        result = mint_one(
            manifest_path=manifest_path, beats_path=config.beats_path,
            beat_id=beat.id, seed=b["video_seed"], endpoint=config.endpoint,
            out_dir=os.path.join(run_dir, "clips"),
            duration_s=config.duration_s, width=config.width,
            height=config.height,
            gate_a_report=gate_a_dir,
            vram_floor_gb=config.vram_floor_gb)
        if result.get("exit_code") == EXIT_GPU_BUSY:
            b["status"] = "paused_gpu_busy"
            note = (f"{beat.id}: GPU busy (vram below floor "
                    f"{config.vram_floor_gb}GB) — run paused, resume later")
            b.setdefault("notes", []).append(note)
            state.doc["status"] = "paused_gpu_busy"
            state.save()
            break
        if not result.get("ok"):
            b["status"] = "mint_failed"
            note = f"{beat.id}: mint failed — {result.get('reason', '?')[:300]}"
            b.setdefault("notes", []).append(note)
            intervene(note)
            state.save()
            continue
        usage["gpu_seconds"] = round(
            usage.get("gpu_seconds", 0.0) + (result.get("wall_s") or 0), 1)
        b["clip"] = {"path": result["clip"], "seed": result["seed"],
                     "sha1": sha1_of(result["clip"]),
                     "wall_s": result.get("wall_s")}
        b["status"] = "minted"
        state.save()

        # 4 ── Gate B + bounded hardened re-mint loop ───────────────────────
        suffix = ""
        while True:
            gb_provider = provider if config.gate_b_vision else None
            report = gate_mint(b["clip"]["path"], beat.id, spec,
                               seed=b["clip"]["seed"],
                               keyframe_path=b["frame"]["path"],
                               provider=gb_provider,
                               workdir=gate_b_dir)
            report["clip_sha1"] = b["clip"]["sha1"]
            cost(report.get("usage", {}).get("cost_usd", 0))
            gate_b_reports.append(report)
            if report["verdict"] == "PASS":
                b["gate_b"] = {"verdict": "PASS",
                               "clip_sha1": b["clip"]["sha1"],
                               "fails": []}
                b["status"] = "gate_b_pass"
                state.save()
                break
            b["gate_b"] = {"verdict": "FAIL",
                           "clip_sha1": b["clip"]["sha1"],
                           "fails": report["fails"]}
            state.save()
            attempt = strikes["mint"] + 1
            plan = plan_reroll(report, b["clip"]["seed"], attempt=attempt,
                               tod_class=b["tod_class"])
            b["rerolls"].append({
                "from_seed": plan.from_seed, "to_seed": plan.to_seed,
                "reason": plan.reason, "round": attempt,
                "allowed": plan.allowed, "note": plan.note})
            if not plan.allowed or strikes["mint"] >= config.max_mint_rounds:
                b["status"] = "escalated" if plan.allowed else "operator_call"
                note = (f"{beat.id}: Gate B FAIL ({plan.reason}) — "
                        f"{plan.note or 'mint rounds exhausted'}")
                b.setdefault("notes", []).append(note)
                intervene(note)
                state.save()
                break
            strikes["mint"] += 1
            usage["retries"] += 1
            b["video_seed"] = plan.to_seed
            suffix = plan.suffix()
            rr = mint_one(
                manifest_path=manifest_path, beats_path=config.beats_path,
                beat_id=beat.id, seed=b["video_seed"],
                endpoint=config.endpoint,
                out_dir=os.path.join(run_dir, "clips"),
                duration_s=config.duration_s, width=config.width,
                height=config.height, gate_a_report=gate_a_dir,
                vram_floor_gb=config.vram_floor_gb,
                prompt_suffix=suffix)
            if rr.get("exit_code") == EXIT_GPU_BUSY:
                b["status"] = "paused_gpu_busy"
                state.doc["status"] = "paused_gpu_busy"
                state.save()
                break
            if not rr.get("ok"):
                b["status"] = "mint_failed"
                intervene(f"{beat.id}: re-mint failed — "
                          f"{rr.get('reason', '?')[:300]}")
                state.save()
                break
            usage["gpu_seconds"] = round(
                usage.get("gpu_seconds", 0.0) + (rr.get("wall_s") or 0), 1)
            b["clip"] = {"path": rr["clip"], "seed": rr["seed"],
                         "sha1": sha1_of(rr["clip"]),
                         "wall_s": rr.get("wall_s")}
            b["status"] = "minted"
            state.save()
        if state.doc["status"] == "paused_gpu_busy":
            break

    # 5 ── ordered master (missing beats DECLARED, never silent) ────────────
    summary_beats = {bid: state.doc["beats"][bid]
                     for bid in [b.id for b in beats]}
    if not dry and gate_b_reports:
        order = [b.id for b in beats]
        reports_by_beat = {r["beat"]: r for r in gate_b_reports}
        ordered = [reports_by_beat[bid] for bid in order if bid in reports_by_beat]
        master_path = os.path.join(run_dir, "master.mp4")
        m = assemble_master(ordered, master_path, gate_b_dir)
        missing = [bid for bid in order
                   if summary_beats[bid].get("status") != "gate_b_pass"]
        m["declared_incomplete"] = bool(missing)
        m["missing_beats"] = [
            {"beat": bid, "status": summary_beats[bid].get("status"),
             "why": "; ".join(summary_beats[bid].get("notes", []))[:300]
             or "; ".join(summary_beats[bid].get("gate_b", {}).get("fails",
                                                                    []))[:300]}
            for bid in missing]
        state.doc["master"] = m

    # ── final usage + report ────────────────────────────────────────────────
    usage["wall_s"] = round(time.time() - t0, 1)
    statuses = {bid: v.get("status")
                for bid, v in state.doc.get("beats", {}).items()
                if bid in {b.id for b in beats}}
    all_pass = bool(statuses) and all(s == "gate_b_pass"
                                      for s in statuses.values())
    if dry:
        all_pass = True  # ship gate already passed; stages are planned
    paused = state.doc["status"] == "paused_gpu_busy"
    state.doc["status"] = ("done" if not paused else "paused_gpu_busy")
    state.doc["statuses"] = statuses
    state.doc["ok"] = all_pass and not paused
    state.save()

    if not dry:
        _write_report(run_dir, state.doc,
                      {bid: state.doc["beats"][bid] for bid in statuses})

    return {"ok": state.doc["ok"], "run_dir": run_dir,
            "state_path": state.path, "statuses": statuses,
            "master": state.doc.get("master"),
            "usage": usage, "paused": paused,
            "human_interventions": usage.get("human_interventions", [])}


def _reload_gate_b(gate_b_dir: str, beat_id: str, b: dict) -> dict:
    path = os.path.join(gate_b_dir, beat_id, "gate_b.json")
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def _write_report(run_dir: str, doc: dict, beats: dict) -> str:
    u = doc["usage"]
    lines = [
        "# Continuity run report",
        "",
        f"run: `{doc['run_id']}` · pipeline {PIPELINE_VERSION} · "
        f"tokenizer: {doc['tokenizer_mode']}",
        "",
        "| beat | status | kf seed | video seed | gate A | gate B | notes |",
        "|---|---|---|---|---|---|---|",
    ]
    for bid, b in beats.items():
        ga = (b.get("gate_a") or {}).get("verdict", "-")
        gb = (b.get("gate_b") or {}).get("verdict", "-")
        notes = "; ".join(b.get("notes", []))[:120]
        lines.append(
            f"| {bid} | {b.get('status')} | s{b.get('keyframe_seed', '-')} "
            f"| s{b.get('video_seed', '-')} | {ga} | {gb} | {notes} |")
    lines += [
        "",
        f"wall {u.get('wall_s')}s · gpu {u.get('gpu_seconds')}s · "
        f"vision ${u.get('vision_cost_usd')} · retries {u.get('retries')}",
        f"human interventions: {len(u.get('human_interventions', []))}",
    ]
    for note in u.get("human_interventions", []):
        lines.append(f"- {note}")
    m = doc.get("master")
    if m:
        lines += [
            "",
            f"master: `{m['master']}` — included {', '.join(m['included'])}"
            + (f"; DECLARED INCOMPLETE, missing: "
               f"{[x['beat'] for x in m['missing_beats']]}"
               if m.get("declared_incomplete") else " (complete story)"),
        ]
    lines += [
        "",
        f"charsheet: {doc['charsheet_definition']}",
    ]
    path = os.path.join(run_dir, "report.md")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    return path
