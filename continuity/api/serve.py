"""T8 — Continuity serve API: HTTP surface over the keyframe + mint lanes.

Serves real runs on this box (GPU local, Comfy at 127.0.0.1:8189):

    python -m continuity.api.serve            # binds ZeroTier iface, port 8191

Every route except ``/`` requires the ``x-agent-secret`` header (shared with
the orchestrator heartbeat). GPU jobs run one-at-a-time (executor=1) so a
public caller can never double-book the card. Artifacts are served back via
``/files/<relative-path>`` under the output root.

Run lifecycle: POST /runs/keyframe | /runs/mint -> {run_id}; poll GET /runs/{id}
until status=done; result carries manifest/clip paths plus /files/ URLs.
"""
from __future__ import annotations

import json
import os
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from continuity.keyframe.lane import (
    DEFAULT_ENDPOINT, ShipGateError, load_deck, render_deck, Deck,
)
from continuity.mint.lane import (
    MintSpec, build_i2v_graph, build_mint_prompt, submit_and_wait,
    upload_keyframe, vram_free_gb,
)

SECRET = os.environ.get("CONTINUITY_SECRET", "")
COMFY = os.environ.get("CONTINUITY_COMFY_URL", DEFAULT_ENDPOINT)
OUT_ROOT = Path(os.environ.get("CONTINUITY_OUT_ROOT", "D:/1.AI/Comfy/output"))
TOKENIZER = os.environ.get(
    "CONTINUITY_T5_TOKENIZER",
    "C:/Users/Owner/ComfyUI-Installs/ComfyUI/ComfyUI/comfy/text_encoders/t5_tokenizer",
)
LISTEN_HOST = os.environ.get("CONTINUITY_LISTEN_HOST", "172.26.227.76")
LISTEN_PORT = int(os.environ.get("CONTINUITY_LISTEN_PORT", "8377"))

app = FastAPI(title="continuity-serve", version="1.0.0")
_executor = ThreadPoolExecutor(max_workers=1)  # GPU exclusivity: one job at a time
RUNS: dict[str, dict] = {}


@app.middleware("http")
async def auth(req: Request, call_next):
    if req.url.path != "/" and req.headers.get("x-agent-secret") != SECRET:
        return JSONResponse({"error": "bad or missing x-agent-secret"}, status_code=401)
    return await call_next(req)


def _files_urls(paths) -> list[str]:
    urls = []
    for p in paths or []:
        try:
            rel = Path(p).resolve().relative_to(OUT_ROOT).as_posix()
            urls.append(f"/files/{rel}")
        except Exception:
            pass
    return urls


def _run_job(run_id: str, fn) -> None:
    job = RUNS[run_id]
    job.update(status="running", started=time.time())
    try:
        result = fn()
        job.update(status="done", finished=time.time(), result=result)
        files = []
        if isinstance(result, dict):
            m = result.get("manifest") or {}
            files += [e.get("file") for e in m.values() if e.get("file")]
            files += [result.get("contact_sheet")] if result.get("contact_sheet") else []
            if result.get("mp4"):
                files.append(result["mp4"])
        job["files"] = _files_urls(files)
    except Exception as e:  # fail loud into the job record
        job.update(status="error", finished=time.time(),
                   error=f"{type(e).__name__}: {e}")
        if isinstance(e, ShipGateError):
            job["no_gpu_spent"] = True


class KeyframeReq(BaseModel):
    deck: dict            # beats.json content (charsheet/style/beats/...)
    prefix: str = "api"
    limit: int | None = None
    dry_run: bool = False
    tokenizer_path: str | None = None


class MintReq(BaseModel):
    beat_id: str
    deck: dict            # same beats.json content (for beat scene text)
    keyframe_path: str
    gate_a_report: str    # REQUIRED (fail-closed): Gate A verdict file/dir;
                          # the report must show PASS for THIS beat AND name
                          # THIS exact keyframe file. Arbitrary path-only
                          # mints are rejected — dirty frames never mint.
    camera: str = "lateral_track"
    vseed: int = 2301
    duration_s: float = 12.25
    width: int = 1248
    height: int = 704
    fps: int = 24


@app.get("/")
def banner():
    return {"service": "continuity-serve", "version": "1.0.0",
            "auth": "x-agent-secret header on every other route",
            "routes": ["/health", "/runs/keyframe", "/runs/mint", "/runs/{id}", "/files/{path}"]}


@app.get("/health")
def health():
    try:
        free = vram_free_gb(COMFY)
        comfy = {"reachable": True, "vram_free_gb": round(free, 2)}
    except Exception as e:
        comfy = {"reachable": False, "error": str(e)[:200]}
    return {"ok": comfy["reachable"], "comfy": comfy, "endpoint": COMFY,
            "runs_total": len(RUNS),
            "runs_open": sum(1 for j in RUNS.values() if j["status"] in ("queued", "running"))}


@app.post("/runs/keyframe")
def run_keyframe(req: KeyframeReq):
    try:
        deck = load_deck_dict(req.deck)
    except Exception as e:
        raise HTTPException(400, f"bad deck: {e}")
    run_id = str(uuid.uuid4())[:8]
    prefix_scoped = f"api_{req.prefix}_{run_id}"   # Comfy saves under output/<prefix_scoped>/
    out_dir = str(OUT_ROOT)
    RUNS[run_id] = {"id": run_id, "kind": "keyframe", "status": "queued",
                    "created": time.time(), "params": {"prefix": prefix_scoped, "limit": req.limit}}
    _executor.submit(_run_job, run_id, lambda: render_deck(
        deck, endpoint=COMFY, out_dir=out_dir,
        tokenizer_path=req.tokenizer_path or TOKENIZER,
        prefix=prefix_scoped, limit=req.limit, dry_run=req.dry_run))
    return {"run_id": run_id, "status_url": f"/runs/{run_id}"}


def load_deck_dict(d: dict) -> Deck:
    import tempfile
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as fh:
        json.dump(d, fh)
        path = fh.name
    return load_deck(path)


def _gated_frame_path(gate_a_report: str, beat_id: str) -> str | None:
    """Frame path recorded in the Gate A verdict for this beat, if any."""
    import glob as _glob
    p = gate_a_report
    if os.path.isdir(p):
        hits = _glob.glob(os.path.join(p, f"*{beat_id}*"))
        if not hits:
            return None
        p = hits[0]
    if not os.path.isfile(p):
        return None
    try:
        with open(p, "r", encoding="utf-8") as fh:
            d = json.load(fh)
        if isinstance(d, dict) and d.get("beat") == beat_id:
            return d.get("frame")
    except Exception:
        pass
    return None


@app.post("/runs/mint")
def run_mint(req: MintReq):
    from continuity.mint.lane import _gate_verdict
    deck = load_deck_dict(req.deck)
    beat = next((b for b in deck.beats if b.id == req.beat_id), None)
    if beat is None:
        raise HTTPException(400, f"beat_id {req.beat_id} not in deck "
                                 f"({[b.id for b in deck.beats]})")
    if not Path(req.keyframe_path).exists():
        raise HTTPException(400, f"keyframe_path not found: {req.keyframe_path}")
    # Gate A enforcement (T15): verdict must be PASS for this beat, and the
    # gated frame must BE this keyframe (a PASS for a different frame is a
    # stale report, not an approval). Fail-closed on anything ambiguous.
    verdict = _gate_verdict(req.gate_a_report, req.beat_id)
    if verdict != "PASS":
        raise HTTPException(
            400, f"Gate A verdict for '{req.beat_id}' is {verdict} — "
                 f"dirty (or unproven) frames never mint. Pass a Gate A "
                 f"report showing PASS for this beat.")
    gated_frame = _gated_frame_path(req.gate_a_report, req.beat_id)
    if gated_frame and os.path.abspath(gated_frame) != os.path.abspath(req.keyframe_path):
        raise HTTPException(
            400, f"Gate A PASS names a different frame ({gated_frame}) than "
                 f"keyframe_path ({req.keyframe_path}) — stale report; "
                 f"re-gate this keyframe before minting")
    run_id = str(uuid.uuid4())[:8]

    def _mint() -> dict:
        free = vram_free_gb(COMFY)
        if free < 12:
            import urllib.request
            r = urllib.request.Request(
                COMFY + "/free",
                data=json.dumps({"unload_models": True, "free_memory": True}).encode(),
                headers={"Content-Type": "application/json"}, method="POST")
            urllib.request.urlopen(r, timeout=60)
            time.sleep(5)
        spec = MintSpec(beat_id=req.beat_id, keyframe_path=req.keyframe_path,
                        camera=req.camera, duration_s=req.duration_s,
                        width=req.width, height=req.height, fps=req.fps,
                        seed=req.vseed)
        prompt = build_mint_prompt(spec, beat.scene)
        up = upload_keyframe(COMFY, req.keyframe_path)
        prefix = f"api_mint/{req.beat_id}_s{spec.seed}"
        graph = build_i2v_graph(spec, prompt, up, filename_prefix=prefix)
        t0 = time.time()
        res = submit_and_wait(COMFY, graph, timeout_s=2400)
        dur = round(time.time() - t0)
        if res.get("error") or not res.get("outputs"):
            raise RuntimeError(f"mint failed after {dur}s: {str(res.get('error'))[:400]}")
        outs = res["outputs"]
        if isinstance(outs[0], (tuple, list)):
            mp4 = str(OUT_ROOT / outs[0][2] / outs[0][1])
        else:
            mp4 = str(outs[0])
        return {"beat": req.beat_id, "mp4": mp4, "vseed": spec.seed,
                "camera": req.camera, "wall_s": dur}

    RUNS[run_id] = {"id": run_id, "kind": "mint", "status": "queued",
                    "created": time.time(),
                    "params": {k: getattr(req, k) for k in
                               ("beat_id", "camera", "vseed", "duration_s")}}
    _executor.submit(_run_job, run_id, _mint)
    return {"run_id": run_id, "status_url": f"/runs/{run_id}"}


@app.get("/runs")
def list_runs():
    return [{"id": j["id"], "kind": j["kind"], "status": j["status"],
             "created": j["created"]} for j in RUNS.values()]


def _no_run(run_id: str):
    raise HTTPException(404, f"no such run: {run_id}")


@app.get("/runs/{run_id}")
def get_run(run_id: str):
    job = RUNS.get(run_id)
    if job is None:
        _no_run(run_id)
    return job


app.mount("/files", StaticFiles(directory=str(OUT_ROOT)), name="files")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=LISTEN_HOST, port=LISTEN_PORT, log_level="info")
