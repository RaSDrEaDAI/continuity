"""Official Comfy SDK endpoint — Comfy API v2 via comfy-sdk (T13).

A second, explicit endpoint class next to the native ``ComfyEndpoint``:

    ComfyEndpoint      native ComfyUI /prompt + /history  (local mode, default)
    ComfySdkEndpoint   official ``comfy_sdk`` client speaking Comfy API v2
                       against comfy-api-proxy (or a managed Comfy API URL)

Mode selection is explicit — ``CONTINUITY_COMFY_MODE=native|sdk`` — so local
generation and the Comfy Developer Platform path are never conflated. The SDK
path exists to satisfy the "built with Comfy Developer Platform" requirement
with a SUPPORTED client (pinned versions in ``pyproject.toml``), while the
engine stays 100% local on this box.

Surface (read from the installed SDK, comfy-sdk==0.5.0):
    WorkflowFactory.from_json(dict) -> Workflow
    Comfy.submit(workflow) -> Job            (Job.id / .status / .outputs)
    comfy_sdk.jobs.Job(low, low_job)         (refresh/status/outputs)
    Output.node_id/.name/.type/.id/.size_bytes/.content_type/.to_file(path)

Pinned: comfy-sdk==0.5.0, comfy-api-proxy==0.1.6 (pyproject [sdk] extra).
"""

from __future__ import annotations

import os
from pathlib import Path

SDK_IMPORT_ERROR = (
    "comfy-sdk is not installed in this environment. Install the [sdk] extra: "
    "pip install 'continuity[api,sdk]' (pins comfy-sdk==0.5.0, "
    "comfy-api-proxy==0.1.6)."
)

# v2 job status strings -> our normalized terminal status
_STATUS_MAP = {
    "success": "success",
    "succeeded": "success",
    "completed": "success",
    "failed": "error",
    "error": "error",
    "cancelled": "cancelled",
    "canceled": "cancelled",
}
_TERMINAL = {"success", "error", "cancelled"}


def _import_sdk():
    try:
        from comfy_sdk.client import Comfy, ComfyLow
        from comfy_sdk.jobs import Job
        from comfy_sdk.workflows import WorkflowFactory
    except ImportError as e:  # pragma: no cover - environment-dependent
        raise ImportError(SDK_IMPORT_ERROR) from e
    return Comfy, ComfyLow, Job, WorkflowFactory


class ComfySdkEndpoint:
    """Comfy API v2 endpoint driven by the official Python SDK (T13).

    ``base_url`` defaults to ``COMFY_BASE_URL`` (the SDK's own env var) and
    should point at comfy-api-proxy (e.g. http://127.0.0.1:8190) or a managed
    Comfy API deployment. ``api_key`` defaults to ``COMFY_API_KEY``; it is
    propagated by the SDK on submit, job polling and asset download.
    """

    def __init__(self, base_url: str | None = None, api_key: str | None = None,
                 client=None, low_client=None, workflow_factory=None,
                 job_cls=None) -> None:
        self.base_url = base_url or os.environ.get(
            "COMFY_BASE_URL", "http://127.0.0.1:8190")
        self.api_key = api_key or os.environ.get("COMFY_API_KEY") or None
        Comfy, ComfyLow, Job, WorkflowFactory = (None, None, None, None)
        if client is None or low_client is None or workflow_factory is None \
                or job_cls is None:
            Comfy, ComfyLow, Job, WorkflowFactory = _import_sdk()
        self._client_cls = client or Comfy
        self._low_cls = low_client or ComfyLow
        self._job_cls = job_cls or Job
        self._wf_factory = workflow_factory or WorkflowFactory
        self._client = None
        self._low = None

    def _ensure(self):
        if self._low is None:
            os.environ["COMFY_BASE_URL"] = self.base_url
            self._low = self._low_cls(self.base_url, self.api_key)
            self._client = self._client_cls(api_key=self.api_key)
        return self._client

    def close(self):
        if self._client is not None:
            try:
                self._client.close()
            finally:
                self._client = None
                self._low = None

    # -- contract (mirrors ComfyEndpoint where possible) --------------------
    def health(self) -> dict:
        """Authenticated probe: the v2 jobs list answers on this transport."""
        self._ensure()
        jobs = self._low.list_jobs()
        n = len(jobs) if isinstance(jobs, list) else 0
        return {"ok": True, "transport": "comfy-api-v2",
                "base_url": self.base_url, "jobs_visible": n}

    def queue(self, workflow: dict) -> str:
        """Submit one API-format workflow dict; return the v2 job id."""
        self._ensure()
        factory = (self._wf_factory() if isinstance(self._wf_factory, type)
                   else self._wf_factory)
        wf = factory.from_json(workflow)
        job = self._client.submit(wf)
        return str(job.id)

    def poll(self, job_id: str, timeout_s: int = 2400, poll_s: float = 1.5) -> dict:
        """Poll the job to a terminal state; return our normalized shape."""
        import time as _t
        self._ensure()
        t0 = _t.time()
        while _t.time() - t0 < timeout_s:
            job = self._job_cls(self._low, self._low.get_job(job_id))
            raw = str(job.status)
            status = _STATUS_MAP.get(raw.lower(), "running")
            if status in _TERMINAL:
                return {"job_id": job_id, "status": status,
                        "raw_status": raw,
                        "error": job.error if status != "success" else None,
                        "outputs": self.describe_outputs(job)}
            _t.sleep(poll_s)
        raise TimeoutError(f"job {job_id} unresolved after {timeout_s}s")

    @staticmethod
    def describe_outputs(job) -> list[dict]:
        """Output descriptors (no bytes) for receipts and traces."""
        outs = []
        for o in job.outputs:
            outs.append({
                "node_id": o.node_id, "name": o.name, "type": o.type,
                "id": o.id, "size_bytes": o.size_bytes,
                "content_type": o.content_type,
            })
        return outs

    def download_outputs(self, job_id: str, dest_dir: str) -> list[str]:
        """Download every output of a job to dest_dir; return written paths."""
        self._ensure()
        Path(dest_dir).mkdir(parents=True, exist_ok=True)
        job = self._job_cls(self._low, self._low.get_job(job_id))
        paths = []
        for o in job.outputs:
            dest = Path(dest_dir) / (o.name or f"{o.node_id}_{o.id}")
            o.to_file(str(dest))
            paths.append(str(dest))
        return paths
