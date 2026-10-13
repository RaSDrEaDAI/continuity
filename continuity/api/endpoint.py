"""Comfy API v2 endpoint — submit/poll for API-format workflows (T8).

One class, endpoint-agnostic:

    local ComfyUI       http://127.0.0.1:8188
    comfy-api-proxy     http://127.0.0.1:8190   (serves Comfy API v2)
    managed Comfy API   https://...             (same surface, credits)

Platform-fit note (DECISIONS.md): the proxy satisfies the "built with Comfy
Developer Platform" requirement while the engine stays 100% local.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import requests


@dataclass
class ComfyEndpoint:
    base_url: str | None = None
    api_key: str | None = None

    def __post_init__(self) -> None:
        self.base_url = self.base_url or os.environ.get(
            "CONTINUITY_COMFY_URL", "http://127.0.0.1:8190"
        )
        self.api_key = self.api_key or os.environ.get("CONTINUITY_COMFY_API_KEY") or None

    def _headers(self) -> dict:
        h = {"Content-Type": "application/json"}
        if self.api_key:
            h["Authorization"] = f"Bearer {self.api_key}"
        return h

    def health(self) -> dict:
        """system_stats — GPU present, VRAM free."""
        r = requests.get(f"{self.base_url}/system_stats", timeout=10)
        r.raise_for_status()
        return r.json()

    def queue(self, workflow: dict) -> str:
        """Submit an API-format workflow, return the prompt id (T8)."""
        r = requests.post(
            f"{self.base_url}/prompt", json={"prompt": workflow},
            headers=self._headers(), timeout=30,
        )
        r.raise_for_status()
        return r.json()["prompt_id"]

    def poll(self, prompt_id: str, timeout_s: int = 2400, poll_s: float = 1.5) -> dict:
        """Poll /history until the job resolves; return status + outputs (T8)."""
        import time as _t
        t0 = _t.time()
        while _t.time() - t0 < timeout_s:
            r = requests.get(f"{self.base_url}/history/{prompt_id}", timeout=30)
            r.raise_for_status()
            hist = r.json().get(prompt_id)
            if hist:
                return {"prompt_id": prompt_id,
                        "status": hist.get("status"),
                        "outputs": hist.get("outputs")}
            _t.sleep(poll_s)
        raise TimeoutError(f"prompt {prompt_id} unresolved after {timeout_s}s")
