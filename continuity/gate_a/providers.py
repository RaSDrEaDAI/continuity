"""Vision providers for Gate A instruments.

The gate never talks to a vision backend directly — it talks to a
:class:`VisionProvider`. Two shipped implementations:

- :class:`OpenAICompatProvider` — any OpenAI-compatible chat endpoint that
  accepts data-URL images (OpenRouter, zai paas/v4, local vLLM, ...).
  Configured from env: ``CONTINUITY_VISION_BASE_URL``,
  ``CONTINUITY_VISION_API_KEY``, ``CONTINUITY_VISION_MODEL``. With
  OpenRouter the response ``usage`` carries a real ``cost`` field which the
  gate records per run.
- :class:`ReplayProvider` — deterministic answers from a JSON file, for
  tests and offline demos. Keys are ``"<frame>|<instrument>"`` with a list
  of answers consumed one per pass.

Bring-your-own vision model keeps the whole stack OSS: the repo ships the
instruments + adjudication, not a vendor lock.
"""

from __future__ import annotations

import base64
import json
import os
import urllib.request
from typing import Protocol

DEFAULT_MODEL = "anthropic/claude-haiku-5.5"  # $0.10/M in — measured $2.6e-5/read


class VisionProvider(Protocol):
    name: str

    def ask(self, image_path: str, prompt: str, key: str | None = None) \
            -> tuple[str, dict]:
        """Protocol: solo image read. ``key`` is an optional replay/cache key."""
        raise NotImplementedError


def _image_data_url(image_path: str, max_px: int = 1568) -> str:
    """Local file -> JPEG data URL, downscaled to max_px on the long edge.

    Solo full-frame reads are the reliable instrument; 1568px keeps detail
    (halter straps are thin) while staying inside cheap token budgets.
    """
    from PIL import Image

    img = Image.open(image_path)
    img = img.convert("RGB")
    if max(img.size) > max_px:
        img.thumbnail((max_px, max_px))
    import io

    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=88)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


class OpenAICompatProvider:
    def __init__(self, base_url: str, api_key: str, model: str,
                 max_tokens: int = 700, timeout: int = 120,
                 disable_reasoning: bool = True):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.disable_reasoning = disable_reasoning
        self.name = f"openai-compat:{model}"

    def ask(self, image_path: str, prompt: str, key: str | None = None) \
            -> tuple[str, dict]:
        body = {
            "model": self.model,
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "image_url",
                     "image_url": {"url": _image_data_url(image_path)}},
                    {"type": "text", "text": prompt},
                ],
            }],
            "max_tokens": self.max_tokens,
            "temperature": 0.0,
        }
        if self.disable_reasoning:
            # Measured Oct 9 2026 on OpenRouter anthropic/claude-haiku-5.5:
            # without this the model burns the whole completion budget on
            # reasoning tokens and message.content comes back EMPTY or cut
            # mid-token ('TO') at exactly max_tokens. Gate reads then parse
            # as None -> every instrument "unusable". Disabling reasoning
            # (plus a 700-token budget for segment inventories) yields
            # clean, parseable answers.
            if "api.z.ai" in self.base_url or "bigmodel.cn" in self.base_url:
                # zai/GLM wants a different body key (measured Oct 14 2026):
                # without thinking-off, glm-4.6v returns HTTP 200 with EMPTY
                # content at finish=length after burning ~4600 reasoning
                # tokens. Same disease, different header.
                body["thinking"] = {"type": "disabled"}
            else:
                body["reasoning"] = {"enabled": False}
        req = urllib.request.Request(
            self.base_url + "/chat/completions",
            data=json.dumps(body).encode(),
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
        )
        with urllib.request.urlopen(req, timeout=self.timeout) as r:
            res = json.load(r)
        text = (res.get("choices") or [{}])[0].get("message", {}).get("content") or ""
        usage = res.get("usage") or {}
        return text, {
            "prompt_tokens": usage.get("prompt_tokens"),
            "completion_tokens": usage.get("completion_tokens"),
            "cost_usd": usage.get("cost"),
        }


class ReplayProvider:
    """Deterministic offline provider. Answers file shape::

        {"<frame_key>|<instrument>": ["pass-1 answer", "pass-2 answer"],
         "default": ["answer", ...]}
    """

    def __init__(self, answers: dict):
        self.answers = answers
        self._cursor: dict[str, int] = {}
        self.name = "replay"

    def ask(self, image_path: str, prompt: str, key: str | None = None) \
            -> tuple[str, dict]:
        k = key or image_path
        lst = self.answers.get(k) or self.answers.get("default")
        if lst is None:
            raise KeyError(f"replay: no answers for {k!r}")
        i = self._cursor.get(k, 0)
        self._cursor[k] = i + 1
        return lst[min(i, len(lst) - 1)], {"prompt_tokens": 0,
                                           "completion_tokens": 0,
                                           "cost_usd": 0.0}


def provider_from_env(env: dict | None = None) -> VisionProvider:
    e = dict(os.environ if env is None else env)
    base = e.get("CONTINUITY_VISION_BASE_URL")
    key = e.get("CONTINUITY_VISION_API_KEY")
    model = e.get("CONTINUITY_VISION_MODEL", DEFAULT_MODEL)
    if base and key:
        return OpenAICompatProvider(base, key, model)
    or_key = e.get("OPENROUTER_API_KEY")
    if or_key:
        return OpenAICompatProvider("https://openrouter.ai/api/v1", or_key, model)
    raise RuntimeError(
        "no vision provider configured: set CONTINUITY_VISION_BASE_URL + "
        "CONTINUITY_VISION_API_KEY (+ optional CONTINUITY_VISION_MODEL), "
        "or OPENROUTER_API_KEY"
    )
