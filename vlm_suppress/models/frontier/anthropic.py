
from __future__ import annotations

import os
import time

from PIL import Image

from vlm_suppress.models.frontier.base import FrontierModel, _pil_to_b64


class ClaudeModel(FrontierModel):
    name = "claude"

    def __init__(self, cfg) -> None:
        import anthropic
        self._model = cfg.anthropic_model
        self._max_retries = cfg.max_retries
        self._client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

    def transcribe(self, image: Image.Image, prompt: str) -> str:
        b64 = _pil_to_b64(image)
        for attempt in range(self._max_retries):
            try:
                msg = self._client.messages.create(
                    model=self._model,
                    max_tokens=1024,
                    messages=[{
                        "role": "user",
                        "content": [
                            {"type": "image",
                             "source": {"type": "base64",
                                        "media_type": "image/png",
                                        "data": b64}},
                            {"type": "text", "text": prompt},
                        ],
                    }],
                )
                return msg.content[0].text or ""
            except Exception:
                if attempt == self._max_retries - 1:
                    raise
                time.sleep(2 ** attempt)
        return ""