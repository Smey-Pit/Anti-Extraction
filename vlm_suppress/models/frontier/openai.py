
from __future__ import annotations

import os
import time

from PIL import Image

from vlm_suppress.models.frontier.base import FrontierModel, _pil_to_b64


class GPT4oModel(FrontierModel):
    name = "gpt4o"

    def __init__(self, cfg) -> None:
        from openai import OpenAI
        self._model = cfg.openai_model
        self._max_retries = cfg.max_retries
        self._client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

    def transcribe(self, image: Image.Image, prompt: str) -> str:
        b64 = _pil_to_b64(image)
        for attempt in range(self._max_retries):
            try:
                resp = self._client.chat.completions.create(
                    model=self._model,
                    temperature=0,
                    messages=[{
                        "role": "user",
                        "content": [
                            {"type": "image_url",
                             "image_url": {"url": f"data:image/png;base64,{b64}"}},
                            {"type": "text", "text": prompt},
                        ],
                    }],
                )
                return resp.choices[0].message.content or ""
            except Exception as e:
                if attempt == self._max_retries - 1:
                    raise
                time.sleep(2 ** attempt)
        return ""
