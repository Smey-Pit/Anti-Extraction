
from __future__ import annotations

import os
import time

from PIL import Image

from vlm_suppress.models.frontier.base import FrontierModel


class GeminiModel(FrontierModel):
    name = "gemini"

    def __init__(self, cfg) -> None:
        import google.generativeai as genai
        genai.configure(api_key=os.environ["GEMINI_API_KEY"])
        self._model = genai.GenerativeModel(cfg.gemini_model)
        self._max_retries = cfg.max_retries

    def transcribe(self, image: Image.Image, prompt: str) -> str:
        import google.generativeai as genai
        for attempt in range(self._max_retries):
            try:
                resp = self._model.generate_content(
                    [prompt, image],
                    generation_config=genai.types.GenerationConfig(temperature=0.0),
                )
                return resp.text or ""
            except Exception:
                if attempt == self._max_retries - 1:
                    raise
                time.sleep(2 ** attempt)
        return ""
