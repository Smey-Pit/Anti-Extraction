
from __future__ import annotations

import base64
import io
import time
from abc import ABC, abstractmethod

from PIL import Image


class FrontierModel(ABC):
    name: str

    @abstractmethod
    def transcribe(self, image: Image.Image, prompt: str) -> str: ...


def _pil_to_b64(image: Image.Image, fmt: str = "PNG") -> str:
    buf = io.BytesIO()
    image.save(buf, format=fmt)
    return base64.b64encode(buf.getvalue()).decode("utf-8")
