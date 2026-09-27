"""Deterministic local/URL image resolution and hashing."""

from __future__ import annotations

import hashlib
import io
from pathlib import Path
from urllib.parse import urlparse
from urllib.request import Request, urlopen

from PIL import Image, ImageOps


def _is_url(value: object) -> bool:
    return isinstance(value, str) and urlparse(value).scheme in {"http", "https"}


class ImageResolver:
    """Resolve image references once and key caches by their actual bytes."""

    def __init__(self, timeout: float = 60.0):
        self.timeout = float(timeout)
        self._cache: dict[str, tuple[Image.Image, str]] = {}

    def load(self, reference) -> tuple[Image.Image, str]:
        key = str(reference)
        if key in self._cache:
            image, digest = self._cache[key]
            return image.copy(), digest

        if isinstance(reference, Image.Image):
            buffer = io.BytesIO()
            ImageOps.exif_transpose(reference).convert("RGB").save(buffer, format="PNG")
            payload = buffer.getvalue()
        elif _is_url(reference):
            request = Request(str(reference), headers={"User-Agent": "face-aging-evaluator/1.0"})
            with urlopen(request, timeout=self.timeout) as response:
                payload = response.read()
        else:
            path = Path(reference).expanduser()
            if not path.is_file():
                raise FileNotFoundError(f"Image not found: {path}")
            payload = path.read_bytes()

        digest = hashlib.sha256(payload).hexdigest()
        image = ImageOps.exif_transpose(Image.open(io.BytesIO(payload))).convert("RGB")
        self._cache[key] = (image.copy(), digest)
        return image, digest

