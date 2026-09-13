"""Load diagnostic source images without silently upsampling low-resolution input."""

from __future__ import annotations

from io import BytesIO
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen

from PIL import Image, ImageOps, UnidentifiedImageError
import torch


_MAX_DOWNLOAD_BYTES = 30 * 1024 * 1024


def _validate_minimum_resolution(width: int, height: int, image_size: int) -> None:
    if image_size <= 0:
        raise ValueError("image_size must be positive")
    if width < image_size or height < image_size:
        raise ValueError(
            f"Source image resolution is {width}x{height}, but image_size={image_size}. "
            "Upsampling is intentionally unsupported; use an image whose width and "
            "height are both at least image_size."
        )


def _download_image(url: str, *, timeout_seconds: float) -> Image.Image:
    request = Request(url, headers={"User-Agent": "longitudinal-face-aging/1.0"})
    try:
        with urlopen(request, timeout=timeout_seconds) as response:
            content_length = response.headers.get("Content-Length")
            if content_length is not None and int(content_length) > _MAX_DOWNLOAD_BYTES:
                raise ValueError("Remote image exceeds the 30 MiB download limit")
            payload = response.read(_MAX_DOWNLOAD_BYTES + 1)
    except (HTTPError, URLError, TimeoutError) as exc:
        raise ValueError(f"Could not download source image from {url!r}: {exc}") from exc
    if len(payload) > _MAX_DOWNLOAD_BYTES:
        raise ValueError("Remote image exceeds the 30 MiB download limit")
    try:
        with Image.open(BytesIO(payload)) as opened:
            return ImageOps.exif_transpose(opened).convert("RGB").copy()
    except (UnidentifiedImageError, OSError) as exc:
        raise ValueError(f"URL did not return a readable image: {url!r}") from exc


def load_sweep_source_image(
    source_image,
    *,
    image_size: int,
    timeout_seconds: float = 15.0,
):
    """Resolve HTTP(S), local, PIL, or tensor input and forbid upsampling."""
    if isinstance(source_image, str):
        parsed = urlparse(source_image)
        if parsed.scheme in {"http", "https"}:
            image = _download_image(source_image, timeout_seconds=timeout_seconds)
        elif "://" in source_image:
            raise ValueError("Only HTTP and HTTPS image URLs are supported")
        else:
            with Image.open(Path(source_image).expanduser()) as opened:
                image = ImageOps.exif_transpose(opened).convert("RGB").copy()
    elif isinstance(source_image, Path):
        with Image.open(source_image.expanduser()) as opened:
            image = ImageOps.exif_transpose(opened).convert("RGB").copy()
    elif isinstance(source_image, Image.Image):
        image = ImageOps.exif_transpose(source_image).convert("RGB").copy()
    elif torch.is_tensor(source_image):
        if source_image.ndim not in {3, 4}:
            raise ValueError("Tensor image must have shape [3,H,W] or [B,3,H,W]")
        height, width = source_image.shape[-2:]
        _validate_minimum_resolution(int(width), int(height), int(image_size))
        return source_image
    else:
        raise TypeError("source_image must be an HTTP(S) URL, local path, PIL image, or tensor")
    _validate_minimum_resolution(*image.size, int(image_size))
    return image
