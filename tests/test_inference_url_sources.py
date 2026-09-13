from __future__ import annotations

from io import BytesIO

import pytest
from PIL import Image
import torch

from src.inference import load_sweep_source_image


class _FakeHTTPResponse(BytesIO):
    def __init__(self, payload: bytes):
        super().__init__(payload)
        self.headers = {"Content-Length": str(len(payload))}

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


def _png_bytes(size=(734, 489)) -> bytes:
    buffer = BytesIO()
    Image.new("RGB", size, "gray").save(buffer, format="PNG")
    return buffer.getvalue()


def test_https_source_is_downloaded_once_in_memory_and_keeps_native_resolution(
    monkeypatch,
):
    payload = _png_bytes()
    calls = []

    def fake_urlopen(request, timeout):
        calls.append((request.full_url, timeout))
        return _FakeHTTPResponse(payload)

    monkeypatch.setattr(
        "src.inference.source_image_loading.urlopen", fake_urlopen
    )
    image = load_sweep_source_image(
        "https://example.test/portrait.jpg", image_size=400
    )

    assert image.mode == "RGB"
    assert image.size == (734, 489)
    assert calls == [("https://example.test/portrait.jpg", 15.0)]


@pytest.mark.parametrize(
    "source",
    [
        Image.new("RGB", (399, 500)),
        torch.zeros(3, 500, 399),
    ],
)
def test_sweep_source_rejects_any_input_that_would_require_upsampling(source):
    with pytest.raises(ValueError, match="Upsampling is intentionally unsupported"):
        load_sweep_source_image(source, image_size=400)


def test_downloaded_low_resolution_image_is_rejected(monkeypatch):
    monkeypatch.setattr(
        "src.inference.source_image_loading.urlopen",
        lambda *args, **kwargs: _FakeHTTPResponse(_png_bytes((307, 361))),
    )
    with pytest.raises(ValueError, match="307x361"):
        load_sweep_source_image("http://example.test/small.jpg", image_size=400)


def test_non_http_url_scheme_is_rejected():
    with pytest.raises(ValueError, match="Only HTTP and HTTPS"):
        load_sweep_source_image("ftp://example.test/portrait.jpg", image_size=400)
