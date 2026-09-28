"""Idempotent setup for the frozen quantitative-metric assets."""

from __future__ import annotations

import hashlib
import shutil
import subprocess
import tempfile
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping


ADAFACE_REPOSITORY = "https://github.com/mk-minchul/AdaFace.git"


@dataclass(frozen=True)
class DownloadSpec:
    """One immutable metric artifact and its basic integrity constraints."""

    url: str | None = None
    google_drive_id: str | None = None
    min_bytes: int = 1
    exact_bytes: int | None = None
    sha256: str | None = None
    hf_repo_id: str | None = None
    hf_filename: str | None = None
    hf_revision: str | None = None


DOWNLOAD_SPECS = {
    "adaface_checkpoint_path": DownloadSpec(
        google_drive_id="1dswnavflETcnAuplZj1IOKKP0eM8ITgT",
        min_bytes=100_000_000,
        exact_bytes=1_526_801_999,
        sha256="0e7a3238d2a50f3fe3860782534928ac7cb2598977cf897f6869fd5ac2493fd0",
        hf_repo_id="VishalMishraTss/AdaFace",
        hf_filename="adaface_ir101_webface12m.ckpt",
        # Pin the exact mirror revision; the SHA256 below is still checked.
        hf_revision="534fa44f9499437645bd18eee376e52fd60bd931",
    ),
    "dex_prototxt_path": DownloadSpec(
        url="https://data.vision.ee.ethz.ch/cvl/rrothe/imdb-wiki/static/age.prototxt",
        min_bytes=1_000,
        exact_bytes=4_824,
    ),
    "dex_checkpoint_path": DownloadSpec(
        url=(
            "https://data.vision.ee.ethz.ch/cvl/rrothe/imdb-wiki/static/"
            "dex_chalearn_iccv2015.caffemodel"
        ),
        min_bytes=100_000_000,
        exact_bytes=538_700_290,
    ),
    "kid_inception_weights_path": DownloadSpec(
        url=(
            "https://github.com/toshas/torch-fidelity/releases/download/v0.2.0/"
            "weights-inception-2015-12-05-6726825d.pth"
        ),
        min_bytes=10_000_000,
        exact_bytes=95_628_359,
        sha256="6726825d0af5f729cebd5821db510b11b1cfad8faad88a03f1befd49fb9129b2",
    ),
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _is_valid_download(path: Path, spec: DownloadSpec) -> bool:
    if not path.is_file() or path.stat().st_size < spec.min_bytes:
        return False
    if spec.exact_bytes is not None and path.stat().st_size != spec.exact_bytes:
        return False
    return spec.sha256 is None or _sha256(path) == spec.sha256


def _download_http(url: str, destination: Path) -> None:
    request = urllib.request.Request(url, headers={"User-Agent": "face-aging-metrics/1.0"})
    with urllib.request.urlopen(request) as response, destination.open("wb") as output:
        shutil.copyfileobj(response, output, length=1024 * 1024)


def _download_google_drive(file_id: str, destination: Path) -> None:
    try:
        import gdown
    except ImportError as error:
        raise RuntimeError(
            "Downloading the official AdaFace checkpoint requires gdown. "
            "Install the evaluation extras with `pip install -e '.[evaluation]'`."
        ) from error
    result = gdown.download(id=file_id, output=str(destination), quiet=False)
    if result is None:
        raise RuntimeError("gdown did not download the AdaFace checkpoint")


def _download_huggingface(spec: DownloadSpec, destination: Path) -> None:
    """Download a pinned fallback artifact into the atomic temporary path."""
    if not spec.hf_repo_id or not spec.hf_filename:
        raise RuntimeError("No Hugging Face fallback is configured for this metric asset")
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as error:
        raise RuntimeError(
            "The AdaFace Google Drive download was unavailable and the Hugging Face "
            "fallback requires huggingface_hub. Install it with `pip install huggingface_hub`."
        ) from error
    downloaded = Path(
        hf_hub_download(
            repo_id=spec.hf_repo_id,
            filename=spec.hf_filename,
            revision=spec.hf_revision,
        )
    )
    if downloaded.resolve() != destination.resolve():
        shutil.copyfile(downloaded, destination)


def _is_file_url_retrieval_error(error: BaseException) -> bool:
    """Recognize gdown's quota/permission retrieval failure without importing gdown."""
    return error.__class__.__name__ == "FileURLRetrievalError"


def _ensure_download(path: Path, spec: DownloadSpec) -> None:
    if _is_valid_download(path, spec):
        print(f"Metric asset ready: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        prefix=f".{path.name}.", suffix=".part", dir=path.parent, delete=False
    ) as handle:
        partial = Path(handle.name)
    print(f"Downloading metric asset: {path.name}")
    try:
        if spec.google_drive_id is not None:
            try:
                _download_google_drive(spec.google_drive_id, partial)
            except Exception as error:
                if not (_is_file_url_retrieval_error(error) and spec.hf_repo_id):
                    raise
                print(
                    "Google Drive did not provide the AdaFace checkpoint; "
                    f"using pinned Hugging Face fallback {spec.hf_repo_id}/{spec.hf_filename}."
                )
                _download_huggingface(spec, partial)
        elif spec.url is not None:
            _download_http(spec.url, partial)
        else:
            raise ValueError(f"No download source configured for {path.name}")
        if not _is_valid_download(partial, spec):
            raise RuntimeError(f"Downloaded metric asset failed integrity checks: {path.name}")
        partial.replace(path)
    except Exception:
        if partial.exists():
            partial.unlink()
        raise
    print(f"Metric asset downloaded: {path}")


def _valid_adaface_repository(path: Path) -> bool:
    return (path / "net.py").is_file() and (path / "face_alignment" / "align.py").is_file()


def _clone_adaface_repository(destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and any(destination.iterdir()):
        raise RuntimeError(
            f"AdaFace path exists but is incomplete: {destination}. "
            "Move it aside or provide a valid AdaFace checkout."
        )
    temporary_root = Path(tempfile.mkdtemp(prefix="adaface-clone-", dir=destination.parent))
    temporary_checkout = temporary_root / "AdaFace"
    try:
        subprocess.run(
            ["git", "clone", "--depth", "1", ADAFACE_REPOSITORY, str(temporary_checkout)],
            check=True,
        )
        if not _valid_adaface_repository(temporary_checkout):
            raise RuntimeError("The cloned AdaFace repository is missing required inference files")
        if destination.exists():
            destination.rmdir()
        temporary_checkout.replace(destination)
    except FileNotFoundError as error:
        raise RuntimeError("git is required to download the AdaFace inference code") from error
    finally:
        shutil.rmtree(temporary_root, ignore_errors=True)


def prepare_metrics_config(
    config: Mapping[str, Any], *, include_kid: bool = True
) -> dict[str, Any]:
    """Download selected evaluator assets and return a runner-ready configuration.

    Models are not imported or loaded here. The returned configuration defaults
    evaluation to CPU, leaving training VRAM untouched. Set ``include_kid=False``
    for the AdaFace/DEX-only path.
    """
    prepared = dict(config)
    specs = DOWNLOAD_SPECS if include_kid else {
        key: spec for key, spec in DOWNLOAD_SPECS.items()
        if key != "kid_inception_weights_path"
    }
    required = {"adaface_repo_path", *specs}
    missing = sorted(required.difference(prepared))
    if missing:
        raise ValueError(f"metrics config is missing required path(s): {', '.join(missing)}")

    repository = Path(prepared["adaface_repo_path"]).expanduser()
    if not _valid_adaface_repository(repository):
        print(f"Downloading AdaFace inference repository: {repository}")
        _clone_adaface_repository(repository)
    else:
        print(f"AdaFace repository ready: {repository}")

    for key, spec in specs.items():
        _ensure_download(Path(prepared[key]).expanduser(), spec)

    prepared.setdefault("device", "cpu")
    prepared["local_files_only"] = True
    return prepared
