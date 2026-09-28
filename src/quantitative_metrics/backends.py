"""Frozen AdaFace, DEX, and KID evaluation backends."""

from __future__ import annotations

import hashlib
import importlib.util
import sys
from pathlib import Path

import numpy as np
from PIL import Image


def _file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def adaface_input_tensor(image: Image.Image):
    """Canonical AdaFace input: aligned 112x112 BGR normalized to [-1, 1]."""
    import torch

    rgb = np.asarray(image.convert("RGB").resize((112, 112), Image.Resampling.BILINEAR))
    bgr = np.ascontiguousarray(rgb[..., ::-1])
    return torch.from_numpy(bgr).permute(2, 0, 1).float().div(127.5).sub(1.0)


def resolve_torch_device(device="auto"):
    import torch

    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(device)


def clean_adaface_state_dict(state):
    """Extract only the AdaFace backbone from Lightning or plain checkpoints."""
    if any(key.startswith("model.") for key in state):
        return {key[len("model."):]: value for key, value in state.items() if key.startswith("model.")}
    if any(key.startswith("module.") for key in state):
        return {key[len("module."):]: value for key, value in state.items() if key.startswith("module.")}
    return dict(state)


def expected_dex_age(logits: np.ndarray) -> np.ndarray:
    """Convert DEX logits to expected ages over classes 0..100."""
    logits = np.asarray(logits, dtype=np.float64)
    if logits.ndim == 1:
        logits = logits[None, :]
    elif logits.ndim > 2:
        logits = logits.reshape(logits.shape[0], -1)
    if logits.shape[1] != 101:
        raise ValueError(f"DEX output must contain 101 age logits, got {logits.shape}")
    row_sums = logits.sum(axis=1, keepdims=True)
    if np.all(logits >= 0.0) and np.allclose(row_sums, 1.0, atol=1e-5):
        probabilities = logits
    else:
        stable = logits - logits.max(axis=1, keepdims=True)
        probabilities = np.exp(stable)
        probabilities /= probabilities.sum(axis=1, keepdims=True)
    return probabilities @ np.arange(101, dtype=np.float64)


class AdaFaceAligner:
    """Official AdaFace MTCNN aligner without importing training modules.

    The repository's convenience ``face_alignment.align`` module can pull in
    AdaFace training-only modules such as ``data.py`` (and therefore optional
    packages like ``pytorch_lightning``/``bcolz``).  Metrics need only the
    inference MTCNN, so load ``mtcnn.py`` directly in an isolated module.
    """

    def __init__(self, repo_path: str | Path, device="auto"):
        import torch

        repo = Path(repo_path).expanduser().resolve()
        if not repo.is_dir():
            raise FileNotFoundError(f"AdaFace repository not found: {repo}")
        mtcnn_path = repo / "face_alignment" / "mtcnn.py"
        if not mtcnn_path.is_file():
            raise FileNotFoundError(f"AdaFace MTCNN implementation not found: {mtcnn_path}")
        resolved_device = resolve_torch_device(device)
        mtcnn_device = "cuda:0" if resolved_device.type == "cuda" else "cpu"
        face_alignment_root = str(mtcnn_path.parent)
        previous_path = list(sys.path)
        sys.path.insert(0, face_alignment_root)
        try:
            spec = importlib.util.spec_from_file_location(
                "_face_aging_adaface_mtcnn", mtcnn_path
            )
            if spec is None or spec.loader is None:
                raise ImportError(f"Could not import AdaFace MTCNN implementation: {mtcnn_path}")
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
        finally:
            sys.path[:] = previous_path
        self._mtcnn = module.MTCNN(device=mtcnn_device, crop_size=(112, 112))

    def __call__(self, image: Image.Image):
        _, faces = self._mtcnn.align_multi(image.convert("RGB"), limit=1)
        if not faces:
            return None
        return faces[0].convert("RGB")


class AdaFaceEncoder:
    def __init__(self, repo_path, checkpoint_path, architecture="ir_101", device="auto"):
        import torch

        repo = Path(repo_path).expanduser().resolve()
        checkpoint_path = Path(checkpoint_path).expanduser().resolve()
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"AdaFace checkpoint not found: {checkpoint_path}")
        net_path = repo / "net.py"
        spec = importlib.util.spec_from_file_location("_face_aging_adaface_net", net_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Could not import AdaFace network definition: {net_path}")
        net = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(net)
        self.device = resolve_torch_device(device)
        self.model = net.build_model(architecture)
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        state = checkpoint.get("state_dict", checkpoint)
        cleaned = clean_adaface_state_dict(state)
        self.model.load_state_dict(cleaned, strict=True)
        self.model.eval().requires_grad_(False).to(self.device)

    def embed_batch(self, images):
        import torch

        inputs = torch.stack([adaface_input_tensor(image) for image in images]).to(self.device)
        with torch.inference_mode():
            output = self.model(inputs)
            features = output[0] if isinstance(output, (tuple, list)) else output
            features = torch.nn.functional.normalize(features.float(), dim=1)
        return features.cpu().numpy()


class DexAgeEstimator:
    """Official DEX Caffe model loaded locally with OpenCV DNN."""

    def __init__(self, prototxt_path, checkpoint_path, device="auto"):
        import cv2

        prototxt = Path(prototxt_path).expanduser().resolve()
        checkpoint = Path(checkpoint_path).expanduser().resolve()
        if not prototxt.is_file() or not checkpoint.is_file():
            raise FileNotFoundError("DEX prototxt and checkpoint must both exist locally")
        if not hasattr(cv2.dnn, "readNetFromCaffe"):
            version = getattr(cv2, "__version__", "unknown")
            raise RuntimeError(
                "DEX evaluation requires OpenCV 4.x with dnn.readNetFromCaffe; "
                f"detected OpenCV {version}. OpenCV 5 removed the Caffe reader. "
                "In Colab, run `%pip uninstall -y opencv-python opencv-python-headless "
                "opencv-contrib-python opencv-contrib-python-headless` followed by "
                "`%pip install 'opencv-python-headless>=4.8,<5'`, then restart the runtime."
            )
        self.cv2 = cv2
        self.net = cv2.dnn.readNetFromCaffe(str(prototxt), str(checkpoint))
        if device == "cuda":
            self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
            self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

    def predict_batch(self, images):
        arrays = [np.asarray(image.convert("RGB"))[..., ::-1] for image in images]
        blob = self.cv2.dnn.blobFromImages(
            arrays, scalefactor=1.0, size=(224, 224), mean=(104.0, 117.0, 123.0), swapRB=False
        )
        self.net.setInput(blob)
        return expected_dex_age(self.net.forward())


class TorchMetricsKid:
    _INCEPTION_FILENAME = "weights-inception-2015-12-05-6726825d.pth"

    def __init__(self, device="auto", feature=2048, inception_weights_path=None):
        import torch

        self.torch = torch
        self.device = resolve_torch_device(device)
        self.feature = int(feature)
        if inception_weights_path is None:
            inception_weights_path = (
                Path(torch.hub.get_dir()) / "checkpoints" / self._INCEPTION_FILENAME
            )
        self.inception_weights_path = Path(inception_weights_path).expanduser().resolve()

    @staticmethod
    def _tensor(images):
        import torch

        resized = [np.asarray(image.convert("RGB").resize((299, 299), Image.Resampling.BILINEAR)) for image in images]
        return torch.from_numpy(np.stack(resized)).permute(0, 3, 1, 2).to(torch.uint8)

    def compute(self, real_images, generated_images, subsets=100, subset_size=None, seed=2026):
        if not self.inception_weights_path.is_file():
            raise FileNotFoundError(
                "KID Inception weights are not available locally. Place "
                f"{self._INCEPTION_FILENAME} at {self.inception_weights_path}; "
                "automatic evaluator downloads are disabled."
            )
        if self.inception_weights_path.name != self._INCEPTION_FILENAME:
            raise ValueError(
                f"inception_weights_path must end with {self._INCEPTION_FILENAME}"
            )
        from torchmetrics.image.kid import KernelInceptionDistance

        n = min(len(real_images), len(generated_images))
        subset_size = min(50, n) if subset_size is None else min(int(subset_size), n)
        original_hub_dir = self.torch.hub.get_dir()
        expected_hub_dir = self.inception_weights_path.parent.parent
        try:
            self.torch.hub.set_dir(str(expected_hub_dir))
            with self.torch.random.fork_rng(devices=[]):
                self.torch.manual_seed(int(seed))
                metric = KernelInceptionDistance(
                    feature=self.feature, subsets=int(subsets), subset_size=subset_size, normalize=False
                ).to(self.device)
                metric.update(self._tensor(real_images).to(self.device), real=True)
                metric.update(self._tensor(generated_images).to(self.device), real=False)
                mean, std = metric.compute()
        finally:
            self.torch.hub.set_dir(original_hub_dir)
        return float(mean.cpu()), float(std.cpu())


def load_quantitative_metrics(
    *,
    adaface_repo_path,
    adaface_checkpoint_path,
    dex_prototxt_path,
    dex_checkpoint_path,
    device="auto",
    adaface_architecture="ir_101",
    kid_feature_dim=2048,
    kid_inception_weights_path=None,
    local_files_only=True,
):
    """Load frozen evaluators exclusively from caller-supplied local files."""
    if not local_files_only:
        raise ValueError("Metric model downloading is intentionally unsupported; use local_files_only=True")
    kid_metric = TorchMetricsKid(
        device=device,
        feature=kid_feature_dim,
        inception_weights_path=kid_inception_weights_path,
    )
    return {
        "aligner": AdaFaceAligner(adaface_repo_path, device=device),
        "identity_encoder": AdaFaceEncoder(
            adaface_repo_path, adaface_checkpoint_path, adaface_architecture, device
        ),
        "age_estimator": DexAgeEstimator(dex_prototxt_path, dex_checkpoint_path, device),
        "kid_metric": kid_metric,
        "metadata": {
            "identity_model": "adaface_r100_webface12m",
            "identity_architecture": adaface_architecture,
            "identity_checkpoint": str(Path(adaface_checkpoint_path).resolve()),
            "identity_checkpoint_sha256": _file_sha256(adaface_checkpoint_path),
            "age_model": "dex_chalearn_iccv2015",
            "age_checkpoint": str(Path(dex_checkpoint_path).resolve()),
            "age_checkpoint_sha256": _file_sha256(dex_checkpoint_path),
            "alignment": "official_adaface_get_aligned_face_112x112",
            "identity_preprocessing": "aligned RGB -> BGR; x/127.5-1",
            "age_preprocessing": "same aligned crop; BGR 224; mean=(104,117,123)",
            "kid_inception_weights": str(kid_metric.inception_weights_path),
        },
    }
