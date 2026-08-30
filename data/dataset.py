"""PyTorch Dataset for supervised longitudinal age editing."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from PIL import Image, ImageOps, UnidentifiedImageError
from torch.utils.data import Dataset

from .indexing import ImageRecord, PairRecord, build_pair_index
from .prompts import build_prompts


def _stable_uint64(*values: object) -> int:
    digest = hashlib.blake2b("|".join(map(str, values)).encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "little")


def _load_image(path: Path, image_size: int, flip: bool) -> torch.Tensor:
    try:
        with Image.open(path) as opened:
            image = ImageOps.exif_transpose(opened).convert("RGB")
            width, height = image.size
            side = min(width, height)
            left = (width - side) // 2
            top = (height - side) // 2
            image = image.crop((left, top, left + side, top + side))
            image = image.resize((image_size, image_size), Image.Resampling.LANCZOS)
            if flip:
                image = ImageOps.mirror(image)
            array = np.asarray(image, dtype=np.float32).copy()
    except (OSError, ValueError, UnidentifiedImageError) as exc:
        raise RuntimeError(f"Failed to decode face image: {path}") from exc
    tensor = torch.from_numpy(array).permute(2, 0, 1).contiguous()
    return tensor.div(127.5).sub(1.0)


class FaceAgingDataset(Dataset):
    """Return real same-person image pairs with exact numeric ages.

    ``random_target`` has one item per source image that has an eligible older
    target. Target choice, optional pair reversal, and paired flipping are
    deterministic functions of ``seed``, ``epoch`` and item index, which makes
    them independent of worker scheduling. Bidirectional sampling never removes
    canonical forward pairs or changes dataset length: it only swaps the order
    in which a selected pair is observed. Call :meth:`set_epoch` before each
    training epoch to resample.
    """

    def __init__(
        self,
        root_dir: str | Path,
        manifest: Sequence[ImageRecord],
        *,
        image_size: int = 256,
        pair_strategy: str = "all",
        min_age_gap: int = 1,
        max_age_gap: int | None = None,
        prompt_style: str = "selfage",
        dynamic_person_word: bool = False,
        horizontal_flip_prob: float = 0.0,
        include_zero_delta_pairs: bool = False,
        zero_delta_pair_prob: float = 0.20,
        include_bidirectional_pairs: bool = False,
        reverse_pair_prob: float = 0.20,
        add_reverse_pairs: bool = False,
        pair_records: Sequence[PairRecord] | None = None,
        seed: int = 42,
    ) -> None:
        if pair_strategy not in {"all", "random_target"}:
            raise ValueError("pair_strategy must be 'all' or 'random_target'")
        if image_size <= 0:
            raise ValueError("image_size must be positive")
        if not 0.0 <= horizontal_flip_prob <= 1.0:
            raise ValueError("horizontal_flip_prob must be in [0, 1]")
        if not 0.0 <= zero_delta_pair_prob <= 1.0:
            raise ValueError("zero_delta_pair_prob must be in [0, 1]")
        if not 0.0 <= reverse_pair_prob <= 1.0:
            raise ValueError("reverse_pair_prob must be in [0, 1]")
        self.root_dir = Path(root_dir).expanduser().resolve()
        self.manifest = list(manifest)
        self.image_size = image_size
        self.pair_strategy = pair_strategy
        self.prompt_style = prompt_style
        self.dynamic_person_word = dynamic_person_word
        self.horizontal_flip_prob = horizontal_flip_prob
        self.include_zero_delta_pairs = bool(include_zero_delta_pairs)
        self.zero_delta_pair_prob = float(zero_delta_pair_prob)
        self.add_reverse_pairs = bool(add_reverse_pairs)
        self.include_bidirectional_pairs = bool(include_bidirectional_pairs or self.add_reverse_pairs)
        self.reverse_pair_prob = float(reverse_pair_prob)
        self.seed = seed
        # A shared tensor propagates set_epoch() to persistent DataLoader workers.
        self._shared_epoch = torch.zeros((), dtype=torch.int64).share_memory_()
        self.all_pairs = list(pair_records) if pair_records is not None else build_pair_index(
            self.manifest, min_age_gap=min_age_gap, max_age_gap=max_age_gap
        )
        candidates: dict[int, list[PairRecord]] = {}
        for pair in self.all_pairs:
            candidates.setdefault(pair.source_index, []).append(pair)
        self._source_candidates = [
            (source_index, tuple(candidates[source_index])) for source_index in sorted(candidates)
        ]
        self._base_observations = (
            len(self.all_pairs) if self.pair_strategy == "all" else len(self._source_candidates)
        )
        self._additive_reverse_indices = (
            [
                index
                for index in range(self._base_observations)
                if _stable_uint64(self.seed, index, "add_reverse_pair") / 2**64
                < self.reverse_pair_prob
            ]
            if self.add_reverse_pairs and self.include_bidirectional_pairs
            else []
        )

    def __len__(self) -> int:
        return self._base_observations + len(self._additive_reverse_indices)

    def set_epoch(self, epoch: int) -> None:
        self._shared_epoch.fill_(int(epoch))

    @property
    def epoch(self) -> int:
        return int(self._shared_epoch.item())

    def pair_for_index(self, index: int) -> PairRecord:
        additive_reverse = False
        if self.add_reverse_pairs and index >= self._base_observations:
            index = self._additive_reverse_indices[index - self._base_observations]
            additive_reverse = True
        if self.pair_strategy == "all":
            pair = self.all_pairs[index]
        else:
            _, candidates = self._source_candidates[index]
            choice = _stable_uint64(self.seed, self.epoch, index, "target") % len(candidates)
            pair = candidates[choice]
        zero_delta_draw = _stable_uint64(
            self.seed, self.epoch, index, "zero_delta"
        ) / 2**64
        if self.include_zero_delta_pairs and zero_delta_draw < self.zero_delta_pair_prob:
            self_index = (
                pair.source_index
                if _stable_uint64(self.seed, self.epoch, index, "zero_delta_image") % 2 == 0
                else pair.target_index
            )
            source = self.manifest[self_index]
            return PairRecord(
                person_id=source.person_id,
                source_index=self_index,
                target_index=self_index,
                source_age=source.age,
                target_age=source.age,
                delta_age=0,
                source_path=source.relative_path,
                target_path=source.relative_path,
            )
        if additive_reverse:
            return PairRecord(
                person_id=pair.person_id,
                source_index=pair.target_index,
                target_index=pair.source_index,
                source_age=pair.target_age,
                target_age=pair.source_age,
                delta_age=-pair.delta_age,
                source_path=pair.target_path,
                target_path=pair.source_path,
            )
        if self.include_bidirectional_pairs and not self.add_reverse_pairs:
            reverse_draw = _stable_uint64(
                self.seed, self.epoch, index, "reverse_pair"
            ) / 2**64
            if reverse_draw < self.reverse_pair_prob:
                return PairRecord(
                    person_id=pair.person_id,
                    source_index=pair.target_index,
                    target_index=pair.source_index,
                    source_age=pair.target_age,
                    target_age=pair.source_age,
                    delta_age=-pair.delta_age,
                    source_path=pair.target_path,
                    target_path=pair.source_path,
                )
        return pair

    def __getitem__(self, index: int) -> dict[str, Any]:
        pair = self.pair_for_index(index)
        source = self.manifest[pair.source_index]
        target = self.manifest[pair.target_index]
        flip_draw = _stable_uint64(self.seed, self.epoch, index, "flip") / 2**64
        flip = flip_draw < self.horizontal_flip_prob
        source_path = source.path(self.root_dir)
        target_path = target.path(self.root_dir)
        source_image = _load_image(source_path, self.image_size, flip)
        target_image = _load_image(target_path, self.image_size, flip)
        prompts = build_prompts(
            source.age,
            target.age,
            prompt_style=self.prompt_style,
            gender=source.gender,
            dynamic_person_word=self.dynamic_person_word,
        )
        return {
            "source_image": source_image,
            "target_image": target_image,
            "source_age": source.age,
            "target_age": target.age,
            "delta_age": target.age - source.age,
            **prompts,
            "person_id": source.person_id,
            "source_path": str(source_path),
            "target_path": str(target_path),
            "source_filename": source.filename,
            "target_filename": target.filename,
            "gender": source.gender,
        }


class CombinedFaceAgingDataset(Dataset):
    """Concatenate primary and complementary sources without dropping primary items."""

    def __init__(self, primary: FaceAgingDataset, complementary: FaceAgingDataset) -> None:
        self.primary = primary
        self.complementary = complementary
        self.primary_observations = len(primary)
        self.complementary_observations = len(complementary)
        self.manifest = [*primary.manifest, *complementary.manifest]
        # Preserve the canonical primary index for existing audits. The
        # complementary index uses a different root and is exposed separately.
        self.all_pairs = primary.all_pairs
        self.complementary_pairs = complementary.all_pairs
        self.include_zero_delta_pairs = primary.include_zero_delta_pairs
        self.zero_delta_pair_prob = primary.zero_delta_pair_prob
        self.include_bidirectional_pairs = primary.include_bidirectional_pairs
        self.reverse_pair_prob = primary.reverse_pair_prob
        self.add_reverse_pairs = primary.add_reverse_pairs
        self.kaggle_reverse_pair_prob = complementary.reverse_pair_prob
        self.training_identity_count = (
            len({row.person_id for row in primary.manifest})
            + len({pair.person_id for pair in complementary.all_pairs})
        )
        self.pair_strategy = "combined"

    def __len__(self) -> int:
        return self.primary_observations + self.complementary_observations

    def set_epoch(self, epoch: int) -> None:
        self.primary.set_epoch(epoch)
        self.complementary.set_epoch(epoch)

    @property
    def epoch(self) -> int:
        return self.primary.epoch

    def pair_for_index(self, index: int) -> PairRecord:
        if index < self.primary_observations:
            return self.primary.pair_for_index(index)
        return self.complementary.pair_for_index(index - self.primary_observations)

    def __getitem__(self, index: int) -> dict[str, Any]:
        if index < self.primary_observations:
            sample = self.primary[index]
            source_dataset = "colombian"
        else:
            sample = self.complementary[index - self.primary_observations]
            source_dataset = "fgnet"
        return {**sample, "source_dataset": source_dataset}


def collate_face_aging_batch(samples: list[dict[str, Any]]) -> dict[str, Any]:
    """Collate tensors/integers while preserving raw strings and optional ``None``."""
    if not samples:
        return {}
    batch: dict[str, Any] = {}
    for key in samples[0]:
        values = [sample[key] for sample in samples]
        if isinstance(values[0], torch.Tensor):
            batch[key] = torch.stack(values)
        elif all(isinstance(value, int) for value in values):
            batch[key] = torch.tensor(values, dtype=torch.long)
        else:
            batch[key] = values
    return batch
