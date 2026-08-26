from __future__ import annotations

from types import SimpleNamespace

import pytest

from src.training import TRAIN_AGGING_MODEL


def _loader(*, bidirectional: bool, probability: float):
    return SimpleNamespace(
        dataset=SimpleNamespace(
            include_bidirectional_pairs=bidirectional,
            reverse_pair_prob=probability,
        )
    )


def test_training_rejects_bidirectional_loader_without_training_flag():
    with pytest.raises(ValueError, match="use_bidirectional_training=True"):
        TRAIN_AGGING_MODEL(
            bundle=None,
            loss_fn=None,
            train_loader=_loader(bidirectional=True, probability=0.20),
            val_loader=None,
        )


def test_training_rejects_bidirectional_flag_without_bidirectional_loader():
    with pytest.raises(ValueError, match="include_bidirectional_pairs=True"):
        TRAIN_AGGING_MODEL(
            bundle=None,
            loss_fn=None,
            train_loader=_loader(bidirectional=False, probability=0.20),
            val_loader=None,
            use_bidirectional_training=True,
            reverse_pair_prob=0.20,
        )


def test_training_rejects_reverse_probability_mismatch():
    with pytest.raises(ValueError, match="must match between"):
        TRAIN_AGGING_MODEL(
            bundle=None,
            loss_fn=None,
            train_loader=_loader(bidirectional=True, probability=0.20),
            val_loader=None,
            use_bidirectional_training=True,
            reverse_pair_prob=0.50,
        )
