"""Small notebook-friendly batch inspection helpers."""

from __future__ import annotations

from typing import Any

import torch


def inspect_batch(batch: dict[str, Any]) -> None:
    """Print the fields most useful for catching pairing/collation mistakes."""
    print("source_image:", tuple(batch["source_image"].shape), batch["source_image"].dtype)
    print("target_image:", tuple(batch["target_image"].shape), batch["target_image"].dtype)
    for key in ("source_age", "target_age", "delta_age"):
        value = batch[key].tolist() if isinstance(batch[key], torch.Tensor) else batch[key]
        print(f"{key}:", value)
    for key in ("source_prompt", "target_prompt", "person_id"):
        print(f"{key}:", batch[key])


def plot_pair_grid(batch: dict[str, Any], max_pairs: int = 4):
    """Display a compact SOURCE | TARGET grid; imports matplotlib only on demand."""
    import matplotlib.pyplot as plt

    count = min(max_pairs, len(batch["source_image"]))
    figure, axes = plt.subplots(count, 2, figsize=(6, 3 * count), squeeze=False)
    for row in range(count):
        for column, key in enumerate(("source_image", "target_image")):
            image = batch[key][row].detach().cpu().clamp(-1, 1).add(1).div(2)
            axes[row, column].imshow(image.permute(1, 2, 0).numpy())
            age_key = "source_age" if column == 0 else "target_age"
            age = batch[age_key][row].item()
            axes[row, column].set_title(f"{'SOURCE' if column == 0 else 'TARGET'} · age {age}")
            axes[row, column].axis("off")
    figure.tight_layout()
    return figure
