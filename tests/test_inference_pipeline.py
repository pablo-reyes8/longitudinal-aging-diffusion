from __future__ import annotations

import pytest
import torch
from PIL import Image

from src.inference import (
    compare_inference_modes,
    encode_image_to_latent,
    generate_age_sweep,
    infer_face_aging,
    infer_face_aging_direct,
    infer_face_aging_inverse,
    prepare_inference_image,
    resolve_inference_strength,
)
from training_fakes import make_training_bundle


def source_image(color=(110, 75, 55), size=(38, 32)):
    return Image.new("RGB", size, color)


def inference_kwargs(bundle, **overrides):
    values = dict(
        bundle=bundle, image=source_image(), target_age=65, source_age=30,
        num_inference_steps=8, strength=0.5, image_size=32,
        seed=442, return_latents=True,
    )
    values.update(overrides)
    return values


def test_direct_determinism_seed_sensitivity_and_finiteness():
    bundle = make_training_bundle()
    first = infer_face_aging_direct(**inference_kwargs(bundle))
    second = infer_face_aging_direct(**inference_kwargs(bundle))
    different = infer_face_aging_direct(**inference_kwargs(bundle, seed=443))
    assert torch.equal(first["latents"], second["latents"])
    assert torch.equal(first["image_tensor"], second["image_tensor"])
    assert not torch.equal(first["latents"], different["latents"])
    assert torch.isfinite(first["latents"]).all() and torch.isfinite(first["image_tensor"]).all()


def test_inverse_determinism_flag_dispatch_and_intermediate_policy():
    bundle = make_training_bundle()
    first = infer_face_aging(**inference_kwargs(bundle, use_inverse_diffusion=True, return_intermediates=True))
    second = infer_face_aging_inverse(**inference_kwargs(bundle, return_intermediates=True))
    compact = infer_face_aging(**inference_kwargs(bundle, mode="direct", return_intermediates=False))
    assert first["mode"] == second["mode"] == "inverse"
    assert torch.equal(first["latents"], second["latents"])
    assert torch.equal(first["inverted_latents"], second["inverted_latents"])
    assert "intermediates" in first and len(first["intermediates"]["inversion_trajectory"]) == 8
    assert "intermediates" not in compact
    assert torch.isfinite(first["image_tensor"]).all()


def test_invalid_mode_is_rejected_and_output_types_work():
    bundle = make_training_bundle()
    try:
        infer_face_aging(**inference_kwargs(bundle, mode="wrong"))
    except ValueError as exc:
        assert "direct" in str(exc) and "inverse" in str(exc)
    else:
        raise AssertionError("invalid inference mode was accepted")
    tensor = infer_face_aging_direct(**inference_kwargs(bundle, output_type="tensor", return_dict=False))
    assert torch.is_tensor(tensor) and tensor.shape == (1, 3, 32, 32)


def test_direct_initialization_is_source_plus_noise_and_strength_distance_increases():
    bundle = make_training_bundle()
    prepared = prepare_inference_image(source_image(), image_size=32)
    source_latent = encode_image_to_latent(bundle, prepared)
    distances = []
    for strength in (0.25, 0.5, 0.75):
        result = infer_face_aging_direct(
            **inference_kwargs(bundle, strength=strength, return_intermediates=True)
        )
        initial = result["intermediates"]["edit_trajectory"][0]
        distances.append(float((initial - source_latent.cpu()).norm()))
        assert not torch.equal(initial, torch.zeros_like(initial))
    assert distances[0] < distances[1] < distances[2]


def test_delta_dependent_strength_policy_symmetry_monotonicity_and_clipping():
    common = dict(
        strength=0.33,
        use_delta_dependent_strength=True,
        base_strength=0.18,
        strength_per_year=0.005,
        min_strength=0.18,
        max_strength=0.40,
    )
    assert resolve_inference_strength(delta_age=0, **common) == 0.18
    assert resolve_inference_strength(delta_age=10, **common) == pytest.approx(0.23)
    assert resolve_inference_strength(delta_age=-10, **common) == pytest.approx(0.23)
    assert resolve_inference_strength(delta_age=100, **common) == 0.40
    assert resolve_inference_strength(delta_age=-100, **common) == 0.40
    assert resolve_inference_strength(
        strength=0.27,
        delta_age=100,
        use_delta_dependent_strength=False,
        base_strength=99,
    ) == 0.27


def test_delta_dependent_strength_reports_effective_value_and_disabled_is_exact():
    bundle = make_training_bundle(seed=445)
    baseline = infer_face_aging_direct(**inference_kwargs(bundle, strength=0.35))
    disabled = infer_face_aging_direct(
        **inference_kwargs(bundle, strength=0.35),
        use_delta_dependent_strength=False,
    )
    adaptive = infer_face_aging_direct(
        **inference_kwargs(bundle, target_age=65, source_age=30),
        use_delta_dependent_strength=True,
        base_strength=0.18,
        strength_per_year=0.005,
        min_strength=0.18,
        max_strength=0.40,
    )
    assert torch.equal(baseline["latents"], disabled["latents"])
    assert adaptive["requested_strength"] == 0.5
    assert adaptive["strength"] == adaptive["metadata"]["effective_strength"] == 0.355


def test_internal_batch_size_two_alignment_and_finite_outputs():
    bundle = make_training_bundle()
    images = torch.stack([
        prepare_inference_image(source_image((100, 60, 40)), image_size=32)[0],
        prepare_inference_image(source_image((40, 90, 130)), image_size=32)[0],
    ])
    result = infer_face_aging_direct(
        **inference_kwargs(bundle, image=images, output_type="tensor")
    )
    assert result["image_tensor"].shape == (2, 3, 32, 32)
    assert result["latents"].shape[0] == 2 and torch.isfinite(result["latents"]).all()
    assert not torch.equal(result["image_tensor"][0], result["image_tensor"][1])


def test_cpu_fp32_needs_no_training_state_and_restores_unet_mode():
    bundle = make_training_bundle()
    bundle.pop("optimizer", None)
    bundle.pop("train_loader", None)
    bundle.pop("lr_scheduler", None)
    bundle["unet"].train()
    result = infer_face_aging_direct(
        **inference_kwargs(bundle, device="cpu", output_type="tensor")
    )
    assert bundle["unet"].training is True
    assert result["image_tensor"].device.type == "cpu"
    assert result["image_tensor"].dtype == torch.float32
    assert torch.isfinite(result["image_tensor"]).all()
    assert all(parameter.grad is None for parameter in bundle["unet"].parameters())


def test_source_and_prompt_conditioning_are_causally_active():
    bundle = make_training_bundle()
    common = inference_kwargs(bundle, seed=81)
    young = infer_face_aging_direct(**{**common, "target_age": 35})
    old = infer_face_aging_direct(**{**common, "target_age": 75})
    other_source = infer_face_aging_direct(**{**common, "image": source_image((30, 140, 180))})
    assert not torch.equal(young["latents"], old["latents"])
    assert not torch.equal(old["latents"], other_source["latents"])


def test_delta_override_preserves_default_and_changes_only_effective_condition():
    bundle = make_training_bundle(seed=118)
    common = inference_kwargs(
        bundle, source_age=26, target_age=65, target_prompt="photo of a person",
        seed=2026, num_inference_steps=3,
    )
    normal = infer_face_aging_direct(**common)
    explicit_default = infer_face_aging_direct(**common, override_delta_age=None)
    text_only = infer_face_aging_direct(**common, override_delta_age=0.0)
    assert torch.equal(normal["image_tensor"], explicit_default["image_tensor"])
    assert normal["metadata"]["true_delta_age"] == 39
    assert normal["metadata"]["delta_age"] == 39
    assert text_only["metadata"]["true_delta_age"] == 39
    assert text_only["metadata"]["delta_age"] == 0
    assert not torch.equal(normal["image_tensor"], text_only["image_tensor"])


def test_direct_referenced_cfg_modes_are_active_and_inverse_remains_legacy():
    bundle = make_training_bundle(seed=119)
    common = inference_kwargs(bundle, source_age=26, target_age=65, seed=2026, num_inference_steps=3)
    source_ref = infer_face_aging_direct(
        **common, text_reference_mode="source_age", age_guidance_scale=3.0,
    )
    generic_ref = infer_face_aging_direct(
        **common, text_reference_mode="generic", age_guidance_scale=3.0,
    )
    null_ref = infer_face_aging_direct(
        **common, text_reference_mode="null", age_guidance_scale=7.0,
    )
    assert source_ref["reference_prompt"] == source_ref["source_prompt"]
    assert generic_ref["reference_prompt"] == "photo of a person"
    assert null_ref["reference_prompt"] == ""
    assert not torch.equal(source_ref["image_tensor"], generic_ref["image_tensor"])
    assert not torch.equal(source_ref["image_tensor"], null_ref["image_tensor"])
    inverse = infer_face_aging_inverse(
        **common, text_reference_mode="source_age", age_guidance_scale=3.0,
    )
    assert inverse["requested_text_reference_mode"] == "source_age"
    assert inverse["text_reference_mode"] == "null"
    assert inverse["age_guidance_scale"] == inverse["text_guidance_scale"]


def test_comparison_grid_and_age_sweep_order(tmp_path):
    bundle = make_training_bundle()
    comparison = compare_inference_modes(
        bundle=bundle, image=source_image(), source_age=30, target_age=65,
        num_inference_steps=4, strength=0.5, image_size=32, seed=1,
        output_path=tmp_path / "comparison.png",
    )
    assert comparison["grid"].size == (96, 60)
    assert (tmp_path / "comparison.png").exists()
    sweep = generate_age_sweep(
        bundle=bundle, image=source_image(), ages=[35, 45, 65],
        source_age=30, mode="direct", num_inference_steps=3, strength=0.5,
        image_size=32, seed=1, output_path=tmp_path / "sweep.png",
    )
    assert sweep["ages"] == [35, 45, 65]
    assert [result["target_age"] for result in sweep["results"]] == [35, 45, 65]
    assert sweep["grid"].size == (96, 60) and (tmp_path / "sweep.png").exists()


def test_training_monitor_sweep_places_source_between_younger_and_older_targets(monkeypatch):
    colors = {16: (0, 255, 0), 35: (0, 0, 255), 65: (255, 0, 0)}

    def fake_inference(*, target_age, **kwargs):
        return {
            "target_age": target_age,
            "image": Image.new("RGB", (16, 16), colors[target_age]),
        }

    monkeypatch.setattr(
        "src.inference.comparison_helpers.infer_face_aging", fake_inference
    )
    sweep = generate_age_sweep(
        bundle=None,
        image=Image.new("RGB", (16, 16), (0, 0, 0)),
        ages=[65, 16, 35],
        source_age=30,
        image_size=16,
        include_source=True,
    )
    # The public result order remains the caller's order; only the grid is chronological.
    assert sweep["ages"] == [65, 16, 35]
    assert [result["target_age"] for result in sweep["results"]] == [65, 16, 35]
    assert [sweep["grid"].getpixel((8 + 16 * index, 36)) for index in range(4)] == [
        colors[16],
        (0, 0, 0),
        colors[35],
        colors[65],
    ]
