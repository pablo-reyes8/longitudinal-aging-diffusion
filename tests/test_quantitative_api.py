from pathlib import Path

import pandas as pd
from PIL import Image
import pytest


def test_build_default_metrics_config_uses_cpu_and_stable_layout(tmp_path):
    from src.quantitative_metrics.api import build_default_metrics_config

    config = build_default_metrics_config(tmp_path)

    assert config["device"] == "cpu"
    assert config["local_files_only"] is True
    assert config["adaface_repo_path"] == tmp_path / "AdaFace"
    assert config["dex_prototxt_path"] == tmp_path / "DEX" / "age.prototxt"
    assert config["kid_inception_weights_path"] == (
        tmp_path / "KID" / "weights-inception-2015-12-05-6726825d.pth"
    )


def test_index_labeled_image_tree_reads_identity_and_age(tmp_path):
    from src.quantitative_metrics.api import index_labeled_image_tree

    for relative in ("id_2/53.jpg", "id_1/35.png", "id_1/54.jpg", "id_1/54_1.jpg"):
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"image")

    indexed = index_labeled_image_tree(tmp_path)

    assert indexed[("id_1", 35)].name == "35.png"
    assert indexed[("id_1", 54)].name == "54.jpg"
    assert indexed[("id_2", 53)].name == "53.jpg"


def test_select_kid_pairs_is_deterministic_and_respects_limits(tmp_path):
    from src.quantitative_metrics.api import select_kid_pairs

    real_root = tmp_path / "real"
    generated_root = tmp_path / "generated"
    for identity in ("id_1", "id_2", "id_3"):
        for age in (20, 30, 40):
            for root in (real_root, generated_root):
                path = root / identity / f"{age}.png"
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b"image")

    first = select_kid_pairs(
        generated_root,
        real_root,
        num_ids=2,
        num_transitions=3,
        seed=17,
    )
    second = select_kid_pairs(
        generated_root,
        real_root,
        num_ids=2,
        num_transitions=3,
        seed=17,
    )

    assert first == second
    assert len(first) == 3
    assert len({pair["identity_id"] for pair in first}) <= 2
    assert all(Path(pair["real_path"]).is_file() for pair in first)
    assert all(Path(pair["generated_path"]).is_file() for pair in first)


def test_evaluate_kid_rejects_a_single_image_pair(tmp_path):
    from src.quantitative_metrics.api import select_kid_pairs

    path = tmp_path / "real" / "id_1" / "35.png"
    path.parent.mkdir(parents=True)
    path.write_bytes(b"image")
    generated = tmp_path / "generated" / "id_1" / "35.png"
    generated.parent.mkdir(parents=True)
    generated.write_bytes(b"image")

    with pytest.raises(ValueError, match="at least two"):
        select_kid_pairs(generated.parent.parent, path.parents[1], num_transitions=1)


def test_evaluate_aging_disables_kid_and_writes_generated_image(tmp_path, monkeypatch):
    import src.inference as inference
    import src.quantitative_metrics.api as api

    checkpoint = tmp_path / "adapter.pt"
    source = tmp_path / "source.png"
    target = tmp_path / "target.png"
    checkpoint.write_bytes(b"checkpoint")
    Image.new("RGB", (8, 8), (10, 20, 30)).save(source)
    Image.new("RGB", (8, 8), (30, 20, 10)).save(target)
    captured = {}

    monkeypatch.setattr(api, "_load_external_metrics", lambda **kwargs: ({"fake": True}, {"ok": True}))
    monkeypatch.setattr(inference, "load_face_aging_inference_bundle", lambda *args, **kwargs: {
        "inference_checkpoint_report": {"loaded_tensors": 1}
    })
    monkeypatch.setattr(inference, "infer_face_aging_direct", lambda **kwargs: {
        "image": Image.new("RGB", (8, 8), (40, 50, 60))
    })

    def fake_evaluate(predictions, *, metrics_bundle, **kwargs):
        captured["metrics_bundle"] = metrics_bundle
        captured["options"] = kwargs
        return {"paths": {"manifest": tmp_path / "manifest.csv"}}

    monkeypatch.setattr(api, "evaluate_aging_outputs", fake_evaluate)
    result = api.evaluate_aging(
        checkpoint,
        source,
        target,
        35,
        53,
        output_dir=tmp_path / "output",
        download_assets=False,
    )

    assert result["generated_path"].is_file()
    assert captured["metrics_bundle"] == {"fake": True}
    assert captured["options"]["compute_kid"] is False


def test_evaluate_aging_inference_uses_exact_target_strength_map(tmp_path, monkeypatch):
    import src.inference as inference
    import src.quantitative_metrics.api as api

    checkpoint = tmp_path / "adapter.pt"
    source = tmp_path / "source.png"
    targets = {}
    checkpoint.write_bytes(b"checkpoint")
    Image.new("RGB", (8, 8), (10, 20, 30)).save(source)
    for age in (35, 53):
        target = tmp_path / f"target_{age}.png"
        Image.new("RGB", (8, 8), (30, 20, 10)).save(target)
        targets[age] = target
    captured = {"maps": [], "options": None}

    monkeypatch.setattr(api, "_load_external_metrics", lambda **kwargs: ({"fake": True}, {"ok": True}))
    monkeypatch.setattr(api, "_load_inference_bundle", lambda *args, **kwargs: ({"fake": True}, False))

    def fake_adaptive(**kwargs):
        captured["maps"].append(kwargs["target_age_strength_map"])
        return {
            "image": Image.new("RGB", (8, 8), (40, 50, 60)),
            "effective_strength": kwargs["target_age_strength_map"][kwargs["target_age"]],
        }

    monkeypatch.setattr(inference, "generate_aged_face_adaptive_strength", fake_adaptive)

    def fake_evaluate(predictions, *, metrics_bundle, **kwargs):
        captured["options"] = kwargs
        return {"paths": {"manifest": tmp_path / "manifest.csv"}}

    monkeypatch.setattr(api, "evaluate_aging_outputs", fake_evaluate)
    result = api.evaluate_aging_inference(
        checkpoint,
        source,
        26,
        [35, 53],
        target_images=targets,
        target_age_strength_map={35: 0.22, 53: 0.27},
        output_dir=tmp_path / "output",
        download_assets=False,
    )

    assert captured["maps"] == [{35: 0.22, 53: 0.27}] * 2
    assert list(result["generated_paths"]) == [35.0, 53.0]
    assert captured["options"]["compute_kid"] is False
    assert result["manifest"]["strength"].tolist() == [0.22, 0.27]


def test_evaluate_aging_inference_picking_scores_base_and_assisted(tmp_path, monkeypatch):
    import src.inference as inference
    import src.quantitative_metrics.api as api

    checkpoint = tmp_path / "adapter.pt"
    source = tmp_path / "source.png"
    target = tmp_path / "target.png"
    checkpoint.write_bytes(b"checkpoint")
    Image.new("RGB", (8, 8), (10, 20, 30)).save(source)
    Image.new("RGB", (8, 8), (30, 20, 10)).save(target)
    base_result = {"image": Image.new("RGB", (8, 8), (1, 2, 3))}
    assisted_result = {"image": Image.new("RGB", (8, 8), (4, 5, 6))}
    diagnostic = pd.DataFrame([{"target_age": 35.0, "age_error": 5.0, "identity_cosine": 0.9}])
    diagnostic.attrs["base_results"] = [base_result]
    diagnostic.attrs["assisted_results"] = [assisted_result]
    diagnostic.attrs["assisted"] = pd.DataFrame([{
        "target_age": 35.0, "age_error": 1.0, "identity_cosine": 0.8,
        "effective_strength": 0.22,
    }])
    captured = {}

    monkeypatch.setattr(api, "_load_external_metrics", lambda **kwargs: ({"fake": True}, {"ok": True}))
    monkeypatch.setattr(api, "_load_inference_bundle", lambda *args, **kwargs: ({"fake": True}, False))
    monkeypatch.setattr(inference, "diagnose_checkpoint_adaptive_age_sweep", lambda **kwargs: diagnostic)

    def fake_evaluate(predictions, *, metrics_bundle, **kwargs):
        captured["options"] = kwargs
        return {"paths": {"manifest": tmp_path / "manifest.csv"}}

    monkeypatch.setattr(api, "evaluate_aging_outputs", fake_evaluate)
    result = api.evaluate_aging_inference_picking(
        checkpoint,
        source,
        26,
        [35],
        target_images={35: target},
        diagnostic_config={"generate_assisted_prompt_variant": True},
        output_dir=tmp_path / "output",
        download_assets=False,
    )

    assert result["selected_variants"] == {35.0: "assisted"}
    assert result["manifest"].iloc[0]["selected_variant"] == "assisted"
    assert captured["options"]["compute_kid"] is False
