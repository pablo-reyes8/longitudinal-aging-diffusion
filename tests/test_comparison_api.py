from __future__ import annotations

from pathlib import Path


def test_comparison_package_exports_only_the_two_high_level_operations():
    from src import comparisong

    assert callable(comparisong.load_aging_models)
    assert callable(comparisong.age_image)
    assert comparisong.__all__ == ["load_aging_models", "age_image"]


def test_fading_loader_uses_stable_runner_and_applies_path_fix_without_network(
    monkeypatch,
    tmp_path,
):
    from src.comparisong import load_models

    cloned = []

    def fake_clone(url, destination):
        cloned.append((url, Path(destination)))
        Path(destination).mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr(load_models, "_clone", fake_clone)
    model_dir = (
        tmp_path
        / "fading_models"
        / "specialized"
        / "finetune_double_prompt_150_random"
    )
    model_dir.mkdir(parents=True)
    (model_dir / "model_index.json").write_text("{}", encoding="utf-8")

    bundle = load_models.load_aging_models(
        root=tmp_path,
        models=("FADING",),
        fading_specialized_path=model_dir,
    )

    stable_root = tmp_path / "FADING_stable"
    assert cloned == [
        ("https://github.com/gh-BumsooKim/FADING_stable.git", stable_root)
    ]
    assert Path(bundle["FADING"]["root"]) == stable_root
    assert Path(bundle["FADING"]["script"]) == stable_root / "age_editing.py"
    assert Path(bundle["FADING"]["specialized_path"]) == model_dir


def test_loader_applies_hrfae_yaml_compatibility_patch_before_return(
    monkeypatch,
    tmp_path,
):
    from src.comparisong import load_models

    hrfae_root = tmp_path / "HRFAE"
    hrfae_root.mkdir()
    test_script = hrfae_root / "test.py"
    test_script.write_text(
        "config = yaml.load(open('./configs/' + opts.config + '.yaml', 'r'))\n",
        encoding="utf-8",
    )
    checkpoint = tmp_path / "hrfae.pth"
    checkpoint.touch()
    monkeypatch.setattr(load_models, "_clone", lambda *args, **kwargs: None)

    load_models.load_aging_models(
        root=tmp_path,
        models=("HRFAE",),
        hrfae_checkpoint_path=checkpoint,
        download_hrfae=False,
    )

    assert "yaml.safe_load(" in test_script.read_text(encoding="utf-8")
    assert "yaml.load(" not in test_script.read_text(encoding="utf-8")


def test_comparison_cli_accepts_documented_minimal_invocation():
    from scripts.compare_models import build_parser

    args = build_parser().parse_args(
        [
            "--image",
            "face.jpg",
            "--source-age",
            "26",
            "--target-ages",
            "8",
            "26",
            "65",
            "--output-dir",
            "outputs/comparison",
        ]
    )

    assert args.source_age == 26
    assert args.target_ages == [8, 26, 65]
    assert args.models == ["SAM", "FRAN", "FADING", "CUSP", "HRFAE"]


def test_age_image_accepts_path_objects_used_by_the_notebook(tmp_path):
    from PIL import Image

    from src.comparisong import age_image

    source = tmp_path / "face.png"
    Image.new("RGB", (32, 32), "gray").save(source)

    result = age_image(
        bundle={"device": "cpu", "baseline_models": []},
        image=source,
        source_age=26,
        target_ages=[26],
        models=[],
        show=False,
        output_dir=tmp_path / "outputs",
    )

    assert result["source_age"] == 26
    assert (tmp_path / "outputs" / "source.png").is_file()
