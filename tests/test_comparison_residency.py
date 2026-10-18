import sys

import pytest

from src.comparisong.resident_models import ModelSession, ACTIVE_MODEL_SESSION


def test_model_moves_once_per_phase_and_releases_even_on_error():
    from src.comparisong.generate_images import _move_model, _release_after_image

    class Model:
        def __init__(self):
            self.moves = []

        def to(self, device):
            self.moves.append(str(device))
            return self

        def eval(self):
            return self

    model = Model()
    bundle = {"SAM": {"model": model}}
    with pytest.raises(RuntimeError, match="image error"):
        with ModelSession(bundle, "SAM"):
            for _ in range(4):
                _move_model(model, "cuda")
                assert _release_after_image(model) is model
            raise RuntimeError("image error")
    assert model.moves == ["cuda", "cpu"]
    assert ACTIVE_MODEL_SESSION.get() is None


def test_completed_model_phase_does_not_activate_any_model():
    class Model:
        def to(self, device):
            raise AssertionError("No pending images, no device transfer")

    with ModelSession({"SAM": {"model": Model()}}, "SAM") as session:
        assert session.worker is None


def test_fading_persistent_process_loads_once_resets_scheduler_and_recovers_after_bad_image(tmp_path):
    # Stand-in for the repository module. The actual CLI shape is preserved;
    # only model inference is replaced, so no GPU/checkpoints/network are used.
    (tmp_path / "fake_models.py").write_text(
        "from pathlib import Path\n"
        "class Pipeline:\n"
        "    def __init__(self, root, scheduler):\n"
        "        self.root = Path(root)\n"
        "        self.scheduler = scheduler\n"
        "        with (self.root / 'loads.txt').open('a') as f: f.write('load\\n')\n"
        "    @classmethod\n"
        "    def from_pretrained(cls, root, scheduler): return cls(root, scheduler)\n"
        "    def to(self, device):\n"
        "        with (self.root / 'moves.txt').open('a') as f: f.write(str(device) + '\\n')\n"
        "        return self\n", encoding="utf-8",
    )
    script = tmp_path / "age_editing.py"
    script.write_text(
        "import argparse\nfrom pathlib import Path\nfrom PIL import Image\n"
        "from fake_models import Pipeline\n"
        "parser = argparse.ArgumentParser()\n"
        "parser.add_argument('--image_path')\nparser.add_argument('--specialized_path')\n"
        "parser.add_argument('--save_aged_dir')\nargs = parser.parse_args()\n"
        "scheduler = {'image': args.image_path}\n"
        "ldm_stable = Pipeline.from_pretrained(args.specialized_path, scheduler=scheduler).to('cuda')\n"
        "assert ldm_stable.scheduler == {'image': args.image_path}\n"
        "ldm_stable.scheduler['used'] = True\n"
        "if args.image_path == 'bad': raise RuntimeError('face not detected')\n"
        "Image.new('RGB', (8, 8), 'gray').save(Path(args.save_aged_dir) / (args.image_path + '.png'))\n",
        encoding="utf-8",
    )
    bundle = {"FADING": {"script": str(script)}}
    with ModelSession(bundle, "FADING") as session:
        for image in ("first", "bad", "second"):
            cmd = [sys.executable, str(script), "--image_path", image,
                   "--specialized_path", str(tmp_path), "--save_aged_dir", str(tmp_path)]
            if image == "bad":
                with pytest.raises(RuntimeError, match="face not detected"):
                    session.run_command(cmd, str(tmp_path), None)
            else:
                session.run_command(cmd, str(tmp_path), None)
            worker = session.worker
        assert (tmp_path / "loads.txt").read_text().splitlines() == ["load"]
        assert (tmp_path / "moves.txt").read_text().splitlines() == ["cuda"]
    assert worker.poll() is not None
    assert (tmp_path / "first.png").exists() and (tmp_path / "second.png").exists()


def test_hrfae_persistent_process_constructs_and_loads_checkpoint_only_once(tmp_path):
    (tmp_path / "trainer.py").write_text(
        "from pathlib import Path\n"
        "class Trainer:\n"
        "    def __init__(self, config):\n"
        "        self.root = Path(config['root'])\n"
        "        with (self.root / 'construct.txt').open('a') as f: f.write('build\\n')\n"
        "    def load_checkpoint(self, path):\n"
        "        with (self.root / 'checkpoints.txt').open('a') as f: f.write(path + '\\n')\n"
        "    def to(self, device):\n"
        "        with (self.root / 'moves.txt').open('a') as f: f.write(str(device) + '\\n')\n"
        "        return self\n", encoding="utf-8",
    )
    script = tmp_path / "test.py"
    script.write_text(
        "import argparse\nfrom pathlib import Path\nfrom trainer import Trainer\n"
        "parser = argparse.ArgumentParser()\nparser.add_argument('--root')\n"
        "parser.add_argument('--target_age')\nopts = parser.parse_args()\n"
        "config = {'root': opts.root}\ntrainer = Trainer(config)\n"
        "trainer.load_checkpoint('same_weights')\ntrainer.to('cuda')\n"
        "Path(opts.root, 'age_' + opts.target_age + '.txt').write_text(opts.target_age)\n",
        encoding="utf-8",
    )
    with ModelSession({"HRFAE": {"script": str(script)}}, "HRFAE") as session:
        for age in (30, 50, 70):
            session.run_command([sys.executable, str(script), "--root", str(tmp_path), "--target_age", str(age)], str(tmp_path), None)
        assert (tmp_path / "construct.txt").read_text().splitlines() == ["build"]
        assert (tmp_path / "checkpoints.txt").read_text().splitlines() == ["same_weights"]
        assert (tmp_path / "moves.txt").read_text().splitlines() == ["cuda"]
    assert len(list(tmp_path.glob("age_*.txt"))) == 3
