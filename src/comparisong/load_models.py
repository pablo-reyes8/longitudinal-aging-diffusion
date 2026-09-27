import os
import sys
import gc
import glob
import zipfile
import subprocess
import torch

from argparse import Namespace
from pathlib import Path
from huggingface_hub import hf_hub_download, snapshot_download


# ============================================================
# General helpers
# ============================================================

def _run(cmd, cwd=None):
    print(" ".join(map(str, cmd)))

    subprocess.run(
        list(map(str, cmd)),
        cwd=cwd,
        check=True,
    )


def _clone(repo_url, dst):

    if not os.path.exists(dst):

        _run([
            "git",
            "clone",
            "--depth",
            "1",
            repo_url,
            dst,
        ])


def _clear_vram():

    gc.collect()

    if torch.cuda.is_available():

        torch.cuda.empty_cache()

        try:
            torch.cuda.ipc_collect()
        except Exception:
            pass


def _purge_modules(prefixes):
    """
    Some of these old repositories use generic module names:
        models
        model
        utils
        scripts
        training

    Purging avoids collisions when several repos are imported
    inside the same Colab runtime.
    """

    for name in list(sys.modules.keys()):

        if any(
            name == p or name.startswith(p + ".")
            for p in prefixes
        ):
            del sys.modules[name]


# ============================================================
# MAIN LOADER
# ============================================================

def load_aging_models(
    root="/content/aging_benchmarks",

    # Models we want prepared.
    models=(
        "SAM",
        "FRAN",
        "FADING",
        "CUSP",
        "HRFAE",
        "Cradle2Cane",
    ),

    # --------------------------------------------------------
    # FADING
    # --------------------------------------------------------

    # Manual path if automatic download fails.
    fading_specialized_path=None,

    # --------------------------------------------------------
    # CUSP
    # --------------------------------------------------------

    # "ffhq_ls" is the broader lifespan model.
    # Alternative: "ffhq_rr"
    cusp_variant="ffhq_ls",

    # If you already downloaded the model manually.
    cusp_checkpoint_path=None,

    # Whether loader should attempt to download the official
    # Google Drive model folder.
    download_cusp=True,

    # --------------------------------------------------------
    # HRFAE
    # --------------------------------------------------------

    # Manual checkpoint / log directory if required.
    hrfae_checkpoint_path=None,

    # Official repo provides logs/001/download.sh
    download_hrfae=True,

    # --------------------------------------------------------
    # Cradle2Cane
    # --------------------------------------------------------

    # IMPORTANT:
    # default False because dependencies are enormous (~tens GB).
    download_cradle=False,

    # Independent paper metrics (AdaFace + DEX + lazy KID).
    load_metrics=False,
    metrics_config=None,
):
    """
    Prepare face-aging baselines.

    Models supported
    ----------------
    SAM
    FRAN reproduction
    FADING
    CUSP
    HRFAE
    Cradle2Cane

    GPU policy
    ----------
    Nothing is intentionally kept resident on GPU here.

    SAM:
        instantiated on CPU.

    FRAN:
        instantiated on CPU.

    FADING:
        repository + checkpoint paths only.

    CUSP:
        repository + pretrained checkpoint paths only.

    HRFAE:
        repository + pretrained model paths only.

    Cradle2Cane:
        repository + dependency paths only.

    Default benchmark intended later:
        SAM
        FRAN
        FADING
        CUSP
        HRFAE

    Cradle2Cane remains supported but is not part of the
    lightweight default benchmark.
    """

    os.makedirs(
        root,
        exist_ok=True,
    )

    requested = {
        x.lower()
        for x in models
    }

    bundle = {
        "root": root,

        "device": (
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        ),

        "SAM": None,
        "FRAN": None,
        "FADING": None,
        "CUSP": None,
        "HRFAE": None,
        "Cradle2Cane": None,
        "quantitative_metrics": None,

        # Default benchmark:
        # Cradle2Cane intentionally excluded.
        "baseline_models": [
            "SAM",
            "FRAN",
            "FADING",
            "CUSP",
            "HRFAE",
        ],
    }

    print(
        "Runtime device:",
        bundle["device"],
    )

    # ========================================================
    # Repository paths
    # ========================================================

    sam_root = os.path.join(
        root,
        "SAM",
    )

    fran_root = os.path.join(
        root,
        "FRAN",
    )

    fading_root = os.path.join(
        root,
        "FADING_stable",
    )

    cusp_root = os.path.join(
        root,
        "CUSP",
    )

    hrfae_root = os.path.join(
        root,
        "HRFAE",
    )

    c2c_root = os.path.join(
        root,
        "Cradle2Cane",
    )

    # ========================================================
    # Clone repositories
    # ========================================================

    if "sam" in requested:

        _clone(
            "https://github.com/yuval-alaluf/SAM.git",
            sam_root,
        )

    if "fran" in requested:

        _clone(
            "https://github.com/timroelofs123/face_reaging.git",
            fran_root,
        )

    if "fading" in requested:

        _clone(
            "https://github.com/gh-BumsooKim/FADING_stable.git",
            fading_root,
        )

    if "cusp" in requested:

        _clone(
            "https://github.com/guillermogotre/CUSP.git",
            cusp_root,
        )

    if "hrfae" in requested:

        _clone(
            "https://github.com/InterDigitalInc/HRFAE.git",
            hrfae_root,
        )

    if "cradle2cane" in requested:

        _clone(
            "https://github.com/byliutao/Cradle2Cane.git",
            c2c_root,
        )

    # ========================================================
    # SAM
    # ========================================================

    if "sam" in requested:

        print("\n" + "=" * 70)
        print("SAM")
        print("=" * 70)

        # ----------------------------------------------------
        # Checkpoint
        # ----------------------------------------------------

        sam_ckpt = hf_hub_download(
            repo_id="aimi-models/sam-aging",
            filename="sam_ffhq_aging.pt",
        )

        # ----------------------------------------------------
        # dlib landmarks
        # ----------------------------------------------------

        landmarks_path = os.path.join(
            root,
            "shape_predictor_68_face_landmarks.dat",
        )

        if not os.path.exists(
            landmarks_path
        ):

            _run([
                "wget",
                "-q",
                (
                    "https://github.com/italojs/"
                    "facial-landmarks-recognition/raw/master/"
                    "shape_predictor_68_face_landmarks.dat"
                ),
                "-O",
                landmarks_path,
            ])

        # ----------------------------------------------------
        # Import SAM
        # ----------------------------------------------------

        _purge_modules([
            "models",
            "datasets",
            "utils",
            "scripts",
            "configs",
        ])

        sys.path.insert(
            0,
            sam_root,
        )

        from models.psp import pSp
        from datasets.augmentations import AgeTransformer
        from utils.common import tensor2im
        from scripts.align_all_parallel import align_face

        import dlib

        ckpt = torch.load(
            sam_ckpt,
            map_location="cpu",
            weights_only=False,
        )

        opts_dict = ckpt[
            "opts"
        ].copy()

        opts_dict[
            "checkpoint_path"
        ] = sam_ckpt

        opts = Namespace(
            **opts_dict
        )

        sam_model = pSp(
            opts
        )

        sam_model.eval()

        # CRITICAL:
        # keep on CPU.
        sam_model = sam_model.to(
            "cpu"
        )

        predictor = (
            dlib.shape_predictor(
                landmarks_path
            )
        )

        bundle["SAM"] = {
            "model": sam_model,
            "checkpoint": sam_ckpt,
            "root": sam_root,
            "predictor": predictor,
            "AgeTransformer": AgeTransformer,
            "tensor2im": tensor2im,
            "align_face": align_face,
        }

        _purge_modules([
            "models",
            "datasets",
            "utils",
            "scripts",
            "configs",
        ])

        try:
            sys.path.remove(
                sam_root
            )
        except ValueError:
            pass

        print(
            "SAM ready [CPU]."
        )

    # ========================================================
    # FRAN reproduction
    # ========================================================

    if "fran" in requested:

        print("\n" + "=" * 70)
        print("FRAN reproduction")
        print("=" * 70)

        fran_ckpt = hf_hub_download(
            repo_id="timroelofs123/face_re-aging",
            filename="best_unet_model.pth",
        )

        _purge_modules([
            "model",
            "models",
            "utils",
            "scripts",
        ])

        # ----------------------------------------------------
        # PATCH torchvision.io.write_video
        # ----------------------------------------------------

        test_functions_path = Path(
            fran_root,
            "scripts",
            "test_functions.py",
        )

        txt = (
            test_functions_path
            .read_text()
        )

        old_import = (
            "from torchvision.io import write_video"
        )

        patched_import = (
            "try:\n"
            "    from torchvision.io import write_video\n"
            "except ImportError:\n"
            "    write_video = None"
        )

        if old_import in txt:

            txt = txt.replace(
                old_import,
                patched_import,
            )

            test_functions_path.write_text(
                txt
            )

            print(
                "Patched torchvision.write_video."
            )

        # ----------------------------------------------------
        # Import FRAN from repository root
        # ----------------------------------------------------

        old_cwd = os.getcwd()

        try:

            # test_functions.py expects:
            # assets/mask1024.jpg
            # relative to repository root.

            os.chdir(
                fran_root
            )

            sys.path.insert(
                0,
                fran_root,
            )

            _purge_modules([
                "model",
                "models",
                "utils",
                "scripts",
            ])

            from model.models import UNet
            from scripts.test_functions import process_image

        finally:

            os.chdir(
                old_cwd
            )

        # ----------------------------------------------------
        # Load model
        # ----------------------------------------------------

        fran_model = UNet()

        state = torch.load(
            fran_ckpt,
            map_location="cpu",
            weights_only=False,
        )

        fran_model.load_state_dict(
            state
        )

        fran_model.eval()

        fran_model = fran_model.to(
            "cpu"
        )

        bundle["FRAN"] = {
            "model": fran_model,
            "checkpoint": fran_ckpt,
            "root": fran_root,
            "process_image": process_image,
            "window_size": 512,
            "stride": 256,
        }

        _purge_modules([
            "model",
            "models",
            "utils",
            "scripts",
        ])

        try:
            sys.path.remove(
                fran_root
            )
        except ValueError:
            pass

        print(
            "FRAN reproduction ready [CPU]."
        )

    # ========================================================
    # FADING
    # ========================================================

    if "fading" in requested:

        print("\n" + "=" * 70)
        print("FADING")
        print("=" * 70)

        fading_models_dir = os.path.join(
            root,
            "fading_models",
        )

        os.makedirs(
            fading_models_dir,
            exist_ok=True,
        )

        # Official Google Drive ID
        drive_id = (
            "1galwrcHq1HoZNfOI4jdJJqVs5ehB_dvO"
        )

        if fading_specialized_path is None:

            download_path = os.path.join(
                fading_models_dir,
                "fading_specialized",
            )

            existing = glob.glob(
                os.path.join(
                    fading_models_dir,
                    "*",
                )
            )

            if not existing:

                print(
                    "Downloading official FADING "
                    "specialized checkpoint..."
                )

                try:

                    _run([
                        "gdown",
                        "--fuzzy",
                        (
                            "https://drive.google.com/"
                            f"file/d/{drive_id}/view"
                        ),
                        "-O",
                        download_path,
                    ])

                except Exception as e:

                    print(
                        "\nAutomatic FADING download failed."
                    )

                    print(
                        "Download the official specialized "
                        "checkpoint and pass:\n"
                        "fading_specialized_path=..."
                    )

                    raise e

            candidates = glob.glob(
                os.path.join(
                    fading_models_dir,
                    "*",
                )
            )

            if not candidates:

                raise RuntimeError(
                    "FADING checkpoint download returned "
                    "no files."
                )

            candidate = (
                candidates[0]
            )

            if zipfile.is_zipfile(
                candidate
            ):

                extract_dir = os.path.join(
                    fading_models_dir,
                    "specialized",
                )

                os.makedirs(
                    extract_dir,
                    exist_ok=True,
                )

                with zipfile.ZipFile(
                    candidate,
                    "r",
                ) as z:

                    z.extractall(
                        extract_dir
                    )

                fading_specialized_path = (
                    extract_dir
                )

            else:

                fading_specialized_path = (
                    candidate
                )

        bundle["FADING"] = {
            "root": fading_root,
            "specialized_path": (
                fading_specialized_path
            ),
            "script": os.path.join(
                fading_root,
                "age_editing.py",
            ),
        }

        print(
            "FADING prepared."
        )

        print(
            "specialized_path =",
            fading_specialized_path,
        )

    # ========================================================
    # CUSP
    # ========================================================

    if "cusp" in requested:

        print("\n" + "=" * 70)
        print("CUSP")
        print("=" * 70)

        cusp_models_dir = os.path.join(
            root,
            "cusp_models",
        )

        os.makedirs(
            cusp_models_dir,
            exist_ok=True,
        )

        if cusp_variant not in {
            "ffhq_ls",
            "ffhq_rr",
        }:

            raise ValueError(
                "cusp_variant must be "
                "'ffhq_ls' or 'ffhq_rr'."
            )

        # ----------------------------------------------------
        # Official pretrained model folders.
        #
        # FFHQ-RR:
        # https://drive.google.com/drive/folders/
        # 1ilazawzdIiNZq_jMxW-Gufae6x7SM66t
        #
        # FFHQ-LS:
        # https://drive.google.com/drive/folders/
        # 1C3zhHNFAXmmBAtbUoKQNP6nwJ5MHgBmw
        # ----------------------------------------------------

        cusp_drive_ids = {
            "ffhq_rr":
                "1ilazawzdIiNZq_jMxW-Gufae6x7SM66t",

            "ffhq_ls":
                "1C3zhHNFAXmmBAtbUoKQNP6nwJ5MHgBmw",
        }

        if cusp_checkpoint_path is None:

            variant_dir = os.path.join(
                cusp_models_dir,
                cusp_variant,
            )

            os.makedirs(
                variant_dir,
                exist_ok=True,
            )

            existing = glob.glob(
                os.path.join(
                    variant_dir,
                    "**",
                    "*",
                ),
                recursive=True,
            )

            existing = [
                x
                for x in existing
                if os.path.isfile(x)
            ]

            if (
                not existing
                and download_cusp
            ):

                print(
                    f"Downloading official CUSP "
                    f"{cusp_variant} pretrained model..."
                )

                drive_id = (
                    cusp_drive_ids[
                        cusp_variant
                    ]
                )

                try:

                    _run([
                        "gdown",
                        "--folder",
                        (
                            "https://drive.google.com/"
                            f"drive/folders/{drive_id}"
                        ),
                        "-O",
                        variant_dir,
                    ])

                except Exception as e:

                    print(
                        "\nAutomatic CUSP download failed."
                    )

                    print(
                        "You can manually download the "
                        "official pretrained model and pass:\n"
                        "cusp_checkpoint_path=..."
                    )

                    raise e

            # Search for likely network checkpoint.
            candidates = []

            for pattern in [
                "**/*.pkl",
                "**/*.pt",
                "**/*.pth",
            ]:

                candidates += glob.glob(
                    os.path.join(
                        variant_dir,
                        pattern,
                    ),
                    recursive=True,
                )

            # CUSP / StyleGAN2-ADA typically uses .pkl.
            pkl_candidates = [
                x
                for x in candidates
                if x.lower().endswith(".pkl")
            ]

            if pkl_candidates:

                cusp_checkpoint_path = sorted(
                    pkl_candidates
                )[-1]

            elif candidates:

                cusp_checkpoint_path = sorted(
                    candidates
                )[-1]

            else:

                cusp_checkpoint_path = (
                    variant_dir
                )

        bundle["CUSP"] = {
            "root": cusp_root,
            "variant": cusp_variant,
            "checkpoint": (
                cusp_checkpoint_path
            ),
            "generate_script": os.path.join(
                cusp_root,
                "generate.py",
            ),
            "projector_script": os.path.join(
                cusp_root,
                "projector.py",
            ),
            "models_dir": (
                cusp_models_dir
            ),
        }

        print(
            "CUSP prepared."
        )

        print(
            "variant =",
            cusp_variant,
        )

        print(
            "checkpoint =",
            cusp_checkpoint_path,
        )

    # ========================================================
    # HRFAE
    # ========================================================

    if "hrfae" in requested:

        print("\n" + "=" * 70)
        print("HRFAE")
        print("=" * 70)

        # Official pretrained model is downloaded from:
        #
        #   HRFAE/logs/001/download.sh
        #
        # The repo's official test command is:
        #
        #   python test.py --config 001 --target_age 65

        hrfae_logs_dir = os.path.join(
            hrfae_root,
            "logs",
            "001",
        )

        os.makedirs(
            hrfae_logs_dir,
            exist_ok=True,
        )

        if hrfae_checkpoint_path is None:

            download_script = os.path.join(
                hrfae_logs_dir,
                "download.sh",
            )

            # Look for already downloaded weights.
            candidates = []

            for pattern in [
                "*.pt",
                "*.pth",
                "*.pkl",
                "*.ckpt",
                "**/*.pt",
                "**/*.pth",
                "**/*.pkl",
                "**/*.ckpt",
            ]:

                candidates += glob.glob(
                    os.path.join(
                        hrfae_logs_dir,
                        pattern,
                    ),
                    recursive=True,
                )

            if (
                not candidates
                and download_hrfae
            ):

                if os.path.exists(
                    download_script
                ):

                    print(
                        "Downloading official HRFAE "
                        "pretrained model..."
                    )

                    try:

                        _run([
                            "bash",
                            "download.sh",
                        ],
                            cwd=hrfae_logs_dir,
                        )

                    except Exception as e:

                        print(
                            "\nAutomatic HRFAE download failed."
                        )

                        print(
                            "Run manually:\n"
                            f"cd {hrfae_logs_dir}\n"
                            "bash download.sh\n\n"
                            "or pass:\n"
                            "hrfae_checkpoint_path=..."
                        )

                        raise e

                else:

                    print(
                        "WARNING: HRFAE download.sh "
                        "was not found."
                    )

            # Search again after download.
            candidates = []

            for pattern in [
                "*.pt",
                "*.pth",
                "*.pkl",
                "*.ckpt",
                "**/*.pt",
                "**/*.pth",
                "**/*.pkl",
                "**/*.ckpt",
            ]:

                candidates += glob.glob(
                    os.path.join(
                        hrfae_logs_dir,
                        pattern,
                    ),
                    recursive=True,
                )

            if candidates:

                hrfae_checkpoint_path = (
                    sorted(candidates)[-1]
                )

            else:

                # test.py mainly works from config/log dir,
                # so preserving directory is still useful.
                hrfae_checkpoint_path = (
                    hrfae_logs_dir
                )

        # Official input/output directories.
        test_input_dir = os.path.join(
            hrfae_root,
            "test",
            "input",
        )

        test_output_dir = os.path.join(
            hrfae_root,
            "test",
            "output",
        )

        os.makedirs(
            test_input_dir,
            exist_ok=True,
        )

        os.makedirs(
            test_output_dir,
            exist_ok=True,
        )

        bundle["HRFAE"] = {
            "root": hrfae_root,
            "checkpoint": (
                hrfae_checkpoint_path
            ),
            "config": "001",
            "script": os.path.join(
                hrfae_root,
                "test.py",
            ),
            "logs_dir": (
                hrfae_logs_dir
            ),
            "input_dir": (
                test_input_dir
            ),
            "output_dir": (
                test_output_dir
            ),
        }

        print(
            "HRFAE prepared."
        )

        print(
            "checkpoint/log path =",
            hrfae_checkpoint_path,
        )

    # ========================================================
    # Cradle2Cane
    # ========================================================

    if "cradle2cane" in requested:

        print("\n" + "=" * 70)
        print("Cradle2Cane")
        print("=" * 70)

        model_root = os.path.join(
            root,
            "c2c_models",
        )

        os.makedirs(
            model_root,
            exist_ok=True,
        )

        paths = {

            "sdxl": os.path.join(
                model_root,
                "sdxl-turbo",
            ),

            "vae": os.path.join(
                model_root,
                "sdxl-vae-fp16-fix",
            ),

            "clip": os.path.join(
                model_root,
                "clip-vit-large-patch14",
            ),

            "official": os.path.join(
                model_root,
                "official",
            ),
        }

        if download_cradle:

            print(
                "WARNING: Cradle2Cane requires "
                "very large dependencies."
            )

            print(
                "Downloading Cradle2Cane dependencies..."
            )

            snapshot_download(
                repo_id=(
                    "stabilityai/"
                    "sdxl-turbo"
                ),
                local_dir=paths[
                    "sdxl"
                ],
            )

            snapshot_download(
                repo_id=(
                    "madebyollin/"
                    "sdxl-vae-fp16-fix"
                ),
                local_dir=paths[
                    "vae"
                ],
            )

            snapshot_download(
                repo_id=(
                    "openai/"
                    "clip-vit-large-patch14"
                ),
                local_dir=paths[
                    "clip"
                ],
            )

            snapshot_download(
                repo_id=(
                    "byliutao/"
                    "Cradle2Cane"
                ),
                local_dir=paths[
                    "official"
                ],
            )

        bundle[
            "Cradle2Cane"
        ] = {

            "root": c2c_root,

            "script": os.path.join(
                c2c_root,
                "infer.py",
            ),

            "models_root": (
                model_root
            ),

            "downloaded": (
                download_cradle
            ),

            "baseline": False,

            **paths,
        }

        if download_cradle:

            print(
                "Cradle2Cane prepared "
                "[dependencies downloaded]."
            )

        else:

            print(
                "Cradle2Cane supported "
                "[dependencies NOT downloaded]."
            )

            print(
                "Use download_cradle=True "
                "only if you explicitly want it."
            )

    # ========================================================
    # Final
    # ========================================================

    bundle = fix_aging_baselines(bundle)

    if load_metrics:
        if not metrics_config:
            raise ValueError(
                "metrics_config is required when load_metrics=True and must contain "
                "local AdaFace and DEX paths."
            )
        from src.quantitative_metrics import load_quantitative_metrics

        metric_options = dict(metrics_config)
        metric_options.setdefault("local_files_only", True)
        bundle["quantitative_metrics"] = load_quantitative_metrics(**metric_options)

    _clear_vram()

    print(
        "\n" + "=" * 70
    )

    print(
        "BUNDLE READY"
    )

    print(
        "=" * 70
    )

    for name in [
        "SAM",
        "FRAN",
        "FADING",
        "CUSP",
        "HRFAE",
        "Cradle2Cane",
    ]:

        print(
            f"{name:15s}:",
            (
                "YES"
                if bundle[
                    name
                ] is not None
                else "NO"
            ),
        )

    print(
        "\nDefault baseline:"
    )

    print(
        " + ".join(
            bundle[
                "baseline_models"
            ]
        )
    )

    print(
        "\nCradle2Cane supported:",
        (
            bundle[
                "Cradle2Cane"
            ] is not None
        ),
    )

    return bundle

# ============================================================
# FIX LEGACY BASELINES AFTER load_aging_models()
# ============================================================

def fix_aging_baselines(bundle):
    """
    Fix compatibility/path issues for legacy aging repositories.

    Fixes
    -----
    FADING:
        Automatically finds the real Diffusers pipeline directory
        containing model_index.json and updates:
            bundle["FADING"]["specialized_path"]

    HRFAE:
        Patches legacy PyYAML:
            yaml.load(...)
        ->
            yaml.safe_load(...)

    Returns
    -------
    bundle
        Same bundle, corrected in place.
    """

    # ========================================================
    # FADING
    # ========================================================

    if bundle.get("FADING") is not None:

        print("\n" + "=" * 70)
        print("FIXING FADING")
        print("=" * 70)

        B = bundle["FADING"]

        current_path = B["specialized_path"]

        print("Current specialized_path:")
        print(current_path)

        # ----------------------------------------------------
        # First check current directory directly
        # ----------------------------------------------------

        direct_model_index = os.path.join(
            current_path,
            "model_index.json",
        )

        if os.path.isfile(direct_model_index):

            print("\nFADING path already valid.")

        else:

            # ------------------------------------------------
            # Search recursively.
            #
            # The downloaded Google Drive artifact often has
            # one or more extra directory levels, e.g.
            #
            # specialized/
            #     specialized_model/
            #         model_index.json
            # ------------------------------------------------

            search_roots = [
                current_path,
                os.path.dirname(current_path),
                os.path.join(
                    bundle["root"],
                    "fading_models",
                ),
            ]

            model_indices = []

            for root in search_roots:

                if not os.path.exists(root):
                    continue

                model_indices.extend(
                    glob.glob(
                        os.path.join(
                            root,
                            "**",
                            "model_index.json",
                        ),
                        recursive=True,
                    )
                )

            # Unique paths
            model_indices = sorted(
                set(model_indices)
            )

            if not model_indices:

                print("\nFADING directory tree:")

                fading_root = os.path.join(
                    bundle["root"],
                    "fading_models",
                )

                for root, dirs, files in os.walk(
                    fading_root
                ):

                    level = root.replace(
                        fading_root,
                        ""
                    ).count(
                        os.sep
                    )

                    indent = "    " * level

                    print(
                        f"{indent}{os.path.basename(root)}/"
                    )

                    for file in files[:20]:

                        print(
                            f"{indent}    {file}"
                        )

                raise FileNotFoundError(
                    "\nCould not find model_index.json anywhere "
                    "inside fading_models.\n\n"
                    "This means the downloaded FADING artifact "
                    "is not a complete Diffusers pretrained pipeline."
                )

            # ------------------------------------------------
            # If several exist, prefer one that contains the
            # expected Diffusers components.
            # ------------------------------------------------

            def score_pipeline(path):

                folder = os.path.dirname(
                    path
                )

                expected = [
                    "unet",
                    "vae",
                    "text_encoder",
                    "tokenizer",
                    "scheduler",
                ]

                return sum(
                    os.path.exists(
                        os.path.join(
                            folder,
                            name,
                        )
                    )
                    for name in expected
                )

            best_model_index = max(
                model_indices,
                key=score_pipeline,
            )

            correct_path = os.path.dirname(
                best_model_index
            )

            B["specialized_path"] = correct_path

            print("\nFound Diffusers pipeline:")
            print(correct_path)

        # ----------------------------------------------------
        # Final validation
        # ----------------------------------------------------

        corrected_path = B[
            "specialized_path"
        ]

        required_file = os.path.join(
            corrected_path,
            "model_index.json",
        )

        if not os.path.isfile(
            required_file
        ):

            raise RuntimeError(
                "FADING fix failed: model_index.json "
                "still not present."
            )

        print("\nFADING FIXED")
        print(
            "specialized_path =",
            corrected_path,
        )

    # ========================================================
    # HRFAE
    # ========================================================

    if bundle.get("HRFAE") is not None:

        print("\n" + "=" * 70)
        print("FIXING HRFAE")
        print("=" * 70)

        B = bundle["HRFAE"]

        test_py = Path(
            B["root"]
        ) / "test.py"

        if not test_py.exists():

            raise FileNotFoundError(
                f"HRFAE test.py not found:\n{test_py}"
            )

        txt = test_py.read_text(
            encoding="utf-8"
        )

        # ----------------------------------------------------
        # Original legacy line:
        #
        # config = yaml.load(
        #     open('./configs/' + opts.config + '.yaml', 'r')
        # )
        #
        # Modern PyYAML requires a Loader.
        # safe_load is sufficient for YAML config files.
        # ----------------------------------------------------

        old = (
            "config = yaml.load("
            "open('./configs/' + opts.config + '.yaml', 'r'))"
        )

        new = (
            "config = yaml.safe_load("
            "open('./configs/' + opts.config + '.yaml', 'r'))"
        )

        if old in txt:

            txt = txt.replace(
                old,
                new,
            )

            test_py.write_text(
                txt,
                encoding="utf-8",
            )

            print(
                "Patched yaml.load -> yaml.safe_load"
            )

        elif "yaml.safe_load(" in txt:

            print(
                "PyYAML patch already applied."
            )

        else:

            # More general fallback in case formatting differs
            txt2 = txt.replace(
                "yaml.load(open('./configs/' + opts.config + '.yaml', 'r'))",
                "yaml.safe_load(open('./configs/' + opts.config + '.yaml', 'r'))",
            )

            if txt2 != txt:

                test_py.write_text(
                    txt2,
                    encoding="utf-8",
                )

                print(
                    "Patched alternate yaml.load formatting."
                )

            else:

                print(
                    "WARNING: Could not automatically locate "
                    "the yaml.load line in HRFAE/test.py."
                )

        print(
            "\nHRFAE FIXED"
        )

    # ========================================================
    # DONE
    # ========================================================

    print("\n" + "=" * 70)
    print("BASELINE FIXES COMPLETE")
    print("=" * 70)

    return bundle



