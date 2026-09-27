import os
import io
import gc
import sys
import glob
import shutil
import tempfile
import subprocess
import requests
import numpy as np
import torch
from pathlib import Path

from PIL import Image


# ============================================================
# GENERAL HELPERS
# ============================================================

def _clear_vram():
    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

        try:
            torch.cuda.ipc_collect()
        except Exception:
            pass


def _load_source_image(image):
    """
    Parameters
    ----------
    image:
        PIL.Image
        local path
        URL

    Returns
    -------
    PIL.Image RGB
    """

    if isinstance(image, Image.Image):
        return image.convert("RGB")

    if isinstance(image, Path):
        return Image.open(image.expanduser()).convert("RGB")

    if isinstance(image, str):

        if image.startswith(("http://", "https://")):

            r = requests.get(
                image,
                timeout=60,
            )

            r.raise_for_status()

            return Image.open(
                io.BytesIO(r.content)
            ).convert("RGB")

        return Image.open(
            image
        ).convert("RGB")

    raise TypeError(
        "image must be PIL.Image, pathlib.Path, local path or URL."
    )


def _to_temp_file(image):
    """
    Save source image temporarily for repositories
    that require a physical path.
    """

    tmp = tempfile.NamedTemporaryFile(
        suffix=".jpg",
        delete=False,
    )

    image.save(
        tmp.name,
        quality=95,
    )

    return tmp.name


def _move_model(model, device):
    model = model.to(device)
    model.eval()
    return model


def _find_images(folder):
    """
    Recursively locate image files.
    """

    if not os.path.exists(folder):
        return []

    patterns = [
        "*.png",
        "*.jpg",
        "*.jpeg",
        "*.webp",
        "**/*.png",
        "**/*.jpg",
        "**/*.jpeg",
        "**/*.webp",
    ]

    found = []

    for pattern in patterns:

        found.extend(
            glob.glob(
                os.path.join(
                    folder,
                    pattern,
                ),
                recursive=True,
            )
        )

    return sorted(
        set(found)
    )


def _find_age_image(folder, age):
    """
    Attempt to identify an image corresponding
    to a requested target age.
    """

    images = _find_images(
        folder
    )

    if not images:
        return None

    age_str = str(
        int(age)
    )

    candidates = []

    for path in images:

        stem = os.path.splitext(
            os.path.basename(path)
        )[0].lower()

        if (
            stem == age_str
            or stem.endswith("_" + age_str)
            or f"age{age_str}" in stem
            or f"age_{age_str}" in stem
            or f"_{age_str}_" in stem
        ):
            candidates.append(
                path
            )

    if candidates:

        return max(
            candidates,
            key=os.path.getmtime,
        )

    return None


def _latest_image(folder):
    """
    Return most recently modified image.
    """

    images = _find_images(
        folder
    )

    if not images:
        return None

    return max(
        images,
        key=os.path.getmtime,
    )


def _save_standard_result(
    image,
    output_dir,
    model_name,
    age,
):
    """
    Save model output with a common naming convention.
    """

    model_dir = os.path.join(
        output_dir,
        model_name,
    )

    os.makedirs(
        model_dir,
        exist_ok=True,
    )

    output_path = os.path.join(
        model_dir,
        f"age_{int(age):03d}.png",
    )

    image.convert(
        "RGB"
    ).save(
        output_path
    )

    return output_path


def _save_model_sweep(
    source,
    source_age,
    model_name,
    model_results,
    target_ages,
    output_dir,
):
    """
    Save ONE horizontal sweep for one model.

    Layout
    ------
    Source | Age 8 | Age 15 | Age 27 | Age 45 | ...
    """

    available_ages = [
        age
        for age in target_ages
        if age in model_results
    ]

    if not available_ages:
        return None

    import matplotlib.pyplot as plt

    n_cols = (
        len(available_ages)
        + 1
    )

    fig, axes = plt.subplots(
        1,
        n_cols,
        figsize=(
            3.2 * n_cols,
            3.8,
        ),
        squeeze=False,
    )

    axes = axes[0]

    # --------------------------------------------------------
    # Source
    # --------------------------------------------------------

    axes[0].imshow(
        source
    )

    axes[0].set_title(
        f"Source\nAge {source_age}"
    )

    axes[0].axis(
        "off"
    )

    # --------------------------------------------------------
    # Target ages
    # --------------------------------------------------------

    for idx, age in enumerate(
        available_ages,
        start=1,
    ):

        axes[idx].imshow(
            model_results[age]
        )

        axes[idx].set_title(
            f"Age {age}"
        )

        axes[idx].axis(
            "off"
        )

    fig.suptitle(
        model_name,
        fontsize=16,
    )

    plt.tight_layout()

    model_dir = os.path.join(
        output_dir,
        model_name,
    )

    os.makedirs(
        model_dir,
        exist_ok=True,
    )

    sweep_path = os.path.join(
        model_dir,
        "sweep.png",
    )

    plt.savefig(
        sweep_path,
        dpi=180,
        bbox_inches="tight",
    )

    plt.close(
        fig
    )

    return sweep_path


def _run_verbose(
    cmd,
    cwd=None,
    env=None,
):
    """
    Run external repository scripts.

    If something fails, print the COMPLETE stdout/stderr
    instead of only returning:

        exit status 1

    This is essential for debugging old research repos.
    """

    print(
        "\nCOMMAND:"
    )

    print(
        " ".join(
            map(str, cmd)
        )
    )

    proc = subprocess.run(
        list(map(str, cmd)),
        cwd=cwd,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )

    if proc.stdout:

        print(
            "\n"
            + proc.stdout
        )

    if proc.returncode != 0:

        raise RuntimeError(
            f"\nCommand failed with exit code "
            f"{proc.returncode}\n\n"
            f"{proc.stdout}"
        )

    return proc


# ============================================================
# MAIN INFERENCE FUNCTION
# ============================================================

def age_image(
    bundle,
    image,
    source_age,
    target_ages,
    gender="male",
    models=None,
    seed=42,
    show=True,
    output_dir="/content/aging_outputs",
    strict=False,
):
    """
    Generate face-aging sweeps sequentially.

    Default benchmark
    -----------------
    SAM
    FRAN
    FADING
    CUSP
    HRFAE

    Cradle2Cane remains supported if explicitly requested.

    Memory policy
    -------------
    SAM:
        CPU -> GPU -> complete sweep -> CPU

    FRAN:
        CPU -> GPU -> complete sweep -> CPU

    FADING:
        isolated subprocess -> process exits -> CUDA released

    CUSP:
        attempted only if executable inference support exists

    HRFAE:
        isolated subprocess -> process exits -> CUDA released

    Cradle2Cane:
        isolated subprocess -> process exits -> CUDA released

    Output
    ------
    output_dir/
        source.png

        SAM/
            age_008.png
            age_015.png
            ...
            sweep.png

        FRAN/
            ...
            sweep.png

        FADING/
            ...
            sweep.png

        HRFAE/
            ...
            sweep.png
    """

    # ========================================================
    # Configuration
    # ========================================================

    device = bundle[
        "device"
    ]

    if models is None:

        models = bundle.get(
            "baseline_models",
            [
                "SAM",
                "FRAN",
                "FADING",
                "CUSP",
                "HRFAE",
            ],
        )

    requested = {
        str(x).lower()
        for x in models
    }

    # ========================================================
    # Normalize ages
    # ========================================================

    if isinstance(
        target_ages,
        (
            int,
            float,
            np.integer,
        ),
    ):

        target_ages = [
            int(target_ages)
        ]

    else:

        target_ages = [
            int(age)
            for age in target_ages
        ]

    # Remove duplicates preserving order

    target_ages = list(
        dict.fromkeys(
            target_ages
        )
    )

    if not target_ages:

        raise ValueError(
            "target_ages cannot be empty."
        )

    # ========================================================
    # Gender
    # ========================================================

    gender = str(
        gender
    ).lower()

    if gender not in {
        "male",
        "female",
    }:

        raise ValueError(
            "gender must be 'male' or 'female'."
        )

    # ========================================================
    # Source image
    # ========================================================

    source = _load_source_image(
        image
    )

    source_path = _to_temp_file(
        source
    )

    os.makedirs(
        output_dir,
        exist_ok=True,
    )

    source.save(
        os.path.join(
            output_dir,
            "source.png",
        )
    )

    results = {

        "source":
            source,

        "source_age":
            int(source_age),

        "target_ages":
            target_ages,

        "output_dir":
            output_dir,

        "status":
            {},
    }

    # ========================================================
    # SAM
    # ========================================================

    if (
        "sam" in requested
        and bundle.get(
            "SAM"
        ) is not None
    ):

        print(
            "\n"
            + "=" * 70
        )

        print(
            "SAM -> GPU"
        )

        print(
            "=" * 70
        )

        B = bundle[
            "SAM"
        ]

        model = B[
            "model"
        ]

        sam_results = {}

        try:

            import torchvision.transforms as T

            # ------------------------------------------------
            # Move to GPU
            # ------------------------------------------------

            model = _move_model(
                model,
                device,
            )

            # ------------------------------------------------
            # Official SAM alignment
            # ------------------------------------------------

            aligned = B[
                "align_face"
            ](
                filepath=source_path,
                predictor=B[
                    "predictor"
                ],
            )

            transform = T.Compose([

                T.Resize(
                    (256, 256)
                ),

                T.ToTensor(),

                T.Normalize(
                    [0.5, 0.5, 0.5],
                    [0.5, 0.5, 0.5],
                ),
            ])

            base_tensor = transform(
                aligned
            )

            # ------------------------------------------------
            # Complete sweep
            # ------------------------------------------------

            for age in target_ages:

                print(
                    f"SAM: "
                    f"{source_age} -> {age}"
                )

                age_transform = B[
                    "AgeTransformer"
                ](
                    target_age=age
                )

                x = age_transform(
                    base_tensor.clone()
                )

                x = (
                    x
                    .unsqueeze(0)
                    .to(device)
                    .float()
                )

                with torch.inference_mode():

                    y = model(
                        x,
                        randomize_noise=False,
                        resize=False,
                    )[0]

                tensor_result = (
                    y[0]
                    if y.ndim == 4
                    else y
                )

                im = B[
                    "tensor2im"
                ](
                    tensor_result
                )

                if not isinstance(
                    im,
                    Image.Image,
                ):

                    im = Image.fromarray(
                        np.asarray(im)
                    )

                im = im.convert(
                    "RGB"
                )

                sam_results[
                    age
                ] = im

                _save_standard_result(
                    image=im,
                    output_dir=output_dir,
                    model_name="SAM",
                    age=age,
                )

                del x
                del y

            # ------------------------------------------------
            # Save model sweep
            # ------------------------------------------------

            _save_model_sweep(
                source=source,
                source_age=source_age,
                model_name="SAM",
                model_results=sam_results,
                target_ages=target_ages,
                output_dir=output_dir,
            )

            results[
                "SAM"
            ] = sam_results

            results[
                "status"
            ][
                "SAM"
            ] = "OK"

        except Exception as e:

            results[
                "status"
            ][
                "SAM"
            ] = (
                f"ERROR: {e}"
            )

            print(
                "\nSAM FAILED:\n"
                f"{e}"
            )

            if strict:
                raise

        finally:

            print(
                "SAM -> CPU"
            )

            try:

                B["model"] = (
                    model.to(
                        "cpu"
                    )
                )

            except Exception:
                pass

            _clear_vram()

    # ========================================================
    # FRAN
    # ========================================================

    if (
        "fran" in requested
        and bundle.get(
            "FRAN"
        ) is not None
    ):

        print(
            "\n"
            + "=" * 70
        )

        print(
            "FRAN reproduction -> GPU"
        )

        print(
            "=" * 70
        )

        B = bundle[
            "FRAN"
        ]

        model = B[
            "model"
        ]

        fran_results = {}

        old_cwd = os.getcwd()

        try:

            # FRAN expects assets/
            # relative to repository root

            os.chdir(
                B["root"]
            )

            model = _move_model(
                model,
                device,
            )

            for age in target_ages:

                print(
                    f"FRAN: "
                    f"{source_age} -> {age}"
                )

                with torch.inference_mode():

                    result = B[
                        "process_image"
                    ](
                        model,
                        source.copy(),

                        video=False,

                        source_age=float(
                            source_age
                        ),

                        target_age=float(
                            age
                        ),

                        window_size=B[
                            "window_size"
                        ],

                        stride=B[
                            "stride"
                        ],
                    )

                if isinstance(
                    result,
                    np.ndarray,
                ):

                    arr = result

                    if arr.dtype != np.uint8:

                        if arr.max() <= 1:

                            arr = (
                                arr
                                * 255
                            )

                        arr = (
                            arr
                            .clip(
                                0,
                                255,
                            )
                            .astype(
                                np.uint8
                            )
                        )

                    result = Image.fromarray(
                        arr
                    )

                result = result.convert(
                    "RGB"
                )

                fran_results[
                    age
                ] = result

                _save_standard_result(
                    image=result,
                    output_dir=output_dir,
                    model_name="FRAN",
                    age=age,
                )

            _save_model_sweep(
                source=source,
                source_age=source_age,
                model_name="FRAN",
                model_results=fran_results,
                target_ages=target_ages,
                output_dir=output_dir,
            )

            results[
                "FRAN"
            ] = fran_results

            results[
                "status"
            ][
                "FRAN"
            ] = "OK"

        except Exception as e:

            results[
                "status"
            ][
                "FRAN"
            ] = (
                f"ERROR: {e}"
            )

            print(
                "\nFRAN FAILED:\n"
                f"{e}"
            )

            if strict:
                raise

        finally:

            os.chdir(
                old_cwd
            )

            print(
                "FRAN reproduction -> CPU"
            )

            try:

                B["model"] = (
                    model.to(
                        "cpu"
                    )
                )

            except Exception:
                pass

            _clear_vram()

    # ========================================================
    # FADING
    # ========================================================

    if (
        "fading" in requested
        and bundle.get(
            "FADING"
        ) is not None
    ):

        print(
            "\n"
            + "=" * 70
        )

        print(
            "FADING -> isolated GPU process"
        )

        print(
            "=" * 70
        )

        B = bundle[
            "FADING"
        ]

        fading_results = {}

        fading_work = (
            tempfile.mkdtemp(
                prefix="fading_"
            )
        )

        try:

            cmd = [

                sys.executable,

                B[
                    "script"
                ],

                "--image_path",
                source_path,

                "--age_init",
                str(
                    int(
                        source_age
                    )
                ),

                "--gender",
                gender,

                "--save_aged_dir",
                fading_work,

                "--specialized_path",
                B[
                    "specialized_path"
                ],

                "--target_ages",

            ] + [

                str(age)
                for age in target_ages
            ]

            env = (
                os.environ.copy()
            )

            env[
                "PYTHONHASHSEED"
            ] = str(
                seed
            )

            _run_verbose(
                cmd=cmd,
                cwd=B[
                    "root"
                ],
                env=env,
            )

            _clear_vram()

            # ------------------------------------------------
            # Collect generated ages
            # ------------------------------------------------

            for age in target_ages:

                output_file = (
                    _find_age_image(
                        fading_work,
                        age,
                    )
                )

                if (
                    output_file
                    is None
                ):

                    print(
                        f"WARNING: "
                        f"FADING output "
                        f"not found for age "
                        f"{age}"
                    )

                    continue

                im = Image.open(
                    output_file
                ).convert(
                    "RGB"
                )

                fading_results[
                    age
                ] = im

                _save_standard_result(
                    image=im,
                    output_dir=output_dir,
                    model_name="FADING",
                    age=age,
                )

            _save_model_sweep(
                source=source,
                source_age=source_age,
                model_name="FADING",
                model_results=fading_results,
                target_ages=target_ages,
                output_dir=output_dir,
            )

            results[
                "FADING"
            ] = fading_results

            if (
                len(
                    fading_results
                )
                ==
                len(
                    target_ages
                )
            ):

                results[
                    "status"
                ][
                    "FADING"
                ] = "OK"

            else:

                results[
                    "status"
                ][
                    "FADING"
                ] = (
                    f"PARTIAL: "
                    f"{len(fading_results)}/"
                    f"{len(target_ages)}"
                )

        except Exception as e:

            results[
                "FADING"
            ] = fading_results

            results[
                "status"
            ][
                "FADING"
            ] = (
                f"ERROR: {e}"
            )

            print(
                "\n"
                + "=" * 70
            )

            print(
                "FADING FAILED"
            )

            print(
                "=" * 70
            )

            print(
                e
            )

            if strict:
                raise

        finally:

            shutil.rmtree(
                fading_work,
                ignore_errors=True,
            )

            _clear_vram()

    # ========================================================
    # CUSP
    # ========================================================

    if (
        "cusp" in requested
        and bundle.get(
            "CUSP"
        ) is not None
    ):

        print(
            "\n"
            + "=" * 70
        )

        print(
            "CUSP"
        )

        print(
            "=" * 70
        )

        B = bundle[
            "CUSP"
        ]

        cusp_results = {}

        try:

            projector_script = B.get(
                "projector_script"
            )

            generate_script = B.get(
                "generate_script"
            )

            # ------------------------------------------------
            # Check whether repo actually contains
            # executable versions of the scripts.
            # ------------------------------------------------

            usable = True

            for script_path in [
                projector_script,
                generate_script,
            ]:

                if (
                    script_path is None
                    or not os.path.exists(
                        script_path
                    )
                ):

                    usable = False
                    break

                with open(
                    script_path,
                    "r",
                    encoding="utf-8",
                    errors="ignore",
                ) as f:

                    content = (
                        f.read()
                    )

                # If virtually everything is commented,
                # don't pretend inference works.

                executable_lines = [

                    line.strip()

                    for line
                    in content.splitlines()

                    if (
                        line.strip()
                        and not line
                        .strip()
                        .startswith(
                            "#"
                        )
                    )
                ]

                if (
                    len(
                        executable_lines
                    )
                    < 10
                ):

                    usable = False
                    break

            if not usable:

                raise RuntimeError(
                    "Current CUSP checkout does not expose "
                    "a directly executable real-image "
                    "projection + aging pipeline. "
                    "Need to port the official notebook "
                    "pipeline before using CUSP as a benchmark."
                )

            # If later we add a valid CUSP runner,
            # this is where it will execute.

            results[
                "CUSP"
            ] = cusp_results

            results[
                "status"
            ][
                "CUSP"
            ] = (
                "NO EXECUTABLE RUNNER YET"
            )

        except Exception as e:

            results[
                "CUSP"
            ] = cusp_results

            results[
                "status"
            ][
                "CUSP"
            ] = (
                f"SKIPPED: {e}"
            )

            print(
                results[
                    "status"
                ][
                    "CUSP"
                ]
            )

            if strict:
                raise

        finally:

            _clear_vram()

    # ========================================================
    # HRFAE
    # ========================================================

    if (
        "hrfae" in requested
        and bundle.get(
            "HRFAE"
        ) is not None
    ):

        print(
            "\n"
            + "=" * 70
        )

        print(
            "HRFAE -> isolated GPU process"
        )

        print(
            "=" * 70
        )

        B = bundle[
            "HRFAE"
        ]

        hrfae_results = {}

        input_dir = B[
            "input_dir"
        ]

        repo_output_dir = B[
            "output_dir"
        ]

        os.makedirs(
            input_dir,
            exist_ok=True,
        )

        os.makedirs(
            repo_output_dir,
            exist_ok=True,
        )

        # ----------------------------------------------------
        # HRFAE expects image inside test/input
        # ----------------------------------------------------

        input_name = (
            "benchmark_source.png"
        )

        hrfae_input = os.path.join(
            input_dir,
            input_name,
        )

        try:

            if os.path.exists(
                hrfae_input
            ):

                os.remove(
                    hrfae_input
                )

            source.save(
                hrfae_input
            )

            env = (
                os.environ.copy()
            )

            env[
                "PYTHONHASHSEED"
            ] = str(
                seed
            )

            # ------------------------------------------------
            # One target age at a time
            # ------------------------------------------------

            for age in target_ages:

                print(
                    f"HRFAE: "
                    f"{source_age} -> {age}"
                )

                # Existing outputs before run

                before = {

                    path:
                        os.path.getmtime(
                            path
                        )

                    for path
                    in _find_images(
                        repo_output_dir
                    )
                }

                cmd = [

                    sys.executable,

                    B[
                        "script"
                    ],

                    "--config",
                    str(
                        B[
                            "config"
                        ]
                    ),

                    "--target_age",
                    str(
                        int(
                            age
                        )
                    ),
                ]

                _run_verbose(
                    cmd=cmd,
                    cwd=B[
                        "root"
                    ],
                    env=env,
                )

                _clear_vram()

                # ------------------------------------------------
                # Detect new / modified output
                # ------------------------------------------------

                after = (
                    _find_images(
                        repo_output_dir
                    )
                )

                changed = []

                for path in after:

                    new_mtime = (
                        os.path.getmtime(
                            path
                        )
                    )

                    old_mtime = (
                        before.get(
                            path
                        )
                    )

                    if (
                        old_mtime is None
                        or
                        new_mtime
                        >
                        old_mtime
                    ):

                        changed.append(
                            path
                        )

                result_path = None

                # Prefer matching source filename

                matching = [

                    path

                    for path
                    in changed

                    if (
                        "benchmark_source"
                        in
                        os.path.basename(
                            path
                        )
                    )
                ]

                if matching:

                    result_path = max(
                        matching,
                        key=os.path.getmtime,
                    )

                elif changed:

                    result_path = max(
                        changed,
                        key=os.path.getmtime,
                    )

                else:

                    # fallback if file overwritten
                    # without detectable modification list

                    candidate = (
                        _latest_image(
                            repo_output_dir
                        )
                    )

                    if candidate:

                        result_path = (
                            candidate
                        )

                if (
                    result_path
                    is None
                ):

                    print(
                        f"WARNING: "
                        f"HRFAE output "
                        f"not found for "
                        f"age {age}"
                    )

                    continue

                im = Image.open(
                    result_path
                ).convert(
                    "RGB"
                )

                hrfae_results[
                    age
                ] = im

                _save_standard_result(
                    image=im,
                    output_dir=output_dir,
                    model_name="HRFAE",
                    age=age,
                )

            _save_model_sweep(
                source=source,
                source_age=source_age,
                model_name="HRFAE",
                model_results=hrfae_results,
                target_ages=target_ages,
                output_dir=output_dir,
            )

            results[
                "HRFAE"
            ] = hrfae_results

            if (
                len(
                    hrfae_results
                )
                ==
                len(
                    target_ages
                )
            ):

                results[
                    "status"
                ][
                    "HRFAE"
                ] = "OK"

            else:

                results[
                    "status"
                ][
                    "HRFAE"
                ] = (
                    f"PARTIAL: "
                    f"{len(hrfae_results)}/"
                    f"{len(target_ages)}"
                )

        except Exception as e:

            results[
                "HRFAE"
            ] = hrfae_results

            results[
                "status"
            ][
                "HRFAE"
            ] = (
                f"ERROR: {e}"
            )

            print(
                "\n"
                + "=" * 70
            )

            print(
                "HRFAE FAILED"
            )

            print(
                "=" * 70
            )

            print(
                e
            )

            if strict:
                raise

        finally:

            try:

                if os.path.exists(
                    hrfae_input
                ):

                    os.remove(
                        hrfae_input
                    )

            except Exception:
                pass

            _clear_vram()

    # ========================================================
    # CRADLE2CANE
    # ========================================================

    if (
        "cradle2cane"
        in requested
        and bundle.get(
            "Cradle2Cane"
        ) is not None
    ):

        print(
            "\n"
            + "=" * 70
        )

        print(
            "Cradle2Cane -> isolated GPU process"
        )

        print(
            "=" * 70
        )

        B = bundle[
            "Cradle2Cane"
        ]

        c2c_results = {}

        # ----------------------------------------------------
        # Avoid accidental 29GB baseline execution
        # ----------------------------------------------------

        if not B.get(
            "downloaded",
            False,
        ):

            message = (
                "SKIPPED: Cradle2Cane dependencies "
                "were not downloaded. "
                "Use download_cradle=True to enable it."
            )

            print(
                message
            )

            results[
                "Cradle2Cane"
            ] = {}

            results[
                "status"
            ][
                "Cradle2Cane"
            ] = message

        else:

            c2c_work = (
                tempfile.mkdtemp(
                    prefix="c2c_"
                )
            )

            try:

                # Filename expected by official pipeline

                c2c_input = os.path.join(
                    c2c_work,
                    (
                        f"{int(source_age)}_"
                        f"{gender}.png"
                    ),
                )

                source.save(
                    c2c_input
                )

                cmd = [

                    sys.executable,

                    B[
                        "script"
                    ],

                    "--input_path",
                    c2c_input,
                ]

                env = (
                    os.environ.copy()
                )

                env[
                    "PYTHONHASHSEED"
                ] = str(
                    seed
                )

                _run_verbose(
                    cmd=cmd,
                    cwd=B[
                        "root"
                    ],
                    env=env,
                )

                _clear_vram()

                candidate_folders = [

                    c2c_work,

                    B[
                        "root"
                    ],

                    os.path.join(
                        B[
                            "root"
                        ],
                        "results",
                    ),

                    os.path.join(
                        B[
                            "root"
                        ],
                        "outputs",
                    ),
                ]

                for age in target_ages:

                    found = None

                    for folder in candidate_folders:

                        found = _find_age_image(
                            folder,
                            age,
                        )

                        if (
                            found
                            is not None
                        ):
                            break

                    if (
                        found
                        is None
                    ):

                        print(
                            f"WARNING: "
                            f"Cradle2Cane output "
                            f"not found for "
                            f"age {age}"
                        )

                        continue

                    im = Image.open(
                        found
                    ).convert(
                        "RGB"
                    )

                    c2c_results[
                        age
                    ] = im

                    _save_standard_result(
                        image=im,
                        output_dir=output_dir,
                        model_name="Cradle2Cane",
                        age=age,
                    )

                _save_model_sweep(
                    source=source,
                    source_age=source_age,
                    model_name="Cradle2Cane",
                    model_results=c2c_results,
                    target_ages=target_ages,
                    output_dir=output_dir,
                )

                results[
                    "Cradle2Cane"
                ] = c2c_results

                if (
                    len(
                        c2c_results
                    )
                    ==
                    len(
                        target_ages
                    )
                ):

                    results[
                        "status"
                    ][
                        "Cradle2Cane"
                    ] = "OK"

                else:

                    results[
                        "status"
                    ][
                        "Cradle2Cane"
                    ] = (
                        f"PARTIAL: "
                        f"{len(c2c_results)}/"
                        f"{len(target_ages)}"
                    )

            except Exception as e:

                results[
                    "Cradle2Cane"
                ] = c2c_results

                results[
                    "status"
                ][
                    "Cradle2Cane"
                ] = (
                    f"ERROR: {e}"
                )

                print(
                    e
                )

                if strict:
                    raise

            finally:

                shutil.rmtree(
                    c2c_work,
                    ignore_errors=True,
                )

                _clear_vram()

    # ========================================================
    # OPTIONAL DISPLAY
    # ========================================================

    if show:

        import matplotlib.pyplot as plt

        for model_name in [
            "SAM",
            "FRAN",
            "FADING",
            "CUSP",
            "HRFAE",
            "Cradle2Cane",
        ]:

            if (
                model_name
                not in results
            ):
                continue

            model_results = (
                results[
                    model_name
                ]
            )

            if not isinstance(
                model_results,
                dict,
            ):
                continue

            if not model_results:
                continue

            available_ages = [

                age

                for age
                in target_ages

                if age
                in model_results
            ]

            if not available_ages:
                continue

            n_cols = (
                len(
                    available_ages
                )
                + 1
            )

            fig, axes = plt.subplots(
                1,
                n_cols,
                figsize=(
                    3.2
                    * n_cols,
                    3.8,
                ),
                squeeze=False,
            )

            axes = (
                axes[0]
            )

            axes[0].imshow(
                source
            )

            axes[0].set_title(
                f"Source\nAge {source_age}"
            )

            axes[0].axis(
                "off"
            )

            for idx, age in enumerate(
                available_ages,
                start=1,
            ):

                axes[idx].imshow(
                    model_results[
                        age
                    ]
                )

                axes[idx].set_title(
                    f"Age {age}"
                )

                axes[idx].axis(
                    "off"
                )

            fig.suptitle(
                model_name,
                fontsize=16,
            )

            plt.tight_layout()

            plt.show()

            plt.close(
                fig
            )

    # ========================================================
    # CLEANUP
    # ========================================================

    try:

        os.remove(
            source_path
        )

    except Exception:
        pass

    _clear_vram()

    # ========================================================
    # STATUS
    # ========================================================

    print(
        "\n"
        + "=" * 70
    )

    print(
        "AGING SWEEP COMPLETE"
    )

    print(
        "=" * 70
    )

    for model_name in [
        "SAM",
        "FRAN",
        "FADING",
        "CUSP",
        "HRFAE",
        "Cradle2Cane",
    ]:

        if (
            model_name.lower()
            in requested
        ):

            status = (
                results[
                    "status"
                ].get(
                    model_name,
                    "NOT RUN",
                )
            )

            print(
                f"{model_name:15s}: "
                f"{status}"
            )

    print(
        "\nOutputs:"
    )

    print(
        output_dir
    )

    return results
