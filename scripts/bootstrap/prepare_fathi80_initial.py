#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path

import h5py
import numpy as np


def get_root() -> Path:
    default_root = Path(__file__).resolve().parents[2]
    return Path(
        os.environ.get("FATHI_BENCHMARK_ROOT", str(default_root))
    ).expanduser().resolve()


def load_config(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")

    return json.loads(path.read_text(encoding="utf-8"))


def logical_path(root: Path, value: str) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute():
        return path
    return root / path


def non_comment_line_count(path: Path) -> int:
    count = 0

    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if stripped and not stripped.startswith("#"):
            count += 1

    return count


def find_dataset(path: Path, expected_shape: tuple[int, int, int]) -> str:
    candidates: list[str] = []

    with h5py.File(path, "r") as h5:
        def visitor(name: str, obj) -> None:
            if isinstance(obj, h5py.Dataset):
                if tuple(obj.shape) == expected_shape:
                    candidates.append(name)

        h5.visititems(visitor)

    if not candidates:
        raise RuntimeError(
            f"No dataset with shape {expected_shape} found in {path}"
        )

    if "samples" in candidates:
        return "samples"

    return candidates[0]


def inspect_h5(
    path: Path,
    expected_shape: tuple[int, int, int],
) -> dict:
    dataset = find_dataset(path, expected_shape)

    with h5py.File(path, "r") as h5:
        array = np.asarray(h5[dataset])

        attrs = {
            key: np.asarray(value).tolist()
            for key, value in h5.attrs.items()
        }

    return {
        "path": str(path),
        "dataset": dataset,
        "shape": list(array.shape),
        "dtype": str(array.dtype),
        "finite": bool(np.isfinite(array).all()),
        "min": float(np.min(array)),
        "max": float(np.max(array)),
        "mean": float(np.mean(array)),
        "file_attrs": attrs,
    }


def overwrite_constant(
    path: Path,
    expected_shape: tuple[int, int, int],
    value: float,
) -> None:
    dataset = find_dataset(path, expected_shape)

    with h5py.File(path, "r+") as h5:
        dtype = h5[dataset].dtype
        h5[dataset][...] = np.full(
            expected_shape,
            value,
            dtype=dtype,
        )


def validate_template(template: Path) -> None:
    required = [
        "input.spec",
        "material.spec",
        "stations.txt",
        "gaussian_stf.txt",
        "mat/h5/Mat_0_Kappa.h5",
        "mat/h5/Mat_0_Mu.h5",
        "mat/h5/Mat_0_Density.h5",
        "sem",
    ]

    missing = [
        item
        for item in required
        if not (template / item).exists()
    ]

    if missing:
        raise RuntimeError(
            "Template is incomplete. Missing:\n  "
            + "\n  ".join(missing)
        )


def runtime_ignore(directory: str, names: list[str]) -> set[str]:
    ignored: set[str] = set()

    runtime_names = {
        "traces",
        "res",
        "logs",
        "prot",
        "mirror",
        "stat.log",
    }

    for name in names:
        if name in runtime_names:
            ignored.add(name)
        elif name.startswith("output."):
            ignored.add(name)

    return ignored


def build_state(
    *,
    state_path: Path,
    accepted_dir_config_value: str,
    lambda_field: np.ndarray,
    mu_field: np.ndarray,
    kappa_field: np.ndarray,
    density_field: np.ndarray,
    initial_j: float,
) -> None:
    state_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        state_path,
        **{
            "lambda": lambda_field,
            "lambda_field": lambda_field,
            "mu": mu_field,
            "kappa": kappa_field,
            "density": density_field,
            "J": np.float64(initial_j),
            "parent_J": np.float64(initial_j),
            "delta_J": np.float64(0.0),
            "iter_k": np.int64(-1),
            "iter": np.int64(0),
            "accepted_from": np.array(
                "bootstrap_fathi80_initial"
            ),
            "accepted_dir": np.array(
                accepted_dir_config_value
            ),
            "transition": np.array(
                "bootstrap_to_iter_000"
            ),
            "descent": np.bool_(True),
            "candidate_misfit_summary": np.array(
                "Canonical Fathi homogeneous 80 MPa initial state"
            ),
        },
    )

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Create the canonical Fathi homogeneous 80 MPa "
            "iter_000 accepted workspace and state."
        )
    )

    parser.add_argument(
        "--config",
        default="configs/fathi80_initial.json",
        help="Initial configuration file.",
    )

    parser.add_argument(
        "--template",
        required=True,
        help="Local SEM3D template workspace.",
    )

    parser.add_argument(
        "--write",
        action="store_true",
        help="Create the accepted workspace and state.",
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace existing bootstrap outputs.",
    )

    args = parser.parse_args()

    root = get_root()

    config_path = logical_path(root, args.config)
    template = Path(args.template).expanduser().resolve()

    config = load_config(config_path)
    validate_template(template)

    material = config["material"]
    grid = config["material_grid"]
    validation = config["fathi_validation"]
    outputs = config["runtime_outputs"]

    expected_shape = tuple(grid["shape"])

    lambda_pa = float(material["lambda_pa"])
    mu_pa = float(material["mu_pa"])
    kappa_pa = float(material["kappa_pa"])
    density_value = float(material["density_kg_m3"])

    computed_kappa = lambda_pa + (2.0 / 3.0) * mu_pa

    if not np.isclose(kappa_pa, computed_kappa):
        raise RuntimeError(
            f"Inconsistent Kappa: stored={kappa_pa}, "
            f"computed={computed_kappa}"
        )

    accepted_config_value = outputs["accepted_dir"]
    state_config_value = outputs["state_file"]

    accepted_dir = logical_path(root, accepted_config_value)
    state_path = logical_path(root, state_config_value)

    station_count = non_comment_line_count(template / "stations.txt")
    expected_receivers = int(
        validation["number_of_physical_receivers"]
    )

    if station_count != expected_receivers:
        raise RuntimeError(
            f"Station count mismatch: "
            f"found={station_count}, expected={expected_receivers}"
        )

    print("FATHI 80 MPA INITIAL BOOTSTRAP")
    print("==============================")
    print()
    print(f"root = {root}")
    print(f"config = {config_path}")
    print(f"template = {template}")
    print(f"accepted_dir = {accepted_dir}")
    print(f"state_path = {state_path}")
    print()
    print(f"lambda = {lambda_pa:.16e} Pa")
    print(f"mu = {mu_pa:.16e} Pa")
    print(f"kappa = {kappa_pa:.16e} Pa")
    print(f"density = {density_value:.16e} kg/m3")
    print(f"shape = {expected_shape}")
    print(f"stations = {station_count}")
    print()

    if not args.write:
        print("PLAN ONLY: no files were written.")
        print("Use --write to create the bootstrap outputs.")
        print("RESULT = PASS_FATHI80_BOOTSTRAP_PLAN")
        return

    if accepted_dir.exists():
        if not args.overwrite:
            raise FileExistsError(
                f"Accepted directory already exists: {accepted_dir}\n"
                "Use --overwrite only after checking the existing output."
            )

        shutil.rmtree(accepted_dir)

    if state_path.exists():
        if not args.overwrite:
            raise FileExistsError(
                f"State already exists: {state_path}\n"
                "Use --overwrite only after checking the existing output."
            )

        state_path.unlink()

    accepted_dir.parent.mkdir(parents=True, exist_ok=True)

    shutil.copytree(
        template,
        accepted_dir,
        ignore=runtime_ignore,
    )

    h5_dir = accepted_dir / "mat/h5"

    kappa_path = h5_dir / "Mat_0_Kappa.h5"
    mu_path = h5_dir / "Mat_0_Mu.h5"
    density_path = h5_dir / "Mat_0_Density.h5"

    overwrite_constant(
        kappa_path,
        expected_shape,
        kappa_pa,
    )

    overwrite_constant(
        mu_path,
        expected_shape,
        mu_pa,
    )

    overwrite_constant(
        density_path,
        expected_shape,
        density_value,
    )

    kappa_info = inspect_h5(kappa_path, expected_shape)
    mu_info = inspect_h5(mu_path, expected_shape)
    density_info = inspect_h5(density_path, expected_shape)

    for label, info in [
        ("Kappa", kappa_info),
        ("Mu", mu_info),
        ("Density", density_info),
    ]:
        if not info["finite"]:
            raise RuntimeError(f"{label} contains non-finite values")

    lambda_field = np.full(
        expected_shape,
        lambda_pa,
        dtype=np.float64,
    )

    mu_field = np.full(
        expected_shape,
        mu_pa,
        dtype=np.float64,
    )

    kappa_field = np.full(
        expected_shape,
        kappa_pa,
        dtype=np.float64,
    )

    density_field = np.full(
        expected_shape,
        density_value,
        dtype=np.float64,
    )

    initial_j = float(
        config.get("validated_reference", {}).get(
            "expected_initial_absolute_misfit",
            np.nan,
        )
    )

    build_state(
        state_path=state_path,
        accepted_dir_config_value=accepted_config_value,
        lambda_field=lambda_field,
        mu_field=mu_field,
        kappa_field=kappa_field,
        density_field=density_field,
        initial_j=initial_j,
    )

    report_dir = root / "reports/tv_extension"
    report_dir.mkdir(parents=True, exist_ok=True)

    report = {
        "config": str(config_path),
        "template": str(template),
        "accepted_dir": str(accepted_dir),
        "state_path": str(state_path),
        "station_count": station_count,
        "material": {
            "lambda_pa": lambda_pa,
            "mu_pa": mu_pa,
            "kappa_pa": kappa_pa,
            "density_kg_m3": density_value,
        },
        "h5": {
            "Kappa": kappa_info,
            "Mu": mu_info,
            "Density": density_info,
        },
        "runtime_outputs_removed": {
            "traces": not (accepted_dir / "traces").exists(),
            "res": not (accepted_dir / "res").exists(),
            "logs": not (accepted_dir / "logs").exists(),
        },
        "result": "PASS_FATHI80_BOOTSTRAP_WRITE",
    }

    json_path = report_dir / "fathi80_bootstrap_summary.json"
    txt_path = report_dir / "fathi80_bootstrap_summary.txt"

    json_path.write_text(
        json.dumps(report, indent=2) + "\n",
        encoding="utf-8",
    )

    lines = [
        "FATHI 80 MPA BOOTSTRAP SUMMARY",
        "==============================",
        "",
        f"template = {template}",
        f"accepted_dir = {accepted_dir}",
        f"state_path = {state_path}",
        f"stations = {station_count}",
        "",
        f"lambda = {lambda_pa:.16e} Pa",
        f"mu = {mu_pa:.16e} Pa",
        f"kappa = {kappa_pa:.16e} Pa",
        f"density = {density_value:.16e} kg/m3",
        "",
        f"Kappa range = {kappa_info['min']:.16e} "
        f"{kappa_info['max']:.16e}",
        f"Mu range = {mu_info['min']:.16e} "
        f"{mu_info['max']:.16e}",
        f"Density range = {density_info['min']:.16e} "
        f"{density_info['max']:.16e}",
        "",
        f"traces absent = {not (accepted_dir / 'traces').exists()}",
        f"res absent = {not (accepted_dir / 'res').exists()}",
        f"logs absent = {not (accepted_dir / 'logs').exists()}",
        "",
        "RESULT = PASS_FATHI80_BOOTSTRAP_WRITE",
    ]

    txt_path.write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )

    print(txt_path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
