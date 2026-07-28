from __future__ import annotations

from pathlib import Path
import argparse
import json

import numpy as np


ROOT = Path(__file__).resolve().parents[2]


def resolve(path: str) -> Path:
    p = Path(path)
    return p if p.is_absolute() else ROOT / p


def load_config(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Missing config: {path}")

    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config",
        default=(
            "benchmark_fathi_tv/config/"
            "tv_config_iter008_to_iter009.json"
        ),
    )

    parser.add_argument(
        "--label",
        default="parent_iter008",
    )

    args = parser.parse_args()

    config_path = resolve(args.config)
    config = load_config(config_path)

    tv_root = (
        resolve(config["tv_transition_dir"])
        / "tv_full_grid"
        / args.label
    )

    output_dir = (
        resolve(config["tv_transition_dir"])
        / "tv_active"
        / args.label
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    lambda_full_path = (
        tv_root / "tv_rhs_lambda_hat_full.npy"
    )

    mu_full_path = (
        tv_root / "tv_rhs_mu_hat_full.npy"
    )

    active_indices_path = resolve(
        config["active_indices"]
    )

    active_coords_path = resolve(
        config["active_coords"]
    )

    lambda_hat_full = np.load(lambda_full_path)
    mu_hat_full = np.load(mu_full_path)

    active_indices = np.load(active_indices_path)
    active_coords = np.load(active_coords_path)

    nx = int(config["mesh"]["nx"])
    ny = int(config["mesh"]["ny"])
    nz = int(config["mesh"]["nz"])

    expected_shape = (nz, ny, nx)
    full_node_count = nx * ny * nz

    if lambda_hat_full.shape != expected_shape:
        raise RuntimeError(
            "Lambda full TV RHS shape mismatch: "
            f"expected {expected_shape}, "
            f"got {lambda_hat_full.shape}"
        )

    if mu_hat_full.shape != expected_shape:
        raise RuntimeError(
            "Mu full TV RHS shape mismatch: "
            f"expected {expected_shape}, "
            f"got {mu_hat_full.shape}"
        )

    if active_indices.ndim != 1:
        raise RuntimeError(
            f"Active indices must be 1D: "
            f"{active_indices.shape}"
        )

    if active_coords.shape != (
        active_indices.size,
        3,
    ):
        raise RuntimeError(
            "Active coordinates and indices mismatch: "
            f"{active_coords.shape}, "
            f"{active_indices.shape}"
        )

    if len(np.unique(active_indices)) != active_indices.size:
        raise RuntimeError(
            "Duplicate active indices detected"
        )

    if active_indices.min() < 0:
        raise RuntimeError(
            "Negative active index detected"
        )

    if active_indices.max() >= full_node_count:
        raise RuntimeError(
            "Active index exceeds full-grid size"
        )

    lambda_reference = float(
        config["parameter_scaling"][
            "lambda_reference_pa"
        ]
    )

    mu_reference = float(
        config["parameter_scaling"][
            "mu_reference_pa"
        ]
    )

    if lambda_reference <= 0.0:
        raise RuntimeError(
            "lambda_reference_pa must be positive"
        )

    if mu_reference <= 0.0:
        raise RuntimeError(
            "mu_reference_pa must be positive"
        )

    lambda_hat_flat = lambda_hat_full.ravel(
        order="C"
    )

    mu_hat_flat = mu_hat_full.ravel(
        order="C"
    )

    lambda_hat_active = lambda_hat_flat[
        active_indices
    ]

    mu_hat_active = mu_hat_flat[
        active_indices
    ]

    # Chain rule:
    #
    # lambda_hat = lambda / lambda_reference
    #
    # dR/dlambda =
    # (1/lambda_reference) dR/dlambda_hat
    lambda_physical_active = (
        lambda_hat_active / lambda_reference
    )

    mu_physical_active = (
        mu_hat_active / mu_reference
    )

    arrays = {
        "tv_rhs_lambda_hat_active.npy":
            lambda_hat_active,
        "tv_rhs_mu_hat_active.npy":
            mu_hat_active,
        "tv_rhs_lambda_physical_active.npy":
            lambda_physical_active,
        "tv_rhs_mu_physical_active.npy":
            mu_physical_active,
        "tv_active_indices.npy":
            active_indices,
        "tv_active_coords.npy":
            active_coords,
    }

    for filename, array in arrays.items():
        np.save(output_dir / filename, array)

    metadata = {
        "config": str(config_path),
        "label": args.label,
        "full_shape": list(expected_shape),
        "full_node_count": full_node_count,
        "active_node_count": int(
            active_indices.size
        ),
        "flatten_order": "C",
        "array_convention": "field[iz, iy, ix]",
        "fastest_axis": "x",
        "lambda_reference_pa": lambda_reference,
        "mu_reference_pa": mu_reference,
        "chain_rule": {
            "lambda": (
                "dR/dlambda = "
                "(1/lambda_reference_pa) "
                "dR/dlambda_hat"
            ),
            "mu": (
                "dR/dmu = "
                "(1/mu_reference_pa) "
                "dR/dmu_hat"
            ),
        },
        "lambda_hat_active_l2": float(
            np.linalg.norm(lambda_hat_active)
        ),
        "mu_hat_active_l2": float(
            np.linalg.norm(mu_hat_active)
        ),
        "lambda_physical_active_l2": float(
            np.linalg.norm(
                lambda_physical_active
            )
        ),
        "mu_physical_active_l2": float(
            np.linalg.norm(
                mu_physical_active
            )
        ),
        "lambda_hat_active_finite": int(
            np.count_nonzero(
                np.isfinite(lambda_hat_active)
            )
        ),
        "mu_hat_active_finite": int(
            np.count_nonzero(
                np.isfinite(mu_hat_active)
            )
        ),
    }

    metadata_path = (
        output_dir
        / "tv_active_metadata.json"
    )

    metadata_path.write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )

    summary = []

    summary.append(
        "TV ACTIVE-SPACE RESTRICTION SUMMARY"
    )
    summary.append(
        "==================================="
    )
    summary.append("")
    summary.append(
        f"label = {args.label}"
    )
    summary.append(
        f"full shape = {expected_shape}"
    )
    summary.append(
        f"full nodes = {full_node_count}"
    )
    summary.append(
        f"active nodes = {active_indices.size}"
    )
    summary.append(
        "flatten convention = "
        "field[iz, iy, ix], C-order, x fastest"
    )
    summary.append("")
    summary.append("Dimensionless TV dual vectors")
    summary.append("-----------------------------")
    summary.append(
        "lambda_hat active L2 = "
        f"{np.linalg.norm(lambda_hat_active):.16e}"
    )
    summary.append(
        "mu_hat active L2 = "
        f"{np.linalg.norm(mu_hat_active):.16e}"
    )
    summary.append("")
    summary.append(
        "Physical-parameter TV dual vectors"
    )
    summary.append(
        "----------------------------------"
    )
    summary.append(
        "lambda physical active L2 = "
        f"{np.linalg.norm(lambda_physical_active):.16e}"
    )
    summary.append(
        "mu physical active L2 = "
        f"{np.linalg.norm(mu_physical_active):.16e}"
    )
    summary.append("")
    summary.append("Chain rule")
    summary.append("----------")
    summary.append(
        "dR/dlambda = "
        "(1/lambda_reference_pa) "
        "dR/dlambda_hat"
    )
    summary.append(
        "dR/dmu = "
        "(1/mu_reference_pa) "
        "dR/dmu_hat"
    )
    summary.append("")
    summary.append("Checks")
    summary.append("------")
    summary.append(
        "active indices unique = "
        f"{len(np.unique(active_indices))}"
    )
    summary.append(
        "lambda finite = "
        f"{np.count_nonzero(np.isfinite(lambda_hat_active))}"
        f" / {active_indices.size}"
    )
    summary.append(
        "mu finite = "
        f"{np.count_nonzero(np.isfinite(mu_hat_active))}"
        f" / {active_indices.size}"
    )
    summary.append("")
    summary.append(
        "RESULT = PASS_TV_ACTIVE_RESTRICTION"
    )

    summary_text = "\n".join(summary) + "\n"

    summary_path = (
        output_dir
        / "tv_active_summary.txt"
    )

    summary_path.write_text(
        summary_text,
        encoding="utf-8",
    )

    print(summary_text)
    print(f"metadata = {metadata_path}")


if __name__ == "__main__":
    main()
