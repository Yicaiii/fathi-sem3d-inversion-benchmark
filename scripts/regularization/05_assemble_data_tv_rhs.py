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
        raise FileNotFoundError(
            f"Missing config: {path}"
        )

    return json.loads(
        path.read_text(encoding="utf-8")
    )


def relative_l2(a: np.ndarray, b: np.ndarray) -> float:
    denominator = max(
        float(np.linalg.norm(b)),
        1e-300,
    )

    return float(
        np.linalg.norm(a - b) / denominator
    )


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

    tv_active_dir = (
        resolve(config["tv_transition_dir"])
        / "tv_active"
        / args.label
    )

    output_dir = (
        resolve(config["tv_transition_dir"])
        / "combined_rhs"
        / args.label
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    data_lambda_path = resolve(
        config["data_rhs_lambda"]
    )

    data_mu_path = resolve(
        config["data_rhs_mu"]
    )

    tv_lambda_path = (
        tv_active_dir
        / "tv_rhs_lambda_physical_active.npy"
    )

    tv_mu_path = (
        tv_active_dir
        / "tv_rhs_mu_physical_active.npy"
    )

    data_lambda = np.load(data_lambda_path)
    data_mu = np.load(data_mu_path)

    tv_lambda = np.load(tv_lambda_path)
    tv_mu = np.load(tv_mu_path)

    if data_lambda.shape != (38440,):
        raise RuntimeError(
            "Unexpected data lambda RHS shape: "
            f"{data_lambda.shape}"
        )

    if data_mu.shape != (38440,):
        raise RuntimeError(
            "Unexpected data mu RHS shape: "
            f"{data_mu.shape}"
        )

    if tv_lambda.shape != data_lambda.shape:
        raise RuntimeError(
            "TV/data lambda shape mismatch: "
            f"{tv_lambda.shape}, "
            f"{data_lambda.shape}"
        )

    if tv_mu.shape != data_mu.shape:
        raise RuntimeError(
            "TV/data mu shape mismatch: "
            f"{tv_mu.shape}, "
            f"{data_mu.shape}"
        )

    for name, array in {
        "data_lambda": data_lambda,
        "data_mu": data_mu,
        "tv_lambda": tv_lambda,
        "tv_mu": tv_mu,
    }.items():
        if not np.all(np.isfinite(array)):
            raise RuntimeError(
                f"Non-finite values in {name}"
            )

    alpha_lambda = float(
        config["tv"]["alpha_lambda"]
    )

    alpha_mu = float(
        config["tv"]["alpha_mu"]
    )

    weighted_tv_lambda = (
        alpha_lambda * tv_lambda
    )

    weighted_tv_mu = (
        alpha_mu * tv_mu
    )

    total_lambda = (
        data_lambda + weighted_tv_lambda
    )

    total_mu = (
        data_mu + weighted_tv_mu
    )

    np.save(
        output_dir / "rhs_data_lambda.npy",
        data_lambda,
    )

    np.save(
        output_dir / "rhs_data_mu.npy",
        data_mu,
    )

    np.save(
        output_dir
        / "rhs_tv_lambda_physical.npy",
        tv_lambda,
    )

    np.save(
        output_dir
        / "rhs_tv_mu_physical.npy",
        tv_mu,
    )

    np.save(
        output_dir
        / "rhs_tv_lambda_weighted.npy",
        weighted_tv_lambda,
    )

    np.save(
        output_dir
        / "rhs_tv_mu_weighted.npy",
        weighted_tv_mu,
    )

    np.save(
        output_dir / "rhs_total_lambda.npy",
        total_lambda,
    )

    np.save(
        output_dir / "rhs_total_mu.npy",
        total_mu,
    )

    lambda_max_difference = float(
        np.max(
            np.abs(total_lambda - data_lambda)
        )
    )

    mu_max_difference = float(
        np.max(
            np.abs(total_mu - data_mu)
        )
    )

    lambda_relative_difference = relative_l2(
        total_lambda,
        data_lambda,
    )

    mu_relative_difference = relative_l2(
        total_mu,
        data_mu,
    )

    lambda_data_norm = float(
        np.linalg.norm(data_lambda)
    )

    mu_data_norm = float(
        np.linalg.norm(data_mu)
    )

    lambda_tv_norm = float(
        np.linalg.norm(tv_lambda)
    )

    mu_tv_norm = float(
        np.linalg.norm(tv_mu)
    )

    lambda_weighted_tv_norm = float(
        np.linalg.norm(weighted_tv_lambda)
    )

    mu_weighted_tv_norm = float(
        np.linalg.norm(weighted_tv_mu)
    )

    metadata = {
        "config": str(config_path),
        "label": args.label,
        "alpha_lambda": alpha_lambda,
        "alpha_mu": alpha_mu,
        "lambda": {
            "data_rhs_l2": lambda_data_norm,
            "physical_tv_rhs_l2": lambda_tv_norm,
            "weighted_tv_rhs_l2": (
                lambda_weighted_tv_norm
            ),
            "weighted_tv_to_data_ratio": (
                lambda_weighted_tv_norm
                / max(lambda_data_norm, 1e-300)
            ),
            "total_vs_data_max_abs": (
                lambda_max_difference
            ),
            "total_vs_data_relative_l2": (
                lambda_relative_difference
            ),
        },
        "mu": {
            "data_rhs_l2": mu_data_norm,
            "physical_tv_rhs_l2": mu_tv_norm,
            "weighted_tv_rhs_l2": (
                mu_weighted_tv_norm
            ),
            "weighted_tv_to_data_ratio": (
                mu_weighted_tv_norm
                / max(mu_data_norm, 1e-300)
            ),
            "total_vs_data_max_abs": (
                mu_max_difference
            ),
            "total_vs_data_relative_l2": (
                mu_relative_difference
            ),
        },
    }

    metadata_path = (
        output_dir
        / "combined_rhs_metadata.json"
    )

    metadata_path.write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )

    alpha_zero = (
        alpha_lambda == 0.0
        and alpha_mu == 0.0
    )

    if alpha_zero:
        regression_ok = (
            np.array_equal(total_lambda, data_lambda)
            and np.array_equal(total_mu, data_mu)
        )
    else:
        regression_ok = True

    summary = []

    summary.append(
        "DATA + TV RHS ASSEMBLY SUMMARY"
    )
    summary.append(
        "=============================="
    )
    summary.append("")
    summary.append(
        f"label = {args.label}"
    )
    summary.append(
        f"alpha_lambda = {alpha_lambda:.16e}"
    )
    summary.append(
        f"alpha_mu = {alpha_mu:.16e}"
    )
    summary.append("")
    summary.append("Lambda")
    summary.append("------")
    summary.append(
        f"data RHS L2 = {lambda_data_norm:.16e}"
    )
    summary.append(
        f"physical TV RHS L2 = "
        f"{lambda_tv_norm:.16e}"
    )
    summary.append(
        f"weighted TV RHS L2 = "
        f"{lambda_weighted_tv_norm:.16e}"
    )
    summary.append(
        "weighted TV / data ratio = "
        f"{lambda_weighted_tv_norm / max(lambda_data_norm, 1e-300):.16e}"
    )
    summary.append(
        "total vs data max abs = "
        f"{lambda_max_difference:.16e}"
    )
    summary.append(
        "total vs data relative L2 = "
        f"{lambda_relative_difference:.16e}"
    )
    summary.append("")
    summary.append("Mu")
    summary.append("--")
    summary.append(
        f"data RHS L2 = {mu_data_norm:.16e}"
    )
    summary.append(
        f"physical TV RHS L2 = "
        f"{mu_tv_norm:.16e}"
    )
    summary.append(
        f"weighted TV RHS L2 = "
        f"{mu_weighted_tv_norm:.16e}"
    )
    summary.append(
        "weighted TV / data ratio = "
        f"{mu_weighted_tv_norm / max(mu_data_norm, 1e-300):.16e}"
    )
    summary.append(
        "total vs data max abs = "
        f"{mu_max_difference:.16e}"
    )
    summary.append(
        "total vs data relative L2 = "
        f"{mu_relative_difference:.16e}"
    )
    summary.append("")
    summary.append("Regression check")
    summary.append("----------------")
    summary.append(
        f"alpha_zero = {alpha_zero}"
    )
    summary.append(
        f"regression_ok = {regression_ok}"
    )
    summary.append("")

    if regression_ok:
        summary.append(
            "RESULT = PASS_COMBINED_RHS"
        )
    else:
        summary.append(
            "RESULT = FAIL_ALPHA_ZERO_REGRESSION"
        )

    summary_text = "\n".join(summary) + "\n"

    summary_path = (
        output_dir
        / "combined_rhs_summary.txt"
    )

    summary_path.write_text(
        summary_text,
        encoding="utf-8",
    )

    print(summary_text)
    print(f"metadata = {metadata_path}")

    if not regression_ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
