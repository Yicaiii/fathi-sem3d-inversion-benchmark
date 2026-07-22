from __future__ import annotations

from pathlib import Path
import argparse
import json

import numpy as np


ROOT = Path(__file__).resolve().parents[2]


def resolve(path: str) -> Path:
    p = Path(path)
    return p if p.is_absolute() else ROOT / p


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    denominator = (
        np.linalg.norm(a)
        * np.linalg.norm(b)
    )

    if denominator == 0.0:
        return float("nan")

    return float(np.dot(a, b) / denominator)


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
        "--j-data",
        type=float,
        default=3.8263972312e-19,
    )

    args = parser.parse_args()

    config_path = resolve(args.config)

    config = json.loads(
        config_path.read_text(encoding="utf-8")
    )

    transition_dir = resolve(
        config["tv_transition_dir"]
    )

    tv_values_path = (
        transition_dir
        / "tv_full_grid"
        / "parent_iter008"
        / "tv_values_and_stats.json"
    )

    combined_dir = (
        transition_dir
        / "combined_rhs"
        / "parent_iter008"
    )

    solve_dir = (
        transition_dir
        / "mtilde_solve"
        / "parent_iter008"
    )

    alpha_lambda = float(
        config["tv"]["alpha_lambda"]
    )

    alpha_mu = float(
        config["tv"]["alpha_mu"]
    )

    solve_metadata_path = (
        solve_dir
        / "mtilde_data_tv_solve_metadata.json"
    )

    if not solve_metadata_path.exists():
        raise FileNotFoundError(
            f"Missing solve metadata: {solve_metadata_path}"
        )

    solve_metadata = json.loads(
        solve_metadata_path.read_text(
            encoding="utf-8"
        )
    )

    solve_alpha_lambda = float(
        solve_metadata["alpha_lambda"]
    )

    solve_alpha_mu = float(
        solve_metadata["alpha_mu"]
    )

    if not np.isclose(
        solve_alpha_lambda,
        alpha_lambda,
        rtol=1e-12,
        atol=0.0,
    ):
        raise RuntimeError(
            "Stale lambda gradient detected: "
            f"config alpha={alpha_lambda:.16e}, "
            f"solve alpha={solve_alpha_lambda:.16e}. "
            "Rerun scripts 05 and 06 before diagnosis."
        )

    if not np.isclose(
        solve_alpha_mu,
        alpha_mu,
        rtol=1e-12,
        atol=0.0,
    ):
        raise RuntimeError(
            "Stale mu gradient detected: "
            f"config alpha={alpha_mu:.16e}, "
            f"solve alpha={solve_alpha_mu:.16e}. "
            "Rerun scripts 05 and 06 before diagnosis."
        )

    tv_values = json.loads(
        tv_values_path.read_text(encoding="utf-8")
    )


    r_data_lambda = np.load(
        combined_dir / "rhs_data_lambda.npy"
    )

    r_data_mu = np.load(
        combined_dir / "rhs_data_mu.npy"
    )

    r_tv_lambda = np.load(
        combined_dir
        / "rhs_tv_lambda_physical.npy"
    )

    r_tv_mu = np.load(
        combined_dir
        / "rhs_tv_mu_physical.npy"
    )

    g_total_lambda = np.load(
        solve_dir / "g_lambda_total.npy"
    )

    g_total_mu = np.load(
        solve_dir / "g_mu_total.npy"
    )

    baseline_dir = (
        resolve(config["baseline_transition_dir"])
        / "mtilde_solve"
    )

    g_data_lambda = np.load(
        baseline_dir
        / "g_lambda_mtilde_q1_interior_solve_rhs_total.npy"
    )

    g_data_mu = np.load(
        baseline_dir
        / "g_mu_mtilde_q1_interior_solve_rhs_total.npy"
    )

    g_tv_weighted_lambda = (
        g_total_lambda - g_data_lambda
    )

    g_tv_weighted_mu = (
        g_total_mu - g_data_mu
    )

    tv_lambda = float(
        tv_values["tv_lambda_hat"]
    )

    tv_mu = float(
        tv_values["tv_mu_hat"]
    )

    j_reg_lambda = (
        alpha_lambda * tv_lambda
    )

    j_reg_mu = (
        alpha_mu * tv_mu
    )

    j_reg_total = (
        j_reg_lambda + j_reg_mu
    )

    print("TV WEIGHT DIAGNOSTIC")
    print("====================")
    print()

    print("Objective scale")
    print("---------------")
    print(
        f"J_data = {args.j_data:.16e}"
    )
    print(
        f"alpha_lambda * R_TV(lambda) = "
        f"{j_reg_lambda:.16e}"
    )
    print(
        f"alpha_mu * R_TV(mu) = "
        f"{j_reg_mu:.16e}"
    )
    print(
        f"J_reg_total = {j_reg_total:.16e}"
    )
    print(
        f"J_reg_total / J_data = "
        f"{j_reg_total / args.j_data:.16e}"
    )
    print()

    print("RHS geometry")
    print("------------")
    print(
        "cos(data_lambda, TV_lambda) = "
        f"{cosine(r_data_lambda, r_tv_lambda):.16e}"
    )
    print(
        "cos(data_mu, TV_mu) = "
        f"{cosine(r_data_mu, r_tv_mu):.16e}"
    )
    print()

    print("Gradient geometry")
    print("-----------------")
    print(
        "cos(g_data_lambda, weighted_g_tv_lambda) = "
        f"{cosine(g_data_lambda, g_tv_weighted_lambda):.16e}"
    )
    print(
        "cos(g_data_mu, weighted_g_tv_mu) = "
        f"{cosine(g_data_mu, g_tv_weighted_mu):.16e}"
    )
    print()

    print(
        "relative weighted TV gradient lambda = "
        f"{np.linalg.norm(g_tv_weighted_lambda) / np.linalg.norm(g_data_lambda):.16e}"
    )
    print(
        "relative weighted TV gradient mu = "
        f"{np.linalg.norm(g_tv_weighted_mu) / np.linalg.norm(g_data_mu):.16e}"
    )
    print()

    print("RESULT = PASS_TV_WEIGHT_DIAGNOSTIC")


if __name__ == "__main__":
    main()
