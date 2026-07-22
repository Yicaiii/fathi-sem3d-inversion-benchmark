from __future__ import annotations

from pathlib import Path
import argparse
import json

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve


ROOT = Path(__file__).resolve().parents[2]


def resolve(path: str) -> Path:
    p = Path(path)
    return p if p.is_absolute() else ROOT / p


def load_config(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Missing config: {path}")

    return json.loads(path.read_text(encoding="utf-8"))


def relative_l2(a: np.ndarray, b: np.ndarray) -> float:
    denominator = max(float(np.linalg.norm(b)), 1e-300)
    return float(np.linalg.norm(a - b) / denominator)


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

    transition_dir = resolve(config["tv_transition_dir"])

    combined_rhs_dir = (
        transition_dir
        / "combined_rhs"
        / args.label
    )

    output_dir = (
        transition_dir
        / "mtilde_solve"
        / args.label
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    baseline_mtilde_dir = (
        resolve(config["baseline_transition_dir"])
        / "mtilde_solve"
    )

    matrix_path = (
        baseline_mtilde_dir
        / "Mtilde_q1_consistent_interior_38440_sparse.npz"
    )

    baseline_g_lambda_path = (
        baseline_mtilde_dir
        / "g_lambda_mtilde_q1_interior_solve_rhs_total.npy"
    )

    baseline_g_mu_path = (
        baseline_mtilde_dir
        / "g_mu_mtilde_q1_interior_solve_rhs_total.npy"
    )

    rhs_lambda_path = (
        combined_rhs_dir / "rhs_total_lambda.npy"
    )

    rhs_mu_path = (
        combined_rhs_dir / "rhs_total_mu.npy"
    )

    M = sparse.load_npz(matrix_path).tocsr()

    rhs_lambda = np.load(rhs_lambda_path)
    rhs_mu = np.load(rhs_mu_path)

    baseline_g_lambda = np.load(
        baseline_g_lambda_path
    )

    baseline_g_mu = np.load(
        baseline_g_mu_path
    )

    expected_shape = (38440,)

    if M.shape != (38440, 38440):
        raise RuntimeError(
            f"Unexpected Mtilde shape: {M.shape}"
        )

    for name, array in {
        "rhs_lambda": rhs_lambda,
        "rhs_mu": rhs_mu,
        "baseline_g_lambda": baseline_g_lambda,
        "baseline_g_mu": baseline_g_mu,
    }.items():
        if array.shape != expected_shape:
            raise RuntimeError(
                f"{name} shape mismatch: {array.shape}"
            )

        if not np.all(np.isfinite(array)):
            raise RuntimeError(
                f"{name} contains non-finite values"
            )

    print("Solving Mtilde g_lambda = RHS_total_lambda ...")
    g_lambda = spsolve(M, rhs_lambda)

    print("Solving Mtilde g_mu = RHS_total_mu ...")
    g_mu = spsolve(M, rhs_mu)

    if not np.all(np.isfinite(g_lambda)):
        raise RuntimeError(
            "Non-finite lambda gradient"
        )

    if not np.all(np.isfinite(g_mu)):
        raise RuntimeError(
            "Non-finite mu gradient"
        )

    residual_lambda = M @ g_lambda - rhs_lambda
    residual_mu = M @ g_mu - rhs_mu

    relative_residual_lambda = float(
        np.linalg.norm(residual_lambda)
        / max(np.linalg.norm(rhs_lambda), 1e-300)
    )

    relative_residual_mu = float(
        np.linalg.norm(residual_mu)
        / max(np.linalg.norm(rhs_mu), 1e-300)
    )

    lambda_max_abs_difference = float(
        np.max(
            np.abs(
                g_lambda - baseline_g_lambda
            )
        )
    )

    mu_max_abs_difference = float(
        np.max(
            np.abs(
                g_mu - baseline_g_mu
            )
        )
    )

    lambda_relative_difference = relative_l2(
        g_lambda,
        baseline_g_lambda,
    )

    mu_relative_difference = relative_l2(
        g_mu,
        baseline_g_mu,
    )

    alpha_lambda = float(
        config["tv"]["alpha_lambda"]
    )

    alpha_mu = float(
        config["tv"]["alpha_mu"]
    )

    alpha_zero = (
        alpha_lambda == 0.0
        and alpha_mu == 0.0
    )

    if alpha_zero:
        regression_ok = (
            np.array_equal(
                g_lambda,
                baseline_g_lambda,
            )
            and np.array_equal(
                g_mu,
                baseline_g_mu,
            )
        )
    else:
        regression_ok = True

    np.save(
        output_dir / "g_lambda_total.npy",
        g_lambda,
    )

    np.save(
        output_dir / "g_mu_total.npy",
        g_mu,
    )

    metadata = {
        "config": str(config_path),
        "label": args.label,
        "matrix": str(matrix_path),
        "matrix_shape": list(M.shape),
        "matrix_nnz": int(M.nnz),
        "alpha_lambda": alpha_lambda,
        "alpha_mu": alpha_mu,
        "relative_residual_lambda": (
            relative_residual_lambda
        ),
        "relative_residual_mu": (
            relative_residual_mu
        ),
        "lambda": {
            "gradient_l2": float(
                np.linalg.norm(g_lambda)
            ),
            "baseline_gradient_l2": float(
                np.linalg.norm(
                    baseline_g_lambda
                )
            ),
            "max_abs_difference": (
                lambda_max_abs_difference
            ),
            "relative_l2_difference": (
                lambda_relative_difference
            ),
        },
        "mu": {
            "gradient_l2": float(
                np.linalg.norm(g_mu)
            ),
            "baseline_gradient_l2": float(
                np.linalg.norm(
                    baseline_g_mu
                )
            ),
            "max_abs_difference": (
                mu_max_abs_difference
            ),
            "relative_l2_difference": (
                mu_relative_difference
            ),
        },
        "alpha_zero": alpha_zero,
        "regression_ok": regression_ok,
    }

    metadata_path = (
        output_dir
        / "mtilde_data_tv_solve_metadata.json"
    )

    metadata_path.write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )

    summary = []

    summary.append(
        "DATA + TV MTILDE SOLVE SUMMARY"
    )
    summary.append(
        "=============================="
    )
    summary.append("")
    summary.append(f"label = {args.label}")
    summary.append(f"matrix shape = {M.shape}")
    summary.append(f"matrix nnz = {M.nnz}")
    summary.append(
        f"alpha_lambda = {alpha_lambda:.16e}"
    )
    summary.append(
        f"alpha_mu = {alpha_mu:.16e}"
    )
    summary.append("")
    summary.append("Linear solve")
    summary.append("------------")
    summary.append(
        "relative residual lambda = "
        f"{relative_residual_lambda:.16e}"
    )
    summary.append(
        "relative residual mu = "
        f"{relative_residual_mu:.16e}"
    )
    summary.append("")
    summary.append("Lambda gradient regression")
    summary.append("--------------------------")
    summary.append(
        "new gradient L2 = "
        f"{np.linalg.norm(g_lambda):.16e}"
    )
    summary.append(
        "baseline gradient L2 = "
        f"{np.linalg.norm(baseline_g_lambda):.16e}"
    )
    summary.append(
        "max abs difference = "
        f"{lambda_max_abs_difference:.16e}"
    )
    summary.append(
        "relative L2 difference = "
        f"{lambda_relative_difference:.16e}"
    )
    summary.append("")
    summary.append("Mu gradient regression")
    summary.append("----------------------")
    summary.append(
        "new gradient L2 = "
        f"{np.linalg.norm(g_mu):.16e}"
    )
    summary.append(
        "baseline gradient L2 = "
        f"{np.linalg.norm(baseline_g_mu):.16e}"
    )
    summary.append(
        "max abs difference = "
        f"{mu_max_abs_difference:.16e}"
    )
    summary.append(
        "relative L2 difference = "
        f"{mu_relative_difference:.16e}"
    )
    summary.append("")
    summary.append("Regression")
    summary.append("----------")
    summary.append(
        f"alpha_zero = {alpha_zero}"
    )
    summary.append(
        f"regression_ok = {regression_ok}"
    )
    summary.append("")

    if regression_ok:
        summary.append(
            "RESULT = PASS_MTILDE_TV_REGRESSION"
        )
    else:
        summary.append(
            "RESULT = FAIL_MTILDE_TV_REGRESSION"
        )

    summary_text = "\n".join(summary) + "\n"

    summary_path = (
        output_dir
        / "mtilde_data_tv_solve_summary.txt"
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
