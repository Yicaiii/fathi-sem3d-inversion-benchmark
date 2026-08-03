from __future__ import annotations

from pathlib import Path
import argparse
import json

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve


ROOT = Path(__file__).resolve().parents[2]


def resolve(path: str) -> Path:
    value = Path(path).expanduser()
    return value.resolve() if value.is_absolute() else (ROOT / value).resolve()


def load_config(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Missing config: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def relative_l2(a: np.ndarray, b: np.ndarray) -> float:
    denominator = max(float(np.linalg.norm(b)), 1e-300)
    return float(np.linalg.norm(a - b) / denominator)


def required_path(config: dict, *keys: str) -> Path:
    for key in keys:
        value = config.get(key)
        if value:
            path = resolve(value)
            if path.exists():
                return path
    raise FileNotFoundError(f"None of the configured paths exist: {keys}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--label",
        default=None,
        help="Output branch label. Default: transition from config.",
    )
    args = parser.parse_args()

    config_path = resolve(args.config)
    config = load_config(config_path)
    label = args.label or config.get("transition", "tv_parent")
    transition_dir = resolve(config["tv_transition_dir"])

    combined_rhs_dir = transition_dir / "combined_rhs" / label
    output_dir = transition_dir / "mtilde_solve" / label
    output_dir.mkdir(parents=True, exist_ok=True)

    matrix_path = required_path(config, "mtilde_matrix", "full_mtilde")
    baseline_g_lambda_path = required_path(config, "baseline_gradient_lambda")
    baseline_g_mu_path = required_path(config, "baseline_gradient_mu")

    rhs_lambda_path = combined_rhs_dir / "rhs_total_lambda.npy"
    rhs_mu_path = combined_rhs_dir / "rhs_total_mu.npy"

    matrix = sparse.load_npz(matrix_path).tocsr()
    rhs_lambda = np.asarray(np.load(rhs_lambda_path), dtype=np.float64)
    rhs_mu = np.asarray(np.load(rhs_mu_path), dtype=np.float64)
    baseline_g_lambda = np.asarray(np.load(baseline_g_lambda_path), dtype=np.float64)
    baseline_g_mu = np.asarray(np.load(baseline_g_mu_path), dtype=np.float64)

    if rhs_lambda.ndim != 1 or rhs_mu.ndim != 1:
        raise RuntimeError(
            "Total RHS arrays must be one-dimensional: "
            f"lambda={rhs_lambda.shape}, mu={rhs_mu.shape}"
        )
    if rhs_lambda.shape != rhs_mu.shape:
        raise RuntimeError(
            "Lambda/mu RHS shape mismatch: "
            f"{rhs_lambda.shape}, {rhs_mu.shape}"
        )

    expected_shape = rhs_lambda.shape
    n_active = int(rhs_lambda.size)
    if matrix.shape != (n_active, n_active):
        raise RuntimeError(
            "Mtilde/RHS dimension mismatch: "
            f"M={matrix.shape}, RHS={expected_shape}"
        )

    arrays = {
        "rhs_lambda": rhs_lambda,
        "rhs_mu": rhs_mu,
        "baseline_g_lambda": baseline_g_lambda,
        "baseline_g_mu": baseline_g_mu,
    }
    for name, array in arrays.items():
        if array.shape != expected_shape:
            raise RuntimeError(
                f"{name} shape mismatch: expected={expected_shape}, got={array.shape}"
            )
        if not np.all(np.isfinite(array)):
            raise RuntimeError(f"{name} contains non-finite values")

    print("Solving Mtilde g_lambda = RHS_total_lambda ...")
    g_lambda = np.asarray(spsolve(matrix, rhs_lambda), dtype=np.float64)
    print("Solving Mtilde g_mu = RHS_total_mu ...")
    g_mu = np.asarray(spsolve(matrix, rhs_mu), dtype=np.float64)

    if not np.all(np.isfinite(g_lambda)):
        raise RuntimeError("Non-finite lambda gradient")
    if not np.all(np.isfinite(g_mu)):
        raise RuntimeError("Non-finite mu gradient")

    residual_lambda = matrix @ g_lambda - rhs_lambda
    residual_mu = matrix @ g_mu - rhs_mu
    relative_residual_lambda = float(
        np.linalg.norm(residual_lambda)
        / max(float(np.linalg.norm(rhs_lambda)), 1e-300)
    )
    relative_residual_mu = float(
        np.linalg.norm(residual_mu)
        / max(float(np.linalg.norm(rhs_mu)), 1e-300)
    )

    lambda_relative_difference = relative_l2(g_lambda, baseline_g_lambda)
    mu_relative_difference = relative_l2(g_mu, baseline_g_mu)
    lambda_max_abs_difference = float(np.max(np.abs(g_lambda - baseline_g_lambda)))
    mu_max_abs_difference = float(np.max(np.abs(g_mu - baseline_g_mu)))

    alpha_lambda = float(config["tv"]["alpha_lambda"])
    alpha_mu = float(config["tv"]["alpha_mu"])
    alpha_zero = alpha_lambda == 0.0 and alpha_mu == 0.0
    tolerance = float(config["tv"].get("alpha_zero_relative_tolerance", 1e-12))
    regression_ok = (
        not alpha_zero
        or (
            lambda_relative_difference <= tolerance
            and mu_relative_difference <= tolerance
        )
    )

    np.save(output_dir / "g_lambda_total.npy", g_lambda)
    np.save(output_dir / "g_mu_total.npy", g_mu)

    metadata = {
        "config": str(config_path),
        "label": label,
        "matrix": str(matrix_path),
        "matrix_shape": list(matrix.shape),
        "matrix_nnz": int(matrix.nnz),
        "active_count": n_active,
        "alpha_lambda": alpha_lambda,
        "alpha_mu": alpha_mu,
        "alpha_zero": alpha_zero,
        "alpha_zero_relative_tolerance": tolerance,
        "alpha_zero_regression_ok": regression_ok,
        "relative_residual_lambda": relative_residual_lambda,
        "relative_residual_mu": relative_residual_mu,
        "lambda_max_abs_difference_from_data": lambda_max_abs_difference,
        "mu_max_abs_difference_from_data": mu_max_abs_difference,
        "lambda_relative_difference_from_data": lambda_relative_difference,
        "mu_relative_difference_from_data": mu_relative_difference,
        "baseline_gradient_lambda": str(baseline_g_lambda_path),
        "baseline_gradient_mu": str(baseline_g_mu_path),
        "g_lambda_total": str(output_dir / "g_lambda_total.npy"),
        "g_mu_total": str(output_dir / "g_mu_total.npy"),
    }
    metadata_path = output_dir / "mtilde_data_tv_solve_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print(f"label = {label}")
    print(f"active_count = {n_active}")
    print(f"relative_residual_lambda = {relative_residual_lambda:.16e}")
    print(f"relative_residual_mu = {relative_residual_mu:.16e}")
    print(
        "lambda_relative_difference_from_data = "
        f"{lambda_relative_difference:.16e}"
    )
    print(
        "mu_relative_difference_from_data = "
        f"{mu_relative_difference:.16e}"
    )
    print(f"metadata = {metadata_path}")

    if alpha_zero and not regression_ok:
        print("RESULT = FAIL_ALPHA_ZERO_REGRESSION")
        raise SystemExit(2)

    print("RESULT = PASS_MTILDE_DATA_TV_SOLVE")


if __name__ == "__main__":
    main()
