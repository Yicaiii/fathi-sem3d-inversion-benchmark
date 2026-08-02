from __future__ import annotations

from pathlib import Path
import argparse
import json

import numpy as np

from scripts.fathi_benchmark.runtime_paths import repository_root, resolve_path


ROOT = repository_root()


def one_match(directory: Path, patterns: list[str]) -> Path:
    for pattern in patterns:
        matches = sorted(directory.glob(pattern))
        if len(matches) == 1:
            return matches[0].resolve()
        if len(matches) > 1:
            raise RuntimeError(
                f"Ambiguous matches in {directory} for {pattern}: {matches}"
            )
    raise RuntimeError(f"No match in {directory} for {patterns}")


def configured_or_match(
    value: str | None,
    directory: Path,
    patterns: list[str],
) -> Path:
    if value:
        path = Path(value).expanduser().resolve()
        if path.exists():
            return path
    return one_match(directory, patterns)


def load_parent_fields(path: Path) -> tuple[np.ndarray, np.ndarray]:
    state = np.load(path, allow_pickle=True)
    mu = np.asarray(state["mu"], dtype=np.float64)
    if "lambda" in state.files:
        lam = np.asarray(state["lambda"], dtype=np.float64)
    elif "lambda_field" in state.files:
        lam = np.asarray(state["lambda_field"], dtype=np.float64)
    elif "kappa" in state.files:
        kappa = np.asarray(state["kappa"], dtype=np.float64)
        lam = kappa - (2.0 / 3.0) * mu
    else:
        raise KeyError(f"Cannot recover lambda. State keys: {state.files}")
    return lam, mu


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--context", required=True)
    parser.add_argument(
        "--strict-config",
        default="benchmark_fathi_strict/config/benchmark_config.json",
    )
    parser.add_argument("--alpha-lambda", type=float, default=0.0)
    parser.add_argument("--alpha-mu", type=float, default=0.0)
    parser.add_argument("--epsilon", type=float, default=1e-3)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    context_path = resolve_path(args.context, base=ROOT)
    context = json.loads(context_path.read_text(encoding="utf-8"))
    strict_path = resolve_path(args.strict_config, base=ROOT)
    strict = json.loads(strict_path.read_text(encoding="utf-8"))

    parent_state = Path(context["parent_state"]).resolve()
    transition_root = Path(context["transition_result_root"]).resolve()
    mtilde_dir = Path(context["mtilde_dir"]).resolve()
    rhs_dir = Path(context["component_rhs_dir"]).resolve()

    active_indices = one_match(
        mtilde_dir,
        ["*interior*indices.npy", "*active*indices.npy", "*indices.npy"],
    )
    active_coords = one_match(
        mtilde_dir,
        ["*interior*coords.npy", "*active*coords.npy", "*coords.npy"],
    )
    matrix = configured_or_match(
        context.get("mtilde_matrix_path"),
        mtilde_dir,
        ["*.npz"],
    )
    data_rhs_lambda = one_match(
        rhs_dir,
        ["*RHS_total_lambda.npy", "*rhs_total_lambda.npy"],
    )
    data_rhs_mu = one_match(
        rhs_dir,
        ["*RHS_total_mu.npy", "*rhs_total_mu.npy"],
    )
    baseline_gradient_lambda = one_match(
        mtilde_dir,
        [
            "g_lambda_mtilde_q1_interior_solve_rhs_total.npy",
            "g_lambda*rhs_total.npy",
            "g_lambda*.npy",
        ],
    )
    baseline_gradient_mu = one_match(
        mtilde_dir,
        [
            "g_mu_mtilde_q1_interior_solve_rhs_total.npy",
            "g_mu*rhs_total.npy",
            "g_mu*.npy",
        ],
    )

    lam, mu = load_parent_fields(parent_state)
    if lam.shape != mu.shape:
        raise RuntimeError(f"lambda/mu shape mismatch: {lam.shape}, {mu.shape}")
    shape = lam.shape
    lambda_reference = float(np.median(lam))
    mu_reference = float(np.median(mu))
    if lambda_reference <= 0 or mu_reference <= 0:
        raise RuntimeError("Median parent references must be positive.")

    domain = strict.get(
        "material_domain_m",
        {
            "x_min": -20.0,
            "x_max": 20.0,
            "y_min": -20.0,
            "y_max": 20.0,
            "z_min": -50.0,
            "z_max": 0.0,
        },
    )
    tv_root = Path(
        context.get(
            "tv_regularization_dir",
            transition_root / "tv_regularization",
        )
    ).resolve()

    objective_value = context.get("residual_summary_txt")
    objective_path = str(Path(objective_value).resolve()) if objective_value else None

    config = {
        "schema_version": 2,
        "transition": context["transition"],
        "iteration_context": str(context_path.resolve()),
        "parent_state": str(parent_state),
        "parent_objective_path": objective_path,
        "baseline_transition_dir": str(transition_root),
        "tv_transition_dir": str(tv_root),
        "full_mtilde": str(matrix),
        "mtilde_matrix": str(matrix),
        "active_indices": str(active_indices),
        "active_coords": str(active_coords),
        "data_rhs_lambda": str(data_rhs_lambda),
        "data_rhs_mu": str(data_rhs_mu),
        "baseline_gradient_lambda": str(baseline_gradient_lambda),
        "baseline_gradient_mu": str(baseline_gradient_mu),
        "mesh": {
            "nx": int(shape[2]),
            "ny": int(shape[1]),
            "nz": int(shape[0]),
            "array_order": "field[iz, iy, ix]",
            "flatten_order": "C",
            "fastest_axis": "x",
            "z_direction": "top_to_bottom",
            **domain,
        },
        "parameter_scaling": {
            "mode": "median_parent",
            "lambda_reference_pa": lambda_reference,
            "mu_reference_pa": mu_reference,
        },
        "tv": {
            "type": "smoothed_isotropic_q1",
            "epsilon_dimensionless": float(args.epsilon),
            "alpha_lambda": float(args.alpha_lambda),
            "alpha_mu": float(args.alpha_mu),
            "weighting_strategy": "explicit",
            "alpha_zero_relative_tolerance": 1e-12,
        },
    }

    output = (
        Path(args.output).expanduser().resolve()
        if args.output
        else tv_root / "tv_config.json"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(config, indent=2), encoding="utf-8")

    print(f"context = {context_path}")
    print(f"transition = {context['transition']}")
    print(f"active_count = {np.load(active_indices).size}")
    print(f"lambda_reference_pa = {lambda_reference:.16e}")
    print(f"mu_reference_pa = {mu_reference:.16e}")
    print(f"config = {output}")
    print("RESULT = PASS_TV_CONFIG_FROM_CONTEXT")


if __name__ == "__main__":
    main()
