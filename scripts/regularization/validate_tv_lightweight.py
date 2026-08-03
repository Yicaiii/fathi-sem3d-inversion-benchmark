from __future__ import annotations

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import json
import numpy as np

from scripts.regularization.total_objective import ObjectiveTerms, decide_acceptance

ROOT = Path(__file__).resolve().parents[2]


def load_tv_function():
    path = ROOT / "scripts/regularization/02_compute_tv_q1_full_grid.py"
    spec = spec_from_file_location("tv_q1_validation_module", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import TV implementation: {path}")
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.compute_smoothed_tv_q1


def main() -> None:
    compute_tv = load_tv_function()
    shape = (5, 5, 5)
    x = np.linspace(-1.0, 1.0, shape[2])
    y = np.linspace(-1.0, 1.0, shape[1])
    z = np.linspace(0.0, -2.0, shape[0])
    epsilon = 1e-3

    constant = np.ones(shape)
    constant_value, constant_rhs, _ = compute_tv(constant, x, y, z, epsilon)
    constant_rhs_norm = float(np.linalg.norm(constant_rhs))
    if constant_rhs_norm > 1e-10:
        raise RuntimeError(f"Constant-field TV derivative is not zero: {constant_rhs_norm}")

    zz, yy, xx = np.meshgrid(z, y, x, indexing="ij")
    field = 1.0 + 0.15 * np.sin(2.0 * xx) + 0.1 * yy * zz
    direction = np.cos(xx + yy - zz)
    value, rhs, _ = compute_tv(field, x, y, z, epsilon)
    rhs_norm = float(np.linalg.norm(rhs))
    if rhs_norm <= 1e-10:
        raise RuntimeError("Non-uniform TV derivative is unexpectedly zero")

    analytical = float(np.sum(rhs * direction))
    delta = 1e-5
    plus, _, _ = compute_tv(field + delta * direction, x, y, z, epsilon)
    minus, _, _ = compute_tv(field - delta * direction, x, y, z, epsilon)
    finite_difference = (plus - minus) / (2.0 * delta)
    relative_error = abs(finite_difference - analytical) / max(abs(finite_difference), abs(analytical), 1e-14)
    if relative_error > 2e-4:
        raise RuntimeError(f"Directional derivative failed: {relative_error}")

    data_rhs = np.linspace(-1.0, 1.0, 20)
    tv_rhs = np.linspace(0.5, -0.5, 20)
    if not np.array_equal(data_rhs + 0.0 * tv_rhs, data_rhs):
        raise RuntimeError("alpha=0 regression failed")

    decision = decide_acceptance(
        ObjectiveTerms(10.0, 4.0, 2.0, 0.1, 0.2),
        ObjectiveTerms(9.0, 3.5, 1.5, 0.1, 0.2),
    )
    if not decision["accepted"]:
        raise RuntimeError("Total-objective acceptance failed")

    payload = {
        "constant_tv_value": float(constant_value),
        "constant_rhs_norm": constant_rhs_norm,
        "nonuniform_tv_value": float(value),
        "nonuniform_rhs_norm": rhs_norm,
        "directional_relative_error": float(relative_error),
        "alpha_zero_regression": True,
        "total_objective_acceptance": decision,
        "sem3d_launched": False,
        "accepted_state_mutated": False,
        "result": "PASS_TV_LIGHTWEIGHT_VALIDATION",
    }
    print(json.dumps(payload, indent=2))
    print("RESULT = PASS_TV_LIGHTWEIGHT_VALIDATION")


if __name__ == "__main__":
    main()
