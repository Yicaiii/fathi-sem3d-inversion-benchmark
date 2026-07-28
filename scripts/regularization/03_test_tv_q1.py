from __future__ import annotations

from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[2]

sys.path.insert(
    0,
    str(ROOT / "scripts/regularization"),
)

from importlib.util import module_from_spec, spec_from_file_location


tv_script = (
    ROOT
    / "scripts/regularization/"
      "02_compute_tv_q1_full_grid.py"
)

spec = spec_from_file_location(
    "tv_q1_module",
    tv_script,
)

if spec is None or spec.loader is None:
    raise RuntimeError(
        f"Cannot import TV module from {tv_script}"
    )

module = module_from_spec(spec)
spec.loader.exec_module(module)

compute_smoothed_tv_q1 = module.compute_smoothed_tv_q1


def directional_test(
    field: np.ndarray,
    direction: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    epsilon: float,
) -> list[dict]:
    value, rhs, _ = compute_smoothed_tv_q1(
        field,
        x,
        y,
        z,
        epsilon,
    )

    analytical = float(
        np.sum(rhs * direction)
    )

    results = []

    for delta in [
        1e-2,
        3e-3,
        1e-3,
        3e-4,
        1e-4,
        3e-5,
        1e-5,
    ]:
        plus, _, _ = compute_smoothed_tv_q1(
            field + delta * direction,
            x,
            y,
            z,
            epsilon,
        )

        minus, _, _ = compute_smoothed_tv_q1(
            field - delta * direction,
            x,
            y,
            z,
            epsilon,
        )

        finite_difference = (
            plus - minus
        ) / (2.0 * delta)

        relative_error = abs(
            finite_difference - analytical
        ) / max(
            abs(finite_difference),
            abs(analytical),
            1e-14,
        )

        results.append(
            {
                "delta": delta,
                "finite_difference": finite_difference,
                "analytical": analytical,
                "relative_error": relative_error,
            }
        )

    return results


def main() -> None:
    nx = 5
    ny = 5
    nz = 5

    x = np.linspace(-1.0, 1.0, nx)
    y = np.linspace(-1.0, 1.0, ny)
    z = np.linspace(0.0, -2.0, nz)

    epsilon = 1e-3

    zz, yy, xx = np.meshgrid(
        z,
        y,
        x,
        indexing="ij",
    )

    constant = np.ones((nz, ny, nx))

    constant_value, constant_rhs, constant_stats = (
        compute_smoothed_tv_q1(
            constant,
            x,
            y,
            z,
            epsilon,
        )
    )

    domain_volume = (
        abs(x[-1] - x[0])
        * abs(y[-1] - y[0])
        * abs(z[-1] - z[0])
    )

    expected_constant_value = (
        epsilon * domain_volume
    )

    constant_value_error = abs(
        constant_value - expected_constant_value
    )

    constant_rhs_norm = float(
        np.linalg.norm(constant_rhs.ravel())
    )

    linear = (
        1.0
        + 0.2 * xx
        - 0.1 * yy
        + 0.05 * zz
    )

    rng = np.random.default_rng(42)

    direction = rng.normal(
        size=linear.shape,
    )

    direction /= np.linalg.norm(
        direction.ravel()
    )

    derivative_results = directional_test(
        linear,
        direction,
        x,
        y,
        z,
        epsilon,
    )

    best_relative_error = min(
        item["relative_error"]
        for item in derivative_results
    )

    layered = np.where(
        zz >= -1.0,
        0.8,
        1.2,
    )

    layered_value, layered_rhs, layered_stats = (
        compute_smoothed_tv_q1(
            layered,
            x,
            y,
            z,
            epsilon,
        )
    )

    print("Q1 TV UNIT TESTS")
    print("================")
    print()
    print("Test 1: constant field")
    print(
        "computed value =",
        f"{constant_value:.16e}",
    )
    print(
        "expected value =",
        f"{expected_constant_value:.16e}",
    )
    print(
        "value abs error =",
        f"{constant_value_error:.16e}",
    )
    print(
        "TV RHS L2 =",
        f"{constant_rhs_norm:.16e}",
    )
    print()
    print("Test 2: directional derivative")

    for item in derivative_results:
        print(
            "delta =",
            f"{item['delta']:.1e}",
            "finite_difference =",
            f"{item['finite_difference']:.16e}",
            "analytical =",
            f"{item['analytical']:.16e}",
            "relative_error =",
            f"{item['relative_error']:.16e}",
        )

    print()
    print(
        "best relative error =",
        f"{best_relative_error:.16e}",
    )
    print()
    print("Test 3: layered field")
    print(
        "TV value =",
        f"{layered_value:.16e}",
    )
    print(
        "TV RHS L2 =",
        f"{np.linalg.norm(layered_rhs.ravel()):.16e}",
    )
    print(
        "max gradient norm =",
        f"{layered_stats['max_gradient_norm']:.16e}",
    )
    print()

    constant_ok = (
        constant_value_error < 1e-10
        and constant_rhs_norm < 1e-10
    )

    derivative_ok = best_relative_error < 1e-6

    layered_ok = (
        np.isfinite(layered_value)
        and layered_value > expected_constant_value
        and np.all(np.isfinite(layered_rhs))
    )

    print("constant_ok =", constant_ok)
    print("derivative_ok =", derivative_ok)
    print("layered_ok =", layered_ok)
    print()

    if constant_ok and derivative_ok and layered_ok:
        print("RESULT = PASS_TV_UNIT_TESTS")
    else:
        print("RESULT = FAIL_TV_UNIT_TESTS")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
