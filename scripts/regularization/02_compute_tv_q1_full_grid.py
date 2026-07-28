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


def load_material_fields(
    state_path: Path,
) -> tuple[np.ndarray, np.ndarray]:
    state = np.load(state_path, allow_pickle=True)

    if "mu" not in state.files:
        raise KeyError(
            f"'mu' missing in {state_path}; keys={state.files}"
        )

    mu = np.asarray(state["mu"], dtype=np.float64)

    if "lambda" in state.files:
        lam = np.asarray(state["lambda"], dtype=np.float64)
    elif "lambda_field" in state.files:
        lam = np.asarray(
            state["lambda_field"],
            dtype=np.float64,
        )
    elif "kappa" in state.files:
        kappa = np.asarray(state["kappa"], dtype=np.float64)
        lam = kappa - (2.0 / 3.0) * mu
    else:
        raise KeyError(
            "Cannot recover lambda from parent state. "
            f"Available keys: {state.files}"
        )

    return lam, mu


def gauss_rule_2() -> tuple[np.ndarray, np.ndarray]:
    point = 1.0 / np.sqrt(3.0)

    points = np.array(
        [-point, point],
        dtype=np.float64,
    )

    weights = np.ones(2, dtype=np.float64)

    return points, weights


def q1_shape_gradients_reference(
    xi: float,
    eta: float,
    zeta: float,
) -> np.ndarray:
    """
    Reference Q1 hexahedron node ordering:

      local node 0: (-1, -1, -1)
      local node 1: (+1, -1, -1)
      local node 2: (-1, +1, -1)
      local node 3: (+1, +1, -1)
      local node 4: (-1, -1, +1)
      local node 5: (+1, -1, +1)
      local node 6: (-1, +1, +1)
      local node 7: (+1, +1, +1)

    The associated material-array ordering is:

      field[iz, iy, ix]

    with x fastest in C-order.
    """

    signs = np.array(
        [
            [-1.0, -1.0, -1.0],
            [+1.0, -1.0, -1.0],
            [-1.0, +1.0, -1.0],
            [+1.0, +1.0, -1.0],
            [-1.0, -1.0, +1.0],
            [+1.0, -1.0, +1.0],
            [-1.0, +1.0, +1.0],
            [+1.0, +1.0, +1.0],
        ],
        dtype=np.float64,
    )

    sx = signs[:, 0]
    sy = signs[:, 1]
    sz = signs[:, 2]

    d_dxi = (
        0.125
        * sx
        * (1.0 + sy * eta)
        * (1.0 + sz * zeta)
    )

    d_deta = (
        0.125
        * sy
        * (1.0 + sx * xi)
        * (1.0 + sz * zeta)
    )

    d_dzeta = (
        0.125
        * sz
        * (1.0 + sx * xi)
        * (1.0 + sy * eta)
    )

    return np.column_stack(
        [d_dxi, d_deta, d_dzeta]
    )


def local_values(
    field: np.ndarray,
    iz: int,
    iy: int,
    ix: int,
) -> np.ndarray:
    """
    Return the eight Q1 nodal values of one structured cell.

    Local ordering matches q1_shape_gradients_reference().
    """

    return np.array(
        [
            field[iz,     iy,     ix],
            field[iz,     iy,     ix + 1],
            field[iz,     iy + 1, ix],
            field[iz,     iy + 1, ix + 1],
            field[iz + 1, iy,     ix],
            field[iz + 1, iy,     ix + 1],
            field[iz + 1, iy + 1, ix],
            field[iz + 1, iy + 1, ix + 1],
        ],
        dtype=np.float64,
    )


def add_local_vector(
    global_vector: np.ndarray,
    local_vector: np.ndarray,
    iz: int,
    iy: int,
    ix: int,
) -> None:
    global_vector[iz,     iy,     ix]     += local_vector[0]
    global_vector[iz,     iy,     ix + 1] += local_vector[1]
    global_vector[iz,     iy + 1, ix]     += local_vector[2]
    global_vector[iz,     iy + 1, ix + 1] += local_vector[3]
    global_vector[iz + 1, iy,     ix]     += local_vector[4]
    global_vector[iz + 1, iy,     ix + 1] += local_vector[5]
    global_vector[iz + 1, iy + 1, ix]     += local_vector[6]
    global_vector[iz + 1, iy + 1, ix + 1] += local_vector[7]


def compute_smoothed_tv_q1(
    normalized_field: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    epsilon: float,
) -> tuple[float, np.ndarray, dict]:
    """
    Compute:

      R_TV(m_hat)
        = integral sqrt(|grad m_hat|^2 + epsilon^2) dx

    and its discrete weak derivative with respect to the
    dimensionless nodal coefficients m_hat.
    """

    if normalized_field.shape != (
        len(z),
        len(y),
        len(x),
    ):
        raise ValueError(
            "Field and coordinate dimensions are inconsistent: "
            f"field={normalized_field.shape}, "
            f"coords={(len(z), len(y), len(x))}"
        )

    if epsilon <= 0.0:
        raise ValueError("epsilon must be strictly positive")

    nz, ny, nx = normalized_field.shape

    tv_rhs = np.zeros_like(
        normalized_field,
        dtype=np.float64,
    )

    gauss_points, gauss_weights = gauss_rule_2()

    tv_value = 0.0
    min_denominator = np.inf
    max_gradient_norm = 0.0
    quadrature_evaluations = 0

    for iz in range(nz - 1):
        hz = abs(float(z[iz + 1] - z[iz]))

        if hz <= 0.0:
            raise RuntimeError(
                f"Invalid z cell length at iz={iz}: {hz}"
            )

        for iy in range(ny - 1):
            hy = abs(float(y[iy + 1] - y[iy]))

            if hy <= 0.0:
                raise RuntimeError(
                    f"Invalid y cell length at iy={iy}: {hy}"
                )

            for ix in range(nx - 1):
                hx = abs(float(x[ix + 1] - x[ix]))

                if hx <= 0.0:
                    raise RuntimeError(
                        f"Invalid x cell length at ix={ix}: {hx}"
                    )

                values = local_values(
                    normalized_field,
                    iz,
                    iy,
                    ix,
                )

                local_rhs = np.zeros(8, dtype=np.float64)

                determinant_jacobian = (
                    hx * hy * hz / 8.0
                )

                inverse_jacobian = np.diag(
                    [
                        2.0 / hx,
                        2.0 / hy,
                        2.0 / hz,
                    ]
                )

                for qz, wz in zip(
                    gauss_points,
                    gauss_weights,
                ):
                    for qy, wy in zip(
                        gauss_points,
                        gauss_weights,
                    ):
                        for qx, wx in zip(
                            gauss_points,
                            gauss_weights,
                        ):
                            grad_reference = (
                                q1_shape_gradients_reference(
                                    qx,
                                    qy,
                                    qz,
                                )
                            )

                            grad_physical = (
                                grad_reference
                                @ inverse_jacobian
                            )

                            grad_m = (
                                values @ grad_physical
                            )

                            gradient_norm = float(
                                np.linalg.norm(grad_m)
                            )

                            denominator = float(
                                np.sqrt(
                                    gradient_norm ** 2
                                    + epsilon ** 2
                                )
                            )

                            weight = (
                                wx
                                * wy
                                * wz
                                * determinant_jacobian
                            )

                            tv_value += (
                                weight * denominator
                            )

                            local_rhs += (
                                weight
                                * (
                                    grad_physical
                                    @ grad_m
                                )
                                / denominator
                            )

                            min_denominator = min(
                                min_denominator,
                                denominator,
                            )

                            max_gradient_norm = max(
                                max_gradient_norm,
                                gradient_norm,
                            )

                            quadrature_evaluations += 1

                add_local_vector(
                    tv_rhs,
                    local_rhs,
                    iz,
                    iy,
                    ix,
                )

    stats = {
        "element_count": int(
            (nx - 1) * (ny - 1) * (nz - 1)
        ),
        "quadrature_points_per_element": 8,
        "quadrature_evaluations": int(
            quadrature_evaluations
        ),
        "min_denominator": float(min_denominator),
        "max_gradient_norm": float(max_gradient_norm),
        "rhs_min": float(np.min(tv_rhs)),
        "rhs_max": float(np.max(tv_rhs)),
        "rhs_maxabs": float(np.max(np.abs(tv_rhs))),
        "rhs_l2": float(np.linalg.norm(tv_rhs.ravel())),
        "rhs_sum": float(np.sum(tv_rhs)),
        "rhs_finite": int(
            np.count_nonzero(np.isfinite(tv_rhs))
        ),
    }

    return float(tv_value), tv_rhs, stats


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
        "--state",
        default=None,
        help=(
            "Optional state override. "
            "Default: parent_state from the config."
        ),
    )

    parser.add_argument(
        "--label",
        default="parent_iter008",
    )

    parser.add_argument(
        "--execute",
        action="store_true",
    )

    args = parser.parse_args()

    config_path = resolve(args.config)
    config = load_config(config_path)

    state_path = resolve(
        args.state or config["parent_state"]
    )

    output_dir = (
        resolve(config["tv_transition_dir"])
        / "tv_full_grid"
        / args.label
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    lam, mu = load_material_fields(state_path)

    nx = int(config["mesh"]["nx"])
    ny = int(config["mesh"]["ny"])
    nz = int(config["mesh"]["nz"])

    expected_shape = (nz, ny, nx)

    if lam.shape != expected_shape:
        raise RuntimeError(
            f"lambda shape mismatch: "
            f"expected {expected_shape}, got {lam.shape}"
        )

    if mu.shape != expected_shape:
        raise RuntimeError(
            f"mu shape mismatch: "
            f"expected {expected_shape}, got {mu.shape}"
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

    epsilon = float(
        config["tv"]["epsilon_dimensionless"]
    )

    if lambda_reference <= 0.0:
        raise RuntimeError(
            "lambda_reference_pa must be positive"
        )

    if mu_reference <= 0.0:
        raise RuntimeError(
            "mu_reference_pa must be positive"
        )

    x = np.linspace(-20.0, 20.0, nx)
    y = np.linspace(-20.0, 20.0, ny)
    z = np.linspace(0.0, -50.0, nz)

    plan = {
        "state": str(state_path),
        "label": args.label,
        "shape": list(expected_shape),
        "coordinates": {
            "x_min": float(x.min()),
            "x_max": float(x.max()),
            "y_min": float(y.min()),
            "y_max": float(y.max()),
            "z_min": float(z.min()),
            "z_max": float(z.max()),
            "dx": float(abs(x[1] - x[0])),
            "dy": float(abs(y[1] - y[0])),
            "dz": float(abs(z[1] - z[0])),
        },
        "lambda_reference_pa": lambda_reference,
        "mu_reference_pa": mu_reference,
        "epsilon_dimensionless": epsilon,
        "element_count": int(
            (nx - 1) * (ny - 1) * (nz - 1)
        ),
        "quadrature": "2x2x2 Gauss-Legendre",
        "derivative_variable": (
            "dimensionless nodal coefficients "
            "m_hat = m / m_reference"
        ),
    }

    plan_path = output_dir / "tv_execution_plan.json"

    plan_path.write_text(
        json.dumps(plan, indent=2),
        encoding="utf-8",
    )

    print("Q1 smoothed-TV full-grid computation")
    print("====================================")
    print()
    print(f"state = {state_path}")
    print(f"label = {args.label}")
    print(f"shape = {expected_shape}")
    print(
        f"elements = "
        f"{nx - 1} x {ny - 1} x {nz - 1} "
        f"= {(nx - 1) * (ny - 1) * (nz - 1)}"
    )
    print(
        "spacing = "
        f"dx={abs(x[1] - x[0])}, "
        f"dy={abs(y[1] - y[0])}, "
        f"dz={abs(z[1] - z[0])}"
    )
    print(f"epsilon = {epsilon}")
    print()
    print(f"plan = {plan_path}")

    if not args.execute:
        print()
        print("No TV assembly was performed.")
        print("Use --execute after reviewing the plan.")
        print("RESULT = PASS_PLAN")
        return

    lambda_hat = lam / lambda_reference
    mu_hat = mu / mu_reference

    print()
    print("Computing lambda TV ...")

    (
        tv_lambda,
        tv_rhs_lambda,
        stats_lambda,
    ) = compute_smoothed_tv_q1(
        lambda_hat,
        x,
        y,
        z,
        epsilon,
    )

    print("Computing mu TV ...")

    (
        tv_mu,
        tv_rhs_mu,
        stats_mu,
    ) = compute_smoothed_tv_q1(
        mu_hat,
        x,
        y,
        z,
        epsilon,
    )

    if not np.all(np.isfinite(tv_rhs_lambda)):
        raise RuntimeError(
            "Non-finite lambda TV RHS detected"
        )

    if not np.all(np.isfinite(tv_rhs_mu)):
        raise RuntimeError(
            "Non-finite mu TV RHS detected"
        )

    lambda_rhs_path = (
        output_dir / "tv_rhs_lambda_hat_full.npy"
    )

    mu_rhs_path = (
        output_dir / "tv_rhs_mu_hat_full.npy"
    )

    np.save(lambda_rhs_path, tv_rhs_lambda)
    np.save(mu_rhs_path, tv_rhs_mu)

    values = {
        "state": str(state_path),
        "label": args.label,
        "regularization_variable": {
            "lambda": (
                "lambda_hat = lambda / "
                f"{lambda_reference:.16e}"
            ),
            "mu": (
                "mu_hat = mu / "
                f"{mu_reference:.16e}"
            ),
        },
        "epsilon_dimensionless": epsilon,
        "tv_lambda_hat": tv_lambda,
        "tv_mu_hat": tv_mu,
        "lambda_stats": stats_lambda,
        "mu_stats": stats_mu,
        "important_note": (
            "The saved TV dual vectors are derivatives "
            "with respect to the dimensionless nodal "
            "coefficients lambda_hat and mu_hat. "
            "They must not yet be added directly to the "
            "existing physical-parameter data RHS."
        ),
    }

    values_path = output_dir / "tv_values_and_stats.json"

    values_path.write_text(
        json.dumps(values, indent=2),
        encoding="utf-8",
    )

    summary = []

    summary.append("Q1 SMOOTHED-TV FULL-GRID SUMMARY")
    summary.append("================================")
    summary.append("")
    summary.append(f"state = {state_path}")
    summary.append(f"label = {args.label}")
    summary.append(f"shape = {expected_shape}")
    summary.append(
        f"full nodes = {nx * ny * nz}"
    )
    summary.append(
        f"Q1 elements = "
        f"{(nx - 1) * (ny - 1) * (nz - 1)}"
    )
    summary.append(
        "quadrature = 2 x 2 x 2 Gauss-Legendre"
    )
    summary.append("")
    summary.append("Parameter scaling")
    summary.append("-----------------")
    summary.append(
        f"lambda_reference_pa = "
        f"{lambda_reference:.16e}"
    )
    summary.append(
        f"mu_reference_pa = "
        f"{mu_reference:.16e}"
    )
    summary.append(
        f"epsilon_dimensionless = {epsilon:.16e}"
    )
    summary.append("")
    summary.append("TV values")
    summary.append("---------")
    summary.append(
        f"R_TV(lambda_hat) = {tv_lambda:.16e}"
    )
    summary.append(
        f"R_TV(mu_hat) = {tv_mu:.16e}"
    )
    summary.append("")
    summary.append("Lambda TV dual vector")
    summary.append("---------------------")

    for key, value in stats_lambda.items():
        summary.append(f"{key} = {value}")

    summary.append("")
    summary.append("Mu TV dual vector")
    summary.append("-----------------")

    for key, value in stats_mu.items():
        summary.append(f"{key} = {value}")

    summary.append("")
    summary.append("Outputs")
    summary.append("-------")
    summary.append(str(lambda_rhs_path))
    summary.append(str(mu_rhs_path))
    summary.append(str(values_path))
    summary.append("")
    summary.append("IMPORTANT")
    summary.append("---------")
    summary.append(
        "The TV vectors are derivatives with respect to "
        "dimensionless coefficients."
    )
    summary.append(
        "Do not add them directly to the existing data RHS "
        "until the parameter-space transformation has been "
        "verified."
    )
    summary.append("")
    summary.append("RESULT = PASS_TV_FULL_GRID")

    summary_text = "\n".join(summary) + "\n"

    summary_path = output_dir / "tv_full_grid_summary.txt"

    summary_path.write_text(
        summary_text,
        encoding="utf-8",
    )

    print()
    print(summary_text)


if __name__ == "__main__":
    main()
