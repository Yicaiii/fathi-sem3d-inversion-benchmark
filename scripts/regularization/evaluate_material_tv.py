from __future__ import annotations

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import argparse
import json

import numpy as np


ROOT = Path(__file__).resolve().parents[2]


def load_tv_function():
    path = (
        ROOT
        / "scripts"
        / "regularization"
        / "02_compute_tv_q1_full_grid.py"
    )

    spec = spec_from_file_location(
        "tv_q1_material_evaluation",
        path,
    )

    if spec is None or spec.loader is None:
        raise RuntimeError(
            f"Cannot import TV implementation: {path}"
        )

    module = module_from_spec(spec)
    spec.loader.exec_module(module)

    return module.compute_smoothed_tv_q1


def reshape_field(
    value: np.ndarray,
    expected_shape: tuple[int, int, int],
    name: str,
) -> np.ndarray:
    array = np.asarray(
        value,
        dtype=np.float64,
    )

    if array.shape == expected_shape:
        result = array
    elif array.size == int(np.prod(expected_shape)):
        result = array.reshape(expected_shape)
    else:
        raise RuntimeError(
            f"{name} shape mismatch: "
            f"value={array.shape}, "
            f"expected={expected_shape}"
        )

    if not np.all(np.isfinite(result)):
        raise RuntimeError(
            f"{name} contains non-finite values"
        )

    return result


def load_material_fields(
    path: Path,
    expected_shape: tuple[int, int, int],
) -> tuple[np.ndarray, np.ndarray]:
    with np.load(
        path,
        allow_pickle=True,
    ) as state:
        keys = list(state.files)

        if "mu" not in keys:
            raise KeyError(
                f"State does not contain mu: {keys}"
            )

        mu = reshape_field(
            state["mu"],
            expected_shape,
            "mu",
        )

        if "lambda" in keys:
            lam_raw = state["lambda"]
        elif "lambda_field" in keys:
            lam_raw = state["lambda_field"]
        elif "lam" in keys:
            lam_raw = state["lam"]
        elif "kappa" in keys:
            kappa = reshape_field(
                state["kappa"],
                expected_shape,
                "kappa",
            )
            lam_raw = (
                kappa
                - (2.0 / 3.0) * mu
            )
        else:
            raise KeyError(
                "State does not contain lambda, "
                "lambda_field, lam, or kappa. "
                f"Available keys: {keys}"
            )

        lam = reshape_field(
            lam_raw,
            expected_shape,
            "lambda",
        )

    return lam, mu


def json_ready(value):
    if isinstance(value, dict):
        return {
            str(key): json_ready(item)
            for key, item in value.items()
        }

    if isinstance(value, (list, tuple)):
        return [
            json_ready(item)
            for item in value
        ]

    if isinstance(value, np.ndarray):
        return value.tolist()

    if isinstance(value, np.generic):
        return value.item()

    return value


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--state",
        required=True,
    )

    parser.add_argument(
        "--config",
        required=True,
    )

    parser.add_argument(
        "--output",
        required=True,
    )

    args = parser.parse_args()

    state_path = Path(
        args.state
    ).expanduser().resolve()

    config_path = Path(
        args.config
    ).expanduser().resolve()

    output_path = Path(
        args.output
    ).expanduser().resolve()

    if not state_path.is_file():
        raise FileNotFoundError(
            f"Missing material state: {state_path}"
        )

    if not config_path.is_file():
        raise FileNotFoundError(
            f"Missing TV config: {config_path}"
        )

    config = json.loads(
        config_path.read_text(
            encoding="utf-8"
        )
    )

    mesh = config["mesh"]
    scaling = config["parameter_scaling"]
    tv_config = config["tv"]

    expected_shape = (
        int(mesh["nz"]),
        int(mesh["ny"]),
        int(mesh["nx"]),
    )

    lam, mu = load_material_fields(
        state_path,
        expected_shape,
    )

    x = np.linspace(
        float(mesh["x_min"]),
        float(mesh["x_max"]),
        expected_shape[2],
    )

    y = np.linspace(
        float(mesh["y_min"]),
        float(mesh["y_max"]),
        expected_shape[1],
    )

    z = np.linspace(
        float(mesh["z_max"]),
        float(mesh["z_min"]),
        expected_shape[0],
    )

    lambda_reference = float(
        scaling["lambda_reference_pa"]
    )

    mu_reference = float(
        scaling["mu_reference_pa"]
    )

    epsilon = float(
        tv_config["epsilon_dimensionless"]
    )

    if lambda_reference <= 0.0:
        raise RuntimeError(
            "lambda_reference_pa must be positive"
        )

    if mu_reference <= 0.0:
        raise RuntimeError(
            "mu_reference_pa must be positive"
        )

    compute_tv = load_tv_function()

    tv_lambda, _, lambda_stats = compute_tv(
        lam / lambda_reference,
        x,
        y,
        z,
        epsilon,
    )

    tv_mu, _, mu_stats = compute_tv(
        mu / mu_reference,
        x,
        y,
        z,
        epsilon,
    )

    payload = {
        "state": str(state_path),
        "config": str(config_path),
        "shape": list(expected_shape),
        "tv_lambda_hat": float(tv_lambda),
        "tv_mu_hat": float(tv_mu),
        "epsilon_dimensionless": epsilon,
        "lambda_reference_pa": (
            lambda_reference
        ),
        "mu_reference_pa": mu_reference,
        "lambda_stats": json_ready(
            lambda_stats
        ),
        "mu_stats": json_ready(
            mu_stats
        ),
        "sem3d_launched": False,
        "accepted_state_mutated": False,
        "result": (
            "PASS_MATERIAL_TV_EVALUATION"
        ),
    }

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path.write_text(
        json.dumps(
            payload,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(
        json.dumps(
            payload,
            indent=2,
        )
    )
    print(f"output = {output_path}")
    print(
        "RESULT = "
        "PASS_MATERIAL_TV_EVALUATION"
    )


if __name__ == "__main__":
    main()
