#!/usr/bin/env python3
from __future__ import annotations

import argparse
import numpy as np

from .common import load_config, output_paths, require, write_json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/fathi_mini_e2e_3600.json")
    args = parser.parse_args()
    root, cfg, _ = load_config(args.config)
    paths = output_paths(root, cfg)
    expected = int(cfg["control_region"]["expected_count"])
    out = paths["component_rhs_dir"]
    arrays = {}
    for component in cfg["adjoint"]["components"]:
        arrays[component] = {
            "lambda": np.load(out / f"mini_RHS_{component}_lambda.npy"),
            "mu": np.load(out / f"mini_RHS_{component}_mu.npy"),
            "coords": np.load(out / f"mini_RHS_{component}_coords.npy"),
        }
        require(arrays[component]["lambda"].shape == (expected,), f"Bad lambda shape {component}")
        require(arrays[component]["mu"].shape == (expected,), f"Bad mu shape {component}")
        require(arrays[component]["coords"].shape == (expected, 3), f"Bad coords shape {component}")
    base_coords = arrays["x"]["coords"]
    for component in ("y", "z"):
        require(np.max(np.abs(arrays[component]["coords"] - base_coords)) <= 1e-8, f"Coordinate mismatch {component}")
    total_lambda = sum(arrays[component]["lambda"] for component in ("x", "y", "z"))
    total_mu = sum(arrays[component]["mu"] for component in ("x", "y", "z"))
    require(np.all(np.isfinite(total_lambda)) and np.all(np.isfinite(total_mu)), "Non-finite total RHS")
    np.save(out / "mini_RHS_total_lambda.npy", total_lambda)
    np.save(out / "mini_RHS_total_mu.npy", total_mu)
    np.save(out / "mini_RHS_total_coords.npy", base_coords)
    write_json(
        out / "mini_RHS_total_summary.json",
        {
            "count": expected,
            "lambda_l2": float(np.linalg.norm(total_lambda)),
            "mu_l2": float(np.linalg.norm(total_mu)),
            "result": "PASS",
        },
    )
    print(f"count = {expected}")
    print(f"lambda_l2 = {np.linalg.norm(total_lambda):.16e}")
    print(f"mu_l2 = {np.linalg.norm(total_mu):.16e}")
    print("RESULT = PASS_MINI_RHS_TOTAL")


if __name__ == "__main__":
    main()
