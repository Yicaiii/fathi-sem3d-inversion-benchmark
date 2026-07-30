#!/usr/bin/env python3
from __future__ import annotations

import argparse
from datetime import datetime

import numpy as np
from scipy.sparse import load_npz, save_npz
from scipy.sparse.linalg import spsolve

from .common import load_config, output_paths, require, resolve, write_json


def key(position):
    return tuple(round(float(value), 8) for value in position)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/fathi_mini_e2e_3600.json")
    args = parser.parse_args()
    root, cfg, _ = load_config(args.config)
    paths = output_paths(root, cfg)
    matrix_path = resolve(root, cfg["mtilde_matrix_path"])
    rhs_dir = paths["component_rhs_dir"]
    out = paths["mtilde_dir"]
    out.mkdir(parents=True, exist_ok=True)

    rhs_lambda = np.load(rhs_dir / "mini_RHS_total_lambda.npy")
    rhs_mu = np.load(rhs_dir / "mini_RHS_total_mu.npy")
    rhs_coords = np.load(rhs_dir / "mini_RHS_total_coords.npy")
    expected = int(cfg["control_region"]["expected_count"])
    require(rhs_lambda.shape == (expected,), f"Bad RHS lambda shape {rhs_lambda.shape}")
    require(rhs_mu.shape == (expected,), f"Bad RHS mu shape {rhs_mu.shape}")
    require(rhs_coords.shape == (expected, 3), f"Bad RHS coords shape {rhs_coords.shape}")

    grid = cfg["full_material_grid"]
    nx, ny, nz = int(grid["nx"]), int(grid["ny"]), int(grid["nz"])
    x = np.linspace(float(grid["x_min_m"]), float(grid["x_max_m"]), nx)
    y = np.linspace(float(grid["y_min_m"]), float(grid["y_max_m"]), ny)
    z = np.linspace(float(grid["z_start_m"]), float(grid["z_end_m"]), nz)
    full_coords = np.asarray([(xx, yy, zz) for zz in z for yy in y for xx in x], dtype=np.float64)

    matrix = load_npz(matrix_path).tocsr()
    require(matrix.shape == (len(full_coords), len(full_coords)), f"Mtilde shape {matrix.shape} != {len(full_coords)}")
    full_map = {key(position): index for index, position in enumerate(full_coords)}
    missing = [key(position) for position in rhs_coords if key(position) not in full_map]
    require(not missing, f"Control coordinates missing from Mtilde grid: {missing[:10]}")
    indices = np.asarray([full_map[key(position)] for position in rhs_coords], dtype=np.int64)
    require(len(np.unique(indices)) == expected, "Duplicate mapped Mtilde indices")
    mapped = full_coords[indices]
    max_diff = float(np.max(np.abs(mapped - rhs_coords)))
    require(max_diff <= 1e-8, f"Mtilde coordinate mapping diff {max_diff}")

    subset = matrix[indices, :][:, indices].tocsr()
    g_lambda = spsolve(subset, rhs_lambda)
    g_mu = spsolve(subset, rhs_mu)
    require(np.all(np.isfinite(g_lambda)) and np.all(np.isfinite(g_mu)), "Non-finite Mtilde gradient")
    rel_lambda = float(np.linalg.norm(subset @ g_lambda - rhs_lambda) / max(np.linalg.norm(rhs_lambda), 1e-300))
    rel_mu = float(np.linalg.norm(subset @ g_mu - rhs_mu) / max(np.linalg.norm(rhs_mu), 1e-300))
    require(rel_lambda < 1e-8 and rel_mu < 1e-8, f"Mtilde solve residuals too large: {rel_lambda}, {rel_mu}")

    save_npz(out / "mini_Mtilde_subset_sparse.npz", subset)
    np.save(out / "mini_Mtilde_subset_indices.npy", indices)
    np.save(out / "mini_Mtilde_subset_coords.npy", rhs_coords)
    np.save(out / "mini_g_lambda.npy", g_lambda)
    np.save(out / "mini_g_mu.npy", g_mu)
    payload = {
        "created": datetime.now().isoformat(),
        "matrix": str(matrix_path),
        "full_shape": matrix.shape,
        "subset_shape": subset.shape,
        "subset_nnz": int(subset.nnz),
        "max_coordinate_diff": max_diff,
        "relative_residual_lambda": rel_lambda,
        "relative_residual_mu": rel_mu,
        "gradient_lambda_maxabs": float(np.max(np.abs(g_lambda))),
        "gradient_mu_maxabs": float(np.max(np.abs(g_mu))),
        "result": "PASS",
    }
    write_json(out / "mini_mtilde_summary.json", payload)
    print(f"subset_shape = {subset.shape}")
    print(f"relative_residual_lambda = {rel_lambda:.16e}")
    print(f"relative_residual_mu = {rel_mu:.16e}")
    print("RESULT = PASS_MINI_MTILDE_SOLVE")


if __name__ == "__main__":
    main()
