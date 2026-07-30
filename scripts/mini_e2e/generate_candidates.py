#!/usr/bin/env python3
from __future__ import annotations

import argparse
from datetime import datetime
import shutil
from pathlib import Path

import h5py
import numpy as np

from .common import load_config, output_paths, require, resolve, write_json


def dataset_name(path: Path, shape: tuple[int, ...]) -> str:
    matches = []
    with h5py.File(path, "r") as handle:
        def visitor(name, obj):
            if isinstance(obj, h5py.Dataset) and tuple(obj.shape) == shape:
                matches.append(name)
        handle.visititems(visitor)
    require(bool(matches), f"No dataset with shape {shape} in {path}")
    return matches[0]


def read_field(path: Path, shape: tuple[int, ...]):
    name = dataset_name(path, shape)
    with h5py.File(path, "r") as handle:
        data = np.asarray(handle[name], dtype=np.float64)
    return data, name


def overwrite(path: Path, name: str, data: np.ndarray) -> None:
    with h5py.File(path, "r+") as handle:
        require(name in handle and tuple(handle[name].shape) == tuple(data.shape), f"Dataset mismatch {path}:{name}")
        handle[name][...] = data


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/fathi_mini_e2e_3600.json")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    root, cfg, _ = load_config(args.config)
    paths = output_paths(root, cfg)
    parent = resolve(root, cfg["parent_accepted_dir"])
    parent_state = resolve(root, cfg["parent_state"])
    shape = tuple(int(value) for value in cfg["material_shape"])
    mtilde = paths["mtilde_dir"]
    indices = np.load(mtilde / "mini_Mtilde_subset_indices.npy").astype(np.int64)
    g_lambda = np.load(mtilde / "mini_g_lambda.npy").astype(np.float64)
    g_mu = np.load(mtilde / "mini_g_mu.npy").astype(np.float64)
    expected = int(cfg["control_region"]["expected_count"])
    require(indices.shape == g_lambda.shape == g_mu.shape == (expected,), "Mini gradient/index shape mismatch")
    require(np.all(np.isfinite(g_lambda)) and np.all(np.isfinite(g_mu)), "Non-finite gradients")
    scale_lambda = float(np.max(np.abs(g_lambda)))
    scale_mu = float(np.max(np.abs(g_mu)))
    require(scale_lambda > 0.0 and scale_mu > 0.0, "Zero gradient scale")

    h5_dir = parent / "mat/h5"
    kappa, kappa_name = read_field(h5_dir / "Mat_0_Kappa.h5", shape)
    mu, mu_name = read_field(h5_dir / "Mat_0_Mu.h5", shape)
    density, density_name = read_field(h5_dir / "Mat_0_Density.h5", shape)
    lam = kappa - (2.0 / 3.0) * mu
    flat_lam = lam.reshape(-1)
    flat_mu = mu.reshape(-1)
    require(indices.min() >= 0 and indices.max() < flat_lam.size, "Gradient indices outside material field")

    candidate_root = paths["candidate_dir"]
    candidate_root.mkdir(parents=True, exist_ok=True)
    records = []
    for step_mpa in cfg["line_search_steps_mpa"]:
        step_mpa = float(step_mpa)
        label = f"{step_mpa:.2f}".replace(".", "p") + "MPa"
        name = f"mini_line_search_pos_mtilde_{label}"
        directory = candidate_root / name
        if directory.exists() and not args.force:
            raise RuntimeError(f"Candidate exists: {directory}; use --force to replace")
        if directory.exists():
            shutil.rmtree(directory)
        candidate_h5 = directory / "mat/h5"
        candidate_h5.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(h5_dir, candidate_h5)

        new_lam_flat = flat_lam.copy()
        new_mu_flat = flat_mu.copy()
        step_pa = step_mpa * 1e6
        new_lam_flat[indices] += step_pa * (g_lambda / scale_lambda)
        new_mu_flat[indices] += step_pa * (g_mu / scale_mu)
        new_lam = new_lam_flat.reshape(shape)
        new_mu = new_mu_flat.reshape(shape)
        new_kappa = new_lam + (2.0 / 3.0) * new_mu
        require(np.all(np.isfinite(new_lam)) and np.all(np.isfinite(new_mu)), f"Non-finite candidate {name}")
        require(float(new_lam.min()) > 0 and float(new_mu.min()) > 0 and float(new_kappa.min()) > 0, f"Non-positive candidate {name}")
        overwrite(candidate_h5 / "Mat_0_Kappa.h5", kappa_name, new_kappa)
        overwrite(candidate_h5 / "Mat_0_Mu.h5", mu_name, new_mu)
        overwrite(candidate_h5 / "Mat_0_Density.h5", density_name, density)

        state_path = directory / f"{name}_state_candidate.npz"
        np.savez_compressed(
            state_path,
            lambda_field=new_lam,
            mu=new_mu,
            kappa=new_kappa,
            density=density,
            parent_state=str(parent_state),
            iter_k=0,
            iter_kp1=1,
            step_mpa=step_mpa,
            gradient_indices=str(mtilde / "mini_Mtilde_subset_indices.npy"),
            direction="positive_mtilde_gradient_maxabs_normalized_mini3600",
        )
        records.append(
            {
                "candidate": name,
                "step_mpa": step_mpa,
                "directory": str(directory),
                "state": str(state_path),
                "max_abs_delta_lambda": float(np.max(np.abs(new_lam - lam))),
                "max_abs_delta_mu": float(np.max(np.abs(new_mu - mu))),
            }
        )

    report = {
        "created": datetime.now().isoformat(),
        "gradient_scale_lambda": scale_lambda,
        "gradient_scale_mu": scale_mu,
        "records": records,
        "result": "PASS",
    }
    write_json(paths["report_dir"] / "candidate_generation.json", report)
    for record in records:
        print(f"{record['candidate']}: step={record['step_mpa']} MPa")
    print("RESULT = PASS_MINI_CANDIDATE_GENERATION")


if __name__ == "__main__":
    main()
