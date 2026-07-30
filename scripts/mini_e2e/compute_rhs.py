#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path
import re

import h5py
import numpy as np

from .common import load_config, output_paths, require, write_json


def read_manifest(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def norm(value: str | float) -> float:
    result = round(float(value), 8)
    return 0.0 if abs(result) < 1e-10 else result


def coordinate(row: dict[str, str]) -> tuple[float, float, float]:
    return norm(row["x"]), norm(row["y"]), norm(row["z"])


def decode_variables(handle: h5py.File) -> list[str]:
    values = handle["Variables"][()] if "Variables" in handle else []
    return [value.decode(errors="ignore").strip() if isinstance(value, bytes) else str(value).strip() for value in values]


def dudx_columns(variables: list[str]) -> list[int]:
    found = {}
    for column, value in enumerate(variables):
        match = re.fullmatch(r"DUDX\s+([1-9])", " ".join(value.split()))
        if match:
            found[int(match.group(1)) - 1] = column
    require(sorted(found) == list(range(9)), f"DUDX 1..9 not found: {variables}")
    return [found[index] for index in range(9)]


def integrate_one(f_arr: np.ndarray, a_arr: np.ndarray, f_cols: list[int], a_cols: list[int]):
    f_time = np.asarray(f_arr[:, 0], dtype=np.float64)
    a_time = np.asarray(a_arr[:, 0], dtype=np.float64)
    final_time = min(float(f_time[-1]), float(a_time[-1]))
    mask = (f_time >= 0.0) & (f_time <= final_time)
    t = f_time[mask]
    tau = final_time - t
    f_d = np.asarray(f_arr[:, f_cols], dtype=np.float64)[mask]
    a_raw = np.asarray(a_arr[:, a_cols], dtype=np.float64)
    a_d = np.column_stack([np.interp(tau, a_time, a_raw[:, index]) for index in range(9)])
    div_f = f_d[:, 0] + f_d[:, 4] + f_d[:, 8]
    div_a = a_d[:, 0] + a_d[:, 4] + a_d[:, 8]
    q_lambda = -div_f * div_a
    diag = 2.0 * (f_d[:, 0] * a_d[:, 0] + f_d[:, 4] * a_d[:, 4] + f_d[:, 8] * a_d[:, 8])
    xy = (f_d[:, 1] + f_d[:, 3]) * (a_d[:, 1] + a_d[:, 3])
    xz = (f_d[:, 2] + f_d[:, 6]) * (a_d[:, 2] + a_d[:, 6])
    yz = (f_d[:, 5] + f_d[:, 7]) * (a_d[:, 5] + a_d[:, 7])
    q_mu = -(diag + xy + xz + yz)
    return float(np.trapezoid(q_lambda, t)), float(np.trapezoid(q_mu, t)), int(len(t)), final_time


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/fathi_mini_e2e_3600.json")
    parser.add_argument("--component", choices=["x", "y", "z"], required=True)
    args = parser.parse_args()
    root, cfg, _ = load_config(args.config)
    paths = output_paths(root, cfg)
    expected = int(cfg["control_region"]["expected_count"])
    fwd_rows = read_manifest(paths["manifest_dir"] / "forward_control_manifest.csv")
    adj_rows = read_manifest(paths["manifest_dir"] / f"adjoint_{args.component}_control_manifest.csv")
    require(len(fwd_rows) == expected and len(adj_rows) == expected, f"Manifest sizes {len(fwd_rows)}, {len(adj_rows)} != {expected}")

    adj_map = {coordinate(row): row for row in adj_rows}
    require(len(adj_map) == expected, "Duplicate adjoint coordinates")
    pairs = [(index, row, adj_map.get(coordinate(row))) for index, row in enumerate(fwd_rows)]
    missing = [coordinate(row) for _, row, adj in pairs if adj is None]
    require(not missing, f"Missing adjoint coordinates: {missing[:10]}")

    rhs_lambda = np.full(expected, np.nan)
    rhs_mu = np.full(expected, np.nan)
    coords = np.full((expected, 3), np.nan)
    steps = np.zeros(expected, dtype=np.int64)
    times = np.full(expected, np.nan)

    by_forward = defaultdict(list)
    for index, fwd, adj in pairs:
        by_forward[fwd["trace_file"]].append((index, fwd, adj))

    failures = []
    done = 0
    for fwd_file, group in sorted(by_forward.items()):
        by_adjoint = defaultdict(list)
        for item in group:
            by_adjoint[item[2]["trace_file"]].append(item)
        with h5py.File(fwd_file, "r") as f_handle:
            f_cols = dudx_columns(decode_variables(f_handle))
            for adj_file, items in sorted(by_adjoint.items()):
                with h5py.File(adj_file, "r") as a_handle:
                    a_cols = dudx_columns(decode_variables(a_handle))
                    for index, fwd, adj in items:
                        try:
                            f_arr = np.asarray(f_handle[fwd["receiver_key"]], dtype=np.float64)
                            a_arr = np.asarray(a_handle[adj["receiver_key"]], dtype=np.float64)
                            rl, rm, ns, final_time = integrate_one(f_arr, a_arr, f_cols, a_cols)
                            rhs_lambda[index] = rl
                            rhs_mu[index] = rm
                            coords[index] = [float(fwd["x"]), float(fwd["y"]), float(fwd["z"])]
                            steps[index] = ns
                            times[index] = final_time
                        except Exception as exc:
                            failures.append((index, repr(exc)))
                        done += 1
                        if done % 500 == 0:
                            print(f"{args.component}: {done}/{expected}", flush=True)

    require(not failures, f"RHS failures: {failures[:10]}")
    require(np.all(np.isfinite(rhs_lambda)) and np.all(np.isfinite(rhs_mu)) and np.all(np.isfinite(coords)), "Non-finite RHS output")
    out = paths["component_rhs_dir"]
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / f"mini_RHS_{args.component}_lambda.npy", rhs_lambda)
    np.save(out / f"mini_RHS_{args.component}_mu.npy", rhs_mu)
    np.save(out / f"mini_RHS_{args.component}_coords.npy", coords)
    payload = {
        "component": args.component,
        "count": expected,
        "lambda_min": float(rhs_lambda.min()),
        "lambda_max": float(rhs_lambda.max()),
        "mu_min": float(rhs_mu.min()),
        "mu_max": float(rhs_mu.max()),
        "time_min": float(times.min()),
        "time_max": float(times.max()),
        "result": "PASS",
    }
    write_json(out / f"mini_RHS_{args.component}_summary.json", payload)
    print(f"component={args.component} count={expected}")
    print("RESULT = PASS_MINI_RHS_COMPONENT")


if __name__ == "__main__":
    main()
