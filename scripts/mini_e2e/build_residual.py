#!/usr/bin/env python3
from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import h5py
import numpy as np

from .common import (
    discover_observed_dir,
    discover_parent_trace_dir,
    load_config,
    load_displacement,
    output_paths,
    require,
    resolve,
    trace_position_map,
    write_json,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/fathi_mini_e2e_3600.json")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    root, cfg, _ = load_config(args.config)
    paths = output_paths(root, cfg)
    parent = resolve(root, cfg["parent_accepted_dir"])
    parent_traces = discover_parent_trace_dir(root, cfg)
    observed_dir = discover_observed_dir(root, cfg, parent_traces)

    out_dir = paths["residual_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)
    out_h5 = out_dir / "mini_residual_timeseries.h5"
    out_json = out_dir / "mini_residual_summary.json"
    out_txt = out_dir / "mini_residual_summary.txt"
    if (out_h5.exists() or out_json.exists()) and not args.force:
        raise RuntimeError(f"Residual outputs already exist in {out_dir}; use --force to replace")

    parent_map = trace_position_map(parent_traces)
    observed_map = trace_position_map(observed_dir)
    subset = cfg["observation_subset"]
    selected = sorted(
        key
        for key in set(parent_map) & set(observed_map)
        if float(subset["x_min_m"]) - 1e-8 <= key[0] <= float(subset["x_max_m"]) + 1e-8
        and float(subset["y_min_m"]) - 1e-8 <= key[1] <= float(subset["y_max_m"]) + 1e-8
        and abs(key[2] - float(subset["z_m"])) <= 1e-8
    )
    expected = int(subset["expected_count"])
    require(len(selected) == expected, f"Selected {len(selected)} receivers, expected {expected}")

    total_j = 0.0
    records = []
    with h5py.File(out_h5, "w") as handle:
        handle.attrs["created"] = datetime.now().isoformat()
        handle.attrs["definition"] = "residual = parent synthetic displacement - true observed displacement"
        handle.attrs["source_sign"] = cfg["adjoint"]["source_sign"]
        for index, coordinate in enumerate(selected):
            observed_entry = observed_map[coordinate]
            parent_entry = parent_map[coordinate]
            t_obs, u_obs = load_displacement(observed_entry)
            t_syn, u_syn = load_displacement(parent_entry)

            t0 = max(float(t_obs[0]), float(t_syn[0]))
            t1 = min(float(t_obs[-1]), float(t_syn[-1]))
            mask = (t_obs >= t0) & (t_obs <= t1)
            t_eval = t_obs[mask]
            require(t_eval.size >= 2, f"Insufficient time overlap at {coordinate}")
            u_true = u_obs[mask, :]
            u_sim = np.column_stack(
                [np.interp(t_eval, t_syn, u_syn[:, comp]) for comp in range(3)]
            )
            residual = u_sim - u_true
            residual_reversed = residual[::-1, :].copy()
            local_j = 0.5 * float(np.trapezoid(np.sum(residual * residual, axis=1), t_eval))
            total_j += local_j

            true_dataset = str(observed_entry["dataset"])
            rid = int(true_dataset.split("_")[1])
            group = handle.create_group(f"station_{index:04d}")
            group.attrs["station_index"] = index
            group.attrs["receiver_id"] = rid
            group.attrs["true_dataset"] = true_dataset
            group.attrs["synthetic_dataset"] = str(parent_entry["dataset"])
            group.attrs["true_file"] = str(observed_entry["file"])
            group.attrs["synthetic_file"] = str(parent_entry["file"])
            group.attrs["local_J"] = local_j
            group.create_dataset("position", data=np.asarray(coordinate, dtype=np.float64))
            group.create_dataset("time_true_grid", data=t_eval)
            group.create_dataset("residual_forward_time_xyz", data=residual)
            group.create_dataset("source_plus_time_reversed_xyz", data=residual_reversed)
            group.create_dataset("source_minus_time_reversed_xyz", data=-residual_reversed)

            records.append(
                {
                    "station_index": index,
                    "receiver_id": rid,
                    "position": coordinate,
                    "true_dataset": true_dataset,
                    "synthetic_dataset": str(parent_entry["dataset"]),
                    "n_time": int(t_eval.size),
                    "local_J": local_j,
                    "max_abs_residual": float(np.max(np.abs(residual))),
                }
            )
        handle.create_dataset("station_positions", data=np.asarray(selected, dtype=np.float64))
        handle.attrs["station_count"] = len(selected)
        handle.attrs["parent_J"] = total_j

    payload = {
        "created": datetime.now().isoformat(),
        "observed_trace_dir": str(observed_dir),
        "parent_trace_dir": str(parent_traces),
        "station_count": len(selected),
        "parent_J": total_j,
        "residual_h5": str(out_h5),
        "records": records,
        "result": "PASS",
    }
    write_json(out_json, payload)
    out_txt.write_text(
        "\n".join(
            [
                "MINI E2E RESIDUAL SUMMARY",
                "=========================",
                f"observed_trace_dir = {observed_dir}",
                f"parent_trace_dir = {parent_traces}",
                f"station_count = {len(selected)}",
                f"parent_J = {total_j:.16e}",
                f"residual_h5 = {out_h5}",
                "RESULT = PASS",
                "",
            ]
        ),
        encoding="utf-8",
    )

    print(out_txt.read_text(encoding="utf-8"), end="")


if __name__ == "__main__":
    main()
