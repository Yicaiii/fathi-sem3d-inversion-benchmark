#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import h5py

from .common import (
    discover_observed_dir,
    discover_parent_trace_dir,
    find_sem3d,
    load_config,
    output_paths,
    require,
    sha256,
    trace_position_map,
    resolve,
    write_json,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/fathi_mini_e2e_3600.json")
    args = parser.parse_args()

    root, cfg, cfg_path = load_config(args.config)
    paths = output_paths(root, cfg)
    parent = resolve(root, cfg["parent_accepted_dir"])
    parent_state = resolve(root, cfg["parent_state"])
    strict = resolve(root, cfg["strict_forward_workspace"])
    matrix = resolve(root, cfg["mtilde_matrix_path"])

    parent_traces = discover_parent_trace_dir(root, cfg)
    observed = discover_observed_dir(root, cfg, parent_traces)
    sem3d = find_sem3d(root, cfg)

    required = [
        parent / "input.spec",
        parent / "stations.txt",
        parent / "mat/h5/Mat_0_Kappa.h5",
        parent / "mat/h5/Mat_0_Mu.h5",
        parent / "mat/h5/Mat_0_Density.h5",
        parent / "sem",
        parent_traces,
        parent_state,
        strict / "input.spec",
        strict / "stations.txt",
        strict / "traces",
        strict / "fin_sem",
        matrix,
        observed,
        sem3d,
    ]
    missing = [str(path) for path in required if not path.exists()]
    require(not missing, "Missing preflight inputs:\n" + "\n".join(missing))

    fin_value = (strict / "fin_sem").read_text(errors="ignore").strip()
    require(fin_value == "1", f"Strict mini fin_sem is not 1: {fin_value!r}")

    station_lines = [
        line
        for line in (strict / "stations.txt").read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    expected_control = int(cfg["control_region"]["expected_count"])
    require(len(station_lines) == expected_control, f"Strict station count {len(station_lines)} != {expected_control}")

    parent_map = trace_position_map(parent_traces)
    observed_map = trace_position_map(observed)
    common = sorted(set(parent_map) & set(observed_map))
    subset = cfg["observation_subset"]
    selected = [
        key
        for key in common
        if float(subset["x_min_m"]) - 1e-8 <= key[0] <= float(subset["x_max_m"]) + 1e-8
        and float(subset["y_min_m"]) - 1e-8 <= key[1] <= float(subset["y_max_m"]) + 1e-8
        and abs(key[2] - float(subset["z_m"])) <= 1e-8
    ]
    expected_obs = int(subset["expected_count"])
    require(len(selected) == expected_obs, f"Selected receiver count {len(selected)} != {expected_obs}")

    dudx_ok = False
    trace_files = sorted((strict / "traces").glob("capteurs.*.h5"))
    require(trace_files, f"No strict trace files in {strict / 'traces'}")
    with h5py.File(trace_files[0], "r") as handle:
        if "Variables" in handle:
            variables = [
                value.decode(errors="ignore").strip() if isinstance(value, bytes) else str(value).strip()
                for value in handle["Variables"][()]
            ]
            dudx_ok = all(any(re.fullmatch(rf"DUDX\s+{i}", value) for value in variables) for i in range(1, 10))
    require(dudx_ok, "Strict mini traces do not expose DUDX 1..9")

    payload = {
        "config": str(cfg_path),
        "parent_accepted": str(parent),
        "parent_state": str(parent_state),
        "parent_trace_dir": str(parent_traces),
        "strict_forward_workspace": str(strict),
        "strict_station_count": len(station_lines),
        "strict_station_sha256": sha256(strict / "stations.txt"),
        "observed_trace_dir": str(observed),
        "parent_trace_count": len(parent_map),
        "observed_trace_count": len(observed_map),
        "common_trace_count": len(common),
        "selected_receiver_count": len(selected),
        "selected_receiver_first": selected[0],
        "selected_receiver_last": selected[-1],
        "mtilde_matrix": str(matrix),
        "sem3d": str(sem3d),
        "dudx_ok": dudx_ok,
        "output_paths": {key: str(value) for key, value in paths.items()},
        "result": "PASS",
    }
    out = paths["report_dir"] / "preflight.json"
    write_json(out, payload)

    print("MINI E2E PREFLIGHT")
    print("==================")
    print(f"config = {cfg_path}")
    print(f"parent accepted = {parent}")
    print(f"parent predicted traces = {parent_traces}")
    print(f"observed traces = {observed}")
    print(f"strict mini = {strict}")
    print(f"control stations = {len(station_lines)}")
    print(f"selected receivers = {len(selected)}")
    print(f"sem3d = {sem3d}")
    print(f"mtilde = {matrix}")
    print(f"report = {out}")
    print("RESULT = PASS_MINI_E2E_PREFLIGHT")


if __name__ == "__main__":
    main()
