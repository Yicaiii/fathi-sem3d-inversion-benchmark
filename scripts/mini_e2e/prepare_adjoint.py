#!/usr/bin/env python3
from __future__ import annotations

import argparse
from datetime import datetime
import re
import shutil
from pathlib import Path

import h5py
import numpy as np

from .common import (
    copy_static_workspace,
    load_config,
    output_paths,
    replace_run_name,
    replace_sources,
    require,
    resolve,
    set_dudx,
    source_block,
    write_json,
)

DIRECTION = {
    "x": (1.0, 0.0, 0.0),
    "y": (0.0, 1.0, 0.0),
    "z": (0.0, 0.0, 1.0),
}


def residual_records(path: Path) -> list[dict]:
    records = []
    with h5py.File(path, "r") as handle:
        for name in sorted(handle.keys()):
            obj = handle[name]
            if not isinstance(obj, h5py.Group) or re.fullmatch(r"station_\d{4}", name) is None:
                continue
            t = np.asarray(obj["time_true_grid"], dtype=np.float64)
            source = np.asarray(obj["source_plus_time_reversed_xyz"], dtype=np.float64)
            position = np.asarray(obj["position"], dtype=np.float64).reshape(3)
            rid = int(obj.attrs["receiver_id"])
            require(source.shape == (len(t), 3), f"Bad residual shape for {name}: {source.shape}")
            records.append(
                {
                    "group": name,
                    "rid": rid,
                    "time": t - t[0],
                    "source": source,
                    "position": position,
                }
            )
    return records


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/fathi_mini_e2e_3600.json")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    root, cfg, _ = load_config(args.config)
    paths = output_paths(root, cfg)
    strict = resolve(root, cfg["strict_forward_workspace"])
    residual_h5 = paths["residual_dir"] / "mini_residual_timeseries.h5"
    require(residual_h5.is_file(), f"Missing residual H5: {residual_h5}")
    records = residual_records(residual_h5)
    expected_receivers = int(cfg["observation_subset"]["expected_count"])
    require(len(records) == expected_receivers, f"Residual records {len(records)} != {expected_receivers}")

    station_source = strict / cfg["control_region"]["station_file"]
    require(station_source.is_file(), f"Missing strict station file: {station_source}")
    station_count = sum(bool(line.strip()) for line in station_source.read_text().splitlines())
    require(station_count == int(cfg["control_region"]["expected_count"]), "Bad control station count")

    component_payload = {}
    for component in cfg["adjoint"]["components"]:
        workspace = paths["adjoint_root"] / component / "batch_000"
        if workspace.exists() and not args.force:
            raise RuntimeError(f"Adjoint workspace exists: {workspace}; use --force to replace")
        copy_static_workspace(strict, workspace, symlink_sem=True)
        shutil.copy2(station_source, workspace / "stations.txt")

        blocks = []
        source_files = []
        for record in records:
            filename = f"s{record['rid']}{component}.txt"
            array = np.column_stack([record["time"], record["source"][:, "xyz".index(component)]])
            require(np.all(np.isfinite(array)), f"Non-finite adjoint source {filename}")
            require(np.all(np.diff(array[:, 0]) > 0), f"Non-increasing time in {filename}")
            np.savetxt(workspace / filename, array, fmt="%.16e")
            blocks.append(source_block(record["position"], DIRECTION[component], filename))
            source_files.append(filename)

        text = (strict / "input.spec").read_text(encoding="utf-8")
        text = replace_run_name(text, f"mini_e2e_adjoint_{component}")
        text = re.sub(r"save_snap\s*=\s*(?:true|false)\s*;", "save_snap = false;", text, count=1)
        text = set_dudx(text, True)
        text = replace_sources(text, blocks)
        (workspace / "input.spec").write_text(text, encoding="utf-8")

        marker = {
            "created": datetime.now().isoformat(),
            "component": component,
            "workspace": str(workspace),
            "template": str(strict),
            "residual_h5": str(residual_h5),
            "source_count": len(source_files),
            "station_count": station_count,
            "batch_count": 1,
            "sem_directory_is_symlink": (workspace / "sem").is_symlink(),
            "source_sign": cfg["adjoint"]["source_sign"],
            "result": "PASS",
        }
        write_json(workspace / "MINI_ADJOINT_PREPARATION.json", marker)
        component_payload[component] = marker

    report = {
        "created": datetime.now().isoformat(),
        "residual_h5": str(residual_h5),
        "components": component_payload,
        "result": "PASS",
    }
    report_path = paths["report_dir"] / "adjoint_preparation.json"
    write_json(report_path, report)

    print("MINI ADJOINT PREPARATION")
    print("========================")
    print(f"receivers/sources per component = {expected_receivers}")
    print(f"control stations per component = {station_count}")
    for component, payload in component_payload.items():
        print(f"{component}: {payload['workspace']}")
    print(f"report = {report_path}")
    print("RESULT = PASS_MINI_ADJOINT_PREPARATION")


if __name__ == "__main__":
    main()
