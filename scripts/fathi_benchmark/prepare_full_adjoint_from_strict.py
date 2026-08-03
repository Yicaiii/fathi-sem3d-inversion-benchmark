#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
from datetime import datetime
from pathlib import Path

import h5py
import numpy as np

from scripts.mini_e2e.common import (
    copy_static_workspace,
    replace_run_name,
    replace_sources,
    require,
    set_dudx,
    source_block,
    write_json,
)

DIRECTIONS = {
    "x": (1.0, 0.0, 0.0),
    "y": (0.0, 1.0, 0.0),
    "z": (0.0, 0.0, 1.0),
}


def resolve(root: Path, value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def get_receiver_id(group: h5py.Group) -> int:
    if "receiver_id" in group.attrs:
        return int(group.attrs["receiver_id"])
    value = group.attrs.get("true_dataset")
    if isinstance(value, bytes):
        value = value.decode()
    match = re.search(r"UU_(\d+)$", str(value))
    require(match is not None, f"Cannot determine receiver id from true_dataset={value}")
    return int(match.group(1))


def load_residual_records(path: Path) -> list[dict]:
    records = []
    with h5py.File(path, "r") as handle:
        for name in sorted(handle.keys()):
            group = handle[name]
            if not isinstance(group, h5py.Group):
                continue
            if re.fullmatch(r"station_\d{4}", name) is None:
                continue
            require("time_true_grid" in group, f"Missing time_true_grid in {name}")
            require("position" in group, f"Missing position in {name}")
            if "source_plus_time_reversed_xyz" in group:
                source_name = "source_plus_time_reversed_xyz"
            elif "residual_time_reversed_xyz" in group:
                source_name = "residual_time_reversed_xyz"
            else:
                raise RuntimeError(f"Missing reversed residual source in {name}")
            time = np.asarray(group["time_true_grid"], dtype=np.float64)
            source = np.asarray(group[source_name], dtype=np.float64)
            position = np.asarray(group["position"], dtype=np.float64).reshape(3)
            rid = get_receiver_id(group)
            require(source.shape == (len(time), 3), f"Bad source shape for {name}: {source.shape}")
            shifted_time = time - time[0]
            require(np.all(np.isfinite(shifted_time)), f"Non-finite time in {name}")
            require(np.all(np.diff(shifted_time) > 0), f"Non-increasing time in {name}")
            require(np.all(np.isfinite(source)), f"Non-finite source in {name}")
            records.append(
                {
                    "name": name,
                    "rid": rid,
                    "time": shifted_time,
                    "source": source,
                    "position": position,
                    "source_dataset": source_name,
                }
            )
    records.sort(key=lambda item: item["rid"])
    receiver_ids = [item["rid"] for item in records]
    require(len(receiver_ids) == len(set(receiver_ids)), "Duplicate receiver ids in residual H5")
    return records


def read_station_lines(path: Path) -> list[str]:
    return [
        line.rstrip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]


def split_ranges(total: int, batch_count: int) -> list[tuple[int, int]]:
    require(batch_count > 0, "batch_count must be positive")
    chunks = np.array_split(np.arange(total, dtype=np.int64), batch_count)
    ranges = []
    for chunk in chunks:
        require(len(chunk) > 0, "batch_count exceeds station count")
        ranges.append((int(chunk[0]), int(chunk[-1]) + 1))
    return ranges


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--context", required=True)
    parser.add_argument("--benchmark-spec", required=True)
    parser.add_argument("--batch-count", required=True, type=int)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    root = Path(os.environ.get("FATHI_BENCHMARK_ROOT", Path.cwd())).expanduser().resolve()
    context_path = resolve(root, args.context)
    profile_path = resolve(root, args.benchmark_spec)

    context = json.loads(context_path.read_text(encoding="utf-8"))
    profile = json.loads(profile_path.read_text(encoding="utf-8"))

    strict_workspace = resolve(root, context["strict_forward_workspace"])
    output_root = resolve(root, context["output_adjoint_batches_dir"])
    residual_value = context.get("residual_h5")
    if residual_value:
        residual_h5 = resolve(root, residual_value)
    else:
        residual_h5 = resolve(
            root,
            Path(context["work_root"])
            / "residual_sources"
            / "454B_strict_residual_timeseries.h5",
        )

    station_spec = profile["receivers"]["strict_full_grid"]
    physical_spec = profile["receivers"]["physical"]
    expected_receivers = int(physical_spec["count"])
    expected_stations = int(station_spec["count"])
    expected_partitions = int(profile["sem3d_mesh"]["partition_count"])
    station_path = strict_workspace / station_spec["file"]

    require(strict_workspace.is_dir(), f"Missing strict workspace: {strict_workspace}")
    require((strict_workspace / "input.spec").is_file(), "Missing strict input.spec")
    require((strict_workspace / "sem").is_dir(), "Missing strict sem directory")
    require((strict_workspace / "mat/h5/Mat_0_Kappa.h5").is_file(), "Missing strict material H5")
    require(residual_h5.is_file(), f"Missing residual H5: {residual_h5}")
    require(station_path.is_file(), f"Missing strict stations: {station_path}")

    records = load_residual_records(residual_h5)
    stations = read_station_lines(station_path)
    ranges = split_ranges(len(stations), args.batch_count)
    mesh_count = len(list((strict_workspace / "sem").glob("mesh4spec.*.h5")))

    require(len(records) == expected_receivers, f"Residual receivers {len(records)} != {expected_receivers}")
    require(len(stations) == expected_stations, f"Strict stations {len(stations)} != {expected_stations}")
    require(mesh_count == expected_partitions, f"Mesh partitions {mesh_count} != {expected_partitions}")

    transition = context["transition"]
    batch_station_counts = [end - start for start, end in ranges]

    print("FULL ADJOINT PREPARATION FROM CURRENT STRICT FORWARD")
    print("====================================================")
    print(f"context = {context_path}")
    print(f"benchmark_spec = {profile_path}")
    print(f"transition = {transition}")
    print(f"strict_workspace = {strict_workspace}")
    print(f"residual_h5 = {residual_h5}")
    print(f"output_root = {output_root}")
    print(f"receiver_count = {len(records)}")
    print(f"station_count = {len(stations)}")
    print(f"batch_count_per_component = {args.batch_count}")
    print(f"batch_station_counts = {batch_station_counts}")
    print(f"mesh_partition_count = {mesh_count}")
    print(f"execute = {args.execute}")
    print(f"force = {args.force}")

    if not args.execute:
        print("RESULT = PASS_FULL_ADJOINT_PREPARATION_PLAN")
        return

    if output_root.exists() or output_root.is_symlink():
        if not args.force:
            raise RuntimeError(f"Output exists: {output_root}; inspect it or use --force")
        backup = output_root.parent / (
            f"{output_root.name}.backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )
        shutil.move(str(output_root), str(backup))
        print(f"backup = {backup}")

    staging = output_root.parent / (
        f".{output_root.name}.staging_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)

    input_template = (strict_workspace / "input.spec").read_text(encoding="utf-8")
    workspace_records = []

    try:
        for component, direction in DIRECTIONS.items():
            component_index = "xyz".index(component)
            for batch_index, (start, end) in enumerate(ranges):
                batch_name = f"batch_{batch_index:03d}"
                workspace = staging / component / batch_name
                copy_static_workspace(strict_workspace, workspace, symlink_sem=True)

                batch_stations = stations[start:end]
                (workspace / station_spec["file"]).write_text(
                    "\n".join(batch_stations) + "\n",
                    encoding="utf-8",
                )

                blocks = []
                for record in records:
                    filename = f"s{record['rid']}{component}.txt"
                    source_array = np.column_stack(
                        [record["time"], record["source"][:, component_index]]
                    )
                    np.savetxt(workspace / filename, source_array, fmt="%.16e")
                    blocks.append(source_block(record["position"], direction, filename))

                input_text = replace_run_name(
                    input_template,
                    f"{transition}_adjoint_{component}_{batch_name}",
                )
                input_text = re.sub(
                    r"save_snap\s*=\s*(?:true|false)\s*;",
                    "save_snap = false;",
                    input_text,
                    count=1,
                )
                input_text = set_dudx(input_text, True)
                input_text = replace_sources(input_text, blocks)
                (workspace / "input.spec").write_text(input_text, encoding="utf-8")

                marker = {
                    "created": datetime.now().isoformat(),
                    "transition": transition,
                    "component": component,
                    "batch": batch_name,
                    "workspace": str(output_root / component / batch_name),
                    "template": str(strict_workspace),
                    "residual_h5": str(residual_h5),
                    "source_count": len(records),
                    "station_count": len(batch_stations),
                    "station_global_start": start,
                    "station_global_end_exclusive": end,
                    "mesh_partition_count": len(
                        list((workspace / "sem").glob("mesh4spec.*.h5"))
                    ),
                    "result": "PASS",
                }
                write_json(workspace / "FULL_ADJOINT_PREPARATION.json", marker)

                require((workspace / "input.spec").is_file(), f"Missing input.spec: {workspace}")
                require(
                    (workspace / "mat/h5/Mat_0_Kappa.h5").is_file(),
                    f"Missing material: {workspace}",
                )
                require(
                    len(read_station_lines(workspace / station_spec["file"]))
                    == len(batch_stations),
                    f"Bad station count: {workspace}",
                )
                require(
                    len(list(workspace.glob(f"s*{component}.txt"))) == len(records),
                    f"Bad source count: {workspace}",
                )
                require(
                    len(list((workspace / "sem").glob("mesh4spec.*.h5")))
                    == expected_partitions,
                    f"Bad mesh count: {workspace}",
                )

                workspace_records.append(marker)

        require(
            len(workspace_records) == len(DIRECTIONS) * args.batch_count,
            "Wrong workspace count",
        )
        staging.replace(output_root)
    except Exception:
        if staging.exists():
            shutil.rmtree(staging)
        raise

    report = {
        "created": datetime.now().isoformat(),
        "context": str(context_path),
        "benchmark_spec": str(profile_path),
        "transition": transition,
        "strict_workspace": str(strict_workspace),
        "residual_h5": str(residual_h5),
        "output_root": str(output_root),
        "receiver_count": len(records),
        "station_count": len(stations),
        "batch_count_per_component": args.batch_count,
        "batch_station_counts": batch_station_counts,
        "mesh_partition_count": mesh_count,
        "workspace_count": len(workspace_records),
        "workspaces": workspace_records,
        "result": "PASS",
    }
    report_path = resolve(
        root,
        Path(context["work_root"])
        / "residual_sources"
        / "FULL_ADJOINT_PREPARATION_FROM_STRICT.json",
    )
    write_json(report_path, report)

    print(f"workspace_count = {len(workspace_records)}")
    print(f"report = {report_path}")
    print("RESULT = PASS_FULL_ADJOINT_PREPARATION_FROM_STRICT")


if __name__ == "__main__":
    main()
