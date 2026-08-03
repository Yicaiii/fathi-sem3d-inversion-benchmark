#!/usr/bin/env python3
"""Prepare a strict full-grid forward workspace from the accepted parent.

This fresh-clone path has no dependency on historical iter_005/006/007/008
workspace templates. It copies only deterministic static solver inputs from
the accepted parent, regenerates the canonical 38,440-point station grid, and
leaves final forward-operator enforcement to enforce_forward_operator.py.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from datetime import datetime
from pathlib import Path

from scripts.bootstrap.generate_sem3d_workspace import (
    generate_stations,
    load_json,
)
from scripts.fathi_benchmark.runtime_paths import repository_root


ROOT = repository_root()


def resolve(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--context", required=True)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    context_path = resolve(args.context)
    context = load_json(context_path)

    parent = resolve(context["input_accepted_dir"])
    workspace = resolve(context["strict_forward_workspace"])
    profile_path = resolve(
        context.get(
            "benchmark_profile_config",
            "configs/fathi_reduced_3x3_12p5.json",
        )
    )
    profile = load_json(profile_path)

    required_parent_files = [
        parent / "mesh.input",
        parent / "mat.dat",
        parent / "mater.in",
        parent / "material.input",
        parent / "material.spec",
        parent / "input.spec",
        parent / profile["forward_operator"]["source_time_function_file"],
        parent / "mat/h5/Mat_0_Kappa.h5",
        parent / "mat/h5/Mat_0_Mu.h5",
        parent / "mat/h5/Mat_0_Density.h5",
    ]
    missing = [path for path in required_parent_files if not path.is_file()]
    require(
        not missing,
        "Accepted parent is missing required static inputs: "
        + ", ".join(str(path) for path in missing),
    )

    mesh_stem = profile["sem3d_mesh"]["mesh_file_stem"]
    expected_partitions = int(profile["sem3d_mesh"]["partition_count"])
    mesh_files = sorted((parent / "sem").glob(f"{mesh_stem}.*.h5"))
    require(
        len(mesh_files) == expected_partitions,
        f"Expected {expected_partitions} mesh partitions in {parent / 'sem'}, "
        f"got {len(mesh_files)}",
    )

    if workspace.exists():
        require(
            args.force,
            f"Strict-forward workspace already exists: {workspace}. "
            "Use --force to replace it.",
        )
        shutil.rmtree(workspace)

    workspace.mkdir(parents=True, exist_ok=False)

    root_files = [
        "mesh.input",
        "mat.dat",
        "mater.in",
        "material.input",
        "material.spec",
        "input.spec",
        profile["forward_operator"]["source_time_function_file"],
    ]
    for name in root_files:
        shutil.copy2(parent / name, workspace / name)

    shutil.copytree(parent / "mat", workspace / "mat")
    shutil.copytree(parent / "sem", workspace / "sem")

    stations_text, stations = generate_stations(
        profile,
        receiver_role="strict_full_grid",
    )
    station_name = profile["receivers"]["strict_full_grid"]["file"]
    station_path = workspace / station_name
    station_path.write_text(stations_text, encoding="utf-8")

    expected_hash = profile["receivers"]["strict_full_grid"]["sha256"]
    actual_hash = sha256_bytes(stations_text.encode("utf-8"))
    require(actual_hash == expected_hash, "Generated station hash mismatch")

    runtime_names = ("traces", "snapshots", "logs", "res", "prot")
    runtime_absent = {
        name: not (workspace / name).exists()
        for name in runtime_names
    }

    checks = {
        "station_count": (
            int(stations.shape[0])
            == int(profile["receivers"]["strict_full_grid"]["count"])
        ),
        "station_sha256": actual_hash == expected_hash,
        "mesh_partition_count": (
            len(list((workspace / "sem").glob(f"{mesh_stem}.*.h5")))
            == expected_partitions
        ),
        "kappa_h5": (workspace / "mat/h5/Mat_0_Kappa.h5").is_file(),
        "mu_h5": (workspace / "mat/h5/Mat_0_Mu.h5").is_file(),
        "density_h5": (workspace / "mat/h5/Mat_0_Density.h5").is_file(),
        "runtime_outputs_absent": all(runtime_absent.values()),
    }
    require(all(checks.values()), f"Strict workspace checks failed: {checks}")

    marker = {
        "created": datetime.now().isoformat(),
        "purpose": "fresh deterministic strict full-grid forward workspace",
        "context": str(context_path),
        "parent_accepted_dir": str(parent),
        "destination": str(workspace),
        "benchmark_profile_config": str(profile_path),
        "station_file": str(station_path),
        "station_count": int(stations.shape[0]),
        "station_sha256": actual_hash,
        "mesh_partition_count": expected_partitions,
        "runtime_outputs_absent": runtime_absent,
        "checks": checks,
        "scientific_status": (
            "STRICT_FORWARD_STATIC_INPUTS_PREPARED_FROM_ACCEPTED_PARENT"
        ),
    }
    marker_path = workspace / "STRICT_FORWARD_PREPARATION.json"
    marker_path.write_text(
        json.dumps(marker, indent=2) + "\n",
        encoding="utf-8",
    )

    print("PREPARE STRICT FORWARD FROM ACCEPTED PARENT")
    print("===========================================")
    print()
    print(f"context = {context_path}")
    print(f"parent = {parent}")
    print(f"workspace = {workspace}")
    print(f"stations = {stations.shape[0]}")
    print(f"station_sha256 = {actual_hash}")
    print(f"mesh_partitions = {expected_partitions}")
    print(f"marker = {marker_path}")
    print()
    print("RESULT = PASS")


if __name__ == "__main__":
    main()
