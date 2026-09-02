"""HISTORICAL_ONLY audit for immutable bytes used by the completed bridge."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


ROUTE_CLASSIFICATION = "HISTORICAL_ONLY"
REUSE_CLASSIFICATION = "HISTORICAL_CERTIFIED_ASSET_REUSE"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def directory_asset(
    *,
    asset_id: str,
    source_run: str,
    source_path: Path,
    bridge_reference_path: Path,
    current_role: str,
    reuse_reason: str,
) -> dict[str, Any]:
    files = []
    for path in sorted(value for value in source_path.rglob("*") if value.is_file()):
        files.append(
            {
                "relative_path": path.relative_to(source_path).as_posix(),
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    digest = hashlib.sha256()
    for item in files:
        digest.update(
            f"{item['sha256']}  {item['relative_path']}\n".encode("utf-8")
        )
    signature = digest.hexdigest()
    if not bridge_reference_path.is_symlink():
        raise RuntimeError(f"bridge reference is not a symlink: {bridge_reference_path}")
    if bridge_reference_path.resolve() != source_path.resolve():
        raise RuntimeError(f"bridge reference target mismatch: {bridge_reference_path}")
    return {
        "asset_id": asset_id,
        "classification": REUSE_CLASSIFICATION,
        "mutable": False,
        "source_run": source_run,
        "source_path": str(source_path.resolve()),
        "bridge_reference_path": str(bridge_reference_path.absolute()),
        "kind": "directory",
        "sha256": signature,
        "content_signature_sha256": signature,
        "file_count": len(files),
        "files": files,
        "current_role": current_role,
        "reuse_reason": reuse_reason,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", required=True)
    parser.add_argument("--engine-config", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    repo = Path(args.repo).expanduser().resolve()
    config = json.loads(
        Path(args.engine_config).expanduser().resolve().read_text(encoding="utf-8")
    )
    source_run = str(config["historical_run_id"])
    current_run = str(config["run_id"])
    completed_transition = "iter_000_to_iter_001"
    source_transition = repo / "results" / source_run / completed_transition
    current_transition = repo / "results" / current_run / completed_transition
    compat = current_transition / "current_t052_external_bridge" / "compat_repo"
    compat_transition = (
        compat / "results" / current_run / completed_transition
    )

    assets = [
        directory_asset(
            asset_id="exact_spatial_operator",
            source_run=source_run,
            source_path=source_transition / "exact_spatial_operator",
            bridge_reference_path=compat_transition / "exact_spatial_operator",
            current_role=(
                "immutable SEM-GLL/H5 interpolation transpose and spatial "
                "quadrature/index operator"
            ),
            reuse_reason=(
                "the CURRENT reproduction uses the identical certified S43 mesh, "
                "GLL topology, control grid, and active-index convention"
            ),
        ),
        directory_asset(
            asset_id="real_s43_compact_topology",
            source_run=source_run,
            source_path=source_transition / "real_s43_compact_topology",
            bridge_reference_path=compat_transition / "real_s43_compact_topology",
            current_role=(
                "immutable solid/PML compact connectivity, coordinates, interface "
                "maps, element IDs, and PML region codes"
            ),
            reuse_reason=(
                "these arrays depend only on the unchanged certified S43 mesh and "
                "partition topology, not on mutable CURRENT material/objective state"
            ),
        ),
    ]

    station_source = (
        repo
        / "data"
        / source_run
        / "iter_001"
        / "forward_dudx_mgcap_full_batches"
        / "strict_full_forward_000"
        / "stations.txt"
    )
    station_current = (
        repo
        / "data"
        / "reproduction"
        / current_run
        / "iterations"
        / "iter_001"
        / "forward_dudx_mgcap_full_batches"
        / "strict_full_forward_000"
        / "stations.txt"
    )
    station_hash = sha256_file(station_source)
    if sha256_file(station_current) != station_hash:
        raise RuntimeError("CURRENT strict station bytes differ from historical source")
    assets.append(
        {
            "asset_id": "strict_full_grid_station_geometry",
            "classification": REUSE_CLASSIFICATION,
            "mutable": False,
            "source_run": source_run,
            "source_path": str(station_source.resolve()),
            "current_copy_path": str(station_current.resolve()),
            "kind": "file",
            "size_bytes": station_source.stat().st_size,
            "sha256": station_hash,
            "current_role": "immutable strict full-grid gradient station geometry",
            "reuse_reason": (
                "the strict grid is a mesh/control-geometry asset; it is not the "
                "physical receiver set and does not depend on the material iterate"
            ),
        }
    )
    payload = {
        "schema_version": 1,
        "result": "PASS_HISTORICAL_IMMUTABLE_OPERATOR_ASSET_AUDIT",
        "classification": REUSE_CLASSIFICATION,
        "current_run": current_run,
        "source_run": source_run,
        "completed_transition": completed_transition,
        "asset_count": len(assets),
        "assets": assets,
        "audit": {
            "historical_bridge_symlink_targets": [
                str((source_transition / "exact_spatial_operator").resolve()),
                str((source_transition / "real_s43_compact_topology").resolve()),
            ],
            "copied_historical_geometry": [str(station_source.resolve())],
            "mutable_current_paths": False,
            "enumeration_complete": True,
        },
        "simulation_runs": 0,
        "sem3d_runs": 0,
    }
    output = Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(output)


if __name__ == "__main__":
    main()
