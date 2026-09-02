"""Validation for explicitly reused immutable historical operator assets."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from scripts.fathi_benchmark.iteration_context import (
    HISTORICAL_REUSE_MARKER,
    validate_historical_asset_reference,
)
from scripts.fathi_benchmark.lbfgs_history import sha256_file


def directory_content_signature(files: list[Mapping[str, Any]]) -> str:
    digest = hashlib.sha256()
    for item in sorted(files, key=lambda value: str(value["relative_path"])):
        digest.update(
            f"{item['sha256']}  {item['relative_path']}\n".encode("utf-8")
        )
    return digest.hexdigest()


def _directory_files(source_path: Path) -> list[dict[str, Any]]:
    return [
        {
            "relative_path": path.relative_to(source_path).as_posix(),
            "size_bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in sorted(value for value in source_path.rglob("*") if value.is_file())
    ]


def _resolve_asset_path(
    asset: Mapping[str, Any],
    *,
    repository_root: Path,
    runtime_root: Path,
) -> Path:
    relative = Path(str(asset.get("source_path", "")))
    if relative.is_absolute():
        raise ValueError(
            f"portable historical source path must be relative: "
            f"{asset.get('asset_id')}"
        )
    bases = {
        "repository_root": repository_root,
        "runtime_root": runtime_root,
    }
    base_name = str(asset.get("path_base", ""))
    if base_name not in bases:
        raise ValueError(f"invalid historical asset path_base: {base_name}")
    base = bases[base_name]
    resolved = (base / relative).resolve()
    try:
        resolved.relative_to(base)
    except ValueError as exc:
        raise ValueError("historical asset path escapes its configured base") from exc
    return resolved


def validate_immutable_asset_manifest(
    path: str | Path,
    *,
    expected_source_run: str,
    repository_root: str | Path,
    runtime_root: str | Path | None = None,
    verify_bytes: bool = True,
) -> dict[str, Any]:
    source = Path(path).expanduser().resolve()
    repository = Path(repository_root).expanduser().resolve()
    runtime = (
        repository
        if runtime_root is None
        else Path(runtime_root).expanduser().resolve()
    )
    manifest = json.loads(source.read_text(encoding="utf-8"))
    assets = manifest.get("assets")
    if not isinstance(assets, list) or not assets:
        raise ValueError("immutable operator-asset manifest has no assets")
    asset_ids = set()
    for asset in assets:
        validate_historical_asset_reference(asset)
        if asset.get("source_run") != expected_source_run:
            raise ValueError(f"historical source-run mismatch: {asset.get('asset_id')}")
        asset_id = str(asset.get("asset_id", ""))
        if not asset_id or asset_id in asset_ids:
            raise ValueError(f"invalid or duplicate historical asset ID: {asset_id}")
        asset_ids.add(asset_id)
        source_path = _resolve_asset_path(
            asset,
            repository_root=repository,
            runtime_root=runtime,
        )
        if asset.get("kind") == "directory":
            if not source_path.is_dir():
                raise ValueError(f"historical directory asset is missing: {source_path}")
            if asset.get("sha256") != asset.get("content_signature_sha256"):
                raise ValueError(f"directory content signature mismatch: {asset_id}")
            if verify_bytes:
                files = _directory_files(source_path)
                if len(files) != int(asset.get("file_count", -1)):
                    raise ValueError(f"historical directory file count mismatch: {asset_id}")
                signature = directory_content_signature(files)
                if signature != asset.get("content_signature_sha256"):
                    raise ValueError(f"historical asset byte mismatch: {source_path}")
        elif asset.get("kind") == "file":
            if not source_path.is_file():
                raise ValueError(f"historical file asset is missing: {source_path}")
            if verify_bytes and sha256_file(source_path) != str(asset["sha256"]):
                raise ValueError(f"historical asset byte mismatch: {source_path}")
        else:
            raise ValueError(f"unsupported historical asset kind: {asset_id}")
        if not asset.get("current_role") or not asset.get("reuse_reason"):
            raise ValueError(f"historical asset lacks role/reuse reason: {asset_id}")
    required = {
        "exact_spatial_operator",
        "real_s43_compact_topology",
        "strict_full_grid_station_geometry",
    }
    if asset_ids != required:
        raise ValueError(
            f"historical asset enumeration incomplete: {asset_ids} != {required}"
        )
    if manifest.get("classification") != HISTORICAL_REUSE_MARKER:
        raise ValueError("manifest classification is not explicit historical reuse")
    manifest["resolved_path_bases"] = {
        "repository_root": str(repository),
        "runtime_root": str(runtime),
    }
    return manifest
