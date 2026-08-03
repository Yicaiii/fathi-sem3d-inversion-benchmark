from __future__ import annotations

from datetime import datetime
from pathlib import Path
import argparse
import hashlib
import json

import numpy as np
from scipy.sparse import load_npz, save_npz

from scripts.fathi_benchmark.runtime_paths import (
    repository_root,
    resolve_path,
)


ROOT = repository_root()

parser = argparse.ArgumentParser()
parser.add_argument(
    "--config",
    default=(
        "benchmark_fathi_strict/config/"
        "benchmark_config_reduced_full_domain_0p45s.json"
    ),
)
parser.add_argument(
    "--force",
    action="store_true",
)
args = parser.parse_args()

config_path = resolve_path(
    args.config,
    base=ROOT,
)

config = json.loads(
    config_path.read_text(encoding="utf-8")
)

profile_path = resolve_path(
    config["benchmark_profile_config"],
    base=ROOT,
)

profile = json.loads(
    profile_path.read_text(encoding="utf-8")
)

full_matrix_path = resolve_path(
    config["mtilde_full_matrix_path"],
    base=ROOT,
)

reduced_matrix_path = resolve_path(
    config["mtilde_matrix_path"],
    base=ROOT,
)

reduced_indices_path = resolve_path(
    config["mtilde_matrix_indices_path"],
    base=ROOT,
)

reduced_coords_path = resolve_path(
    config["mtilde_matrix_coords_path"],
    base=ROOT,
)

manifest_path = reduced_matrix_path.parent / (
    "reduced_mtilde_manifest.json"
)

outputs = [
    reduced_matrix_path,
    reduced_indices_path,
    reduced_coords_path,
    manifest_path,
]

existing = [
    path
    for path in outputs
    if path.exists()
]

if existing and not args.force:
    print(
        "Refusing to overwrite existing reduced "
        "Mtilde outputs:"
    )

    for path in existing:
        print(" ", path)

    print(
        "Re-run with --force only after checking "
        "the existing artifact."
    )

    raise SystemExit(3)

grid = profile["receivers"]["strict_full_grid"]

nx_control = int(grid["shape_zyx"][2])
ny_control = int(grid["shape_zyx"][1])
nz_control = int(grid["shape_zyx"][0])

x_values = (
    float(grid["x_start_m"])
    + float(grid["x_spacing_m"])
    * np.arange(nx_control)
)

y_values = (
    float(grid["y_start_m"])
    + float(grid["y_spacing_m"])
    * np.arange(ny_control)
)

z_values = (
    float(grid["z_start_m"])
    + float(grid["z_spacing_m"])
    * np.arange(nz_control)
)

coords = np.asarray(
    [
        (x, y, z)
        for z in z_values
        for y in y_values
        for x in x_values
    ],
    dtype=np.float64,
)

expected_n = int(
    config["interior_gradient_size"]
)

if coords.shape != (expected_n, 3):
    raise RuntimeError(
        "Reduced coordinate count mismatch: "
        f"{coords.shape} vs {(expected_n, 3)}"
    )

nx_full, ny_full, nz_full = 33, 33, 41

x_full = np.linspace(
    -20.0,
    20.0,
    nx_full,
)

y_full = np.linspace(
    -20.0,
    20.0,
    ny_full,
)

z_full = np.linspace(
    0.0,
    -50.0,
    nz_full,
)

full_coords = np.asarray(
    [
        (x, y, z)
        for z in z_full
        for y in y_full
        for x in x_full
    ],
    dtype=np.float64,
)


def key(point) -> tuple[float, float, float]:
    return tuple(
        round(float(value), 8)
        for value in point
    )


full_map = {
    key(point): index
    for index, point in enumerate(full_coords)
}

missing_coords = [
    key(point)
    for point in coords
    if key(point) not in full_map
]

if missing_coords:
    raise RuntimeError(
        "Reduced coordinates are not present in "
        "the full material grid. First missing: "
        f"{missing_coords[:10]}"
    )

indices = np.asarray(
    [
        full_map[key(point)]
        for point in coords
    ],
    dtype=np.int64,
)

if np.unique(indices).size != expected_n:
    raise RuntimeError(
        "Reduced full-grid indices are not unique"
    )

mapped = full_coords[indices]

max_coord_diff = float(
    np.max(np.abs(mapped - coords))
)

if max_coord_diff > 1e-8:
    raise RuntimeError(
        "Coordinate mapping mismatch: "
        f"{max_coord_diff}"
    )

full_matrix = load_npz(
    full_matrix_path
).tocsr()

if full_matrix.shape != (
    full_coords.shape[0],
    full_coords.shape[0],
):
    raise RuntimeError(
        "Unexpected full Mtilde shape: "
        f"{full_matrix.shape}"
    )

reduced_matrix = full_matrix[
    indices, :
][
    :, indices
].tocsr()

if reduced_matrix.shape != (
    expected_n,
    expected_n,
):
    raise RuntimeError(
        "Unexpected reduced Mtilde shape: "
        f"{reduced_matrix.shape}"
    )

symmetry_delta = (
    reduced_matrix
    - reduced_matrix.transpose()
).tocsr()

max_symmetry_error = (
    float(np.max(np.abs(symmetry_delta.data)))
    if symmetry_delta.nnz
    else 0.0
)

if max_symmetry_error > 1e-12:
    raise RuntimeError(
        "Reduced Mtilde is not symmetric: "
        f"{max_symmetry_error}"
    )

if not np.all(
    np.isfinite(reduced_matrix.data)
):
    raise RuntimeError(
        "Reduced Mtilde contains non-finite values"
    )

if np.min(reduced_matrix.diagonal()) <= 0.0:
    raise RuntimeError(
        "Reduced Mtilde has a non-positive diagonal"
    )

reduced_matrix_path.parent.mkdir(
    parents=True,
    exist_ok=True,
)

save_npz(
    reduced_matrix_path,
    reduced_matrix,
)

np.save(
    reduced_indices_path,
    indices,
)

np.save(
    reduced_coords_path,
    coords,
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as handle:
        for chunk in iter(
            lambda: handle.read(1024 * 1024),
            b"",
        ):
            digest.update(chunk)

    return digest.hexdigest()


manifest = {
    "created": datetime.now().isoformat(),
    "config": str(config_path),
    "profile": str(profile_path),
    "full_matrix": {
        "path": str(full_matrix_path),
        "shape": list(full_matrix.shape),
        "nnz": int(full_matrix.nnz),
        "sha256": sha256(full_matrix_path),
    },
    "reduced_matrix": {
        "path": str(reduced_matrix_path),
        "shape": list(reduced_matrix.shape),
        "nnz": int(reduced_matrix.nnz),
        "sha256": sha256(reduced_matrix_path),
    },
    "indices": {
        "path": str(reduced_indices_path),
        "shape": list(indices.shape),
        "min": int(indices.min()),
        "max": int(indices.max()),
        "unique": int(np.unique(indices).size),
        "sha256": sha256(reduced_indices_path),
    },
    "coords": {
        "path": str(reduced_coords_path),
        "shape": list(coords.shape),
        "max_mapping_error": max_coord_diff,
        "sha256": sha256(reduced_coords_path),
    },
    "max_symmetry_error": max_symmetry_error,
    "sem3d_launched": False,
    "accepted_state_mutated": False,
    "result": "PASS",
}

manifest_path.write_text(
    json.dumps(manifest, indent=2) + "\n",
    encoding="utf-8",
)

print("Reduced Mtilde artifact")
print("========================")
print(f"full_matrix = {full_matrix_path}")
print(f"full_shape = {full_matrix.shape}")
print(f"reduced_matrix = {reduced_matrix_path}")
print(f"reduced_shape = {reduced_matrix.shape}")
print(f"reduced_nnz = {reduced_matrix.nnz}")
print(f"indices = {reduced_indices_path}")
print(f"coords = {reduced_coords_path}")
print(f"max_coord_diff = {max_coord_diff:.16e}")
print(
    "max_symmetry_error = "
    f"{max_symmetry_error:.16e}"
)
print("SEM3D launched = False")
print("accepted state mutated = False")
print("RESULT = PASS_REDUCED_MTILDE_BUILD")
