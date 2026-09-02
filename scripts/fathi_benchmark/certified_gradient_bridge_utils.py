"""Generic helpers for the certified external-gradient bridge."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from scripts.fathi_benchmark.runtime_paths import (
    iteration_runtime_paths,
    runtime_resolve_path,
)

def relative_error(a: float, b: float) -> float:
    return float(
        abs(a - b)
        / max(abs(a), abs(b), np.finfo(np.float64).tiny)
    )

def array_stats(value: np.ndarray) -> dict[str, object]:
    array = np.asarray(value, dtype=np.float64)
    return {
        "shape": list(array.shape),
        "finite": bool(np.all(np.isfinite(array))),
        "l2": float(np.linalg.norm(array)),
        "max_abs": float(np.max(np.abs(array))),
        "sum": float(np.sum(array)),
        "nonzero": int(np.count_nonzero(array)),
    }

def configured_path(value: str | Path, repo: Path) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute():
        return path.resolve()
    return runtime_resolve_path(
        path,
        repo_root=repo,
        prefer_existing_legacy=True,
    )

def material_grid_coordinates(
    config: dict[str, object],
) -> tuple[tuple[int, int, int], np.ndarray]:
    grid = config["material_grid"]
    if grid.get("array_order") != "field[iz, iy, ix]":
        raise RuntimeError("unsupported material-grid array order")
    if grid.get("flatten_order") != "C":
        raise RuntimeError("unsupported material-grid flatten order")
    shape = tuple(int(value) for value in grid["shape"])
    if len(shape) != 3:
        raise RuntimeError(f"invalid material-grid shape: {shape}")
    nz, ny, nx = shape
    domain = config["domain"]
    x = np.linspace(domain["x_min_m"], domain["x_max_m"], nx)
    y = np.linspace(domain["y_min_m"], domain["y_max_m"], ny)
    z = np.linspace(domain["z_min_m"], domain["z_max_m"], nz)
    coords = np.asarray(
        [(xv, yv, zv) for zv in z for yv in y for xv in x],
        dtype=np.float64,
    )
    return shape, coords

def trilinear_transpose(
    values: np.ndarray,
    xyz: np.ndarray,
    shape: tuple[int, int, int],
    bounds: tuple[float, float, float, float, float, float],
    batch_size: int,
) -> np.ndarray:
    """Transpose of the production C-order trilinear sampler."""
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    xyz = np.asarray(xyz, dtype=np.float64).reshape(-1, 3)
    if values.shape != (xyz.shape[0],):
        raise RuntimeError("trilinear transpose value/coordinate mismatch")
    nz, ny, nx = shape
    xmin, xmax, ymin, ymax, zmin, zmax = bounds
    output = np.zeros(int(np.prod(shape)), dtype=np.float64)

    for start in range(0, len(values), batch_size):
        end = min(start + batch_size, len(values))
        points = xyz[start:end]
        local_values = values[start:end]
        x = np.clip(points[:, 0], xmin, xmax)
        y = np.clip(points[:, 1], ymin, ymax)
        z = np.clip(points[:, 2], zmin, zmax)
        fx = (x - xmin) / (xmax - xmin) * (nx - 1)
        fy = (y - ymin) / (ymax - ymin) * (ny - 1)
        fz = (z - zmin) / (zmax - zmin) * (nz - 1)
        ix0 = np.floor(fx).astype(np.int64)
        iy0 = np.floor(fy).astype(np.int64)
        iz0 = np.floor(fz).astype(np.int64)
        ix1 = np.minimum(ix0 + 1, nx - 1)
        iy1 = np.minimum(iy0 + 1, ny - 1)
        iz1 = np.minimum(iz0 + 1, nz - 1)
        tx = fx - ix0
        ty = fy - iy0
        tz = fz - iz0

        for iz, wz in ((iz0, 1.0 - tz), (iz1, tz)):
            for iy, wy in ((iy0, 1.0 - ty), (iy1, ty)):
                for ix, wx in ((ix0, 1.0 - tx), (ix1, tx)):
                    flat = (iz * ny + iy) * nx + ix
                    np.add.at(output, flat, local_values * wz * wy * wx)
    return output

def build_solid_row_map(
    solid_xyz: np.ndarray,
    solid_conn: np.ndarray,
    row_sem_element: np.ndarray,
    row_xyz: np.ndarray,
    batch_size: int,
) -> tuple[np.ndarray, dict[str, object]]:
    """Map stored solid element-local order into the rows of P_solid."""
    local_xyz = solid_xyz[solid_conn]
    element_count, local_count = solid_conn.shape
    row_count = element_count * local_count
    if row_sem_element.shape != (row_count,) or row_xyz.shape != (
        row_count,
        3,
    ):
        raise RuntimeError("invalid solid interpolation row metadata")
    counts = np.bincount(row_sem_element, minlength=element_count)
    if not np.all(counts == local_count):
        raise RuntimeError("solid interpolation rows per element mismatch")
    p_centers = np.zeros((element_count, 3), dtype=np.float64)
    np.add.at(p_centers, row_sem_element, row_xyz)
    p_centers /= counts[:, None]
    solid_centers = np.mean(local_xyz, axis=1)
    direct_center_error = float(np.max(np.abs(p_centers - solid_centers)))
    if direct_center_error <= 1e-12:
        p_to_solid = np.arange(element_count, dtype=np.int64)
        mode = "direct_sem_element_order"
    else:
        lookup = {
            tuple(np.round(row, 12)): index
            for index, row in enumerate(solid_centers)
        }
        if len(lookup) != element_count:
            raise RuntimeError("duplicate solid element centers")
        p_to_solid = np.asarray(
            [lookup[tuple(np.round(row, 12))] for row in p_centers],
            dtype=np.int64,
        )
        mode = "coordinate_center_mapping"
    mapped_center_error = float(
        np.max(np.abs(p_centers - solid_centers[p_to_solid]))
    )
    row_map = np.empty(row_count, dtype=np.int64)
    local_coordinate_error = 0.0
    for start in range(0, row_count, batch_size):
        end = min(start + batch_size, row_count)
        solid_element = p_to_solid[row_sem_element[start:end]]
        candidates = local_xyz[solid_element]
        distance = np.max(
            np.abs(candidates - row_xyz[start:end, None, :]), axis=2
        )
        local_id = np.argmin(distance, axis=1)
        local_coordinate_error = max(
            local_coordinate_error,
            float(np.max(distance[np.arange(end - start), local_id])),
        )
        row_map[start:end] = solid_element * local_count + local_id
    unique_count = int(np.unique(row_map).size)
    bijective = bool(
        unique_count == row_count
        and row_map.min() == 0
        and row_map.max() == row_count - 1
    )
    return row_map, {
        "mode": mode,
        "direct_center_error": direct_center_error,
        "mapped_center_error": mapped_center_error,
        "local_coordinate_error": local_coordinate_error,
        "unique_count": unique_count,
        "bijective": bijective,
    }
