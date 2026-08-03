from __future__ import annotations

from argparse import ArgumentParser
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator
import hashlib
import json
import os
import shutil
import tempfile
import time

import numpy as np
from scipy import sparse
from scipy.sparse import load_npz, save_npz


def repository_root() -> Path:
    import subprocess

    return Path(
        subprocess.check_output(
            ["git", "rev-parse", "--show-toplevel"],
            text=True,
        ).strip()
    ).resolve()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def resolve_path(value: str | Path, base: Path) -> Path:
    path = Path(value).expanduser()

    if not path.is_absolute():
        path = base / path

    return path.resolve()


def runtime_root(repo: Path) -> Path:
    value = os.environ.get("FATHI_RUNTIME_ROOT")

    if value:
        return resolve_path(value, repo)

    return repo


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)

    return digest.hexdigest()


def csr_content_sha256(matrix: sparse.csr_matrix) -> str:
    matrix = matrix.tocsr()
    digest = hashlib.sha256()

    for array in (
        matrix.indptr.astype(np.int64, copy=False),
        matrix.indices.astype(np.int64, copy=False),
        matrix.data.astype(np.float64, copy=False),
    ):
        digest.update(array.tobytes(order="C"))

    return digest.hexdigest()


def one_dimensional_q1_mass(
    node_count: int,
    spacing: float,
) -> sparse.csr_matrix:
    if node_count < 2:
        raise RuntimeError("Q1 mass matrix requires at least two nodes")

    if spacing <= 0.0:
        raise RuntimeError("Q1 grid spacing must be positive")

    diagonal = np.full(
        node_count,
        2.0 * spacing / 3.0,
        dtype=np.float64,
    )
    diagonal[0] = spacing / 3.0
    diagonal[-1] = spacing / 3.0

    off_diagonal = np.full(
        node_count - 1,
        spacing / 6.0,
        dtype=np.float64,
    )

    return sparse.diags(
        [off_diagonal, diagonal, off_diagonal],
        offsets=[-1, 0, 1],
        format="csr",
        dtype=np.float64,
    )


def structured_grid(
    spec: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    grid = spec["full_grid"]
    nz, ny, nx = [
        int(value)
        for value in grid["shape_zyx"]
    ]

    x = np.linspace(
        float(grid["x_min_m"]),
        float(grid["x_max_m"]),
        nx,
        dtype=np.float64,
    )
    y = np.linspace(
        float(grid["y_min_m"]),
        float(grid["y_max_m"]),
        ny,
        dtype=np.float64,
    )
    z = np.linspace(
        float(grid["z_top_m"]),
        float(grid["z_bottom_m"]),
        nz,
        dtype=np.float64,
    )

    zz, yy, xx = np.meshgrid(
        z,
        y,
        x,
        indexing="ij",
    )

    coords = np.column_stack(
        [
            xx.ravel(order="C"),
            yy.ravel(order="C"),
            zz.ravel(order="C"),
        ]
    ).astype(np.float64)

    return x, y, z, coords


def generate_full_matrix(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
) -> sparse.csr_matrix:
    mx = one_dimensional_q1_mass(
        len(x),
        abs(float(x[1] - x[0])),
    )
    my = one_dimensional_q1_mass(
        len(y),
        abs(float(y[1] - y[0])),
    )
    mz = one_dimensional_q1_mass(
        len(z),
        abs(float(z[1] - z[0])),
    )

    return sparse.kron(
        mz,
        sparse.kron(my, mx, format="csr"),
        format="csr",
    ).astype(np.float64)


def validate_matrix_and_coords(
    matrix: sparse.csr_matrix,
    coords: np.ndarray,
    label: str,
) -> None:
    if matrix.shape[0] != matrix.shape[1]:
        raise RuntimeError(f"{label} matrix is not square: {matrix.shape}")

    if coords.ndim != 2 or coords.shape[1] < 3:
        raise RuntimeError(f"{label} coordinates are invalid: {coords.shape}")

    if matrix.shape[0] != coords.shape[0]:
        raise RuntimeError(
            f"{label} matrix/coordinate mismatch: "
            f"{matrix.shape} versus {coords.shape}"
        )

    if not np.all(np.isfinite(matrix.data)):
        raise RuntimeError(f"{label} matrix contains non-finite values")

    if np.min(matrix.diagonal()) <= 0.0:
        raise RuntimeError(f"{label} matrix diagonal is not positive")


def map_active_indices(
    full_coords: np.ndarray,
    active_coords: np.ndarray,
    decimals: int,
) -> np.ndarray:
    rounded_full = np.round(
        full_coords[:, :3],
        decimals=decimals,
    )
    rounded_active = np.round(
        active_coords[:, :3],
        decimals=decimals,
    )

    lookup: dict[tuple[float, float, float], int] = {}

    for index, row in enumerate(rounded_full):
        key = tuple(float(value) for value in row)

        if key in lookup:
            raise RuntimeError(f"Duplicate full-grid coordinate: {key}")

        lookup[key] = index

    indices = []

    for row in rounded_active:
        key = tuple(float(value) for value in row)

        if key not in lookup:
            raise RuntimeError(
                "Active control coordinate is absent from the full grid: "
                f"{key}"
            )

        indices.append(lookup[key])

    result = np.asarray(indices, dtype=np.int64)

    if np.unique(result).size != result.size:
        raise RuntimeError("Active control indices are not unique")

    return result


@contextmanager
def artifact_lock(
    root: Path,
    timeout_seconds: float = 120.0,
) -> Iterator[None]:
    lock_path = root.parent / f".{root.name}.lock"
    root.parent.mkdir(parents=True, exist_ok=True)
    start = time.monotonic()
    descriptor = None

    while descriptor is None:
        try:
            descriptor = os.open(
                lock_path,
                os.O_CREAT | os.O_EXCL | os.O_WRONLY,
            )
        except FileExistsError:
            if time.monotonic() - start > timeout_seconds:
                raise TimeoutError(
                    f"Timed out waiting for artifact lock: {lock_path}"
                )

            time.sleep(0.25)

    try:
        os.write(descriptor, str(os.getpid()).encode())
        yield
    finally:
        os.close(descriptor)
        lock_path.unlink(missing_ok=True)


def atomic_bundle_write(
    root: Path,
    matrix: sparse.csr_matrix,
    coords: np.ndarray,
    manifest: dict[str, Any],
    indices: np.ndarray | None = None,
) -> None:
    root.parent.mkdir(parents=True, exist_ok=True)

    temporary = Path(
        tempfile.mkdtemp(
            prefix=f"{root.name}.tmp.",
            dir=root.parent,
        )
    )

    try:
        matrix_path = temporary / "Mtilde.npz"
        coords_path = temporary / "coords.npy"
        manifest_path = temporary / "manifest.json"

        save_npz(matrix_path, matrix.tocsr())
        np.save(coords_path, coords)

        if indices is not None:
            np.save(
                temporary / "active_indices.npy",
                indices,
            )

        complete_manifest = {
            **manifest,
            "matrix_shape": list(matrix.shape),
            "nnz": int(matrix.nnz),
            "dtype": str(matrix.dtype),
            "coords_shape": list(coords.shape),
            "matrix_file_sha256": sha256_file(matrix_path),
            "matrix_content_sha256": csr_content_sha256(matrix),
            "coords_file_sha256": sha256_file(coords_path),
        }

        if indices is not None:
            complete_manifest["indices_file_sha256"] = sha256_file(
                temporary / "active_indices.npy"
            )

        manifest_path.write_text(
            json.dumps(
                complete_manifest,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )

        if root.exists():
            shutil.rmtree(root)

        temporary.rename(root)
    except Exception:
        shutil.rmtree(
            temporary,
            ignore_errors=True,
        )
        raise


def compare_legacy_reduced(
    config: dict[str, Any],
    repo: Path,
    matrix: sparse.csr_matrix,
    coords: np.ndarray,
    indices: np.ndarray,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "legacy_matrix_checked": False,
        "legacy_coords_checked": False,
        "legacy_indices_checked": False,
    }

    matrix_value = config.get("mtilde_matrix_path")

    if matrix_value:
        legacy_path = resolve_path(matrix_value, repo)

        if legacy_path.is_file():
            legacy = load_npz(legacy_path).tocsr()

            if legacy.shape != matrix.shape:
                raise RuntimeError(
                    "Legacy reduced matrix shape differs from generated artifact: "
                    f"{legacy.shape} versus {matrix.shape}"
                )

            difference = (legacy - matrix).tocsr()
            max_abs = (
                float(np.max(np.abs(difference.data)))
                if difference.nnz
                else 0.0
            )

            if max_abs > 1.0e-12:
                raise RuntimeError(
                    "Legacy reduced matrix differs from generated artifact: "
                    f"max_abs_difference={max_abs}"
                )

            result["legacy_matrix_checked"] = True
            result["legacy_matrix_path"] = str(legacy_path)
            result["legacy_matrix_max_abs_difference"] = max_abs

    coords_value = config.get("mtilde_matrix_coords_path")

    if coords_value:
        legacy_path = resolve_path(coords_value, repo)

        if legacy_path.is_file():
            legacy = np.load(legacy_path)

            if legacy.shape != coords.shape:
                raise RuntimeError(
                    "Legacy coordinate shape differs from generated artifact: "
                    f"{legacy.shape} versus {coords.shape}"
                )

            if not np.allclose(
                legacy[:, :3],
                coords[:, :3],
                atol=1.0e-10,
                rtol=0.0,
            ):
                raise RuntimeError(
                    "Legacy reduced coordinates differ from generated artifact"
                )

            result["legacy_coords_checked"] = True
            result["legacy_coords_path"] = str(legacy_path)

    indices_value = config.get("mtilde_matrix_indices_path")

    if indices_value:
        legacy_path = resolve_path(indices_value, repo)

        if legacy_path.is_file():
            legacy = np.load(legacy_path).astype(
                np.int64,
                copy=False,
            )

            if not np.array_equal(legacy, indices):
                raise RuntimeError(
                    "Legacy reduced indices differ from generated artifact"
                )

            result["legacy_indices_checked"] = True
            result["legacy_indices_path"] = str(legacy_path)

    return result


def path_for_config(path: Path, repo: Path) -> str:
    try:
        return path.relative_to(repo).as_posix()
    except ValueError:
        return str(path)


def parser() -> ArgumentParser:
    result = ArgumentParser(
        description="Ensure, validate, and register Mtilde artifacts",
    )
    result.add_argument("--config", required=True)
    result.add_argument("--context", required=True)
    result.add_argument("--execute", action="store_true")
    result.add_argument("--force", action="store_true")
    return result


def main() -> int:
    args = parser().parse_args()
    repo = repository_root()
    config_path = resolve_path(args.config, repo)
    context_path = resolve_path(args.context, repo)

    config = load_json(config_path)
    context = load_json(context_path)

    spec = config.get("mtilde_artifact")

    if not isinstance(spec, dict):
        raise SystemExit(
            "Config is missing the mtilde_artifact block. "
            "Run the phase-3 installer first."
        )

    runtime = runtime_root(repo)
    full_root = resolve_path(
        spec["full_artifact_root"],
        runtime,
    )
    active_root = resolve_path(
        spec["active_artifact_root"],
        runtime,
    )

    x, y, z, full_coords = structured_grid(spec)
    expected_full_n = full_coords.shape[0]

    station_file_value = spec.get("active_station_file")

    if station_file_value:
        active_station_file = resolve_path(
            station_file_value,
            repo,
        )
    else:
        forward_workspace = resolve_path(
            context["strict_forward_workspace"],
            repo,
        )
        active_station_file = (
            forward_workspace
            / spec.get(
                "active_station_filename",
                "stations.txt",
            )
        )

    active_coords = np.loadtxt(
        active_station_file,
        ndmin=2,
    )[:, :3].astype(np.float64)

    plan = {
        "config": str(config_path),
        "context": str(context_path),
        "runtime_root": str(runtime),
        "full_artifact_root": str(full_root),
        "active_artifact_root": str(active_root),
        "active_station_file": str(active_station_file),
        "full_coordinate_count": int(expected_full_n),
        "active_coordinate_count": int(active_coords.shape[0]),
    }

    if not args.execute:
        print("Mtilde artifact manager")
        print("========================")
        print(json.dumps(plan, indent=2))
        print("RESULT = PASS_MTILDE_PLAN")
        return 0

    with artifact_lock(full_root):
        full_matrix_path = full_root / "Mtilde.npz"
        full_coords_path = full_root / "coords.npy"

        valid_existing_full = (
            full_matrix_path.is_file()
            and full_coords_path.is_file()
            and not args.force
        )

        if valid_existing_full:
            full_matrix = load_npz(
                full_matrix_path
            ).tocsr()
            stored_full_coords = np.load(
                full_coords_path
            )

            validate_matrix_and_coords(
                full_matrix,
                stored_full_coords,
                "full",
            )

            if not np.allclose(
                stored_full_coords,
                full_coords,
                atol=1.0e-10,
                rtol=0.0,
            ):
                raise RuntimeError(
                    "Stored full Mtilde coordinates do not match config grid"
                )

            full_source = "validated_existing_artifact"
        else:
            legacy_full_value = (
                spec.get("legacy_full_matrix_path")
                or config.get("mtilde_full_matrix_path")
            )

            if legacy_full_value:
                legacy_full_path = resolve_path(
                    legacy_full_value,
                    repo,
                )
            else:
                legacy_full_path = Path()

            if legacy_full_value and legacy_full_path.is_file():
                full_matrix = load_npz(
                    legacy_full_path
                ).tocsr()
                full_source = "registered_legacy_full_matrix"
            else:
                full_matrix = generate_full_matrix(
                    x,
                    y,
                    z,
                )
                full_source = "generated_from_config_grid"

            validate_matrix_and_coords(
                full_matrix,
                full_coords,
                "full",
            )

            if full_matrix.shape != (
                expected_full_n,
                expected_full_n,
            ):
                raise RuntimeError(
                    "Full Mtilde shape does not match configured grid: "
                    f"{full_matrix.shape}"
                )

            atomic_bundle_write(
                full_root,
                full_matrix,
                full_coords,
                {
                    "artifact_id": spec["full_artifact_id"],
                    "artifact_kind": "full_structured_q1_consistent_mass",
                    "source": full_source,
                    "shape_zyx": spec["full_grid"]["shape_zyx"],
                    "axis_order": "zyx",
                    "flatten_order": "C",
                    "coordinate_order": "x-fastest",
                    "coordinate_columns": ["x", "y", "z"],
                    "grid": spec["full_grid"],
                },
            )

    full_matrix = load_npz(
        full_root / "Mtilde.npz"
    ).tocsr()
    full_coords = np.load(
        full_root / "coords.npy"
    )

    decimals = int(
        spec.get(
            "coordinate_decimals",
            10,
        )
    )

    active_indices = map_active_indices(
        full_coords,
        active_coords,
        decimals,
    )

    active_matrix = full_matrix[
        active_indices
    ][:, active_indices].tocsr()

    validate_matrix_and_coords(
        active_matrix,
        active_coords,
        "active",
    )

    comparison = compare_legacy_reduced(
        config,
        repo,
        active_matrix,
        active_coords,
        active_indices,
    )

    with artifact_lock(active_root):
        atomic_bundle_write(
            active_root,
            active_matrix,
            active_coords,
            {
                "artifact_id": spec["active_artifact_id"],
                "artifact_kind": "active_principal_submatrix",
                "parent_artifact_id": spec["full_artifact_id"],
                "parent_artifact_root": str(full_root),
                "active_station_file": str(active_station_file),
                "coordinate_decimals": decimals,
                "axis_order": "zyx",
                "flatten_order": "C",
                "coordinate_order": "x-fastest",
                "coordinate_columns": ["x", "y", "z"],
                "legacy_comparison": comparison,
            },
            indices=active_indices,
        )

    active_matrix_path = active_root / "Mtilde.npz"
    active_coords_path = active_root / "coords.npy"
    active_indices_path = active_root / "active_indices.npy"
    active_manifest_path = active_root / "manifest.json"

    context["mtilde_full_artifact_root"] = str(full_root)
    context["mtilde_matrix_path"] = str(active_matrix_path)
    context["mtilde_matrix_coords_path"] = str(active_coords_path)
    context["mtilde_matrix_indices_path"] = str(active_indices_path)
    context["mtilde_manifest_path"] = str(active_manifest_path)

    context_path.write_text(
        json.dumps(context, indent=2) + "\n",
        encoding="utf-8",
    )

    config["mtilde_matrix_path"] = path_for_config(
        active_matrix_path,
        repo,
    )
    config["mtilde_matrix_coords_path"] = path_for_config(
        active_coords_path,
        repo,
    )
    config["mtilde_matrix_indices_path"] = path_for_config(
        active_indices_path,
        repo,
    )
    config["mtilde_manifest_path"] = path_for_config(
        active_manifest_path,
        repo,
    )

    config_path.write_text(
        json.dumps(config, indent=2) + "\n",
        encoding="utf-8",
    )

    result = {
        **plan,
        "full_source": full_source,
        "full_matrix_shape": list(full_matrix.shape),
        "full_nnz": int(full_matrix.nnz),
        "full_matrix_content_sha256": csr_content_sha256(full_matrix),
        "active_matrix_shape": list(active_matrix.shape),
        "active_nnz": int(active_matrix.nnz),
        "active_indices_unique": (
            int(np.unique(active_indices).size)
            == int(active_indices.size)
        ),
        "active_matrix_content_sha256": csr_content_sha256(active_matrix),
        "legacy_comparison": comparison,
        "context_updated": True,
        "config_updated": True,
    }

    print("Mtilde artifact manager")
    print("========================")
    print(json.dumps(result, indent=2))
    print("RESULT = PASS_MTILDE_READY")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
