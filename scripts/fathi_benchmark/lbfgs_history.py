"""Provenance-checked construction of physical Mtilde L-BFGS history."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
from typing import Any, Mapping, Sequence

import h5py
import numpy as np
from scipy.sparse import load_npz

from scripts.fathi_benchmark.physical_space_optimizer import (
    CurvatureAudit,
    VectorPair,
    audit_curvature_pair,
    physical_curvature_pair,
)
from scripts.fathi_benchmark.iteration_context import IterationIdentity
from scripts.fathi_benchmark.current_pipeline_contracts import (
    registered_gradient_result,
)


def waiting_for_gradient_status(iteration: int) -> str:
    if int(iteration) < 0:
        raise ValueError("gradient iteration must be non-negative")
    return f"BLOCKED_WAITING_FOR_ITER{int(iteration):03d}_GRADIENT"


NO_HISTORY_REQUIRED = "NO_HISTORY_REQUIRED"
BLOCKED_WAITING_FOR_HISTORY_AUDIT = "BLOCKED_WAITING_FOR_HISTORY_AUDIT"
HISTORY_OUTCOME_ACCEPTED = "ACCEPTED"
HISTORY_OUTCOME_REJECTED = "REJECTED"
WAITING_FOR_CHILD_GRADIENT = waiting_for_gradient_status(1)
# Deprecated compatibility alias. Production code uses waiting_for_gradient_status.


class HistoryBuildBlocked(RuntimeError):
    def __init__(self, status: str, detail: str):
        super().__init__(detail)
        self.status = status
        self.detail = detail


def sha256_file(path: str | Path) -> str:
    source = Path(path)
    digest = hashlib.sha256()
    with source.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _resolve(repo: Path, value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (repo / path).resolve()


def _artifact_path(
    repo: Path, spec: Mapping[str, Any], *, name: str
) -> Path:
    if not isinstance(spec, Mapping) or not spec.get("path") or not spec.get("sha256"):
        raise ValueError(f"{name} requires path and sha256 provenance")
    path = _resolve(repo, str(spec["path"]))
    if not path.is_file():
        raise FileNotFoundError(f"missing {name}: {path}")
    actual = sha256_file(path)
    if actual != str(spec["sha256"]):
        raise ValueError(
            f"{name} SHA-256 mismatch: declared={spec['sha256']} actual={actual}"
        )
    return path


def _load_vector(
    repo: Path,
    spec: Mapping[str, Any],
    *,
    name: str,
    dtype=np.float64,
) -> tuple[np.ndarray, Path]:
    path = _artifact_path(repo, spec, name=name)
    value = np.asarray(np.load(path), dtype=dtype)
    if value.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional: {value.shape}")
    if not np.all(np.isfinite(value)):
        raise ValueError(f"{name} contains non-finite values")
    return value, path


def _load_coords(
    repo: Path, spec: Mapping[str, Any], *, name: str
) -> tuple[np.ndarray, Path]:
    path = _artifact_path(repo, spec, name=name)
    value = np.asarray(np.load(path), dtype=np.float64)
    if value.ndim != 2 or value.shape[1] != 3:
        raise ValueError(f"{name} must have shape (n, 3): {value.shape}")
    if not np.all(np.isfinite(value)):
        raise ValueError(f"{name} contains non-finite values")
    return value, path


@dataclass(frozen=True)
class CanonicalOrdering:
    active_indices: np.ndarray
    coordinates: np.ndarray
    active_h5_indices: np.ndarray | None = None

    def validate(self, *, vector_size: int, name: str) -> None:
        if self.active_indices.shape != (vector_size,):
            raise ValueError(
                f"{name} active index shape mismatch: {self.active_indices.shape}"
            )
        if self.coordinates.shape != (vector_size, 3):
            raise ValueError(
                f"{name} coordinate shape mismatch: {self.coordinates.shape}"
            )
        if np.unique(self.active_indices).size != vector_size:
            raise ValueError(f"{name} contains duplicate active indices")
        if self.active_h5_indices is not None:
            if self.active_h5_indices.shape != (vector_size,):
                raise ValueError(
                    f"{name} active H5 index shape mismatch: "
                    f"{self.active_h5_indices.shape}"
                )
            if np.unique(self.active_h5_indices).size != vector_size:
                raise ValueError(f"{name} contains duplicate active H5 indices")


def require_canonical_identity(
    left: CanonicalOrdering,
    right: CanonicalOrdering,
    *,
    left_name: str,
    right_name: str,
) -> None:
    if not np.array_equal(left.active_indices, right.active_indices):
        raise ValueError(
            f"canonical active index identity failed: {left_name} != {right_name}"
        )
    if not np.array_equal(left.coordinates, right.coordinates):
        raise ValueError(
            f"canonical coordinate identity failed: {left_name} != {right_name}"
        )
    if (
        left.active_h5_indices is not None
        and right.active_h5_indices is not None
        and not np.array_equal(left.active_h5_indices, right.active_h5_indices)
    ):
        raise ValueError(
            f"canonical active H5 index identity failed: "
            f"{left_name} != {right_name}"
        )


def _material_field(path: Path, dataset: str) -> np.ndarray:
    with h5py.File(path, "r") as handle:
        if dataset not in handle:
            raise ValueError(f"missing material dataset {dataset!r}: {path}")
        value = np.asarray(handle[dataset], dtype=np.float64)
    if not np.all(np.isfinite(value)):
        raise ValueError(f"non-finite material field: {path}")
    return value


def _load_model(
    repo: Path,
    spec: Mapping[str, Any],
    material_config: Mapping[str, Any],
    *,
    name: str,
) -> tuple[VectorPair, CanonicalOrdering, dict[str, Any]]:
    material_dir = _resolve(repo, str(spec.get("material_dir", "")))
    if not material_dir.is_dir():
        raise FileNotFoundError(f"missing {name} accepted material: {material_dir}")
    files = material_config.get("files")
    hashes = spec.get("material_sha256")
    if not isinstance(files, Mapping) or not isinstance(hashes, Mapping):
        raise ValueError(f"{name} requires configured files and material_sha256")
    dataset = str(material_config.get("dataset", ""))
    if not dataset:
        raise ValueError("material dataset is not configured")
    resolved = {}
    for component in ("kappa", "mu", "density"):
        filename = str(files[component])
        path = material_dir / filename
        expected = str(hashes.get(component, ""))
        if not path.is_file() or not expected:
            raise ValueError(f"{name} lacks {component} material provenance")
        actual = sha256_file(path)
        if actual != expected:
            raise ValueError(f"{name} {component} material SHA-256 mismatch")
        resolved[component] = path

    kappa = _material_field(resolved["kappa"], dataset)
    mu = _material_field(resolved["mu"], dataset)
    density = _material_field(resolved["density"], dataset)
    if kappa.shape != mu.shape or kappa.shape != density.shape:
        raise ValueError(f"{name} material shapes differ")
    active_h5, active_h5_path = _load_vector(
        repo,
        spec["active_h5_indices"],
        name=f"{name}.active_h5_indices",
        dtype=np.int64,
    )
    active, active_path = _load_vector(
        repo,
        spec["active_indices"],
        name=f"{name}.active_indices",
        dtype=np.int64,
    )
    coords, coords_path = _load_coords(
        repo, spec["coordinates"], name=f"{name}.coordinates"
    )
    full_count = kappa.size
    if (
        active_h5.size != active.size
        or np.unique(active_h5).size != active_h5.size
        or (active_h5.size and (active_h5.min() < 0 or active_h5.max() >= full_count))
    ):
        raise ValueError(f"{name} active H5 mapping is invalid")
    ordering = CanonicalOrdering(active, coords, active_h5)
    ordering.validate(vector_size=active_h5.size, name=name)
    lambda_full = kappa.reshape(-1) - (2.0 / 3.0) * mu.reshape(-1)
    pair = (
        np.asarray(lambda_full[active_h5], dtype=np.float64),
        np.asarray(mu.reshape(-1)[active_h5], dtype=np.float64),
    )
    if not all(np.all(np.isfinite(value)) for value in pair):
        raise ValueError(f"{name} active physical model contains non-finite values")
    return pair, ordering, {
        "material_dir": str(material_dir),
        "material_sha256": dict(hashes),
        "active_h5_indices": str(active_h5_path),
        "active_h5_indices_sha256": str(spec["active_h5_indices"]["sha256"]),
        "active_indices": str(active_path),
        "active_indices_sha256": str(spec["active_indices"]["sha256"]),
        "coordinates": str(coords_path),
        "coordinates_sha256": str(spec["coordinates"]["sha256"]),
    }


def load_gradient_artifact(
    repo: Path,
    spec: Mapping[str, Any],
    *,
    name: str,
) -> tuple[VectorPair, CanonicalOrdering, Any, dict[str, Any]]:
    if "result" in spec:
        iteration = int(spec.get("iteration", -1))
        expected = registered_gradient_result(iteration)
        if spec.get("result") != expected:
            raise ValueError(
                f"{name} registered result mismatch: "
                f"{spec.get('result')!r} != {expected!r}"
            )
        if int(spec.get("parent_iteration", -1)) != iteration:
            raise ValueError(f"{name} parent iteration mismatch")
        identity = IterationIdentity.from_parent(str(spec.get("run_id", "")), iteration)
        if (
            int(spec.get("child_iteration", -1)) != identity.child_iteration
            or spec.get("transition") != identity.transition_id
        ):
            raise ValueError(f"{name} registered transition mismatch")
    lam, lam_path = _load_vector(
        repo, spec["lambda"], name=f"{name}.lambda"
    )
    mu, mu_path = _load_vector(repo, spec["mu"], name=f"{name}.mu")
    active, active_path = _load_vector(
        repo,
        spec["active_indices"],
        name=f"{name}.active_indices",
        dtype=np.int64,
    )
    coords, coords_path = _load_coords(
        repo, spec["coordinates"], name=f"{name}.coordinates"
    )
    active_h5 = None
    active_h5_path = None
    if "active_h5_indices" in spec:
        active_h5, active_h5_path = _load_vector(
            repo,
            spec["active_h5_indices"],
            name=f"{name}.active_h5_indices",
            dtype=np.int64,
        )
    if lam.shape != mu.shape:
        raise ValueError(f"{name} lambda/mu vector shapes differ")
    ordering = CanonicalOrdering(active, coords, active_h5)
    ordering.validate(vector_size=lam.size, name=name)
    mtilde_path = _artifact_path(repo, spec["mtilde"], name=f"{name}.mtilde")
    matrix = load_npz(mtilde_path).tocsr().astype(np.float64)
    if matrix.shape != (lam.size, lam.size):
        raise ValueError(f"{name} Mtilde shape mismatch: {matrix.shape}")
    if not np.all(np.isfinite(matrix.data)):
        raise ValueError(f"{name} Mtilde contains non-finite values")
    provenance = {
        "lambda": str(lam_path),
        "lambda_sha256": str(spec["lambda"]["sha256"]),
        "mu": str(mu_path),
        "mu_sha256": str(spec["mu"]["sha256"]),
        "active_indices": str(active_path),
        "active_indices_sha256": str(spec["active_indices"]["sha256"]),
        "coordinates": str(coords_path),
        "coordinates_sha256": str(spec["coordinates"]["sha256"]),
        "mtilde": str(mtilde_path),
        "mtilde_sha256": str(spec["mtilde"]["sha256"]),
    }
    if active_h5_path is not None:
        provenance.update(
            {
                "active_h5_indices": str(active_h5_path),
                "active_h5_indices_sha256": str(
                    spec["active_h5_indices"]["sha256"]
                ),
            }
        )
    if "result" in spec:
        provenance.update(
            {
                "registered_result": str(spec["result"]),
                "registration_signature_sha256": str(
                    spec.get("registration_signature_sha256", "")
                ),
                "parent_accepted_model": dict(
                    spec.get("parent_accepted_model", {})
                ),
            }
        )
    return (lam, mu), ordering, matrix, provenance


@dataclass(frozen=True)
class HistoryBuildResult:
    from_iteration: int
    to_iteration: int
    s_pair: VectorPair
    y_pair: VectorPair
    audit: CurvatureAudit
    provenance: Mapping[str, Any]


def history_build_status(request: Mapping[str, Any], repo: str | Path) -> dict[str, Any]:
    """Return the expected block before the child corrected gradient exists."""

    root = Path(repo).expanduser().resolve()
    waiting = waiting_for_gradient_status(int(request["child_iteration"]))
    child = request.get("child_gradient")
    if not isinstance(child, Mapping):
        return {
            "status": waiting,
            "reason": "child corrected-gradient manifest is absent",
            "history_pair_created": False,
        }
    for key in ("lambda", "mu", "active_indices", "coordinates", "mtilde"):
        spec = child.get(key)
        if not isinstance(spec, Mapping) or not spec.get("path"):
            return {
                "status": waiting,
                "reason": f"child corrected-gradient artifact is unspecified: {key}",
                "history_pair_created": False,
            }
        if not _resolve(root, str(spec["path"])).is_file():
            return {
                "status": waiting,
                "reason": f"child corrected-gradient artifact does not exist: {key}",
                "history_pair_created": False,
            }
    return {"status": "READY_TO_BUILD_HISTORY", "history_pair_created": False}


def build_history_pair(
    request: Mapping[str, Any],
    *,
    repo: str | Path,
    material_config: Mapping[str, Any],
    curvature_relative_tolerance: float,
) -> HistoryBuildResult:
    """Build and audit one real accepted-history pair.

    No output is admitted to optimizer memory unless the returned audit is
    accepted.  Missing child-gradient data is an expected blocking state.
    """

    status = history_build_status(request, repo)
    if status["status"] != "READY_TO_BUILD_HISTORY":
        raise HistoryBuildBlocked(status["status"], str(status["reason"]))
    root = Path(repo).expanduser().resolve()
    parent_iteration = int(request["parent_iteration"])
    child_iteration = int(request["child_iteration"])
    if child_iteration != parent_iteration + 1:
        raise ValueError("history request must connect consecutive iterations")

    parent_model, model0_order, model0_provenance = _load_model(
        root, request["parent_model"], material_config, name="parent_model"
    )
    child_model, model1_order, model1_provenance = _load_model(
        root, request["child_model"], material_config, name="child_model"
    )
    parent_gradient, grad0_order, mtilde0, grad0_provenance = load_gradient_artifact(
        root, request["parent_gradient"], name="parent_gradient"
    )
    child_gradient, grad1_order, mtilde1, grad1_provenance = load_gradient_artifact(
        root, request["child_gradient"], name="child_gradient"
    )

    require_canonical_identity(
        model0_order,
        model1_order,
        left_name="parent_model",
        right_name="child_model",
    )
    require_canonical_identity(
        grad0_order,
        grad1_order,
        left_name="parent_gradient",
        right_name="child_gradient",
    )
    require_canonical_identity(
        model0_order,
        grad0_order,
        left_name="accepted_models",
        right_name="corrected_gradients",
    )
    if str(request["parent_gradient"]["mtilde"]["sha256"]) != str(
        request["child_gradient"]["mtilde"]["sha256"]
    ):
        raise ValueError("parent/child Mtilde SHA-256 identity failed")
    if Path(grad0_provenance["mtilde"]).resolve() != Path(
        grad1_provenance["mtilde"]
    ).resolve():
        raise ValueError("parent/child Mtilde path identity failed")
    if mtilde0.shape != mtilde1.shape or not np.array_equal(
        mtilde0.indptr, mtilde1.indptr
    ) or not np.array_equal(mtilde0.indices, mtilde1.indices) or not np.array_equal(
        mtilde0.data, mtilde1.data
    ):
        raise ValueError("parent/child Mtilde content identity failed")

    s_pair, y_pair = physical_curvature_pair(
        parent_model, child_model, parent_gradient, child_gradient
    )
    audit = audit_curvature_pair(
        s_pair,
        y_pair,
        mtilde0,
        relative_tolerance=float(curvature_relative_tolerance),
    )
    return HistoryBuildResult(
        from_iteration=parent_iteration,
        to_iteration=child_iteration,
        s_pair=s_pair,
        y_pair=y_pair,
        audit=audit,
        provenance={
            "parent_model": model0_provenance,
            "child_model": model1_provenance,
            "parent_gradient": grad0_provenance,
            "child_gradient": grad1_provenance,
            "canonical_active_index_identity": True,
            "canonical_coordinate_identity": True,
            "mtilde_identity": True,
        },
    )


def accepted_history_pairs(
    results: Sequence[HistoryBuildResult], *, memory_limit: int
) -> list[tuple[VectorPair, VectorPair]]:
    if int(memory_limit) <= 0:
        raise ValueError("memory_limit must be positive")
    accepted = [
        (value.s_pair, value.y_pair) for value in results if value.audit.accepted
    ]
    return accepted[-int(memory_limit) :]


def _history_identity(from_iteration: int, to_iteration: int) -> IterationIdentity:
    return IterationIdentity("optimizer_history", from_iteration, to_iteration)


def _verified_identity_hashes(provenance: Mapping[str, Any]) -> dict[str, str]:
    groups = {
        "canonical_active_indices_sha256": [
            provenance[name]["active_indices_sha256"]
            for name in (
                "parent_model",
                "child_model",
                "parent_gradient",
                "child_gradient",
            )
        ],
        "coordinates_sha256": [
            provenance[name]["coordinates_sha256"]
            for name in (
                "parent_model",
                "child_model",
                "parent_gradient",
                "child_gradient",
            )
        ],
        "mtilde_sha256": [
            provenance[name]["mtilde_sha256"]
            for name in ("parent_gradient", "child_gradient")
        ],
    }
    result = {}
    for key, values in groups.items():
        if not values or any(str(value) != str(values[0]) for value in values[1:]):
            raise ValueError(f"history provenance identity mismatch: {key}")
        result[key] = str(values[0])
    return result


def _history_arrays(result: HistoryBuildResult) -> dict[str, np.ndarray]:
    arrays = {
        "s_lambda": np.asarray(result.s_pair[0], dtype=np.float64),
        "s_mu": np.asarray(result.s_pair[1], dtype=np.float64),
        "y_lambda": np.asarray(result.y_pair[0], dtype=np.float64),
        "y_mu": np.asarray(result.y_pair[1], dtype=np.float64),
    }
    shapes = {value.shape for value in arrays.values()}
    if len(shapes) != 1 or any(value.ndim != 1 for value in arrays.values()):
        raise ValueError("history checkpoint arrays have inconsistent shapes")
    if not all(np.all(np.isfinite(value)) for value in arrays.values()):
        raise ValueError("history checkpoint arrays contain non-finite values")
    return arrays


def _canonical_json_sha256(value: Mapping[str, Any]) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _curvature_outcome_manifest(
    result: HistoryBuildResult,
    *,
    identity_hashes: Mapping[str, str],
) -> dict[str, Any]:
    provenance = dict(result.provenance)
    return {
        "schema_version": 1,
        "record_type": "MTILDE_CURVATURE_OUTCOME",
        "from_iteration": int(result.from_iteration),
        "to_iteration": int(result.to_iteration),
        "transition": _history_identity(
            result.from_iteration, result.to_iteration
        ).transition_id,
        "status": (
            HISTORY_OUTCOME_ACCEPTED
            if result.audit.accepted
            else HISTORY_OUTCOME_REJECTED
        ),
        "sMy": float(result.audit.s_m_y),
        "curvature_threshold": float(result.audit.threshold),
        "curvature_reason": str(result.audit.reason),
        "active_indices_sha256": str(
            identity_hashes["canonical_active_indices_sha256"]
        ),
        "coordinates_sha256": str(identity_hashes["coordinates_sha256"]),
        "mtilde_sha256": str(identity_hashes["mtilde_sha256"]),
        "provenance": provenance,
        "provenance_sha256": _canonical_json_sha256(provenance),
    }


def _write_curvature_outcome(
    directory: Path, manifest: Mapping[str, Any]
) -> None:
    (directory / "curvature_outcome.json").write_text(
        json.dumps(dict(manifest), indent=2) + "\n", encoding="utf-8"
    )


def _load_persisted_pair(
    pair_dir: Path,
    *,
    expected_active_indices_sha256: str,
    expected_coordinates_sha256: str,
    expected_mtilde_sha256: str,
) -> tuple[tuple[VectorPair, VectorPair], dict[str, Any]] | None:
    manifest_path = pair_dir / "curvature_pair.json"
    if not manifest_path.is_file():
        raise ValueError(f"history checkpoint lacks manifest: {pair_dir}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("accepted") is not True:
        return None
    expected_hashes = {
        "canonical_active_indices_sha256": expected_active_indices_sha256,
        "coordinates_sha256": expected_coordinates_sha256,
        "mtilde_sha256": expected_mtilde_sha256,
    }
    for key, expected in expected_hashes.items():
        if str(manifest.get(key, "")) != str(expected):
            raise ValueError(f"persisted history {key} mismatch: {pair_dir}")
    from_iteration = int(manifest["from_iteration"])
    to_iteration = int(manifest["to_iteration"])
    if to_iteration != from_iteration + 1:
        raise ValueError(f"persisted history is not consecutive: {pair_dir}")
    if not np.isfinite(float(manifest["sMy"])) or float(manifest["sMy"]) <= 0.0:
        raise ValueError(f"persisted accepted history has invalid sMy: {pair_dir}")
    if not np.isfinite(float(manifest["curvature_threshold"])):
        raise ValueError(f"persisted history threshold is non-finite: {pair_dir}")
    arrays = {}
    for name in ("s_lambda", "s_mu", "y_lambda", "y_mu"):
        artifact = manifest["arrays"][name]
        relative = Path(str(artifact["path"]))
        if relative.is_absolute() or relative.parent != Path("."):
            raise ValueError(f"persisted history array path is not local: {relative}")
        path = pair_dir / relative
        if sha256_file(path) != str(artifact["sha256"]):
            raise ValueError(f"persisted history array SHA-256 mismatch: {path}")
        value = np.asarray(np.load(path), dtype=np.float64)
        if value.ndim != 1 or not np.all(np.isfinite(value)):
            raise ValueError(f"invalid persisted history vector: {path}")
        arrays[name] = value
    shapes = {value.shape for value in arrays.values()}
    if len(shapes) != 1:
        raise ValueError(f"persisted history vector shapes differ: {pair_dir}")
    return (
        (arrays["s_lambda"], arrays["s_mu"]),
        (arrays["y_lambda"], arrays["y_mu"]),
    ), manifest


def _load_validated_curvature_outcome(
    pair_dir: Path,
    *,
    expected_from_iteration: int,
    expected_to_iteration: int,
    expected_active_indices_sha256: str,
    expected_coordinates_sha256: str,
    expected_mtilde_sha256: str,
) -> dict[str, Any] | None:
    outcome_path = pair_dir / "curvature_outcome.json"
    if not outcome_path.is_file():
        return None
    manifest = json.loads(outcome_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict):
        raise ValueError(f"curvature outcome must be an object: {outcome_path}")
    from_iteration = int(manifest.get("from_iteration", -1))
    to_iteration = int(manifest.get("to_iteration", -1))
    if (from_iteration, to_iteration) != (
        int(expected_from_iteration),
        int(expected_to_iteration),
    ):
        raise ValueError(f"curvature outcome iteration identity mismatch: {pair_dir}")
    identity = _history_identity(from_iteration, to_iteration)
    if pair_dir.name != identity.transition_id:
        raise ValueError(f"curvature outcome directory identity mismatch: {pair_dir}")
    if manifest.get("transition") != identity.transition_id:
        raise ValueError(f"curvature outcome transition identity mismatch: {pair_dir}")
    status = str(manifest.get("status", ""))
    if status not in {HISTORY_OUTCOME_ACCEPTED, HISTORY_OUTCOME_REJECTED}:
        raise ValueError(f"invalid curvature outcome status: {pair_dir}")
    s_m_y = float(manifest.get("sMy", float("nan")))
    threshold = float(manifest.get("curvature_threshold", float("nan")))
    if not np.isfinite(s_m_y) or not np.isfinite(threshold) or threshold < 0.0:
        raise ValueError(f"non-finite curvature outcome audit: {pair_dir}")
    if not str(manifest.get("curvature_reason", "")).strip():
        raise ValueError(f"curvature outcome lacks audit reason: {pair_dir}")

    expected_hashes = {
        "active_indices_sha256": str(expected_active_indices_sha256),
        "coordinates_sha256": str(expected_coordinates_sha256),
        "mtilde_sha256": str(expected_mtilde_sha256),
    }
    for key, expected in expected_hashes.items():
        if str(manifest.get(key, "")) != expected:
            raise ValueError(f"curvature outcome {key} mismatch: {pair_dir}")
    provenance = manifest.get("provenance")
    if not isinstance(provenance, Mapping) or not provenance:
        raise ValueError(f"curvature outcome lacks provenance: {pair_dir}")
    if str(manifest.get("provenance_sha256", "")) != _canonical_json_sha256(
        provenance
    ):
        raise ValueError(f"curvature outcome provenance SHA-256 mismatch: {pair_dir}")
    provenance_hashes = _verified_identity_hashes(provenance)
    if (
        provenance_hashes["canonical_active_indices_sha256"]
        != expected_hashes["active_indices_sha256"]
        or provenance_hashes["coordinates_sha256"]
        != expected_hashes["coordinates_sha256"]
        or provenance_hashes["mtilde_sha256"]
        != expected_hashes["mtilde_sha256"]
    ):
        raise ValueError(f"curvature outcome provenance identity mismatch: {pair_dir}")

    pair_manifest_path = pair_dir / "curvature_pair.json"
    if status == HISTORY_OUTCOME_ACCEPTED:
        if s_m_y <= 0.0 or s_m_y <= threshold:
            raise ValueError(f"accepted curvature outcome is not admissible: {pair_dir}")
        loaded = _load_persisted_pair(
            pair_dir,
            expected_active_indices_sha256=expected_active_indices_sha256,
            expected_coordinates_sha256=expected_coordinates_sha256,
            expected_mtilde_sha256=expected_mtilde_sha256,
        )
        if loaded is None:
            raise ValueError(f"accepted curvature outcome lacks accepted pair: {pair_dir}")
        _, pair_manifest = loaded
        if (
            int(pair_manifest["from_iteration"]) != from_iteration
            or int(pair_manifest["to_iteration"]) != to_iteration
            or float(pair_manifest["sMy"]) != s_m_y
            or float(pair_manifest["curvature_threshold"]) != threshold
            or str(pair_manifest.get("curvature_reason", ""))
            != str(manifest["curvature_reason"])
            or _canonical_json_sha256(pair_manifest.get("provenance", {}))
            != str(manifest["provenance_sha256"])
        ):
            raise ValueError(f"accepted pair/outcome audit mismatch: {pair_dir}")
    elif pair_manifest_path.exists():
        raise ValueError(f"rejected curvature outcome contains an accepted pair: {pair_dir}")
    return manifest


def load_curvature_outcome(
    history_root: str | Path,
    *,
    from_iteration: int,
    to_iteration: int,
    expected_active_indices_sha256: str,
    expected_coordinates_sha256: str,
    expected_mtilde_sha256: str,
) -> dict[str, Any] | None:
    """Load one explicit, hash/provenance-validated curvature outcome."""

    identity = _history_identity(from_iteration, to_iteration)
    pair_dir = Path(history_root).expanduser().resolve() / identity.transition_id
    if not pair_dir.exists():
        return None
    if not pair_dir.is_dir():
        raise ValueError(f"history outcome path is not a directory: {pair_dir}")
    return _load_validated_curvature_outcome(
        pair_dir,
        expected_from_iteration=from_iteration,
        expected_to_iteration=to_iteration,
        expected_active_indices_sha256=expected_active_indices_sha256,
        expected_coordinates_sha256=expected_coordinates_sha256,
        expected_mtilde_sha256=expected_mtilde_sha256,
    )


def require_newest_curvature_outcome(
    history_root: str | Path,
    *,
    parent_iteration: int,
    expected_active_indices_sha256: str,
    expected_coordinates_sha256: str,
    expected_mtilde_sha256: str,
) -> dict[str, Any]:
    """Require the explicit ``(k-1) -> k`` audit outcome for parent ``k``."""

    parent = int(parent_iteration)
    if parent <= 0:
        raise ValueError("newest curvature outcome is required only for parent > 0")
    outcome = load_curvature_outcome(
        history_root,
        from_iteration=parent - 1,
        to_iteration=parent,
        expected_active_indices_sha256=expected_active_indices_sha256,
        expected_coordinates_sha256=expected_coordinates_sha256,
        expected_mtilde_sha256=expected_mtilde_sha256,
    )
    if outcome is None:
        raise HistoryBuildBlocked(
            BLOCKED_WAITING_FOR_HISTORY_AUDIT,
            f"curvature outcome is absent for iter_{parent - 1:03d}_to_iter_{parent:03d}",
        )
    return outcome


def persist_accepted_history_pair(
    result: HistoryBuildResult,
    *,
    history_root: str | Path,
) -> Path:
    """Atomically persist one accepted, provenance-verified curvature pair."""

    if not result.audit.accepted:
        raise ValueError("rejected curvature pair must not enter persisted memory")
    identity = _history_identity(result.from_iteration, result.to_iteration)
    root = Path(history_root).expanduser().resolve()
    destination = root / identity.transition_id
    hashes = _verified_identity_hashes(result.provenance)
    arrays = _history_arrays(result)
    outcome = _curvature_outcome_manifest(result, identity_hashes=hashes)

    if destination.exists():
        loaded = _load_persisted_pair(
            destination,
            expected_active_indices_sha256=hashes[
                "canonical_active_indices_sha256"
            ],
            expected_coordinates_sha256=hashes["coordinates_sha256"],
            expected_mtilde_sha256=hashes["mtilde_sha256"],
        )
        if loaded is None:
            raise ValueError("existing history checkpoint is not accepted")
        pair, manifest = loaded
        existing = {
            "s_lambda": pair[0][0],
            "s_mu": pair[0][1],
            "y_lambda": pair[1][0],
            "y_mu": pair[1][1],
        }
        if not all(np.array_equal(existing[name], arrays[name]) for name in arrays):
            raise ValueError("existing history checkpoint arrays differ")
        if (
            float(manifest["sMy"]) != float(result.audit.s_m_y)
            or float(manifest["curvature_threshold"])
            != float(result.audit.threshold)
        ):
            raise ValueError("existing history checkpoint curvature audit differs")
        existing_outcome = _load_validated_curvature_outcome(
            destination,
            expected_from_iteration=result.from_iteration,
            expected_to_iteration=result.to_iteration,
            expected_active_indices_sha256=hashes[
                "canonical_active_indices_sha256"
            ],
            expected_coordinates_sha256=hashes["coordinates_sha256"],
            expected_mtilde_sha256=hashes["mtilde_sha256"],
        )
        if existing_outcome is None:
            raise ValueError("existing accepted pair lacks curvature outcome")
        if _canonical_json_sha256(existing_outcome) != _canonical_json_sha256(
            outcome
        ):
            raise ValueError("existing accepted curvature outcome differs")
        return destination

    root.mkdir(parents=True, exist_ok=True)
    temporary = root / f".{identity.transition_id}.tmp.{os.getpid()}"
    if temporary.exists():
        raise FileExistsError(f"history temporary path exists: {temporary}")
    temporary.mkdir()
    try:
        artifacts = {}
        for name, value in arrays.items():
            filename = f"{name}.npy"
            path = temporary / filename
            np.save(path, value)
            artifacts[name] = {
                "path": filename,
                "sha256": sha256_file(path),
            }
        manifest = {
            "schema_version": 1,
            "result": "PASS_ACCEPTED_MTILDE_CURVATURE_PAIR",
            "from_iteration": int(result.from_iteration),
            "to_iteration": int(result.to_iteration),
            "transition": identity.transition_id,
            "arrays": artifacts,
            **hashes,
            "sMy": float(result.audit.s_m_y),
            "curvature_threshold": float(result.audit.threshold),
            "accepted": True,
            "curvature_reason": result.audit.reason,
            "provenance": dict(result.provenance),
        }
        manifest_path = temporary / "curvature_pair.json"
        manifest_path.write_text(
            json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
        )
        _write_curvature_outcome(temporary, outcome)
        temporary.replace(destination)
    except Exception:
        if temporary.exists() and temporary.parent == root:
            shutil.rmtree(temporary)
        raise
    return destination


def persist_curvature_outcome(
    result: HistoryBuildResult,
    *,
    history_root: str | Path,
) -> Path:
    """Persist every completed audit; rejected outcomes contain no s/y arrays."""

    if result.audit.accepted:
        return persist_accepted_history_pair(result, history_root=history_root)
    identity = _history_identity(result.from_iteration, result.to_iteration)
    root = Path(history_root).expanduser().resolve()
    destination = root / identity.transition_id
    hashes = _verified_identity_hashes(result.provenance)
    outcome = _curvature_outcome_manifest(result, identity_hashes=hashes)
    if destination.exists():
        existing = _load_validated_curvature_outcome(
            destination,
            expected_from_iteration=result.from_iteration,
            expected_to_iteration=result.to_iteration,
            expected_active_indices_sha256=hashes[
                "canonical_active_indices_sha256"
            ],
            expected_coordinates_sha256=hashes["coordinates_sha256"],
            expected_mtilde_sha256=hashes["mtilde_sha256"],
        )
        if existing is None:
            raise ValueError("existing history directory lacks curvature outcome")
        if _canonical_json_sha256(existing) != _canonical_json_sha256(outcome):
            raise ValueError("existing rejected curvature outcome differs")
        return destination

    root.mkdir(parents=True, exist_ok=True)
    temporary = root / f".{identity.transition_id}.tmp.{os.getpid()}"
    if temporary.exists():
        raise FileExistsError(f"history temporary path exists: {temporary}")
    temporary.mkdir()
    try:
        _write_curvature_outcome(temporary, outcome)
        temporary.replace(destination)
    except Exception:
        if temporary.exists() and temporary.parent == root:
            shutil.rmtree(temporary)
        raise
    return destination


def load_persisted_history(
    history_root: str | Path,
    *,
    parent_iteration: int,
    memory_limit: int,
    expected_active_indices_sha256: str,
    expected_coordinates_sha256: str,
    expected_mtilde_sha256: str,
) -> list[tuple[VectorPair, VectorPair]]:
    """Restore newest hash-verified accepted history for L-BFGS two-loop input."""

    if int(parent_iteration) < 0:
        raise ValueError("parent_iteration must be non-negative")
    if int(memory_limit) <= 0:
        raise ValueError("memory_limit must be positive")
    root = Path(history_root).expanduser().resolve()
    if not root.exists():
        return []
    restored = []
    seen = set()
    for pair_dir in sorted(path for path in root.iterdir() if path.is_dir()):
        if pair_dir.name.startswith("."):
            continue
        match = re.fullmatch(r"iter_(\d+)_to_iter_(\d+)", pair_dir.name)
        if match is None:
            raise ValueError(f"unrecognized optimizer-history directory: {pair_dir}")
        from_iteration, to_iteration = (int(value) for value in match.groups())
        outcome = _load_validated_curvature_outcome(
            pair_dir,
            expected_from_iteration=from_iteration,
            expected_to_iteration=to_iteration,
            expected_active_indices_sha256=expected_active_indices_sha256,
            expected_coordinates_sha256=expected_coordinates_sha256,
            expected_mtilde_sha256=expected_mtilde_sha256,
        )
        if outcome is None:
            raise ValueError(f"history pair lacks explicit audit outcome: {pair_dir}")
        if outcome["status"] == HISTORY_OUTCOME_REJECTED:
            continue
        loaded = _load_persisted_pair(
            pair_dir,
            expected_active_indices_sha256=expected_active_indices_sha256,
            expected_coordinates_sha256=expected_coordinates_sha256,
            expected_mtilde_sha256=expected_mtilde_sha256,
        )
        if loaded is None:
            raise ValueError(f"accepted history outcome lacks pair: {pair_dir}")
        pair, manifest = loaded
        key = (from_iteration, to_iteration)
        if key in seen:
            raise ValueError(f"duplicate persisted history pair: {key}")
        seen.add(key)
        if key[1] <= int(parent_iteration):
            restored.append((key, pair))
    restored.sort(key=lambda item: item[0][1])
    return [pair for _, pair in restored[-int(memory_limit) :]]
