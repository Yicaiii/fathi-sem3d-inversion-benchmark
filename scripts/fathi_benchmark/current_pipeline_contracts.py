"""Exact, iteration-specific contracts for the CURRENT production pipeline.

This module contains no numerical kernels and launches no subprocesses.  It is
the single naming and SHA/provenance vocabulary shared by CURRENT producers and
consumers.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping

from scripts.fathi_benchmark.iteration_context import IterationPaths


CURRENT_RUN_ID = "fathi_s43_repro_p20_t052"
SCHEMA_VERSION = 1


def _iteration(value: int) -> int:
    result = int(value)
    if result < 0 or result != value:
        raise ValueError("iteration must be a non-negative integer")
    return result


def accepted_model_result(iteration: int) -> str:
    """Canonical CURRENT accepted-model result.

    The prefix intentionally matches the already-frozen iter_001 artifact and
    remains a single dynamic spelling for subsequent CURRENT iterations.
    """

    return f"PASS_CURRENT_T052_ITER{_iteration(iteration):03d}_ACCEPTED_MODEL"


def retained_primal_result(iteration: int) -> str:
    value = _iteration(iteration)
    return f"PASS_ITER{value:03d}_ACCEPTED_EXTERNAL_PRIMAL_FOR_G{value}"


def exact_reverse_result(iteration: int) -> str:
    return f"PASS_ITER{_iteration(iteration):03d}_EXACT_REVERSE_MATERIAL_COVECTOR"


def gradient_bridge_result(iteration: int) -> str:
    return f"PASS_ITER{_iteration(iteration):03d}_CORRECTED_PHYSICAL_GRADIENT_BRIDGE"


def registered_gradient_result(iteration: int) -> str:
    return f"PASS_ITER{_iteration(iteration):03d}_REGISTERED_PHYSICAL_GRADIENT"


def optimizer_direction_result(iteration: int) -> str:
    return f"PASS_ITER{_iteration(iteration):03d}_PHYSICAL_LBFGS_EQ25_DIRECTION"


def candidate_generated_result(parent_iteration: int, trial_index: int) -> str:
    parent = _iteration(parent_iteration)
    trial = _iteration(trial_index)
    return f"PASS_ITER{parent:03d}_ALPHA_{trial:03d}_CANDIDATE_GENERATED"


def candidate_objective_result(parent_iteration: int, trial_index: int) -> str:
    parent = _iteration(parent_iteration)
    trial = _iteration(trial_index)
    return f"PASS_ITER{parent:03d}_ALPHA_{trial:03d}_EXTERNAL_OBJECTIVE"


def armijo_ready_result(parent_iteration: int) -> str:
    return f"PASS_ITER{_iteration(parent_iteration):03d}_EXTERNAL_ARMIJO_READY"


def armijo_trial_result(
    parent_iteration: int, trial_index: int, *, accepted: bool
) -> str:
    parent = _iteration(parent_iteration)
    trial = _iteration(trial_index)
    outcome = "ACCEPTED" if accepted else "REJECTED"
    return f"{outcome}_ITER{parent:03d}_ALPHA_{trial:03d}_ARMIJO"


def armijo_search_result(parent_iteration: int, *, accepted: bool) -> str:
    parent = _iteration(parent_iteration)
    outcome = "PASS" if accepted else "FAIL"
    suffix = "ACCEPTED" if accepted else "EXHAUSTED"
    return f"{outcome}_ITER{parent:03d}_EXTERNAL_ARMIJO_{suffix}"


def promotion_result(parent_iteration: int, child_iteration: int) -> str:
    parent = _iteration(parent_iteration)
    child = _iteration(child_iteration)
    if child != parent + 1:
        raise ValueError("promotion must connect consecutive iterations")
    return f"PASS_ITER{parent:03d}_TO_ITER{child:03d}_PROMOTED"


def identity(paths: IterationPaths) -> dict[str, Any]:
    value = paths.identity
    return {
        "run_id": value.run_id,
        "parent_iteration": value.parent_iteration,
        "child_iteration": value.child_iteration,
        "transition": value.transition_id,
    }


def require_identity(
    manifest: Mapping[str, Any],
    paths: IterationPaths,
    *,
    label: str,
    iteration_field: str | None = None,
) -> None:
    wanted = identity(paths)
    actual = {
        "run_id": manifest.get("run_id"),
        "parent_iteration": int(manifest.get("parent_iteration", -1)),
        "child_iteration": int(manifest.get("child_iteration", -1)),
        "transition": manifest.get("transition"),
    }
    if actual != wanted:
        raise ValueError(f"{label} identity mismatch: {actual} != {wanted}")
    if iteration_field is not None and int(
        manifest.get(iteration_field, -1)
    ) != paths.identity.parent_iteration:
        raise ValueError(f"{label} {iteration_field} mismatch")


def require_result(
    manifest: Mapping[str, Any], expected: str, *, label: str
) -> None:
    actual = manifest.get("result")
    if actual != expected:
        raise ValueError(f"{label} result mismatch: {actual!r} != {expected!r}")


def sha256_file(path: str | Path) -> str:
    source = Path(path)
    digest = hashlib.sha256()
    with source.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha256(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def resolve_record_path(repo: str | Path, record: Mapping[str, Any]) -> Path:
    if not isinstance(record, Mapping) or not record.get("path"):
        raise ValueError("artifact record requires path")
    path = Path(str(record["path"])).expanduser()
    root = Path(repo).expanduser().resolve()
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def artifact_record(path: str | Path, *, repo: str | Path) -> dict[str, str]:
    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    root = Path(repo).expanduser().resolve()
    try:
        recorded = str(source.relative_to(root))
    except ValueError:
        recorded = str(source)
    return {"path": recorded, "sha256": sha256_file(source)}


def verify_artifact_record(
    repo: str | Path,
    record: Mapping[str, Any],
    *,
    label: str,
    expected_path: str | Path | None = None,
) -> Path:
    if not isinstance(record, Mapping) or not record.get("sha256"):
        raise ValueError(f"{label} requires path and sha256")
    path = resolve_record_path(repo, record)
    if expected_path is not None and path != Path(expected_path).resolve():
        raise ValueError(f"{label} path mismatch")
    if not path.is_file():
        raise FileNotFoundError(f"missing {label}: {path}")
    actual = sha256_file(path)
    if actual != str(record["sha256"]):
        raise ValueError(f"{label} SHA-256 mismatch")
    return path


def material_signature(material_hashes: Mapping[str, str]) -> str:
    normalized = {str(key): str(value) for key, value in material_hashes.items()}
    if not normalized or any(len(value) != 64 for value in normalized.values()):
        raise ValueError("material hashes must be a non-empty SHA256 mapping")
    return canonical_sha256(normalized)


def atomic_json(path: str | Path, payload: Mapping[str, Any]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".tmp")
    temporary.write_text(
        json.dumps(dict(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, destination)
