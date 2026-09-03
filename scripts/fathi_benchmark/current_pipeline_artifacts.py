"""Thin durable producers for the CURRENT optimizer-to-acceptance chain.

All numerical decisions delegate to the already-certified physical optimizer
and Armijo primitives. The functions here validate identity/order/SHA
contracts, persist artifacts, and provide resumable orchestration. They never
launch SEM3D or MPI.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
import shutil
from typing import Any, Callable, Mapping, Sequence

import h5py
import numpy as np

from scripts.fathi_benchmark.current_pipeline_contracts import (
    SCHEMA_VERSION,
    accepted_model_result,
    armijo_ready_result,
    armijo_search_result,
    armijo_trial_result,
    artifact_record,
    atomic_json,
    candidate_generated_result,
    candidate_objective_result,
    canonical_sha256,
    material_signature,
    optimizer_direction_result,
    promotion_result,
    registered_gradient_result,
    require_identity,
    require_result,
    sha256_file,
    verify_artifact_record,
)
from scripts.fathi_benchmark.external_armijo import (
    ArmijoParameters,
    armijo_decision,
    candidate_namespace,
)
from scripts.fathi_benchmark.iteration_context import IterationPaths
from scripts.fathi_benchmark.lbfgs_history import load_gradient_artifact
from scripts.fathi_benchmark.physical_space_optimizer import joint_mtilde_inner
from scripts.fathi_benchmark.physical_space_optimizer import (
    apply_lambda_bias_euclidean,
    lambda_bias_weight,
)


DIRECTION_SUMMARY_NAME = "direction_summary.json"
CANDIDATE_SUMMARY_NAME = "candidate_summary.json"
TRIAL_SUMMARY_NAME = "trial_summary.json"
ARMIJO_SUMMARY_NAME = "armijo_summary.json"


def _json(path: str | Path) -> dict[str, Any]:
    source = Path(path).expanduser().resolve()
    value = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON manifest must be an object: {source}")
    return value


def _relative(path: Path, repo: Path) -> str:
    source = path.resolve()
    try:
        return str(source.relative_to(repo.resolve()))
    except ValueError:
        return str(source)


def _record_for_final(
    temporary_path: Path, final_path: Path, *, repo: Path
) -> dict[str, str]:
    return {
        "path": _relative(final_path, repo),
        "sha256": sha256_file(temporary_path),
    }


def _load_array(
    repo: Path,
    record: Mapping[str, Any],
    *,
    label: str,
    dtype: Any,
    ndim: int,
) -> np.ndarray:
    path = verify_artifact_record(repo, record, label=label)
    value = np.asarray(np.load(path), dtype=dtype)
    if value.ndim != ndim or not np.all(np.isfinite(value)):
        raise ValueError(f"{label} array contract mismatch: {value.shape}")
    return value


def _accepted_material_hashes(
    summary: Mapping[str, Any], material_config: Mapping[str, Any]
) -> dict[str, str]:
    declared = summary.get("material_sha256")
    files = material_config.get("files")
    if not isinstance(declared, Mapping) or not isinstance(files, Mapping):
        raise ValueError("accepted model lacks material hash contract")
    result = {}
    for component in ("kappa", "mu", "density"):
        filename = str(files[component])
        value = declared.get(filename, declared.get(component))
        if not value:
            raise ValueError(f"accepted model lacks {component} SHA256")
        result[component] = str(value)
    return result


def _validate_accepted_parent(
    repo: Path,
    paths: IterationPaths,
    record: Mapping[str, Any],
    material_config: Mapping[str, Any],
) -> tuple[Path, dict[str, Any], dict[str, str]]:
    expected = paths.parent_accepted / "accepted_summary.json"
    path = verify_artifact_record(
        repo, record, label="accepted parent", expected_path=expected
    )
    summary = _json(path)
    iteration = paths.identity.parent_iteration
    if (
        summary.get("result") != accepted_model_result(iteration)
        or summary.get("run") != paths.identity.run_id
        or int(summary.get("iter", -1)) != iteration
    ):
        raise ValueError("accepted parent result/identity mismatch")
    hashes = _accepted_material_hashes(summary, material_config)
    material_dir = paths.parent_accepted / str(material_config["directory"])
    for component, expected_hash in hashes.items():
        material_path = material_dir / str(material_config["files"][component])
        if sha256_file(material_path) != expected_hash:
            raise ValueError(f"accepted parent {component} SHA256 mismatch")
    return path, summary, hashes


def _registered_gradient(
    repo: Path, paths: IterationPaths, record: Mapping[str, Any]
) -> tuple[Path, dict[str, Any]]:
    expected = paths.gradient_root / "registered_gradient.json"
    path = verify_artifact_record(
        repo, record, label="registered gradient", expected_path=expected
    )
    manifest = _json(path)
    require_result(
        manifest,
        registered_gradient_result(paths.identity.parent_iteration),
        label="registered gradient",
    )
    require_identity(manifest, paths, label="registered gradient")
    return path, manifest


def _history_records(
    repo: Path,
    paths: IterationPaths,
    records: Sequence[Mapping[str, Any]],
    gradient: Mapping[str, Any],
) -> list[dict[str, Any]]:
    parent = paths.identity.parent_iteration
    loaded = []
    for index, record in enumerate(records):
        path = verify_artifact_record(repo, record, label=f"history outcome {index}")
        payload = _json(path)
        start = int(payload.get("from_iteration", -1))
        end = int(payload.get("to_iteration", -1))
        if end != start + 1 or end > parent:
            raise ValueError("history outcome iteration mismatch")
        if payload.get("status") not in {"ACCEPTED", "REJECTED"}:
            raise ValueError("history outcome lacks explicit accepted/rejected status")
        expected_hashes = {
            "active_indices_sha256": gradient["active_indices"]["sha256"],
            "coordinates_sha256": gradient["coordinates"]["sha256"],
            "mtilde_sha256": gradient["mtilde"]["sha256"],
        }
        for key, expected in expected_hashes.items():
            if str(payload.get(key, "")) != str(expected):
                raise ValueError(f"history outcome {key} mismatch")
        loaded.append({"record": dict(record), "payload": payload})
    if parent == 0:
        if loaded:
            raise ValueError("iter0 H0 direction must not consume history")
    elif not any(
        int(item["payload"]["from_iteration"]) == parent - 1
        and int(item["payload"]["to_iteration"]) == parent
        for item in loaded
    ):
        raise ValueError("newest accepted/rejected curvature outcome is absent")
    return loaded


def persist_optimizer_direction(
    *,
    repo: str | Path,
    paths: IterationPaths,
    material_config: Mapping[str, Any],
    optimizer_manifest: Mapping[str, Any],
    direction_result: Any,
) -> Path:
    """Persist an already-computed physical L-BFGS plus Eq.25 result."""

    root = Path(repo).expanduser().resolve()
    require_identity(optimizer_manifest, paths, label="optimizer request")
    gradient_path, gradient = _registered_gradient(
        root, paths, optimizer_manifest["registered_gradient_manifest"]
    )
    gradient_pair, _, mtilde_matrix, _ = load_gradient_artifact(
        root, gradient, name="registered_current_gradient"
    )
    parent_path, _, _ = _validate_accepted_parent(
        root,
        paths,
        optimizer_manifest["accepted_parent_summary"],
        material_config,
    )
    history = _history_records(
        root, paths, optimizer_manifest.get("history_outcomes", []), gradient
    )
    coordinates = _load_array(
        root,
        gradient["coordinates"],
        label="gradient coordinates",
        dtype=np.float64,
        ndim=2,
    )
    active_indices = _load_array(
        root,
        gradient["active_indices"],
        label="canonical active indices",
        dtype=np.int64,
        ndim=1,
    )
    active_h5 = _load_array(
        root,
        gradient["active_h5_indices"],
        label="active H5 indices",
        dtype=np.int64,
        ndim=1,
    )
    if coordinates.shape != (active_indices.size, 3) or active_h5.shape != (
        active_indices.size,
    ):
        raise ValueError("direction ordering inputs have inconsistent shapes")

    raw = tuple(np.asarray(value, dtype=np.float64) for value in direction_result.raw_direction)
    biased = tuple(
        np.asarray(value, dtype=np.float64)
        for value in direction_result.biased_direction
    )
    if len(raw) != 2 or len(biased) != 2:
        raise ValueError("direction result must contain lambda/mu pairs")
    if any(
        value.shape != active_indices.shape or not np.all(np.isfinite(value))
        for value in (*raw, *biased)
    ):
        raise ValueError("direction vector shape/finite contract failed")
    expected_weight = lambda_bias_weight(paths.identity.parent_iteration)
    if float(direction_result.lambda_bias_weight) != float(expected_weight):
        raise ValueError("Eq.25 weight differs from dynamic parent iteration")
    expected_biased = apply_lambda_bias_euclidean(raw, weight=expected_weight)
    if not all(
        np.array_equal(actual, expected)
        for actual, expected in zip(biased, expected_biased)
    ):
        raise ValueError("durable biased direction differs from Eq.25 output")
    slope = float(
        joint_mtilde_inner(gradient_pair, biased, mtilde_matrix)
    )
    reported_slope = float(direction_result.slope)
    if not math.isclose(
        slope,
        reported_slope,
        rel_tol=32.0 * np.finfo(np.float64).eps,
        abs_tol=0.0,
    ):
        raise ValueError("reported direction slope differs from g.T Mtilde p")
    if not math.isfinite(slope) or slope >= 0.0:
        raise ValueError("physical direction must have negative Mtilde slope")
    parent = paths.identity.parent_iteration
    destination = (
        paths.optimizer_root / f"iter_{parent:03d}_lbfgs_eq25_direction"
    ).resolve()
    input_payload = {
        "identity": {
            "run_id": paths.identity.run_id,
            "parent_iteration": parent,
            "child_iteration": paths.identity.child_iteration,
            "transition": paths.identity.transition_id,
        },
        "registered_gradient_manifest": artifact_record(gradient_path, repo=root),
        "accepted_parent_summary": artifact_record(parent_path, repo=root),
        "history_outcomes": [item["record"] for item in history],
        "mtilde": gradient["mtilde"],
        "coordinates": gradient["coordinates"],
        "active_indices": gradient["active_indices"],
        "active_h5_indices": gradient["active_h5_indices"],
        "lambda_bias_weight": float(direction_result.lambda_bias_weight),
        "h0_or_history_scale": float(direction_result.h0_or_history_scale),
        "slope": slope,
    }
    input_signature = canonical_sha256(input_payload)
    summary_path = destination / DIRECTION_SUMMARY_NAME
    if destination.exists():
        existing = _json(summary_path)
        if (
            existing.get("result") != optimizer_direction_result(parent)
            or existing.get("input_signature_sha256") != input_signature
        ):
            raise ValueError("existing CURRENT direction conflicts")
        expected_outputs = {
            "raw_lambda": destination / "p_lambda_raw_phys.npy",
            "raw_mu": destination / "p_mu_raw_phys.npy",
            "biased_lambda": destination / "p_lambda_biased_phys.npy",
            "biased_mu": destination / "p_mu_biased_phys.npy",
            "coordinates": destination / "direction_coords.npy",
        }
        for name, expected_path in expected_outputs.items():
            verify_artifact_record(
                root,
                existing["artifacts"][name],
                label=f"existing direction {name}",
                expected_path=expected_path,
            )
        return summary_path

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp.{os.getpid()}")
    if temporary.exists():
        raise FileExistsError(temporary)
    temporary.mkdir()
    try:
        arrays = {
            "p_lambda_raw_phys.npy": raw[0],
            "p_mu_raw_phys.npy": raw[1],
            "p_lambda_biased_phys.npy": biased[0],
            "p_mu_biased_phys.npy": biased[1],
        }
        for name, value in arrays.items():
            np.save(temporary / name, value)
        coordinate_source = verify_artifact_record(
            root, gradient["coordinates"], label="canonical coordinates"
        )
        shutil.copy2(coordinate_source, temporary / "direction_coords.npy")
        artifacts = {
            "raw_lambda": _record_for_final(
                temporary / "p_lambda_raw_phys.npy",
                destination / "p_lambda_raw_phys.npy",
                repo=root,
            ),
            "raw_mu": _record_for_final(
                temporary / "p_mu_raw_phys.npy",
                destination / "p_mu_raw_phys.npy",
                repo=root,
            ),
            "biased_lambda": _record_for_final(
                temporary / "p_lambda_biased_phys.npy",
                destination / "p_lambda_biased_phys.npy",
                repo=root,
            ),
            "biased_mu": _record_for_final(
                temporary / "p_mu_biased_phys.npy",
                destination / "p_mu_biased_phys.npy",
                repo=root,
            ),
            "coordinates": _record_for_final(
                temporary / "direction_coords.npy",
                destination / "direction_coords.npy",
                repo=root,
            ),
        }
        summary = {
            "schema_version": SCHEMA_VERSION,
            "result": optimizer_direction_result(parent),
            "run_id": paths.identity.run_id,
            "iteration": parent,
            "parent_iteration": parent,
            "child_iteration": paths.identity.child_iteration,
            "transition": paths.identity.transition_id,
            "units": "physical Pa",
            "coordinate_system": "physical active controls",
            "normalization": "none",
            "lambda_bias_weight": float(direction_result.lambda_bias_weight),
            "mtilde_slope": slope,
            "h0_or_history_scale": float(direction_result.h0_or_history_scale),
            "registered_gradient_manifest": artifact_record(
                gradient_path, repo=root
            ),
            "accepted_parent_summary": artifact_record(parent_path, repo=root),
            "mtilde": dict(gradient["mtilde"]),
            "active_indices": dict(gradient["active_indices"]),
            "active_h5_indices": dict(gradient["active_h5_indices"]),
            "canonical_coordinates": dict(gradient["coordinates"]),
            "history_outcomes": [item["record"] for item in history],
            "artifacts": artifacts,
            "input_signature_sha256": input_signature,
        }
        atomic_json(temporary / DIRECTION_SUMMARY_NAME, summary)
        temporary.replace(destination)
    except Exception:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise
    return summary_path


def _copy_parent_workspace(source: Path, destination: Path) -> None:
    excluded = {
        "accepted_summary.json",
        "traces",
        "snapshots",
        "snapshot",
        "prot",
        "res",
        "output.solver",
        "output.err",
        "fin_sem",
    }

    def ignore(_: str, names: list[str]) -> set[str]:
        return {name for name in names if name in excluded or name.startswith("output.")}

    shutil.copytree(source, destination, ignore=ignore)


def generate_raw_alpha_candidate(
    *,
    repo: str | Path,
    paths: IterationPaths,
    material_config: Mapping[str, Any],
    accepted_parent_record: Mapping[str, Any],
    direction_record: Mapping[str, Any],
    parameters: ArmijoParameters,
    trial_index: int,
    alpha: float,
) -> Path:
    """Generate m_candidate = m_parent + alpha*p in canonical active order."""

    root = Path(repo).expanduser().resolve()
    parent_path, _, parent_hashes = _validate_accepted_parent(
        root, paths, accepted_parent_record, material_config
    )
    direction_path = verify_artifact_record(
        root, direction_record, label="durable direction"
    )
    direction = _json(direction_path)
    parent = paths.identity.parent_iteration
    require_result(
        direction, optimizer_direction_result(parent), label="durable direction"
    )
    require_identity(direction, paths, label="durable direction")
    _, registered = _registered_gradient(
        root, paths, direction["registered_gradient_manifest"]
    )
    for key, direction_key in (
        ("coordinates", "canonical_coordinates"),
        ("active_indices", "active_indices"),
        ("active_h5_indices", "active_h5_indices"),
        ("mtilde", "mtilde"),
    ):
        if dict(direction[direction_key]) != dict(registered[key]):
            raise ValueError(f"direction {direction_key} differs from registered gradient")
    expected_alpha = float(alpha)
    if not math.isfinite(expected_alpha) or expected_alpha <= 0.0:
        raise ValueError("alpha must be finite and positive")
    index = int(trial_index)
    if index < 0 or index > parameters.maximum_backtracks:
        raise ValueError("trial index is outside configured Armijo schedule")
    scheduled_alpha = parameters.alpha0 * parameters.rho**index
    if expected_alpha != float(scheduled_alpha):
        raise ValueError("candidate alpha differs from configured Armijo schedule")

    p_lambda = _load_array(
        root,
        direction["artifacts"]["biased_lambda"],
        label="biased lambda direction",
        dtype=np.float64,
        ndim=1,
    )
    p_mu = _load_array(
        root,
        direction["artifacts"]["biased_mu"],
        label="biased mu direction",
        dtype=np.float64,
        ndim=1,
    )
    active_h5 = _load_array(
        root,
        direction["active_h5_indices"],
        label="active H5 indices",
        dtype=np.int64,
        ndim=1,
    )
    if p_lambda.shape != p_mu.shape or p_lambda.shape != active_h5.shape:
        raise ValueError("candidate direction/mapping shapes differ")
    destination = candidate_namespace(paths, index, expected_alpha)
    input_payload = {
        "accepted_parent": dict(accepted_parent_record),
        "direction": dict(direction_record),
        "trial_index": index,
        "alpha": expected_alpha,
        "armijo_parameters": {
            "alpha0": parameters.alpha0,
            "rho": parameters.rho,
            "c1": parameters.c1,
            "maximum_backtracks": parameters.maximum_backtracks,
        },
        "active_indices": direction["active_indices"],
        "active_h5_indices": direction["active_h5_indices"],
        "coordinates": direction["canonical_coordinates"],
    }
    input_signature = canonical_sha256(input_payload)
    summary_path = destination / CANDIDATE_SUMMARY_NAME
    if destination.exists():
        existing = _json(summary_path)
        if (
            existing.get("result") != candidate_generated_result(parent, index)
            or existing.get("input_signature_sha256") != input_signature
        ):
            raise ValueError("existing candidate conflicts")
        existing_hashes = {}
        for component, record in existing["candidate_material"].items():
            expected_path = (
                destination
                / str(material_config["directory"])
                / str(material_config["files"][component])
            )
            verify_artifact_record(
                root,
                record,
                label=f"existing candidate {component}",
                expected_path=expected_path,
            )
            existing_hashes[component] = str(record["sha256"])
        if material_signature(existing_hashes) != existing.get(
            "candidate_material_signature_sha256"
        ):
            raise ValueError("existing candidate material signature mismatch")
        verify_artifact_record(
            root,
            existing["candidate_state"],
            label="existing candidate state",
            expected_path=destination / "candidate_state.npz",
        )
        return summary_path

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp.{os.getpid()}")
    if temporary.exists():
        raise FileExistsError(temporary)
    try:
        _copy_parent_workspace(paths.parent_accepted, temporary)
        material_dir = temporary / str(material_config["directory"])
        files = {
            component: material_dir / str(material_config["files"][component])
            for component in ("kappa", "mu", "density")
        }
        dataset = str(material_config["dataset"])
        with h5py.File(files["kappa"], "r+") as kappa_h5, h5py.File(
            files["mu"], "r+"
        ) as mu_h5:
            kappa = np.asarray(kappa_h5[dataset], dtype=np.float64)
            mu = np.asarray(mu_h5[dataset], dtype=np.float64)
            flat_kappa = kappa.reshape(-1)
            flat_mu = mu.reshape(-1)
            if active_h5.size and (
                active_h5.min() < 0 or active_h5.max() >= flat_mu.size
            ):
                raise ValueError("active H5 mapping is outside the material field")
            parent_lambda = flat_kappa[active_h5] - (2.0 / 3.0) * flat_mu[active_h5]
            candidate_lambda = parent_lambda + expected_alpha * p_lambda
            candidate_mu = flat_mu[active_h5] + expected_alpha * p_mu
            if not np.all(np.isfinite(candidate_lambda)) or not np.all(
                np.isfinite(candidate_mu)
            ):
                raise ValueError("candidate contains non-finite material values")
            flat_mu[active_h5] = candidate_mu
            flat_kappa[active_h5] = candidate_lambda + (2.0 / 3.0) * candidate_mu
            mu_h5[dataset][...] = mu
            kappa_h5[dataset][...] = kappa

        candidate_state = temporary / "candidate_state.npz"
        np.savez(
            candidate_state,
            lambda_active=candidate_lambda,
            mu_active=candidate_mu,
            active_h5_indices=active_h5,
            alpha=np.float64(expected_alpha),
            trial_index=np.int64(index),
        )
        material_records = {
            component: _record_for_final(
                path,
                destination / str(material_config["directory"]) / path.name,
                repo=root,
            )
            for component, path in files.items()
        }
        candidate_hashes = {
            component: record["sha256"]
            for component, record in material_records.items()
        }
        if candidate_hashes["density"] != parent_hashes["density"]:
            raise ValueError("candidate density differs from accepted parent")
        summary = {
            "schema_version": SCHEMA_VERSION,
            "result": candidate_generated_result(parent, index),
            "run_id": paths.identity.run_id,
            "parent_iteration": parent,
            "child_iteration": paths.identity.child_iteration,
            "transition": paths.identity.transition_id,
            "trial_index": index,
            "alpha": expected_alpha,
            "formula": "m_candidate = m_parent + alpha * p_parent",
            "normalization": "none",
            "accepted_parent_summary": artifact_record(parent_path, repo=root),
            "parent_material_sha256": parent_hashes,
            "direction_summary": artifact_record(direction_path, repo=root),
            "canonical_coordinates": dict(direction["canonical_coordinates"]),
            "active_indices": dict(direction["active_indices"]),
            "active_h5_indices": dict(direction["active_h5_indices"]),
            "candidate_material": material_records,
            "candidate_material_signature_sha256": material_signature(
                candidate_hashes
            ),
            "candidate_state": _record_for_final(
                candidate_state, destination / candidate_state.name, repo=root
            ),
            "input_signature_sha256": input_signature,
        }
        atomic_json(temporary / CANDIDATE_SUMMARY_NAME, summary)
        temporary.replace(destination)
    except Exception:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise
    return summary_path


@dataclass(frozen=True)
class CandidateEvaluation:
    candidate_material_signature_sha256: str
    current_receiver: Mapping[str, Any]
    true_receiver: Mapping[str, Any]
    objective: float
    sample_count: int
    receiver_count: int
    component_count: int
    dt: float


def evaluate_candidate_external(
    *,
    repo: str | Path,
    paths: IterationPaths,
    runtime_config: Mapping[str, Any],
    reference_manifest: str | Path,
    candidate_summary_path: str | Path,
    trial_directory: str | Path,
    batch_size: int = 2048,
    checkpoint_interval: int = 100,
) -> CandidateEvaluation:
    """Delegate one candidate evaluation to the certified external operator.

    This function is the only numerical boundary in this module. It calls the
    existing ExternalForwardDriver/run_external_forward implementation and
    never invokes SEM3D or MPI. Its checkpoint path makes an interrupted trial
    resumable.
    """

    from scripts.exact_adjoint.certify_exact_adjoint_with_fixed_dt_fd import (
        trapezoid_weights,
    )
    from scripts.exact_adjoint.s43_external_forward import (
        ExternalForwardDriver,
        load_certified_reference,
        run_external_forward,
    )

    root = Path(repo).expanduser().resolve()
    candidate_path = Path(candidate_summary_path).expanduser().resolve()
    candidate = _json(candidate_path)
    parent = paths.identity.parent_iteration
    trial_index = int(candidate["trial_index"])
    require_result(
        candidate,
        candidate_generated_result(parent, trial_index),
        label="candidate external-forward input",
    )
    require_identity(candidate, paths, label="candidate external-forward input")
    material_dir = candidate_path.parent / str(
        runtime_config.get("material_directory", "mat/h5")
    )
    if not material_dir.is_dir():
        material_dir = candidate_path.parent / "mat" / "h5"
    actual_hashes = {
        component: sha256_file(
            verify_artifact_record(
                root, record, label=f"candidate {component}"
            )
        )
        for component, record in candidate["candidate_material"].items()
    }
    if material_signature(actual_hashes) != candidate[
        "candidate_material_signature_sha256"
    ]:
        raise ValueError("candidate material signature mismatch before evaluation")
    reference_path = Path(reference_manifest).expanduser().resolve()
    load_certified_reference(root, paths.identity.run_id, reference_path)
    driver = ExternalForwardDriver(
        root,
        paths.identity.run_id,
        material_dir,
        batch_size=int(batch_size),
        reference_manifest=reference_path,
    )
    forward = runtime_config["forward_operator"]
    sample_count = int(forward["expected_sample_count"])
    receiver_count = int(forward["physical_receiver_count"])
    component_count = int(forward["dimension"])
    if driver.receiver_count != receiver_count:
        raise ValueError("candidate receiver count differs from CURRENT contract")
    trial_dir = Path(trial_directory).expanduser().resolve()
    trial_dir.mkdir(parents=True, exist_ok=True)
    current_path = trial_dir / "candidate_external_receiver.npy"
    checkpoint = trial_dir / "checkpoint" / "candidate_latest.npz"
    # run_external_forward has a fixed primal-state label contract.  The
    # output file may be a candidate receiver, but the state label itself
    # remains ``primal`` unless tangent directions are explicitly supplied.
    run_external_forward(
        driver,
        sample_count,
        {"primal": current_path},
        checkpoint,
        checkpoint_interval=int(checkpoint_interval),
    )
    current = np.asarray(np.load(current_path), dtype=np.float64)
    true_path = Path(driver.paths["true_external"]).resolve()
    truth = np.asarray(np.load(true_path), dtype=np.float64)
    expected_shape = (sample_count, receiver_count, component_count)
    if current.shape != expected_shape or truth.shape != expected_shape:
        raise ValueError("candidate/TRUE receiver shape mismatch")
    if not np.all(np.isfinite(current)) or not np.all(np.isfinite(truth)):
        raise ValueError("candidate/TRUE receiver contains non-finite values")
    weights = trapezoid_weights(
        np.arange(sample_count, dtype=np.float64) * float(driver.dt)
    )
    residual = current - truth
    objective = 0.5 * float(
        np.sum(weights[:, None, None] * residual * residual)
    )
    return CandidateEvaluation(
        candidate_material_signature_sha256=str(
            candidate["candidate_material_signature_sha256"]
        ),
        current_receiver=artifact_record(current_path, repo=root),
        true_receiver=artifact_record(true_path, repo=root),
        objective=objective,
        sample_count=sample_count,
        receiver_count=receiver_count,
        component_count=component_count,
        dt=float(driver.dt),
    )


def persist_armijo_trial(
    *,
    repo: str | Path,
    paths: IterationPaths,
    armijo_manifest: Mapping[str, Any],
    candidate_summary_path: str | Path,
    evaluation: CandidateEvaluation,
    trial_directory: str | Path,
) -> Path:
    """Bind one certified candidate evaluation to one exact Armijo decision."""

    root = Path(repo).expanduser().resolve()
    parent = paths.identity.parent_iteration
    require_result(
        armijo_manifest, armijo_ready_result(parent), label="Armijo manifest"
    )
    require_identity(armijo_manifest, paths, label="Armijo manifest")
    _validate_armijo_inputs(root, paths, armijo_manifest)
    candidate_path = Path(candidate_summary_path).expanduser().resolve()
    candidate = _json(candidate_path)
    index = int(candidate["trial_index"])
    require_result(
        candidate,
        candidate_generated_result(parent, index),
        label="candidate",
    )
    require_identity(candidate, paths, label="candidate")
    if (
        candidate["direction_summary"] != armijo_manifest["direction_artifact"]
        or candidate["accepted_parent_summary"]
        != armijo_manifest["parent_accepted_artifact"]
    ):
        raise ValueError("candidate parent/direction differs from Armijo contract")
    scheduled = armijo_manifest["parameters"]
    if (
        float(candidate["alpha"])
        != float(scheduled["alpha0"]) * float(scheduled["rho"]) ** index
    ):
        raise ValueError("candidate alpha differs from Armijo schedule")
    if evaluation.candidate_material_signature_sha256 != str(
        candidate["candidate_material_signature_sha256"]
    ):
        raise ValueError("evaluated candidate SHA does not match candidate manifest")
    trial_dir = Path(trial_directory).expanduser().resolve()
    expected_dir = (
        paths.line_search_root / "trials" / candidate_path.parent.name
    ).resolve()
    if trial_dir != expected_dir:
        raise ValueError("trial directory is not canonical")
    current_path = verify_artifact_record(
        root, evaluation.current_receiver, label="candidate receiver"
    )
    if current_path != trial_dir / "candidate_external_receiver.npy":
        raise ValueError("candidate receiver path is not canonical")
    true_path = verify_artifact_record(
        root, evaluation.true_receiver, label="true receiver"
    )
    expected_true = armijo_manifest["true_receiver_artifact"]
    if dict(evaluation.true_receiver) != dict(expected_true):
        raise ValueError("evaluated TRUE receiver differs from Armijo contract")
    objective = float(evaluation.objective)
    parent_objective = float(armijo_manifest["parent_objective"])
    alpha = float(candidate["alpha"])
    parameters = armijo_manifest["parameters"]
    decision = armijo_decision(
        parent_objective=parent_objective,
        candidate_objective=objective,
        slope=float(armijo_manifest["slope"]),
        alpha=alpha,
        c1=float(parameters["c1"]),
    )
    input_payload = {
        "armijo_signature": armijo_manifest["input_signature_sha256"],
        "candidate_summary": artifact_record(candidate_path, repo=root),
        "candidate_receiver": dict(evaluation.current_receiver),
        "true_receiver": dict(evaluation.true_receiver),
        "objective": objective,
    }
    signature = canonical_sha256(input_payload)
    summary_path = trial_dir / TRIAL_SUMMARY_NAME
    if summary_path.is_file():
        existing = _json(summary_path)
        if existing.get("input_signature_sha256") != signature:
            raise ValueError("existing trial conflicts")
        return summary_path
    trial_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": SCHEMA_VERSION,
        "result": armijo_trial_result(parent, index, accepted=decision["accepted"]),
        "objective_result": candidate_objective_result(parent, index),
        "run_id": paths.identity.run_id,
        "parent_iteration": parent,
        "child_iteration": paths.identity.child_iteration,
        "transition": paths.identity.transition_id,
        "trial_index": index,
        "alpha": alpha,
        "accepted": decision["accepted"],
        "strict_descent": decision["strict_descent"],
        "armijo": decision["armijo"],
        "armijo_rhs": decision["armijo_rhs"],
        "c1": float(parameters["c1"]),
        "rho": float(parameters["rho"]),
        "parent_objective": parent_objective,
        "candidate_objective": objective,
        "slope": float(armijo_manifest["slope"]),
        "parent_accepted_model": dict(
            armijo_manifest["parent_accepted_artifact"]
        ),
        "registered_gradient": dict(armijo_manifest["gradient_artifact"]),
        "durable_direction": dict(armijo_manifest["direction_artifact"]),
        "candidate_summary": artifact_record(candidate_path, repo=root),
        "candidate_material_signature_sha256": str(
            candidate["candidate_material_signature_sha256"]
        ),
        "candidate_receiver": artifact_record(current_path, repo=root),
        "true_receiver": artifact_record(true_path, repo=root),
        "sample_count": int(evaluation.sample_count),
        "receiver_count": int(evaluation.receiver_count),
        "component_count": int(evaluation.component_count),
        "dt": float(evaluation.dt),
        "input_signature_sha256": signature,
    }
    atomic_json(summary_path, payload)
    return summary_path


CandidateProvider = Callable[[int, float], Path]
EvaluationProvider = Callable[[Path, Path], CandidateEvaluation]


def _accepted_objective(summary: Mapping[str, Any]) -> float:
    objective = summary.get("objective")
    if not isinstance(objective, Mapping):
        raise ValueError("accepted parent lacks objective")
    value = objective.get("accepted", objective.get("J_external"))
    result = float(value)
    if not math.isfinite(result) or result < 0.0:
        raise ValueError("accepted parent objective is invalid")
    return result


def _validate_armijo_inputs(
    repo: Path,
    paths: IterationPaths,
    manifest: Mapping[str, Any],
) -> dict[str, Any]:
    parent = paths.identity.parent_iteration
    require_result(manifest, armijo_ready_result(parent), label="Armijo manifest")
    require_identity(manifest, paths, label="Armijo manifest")
    parent_path = verify_artifact_record(
        repo,
        manifest["parent_accepted_artifact"],
        label="Armijo accepted parent",
        expected_path=paths.parent_accepted / "accepted_summary.json",
    )
    parent_summary = _json(parent_path)
    if (
        parent_summary.get("result") != accepted_model_result(parent)
        or parent_summary.get("run") != paths.identity.run_id
        or int(parent_summary.get("iter", -1)) != parent
    ):
        raise ValueError("Armijo accepted parent result/identity mismatch")
    if float(manifest["parent_objective"]) != _accepted_objective(parent_summary):
        raise ValueError("Armijo parent objective differs from accepted J_k")
    gradient_path, _ = _registered_gradient(
        repo, paths, manifest["gradient_artifact"]
    )
    direction_path = verify_artifact_record(
        repo, manifest["direction_artifact"], label="Armijo direction"
    )
    direction = _json(direction_path)
    require_result(
        direction, optimizer_direction_result(parent), label="Armijo direction"
    )
    require_identity(direction, paths, label="Armijo direction")
    if direction["registered_gradient_manifest"] != artifact_record(
        gradient_path, repo=repo
    ):
        raise ValueError("Armijo direction and gradient provenance differ")
    if direction["accepted_parent_summary"] != artifact_record(
        parent_path, repo=repo
    ):
        raise ValueError("Armijo direction and parent provenance differ")
    if float(direction["mtilde_slope"]) != float(manifest["slope"]):
        raise ValueError("Armijo slope differs from durable direction")
    verify_artifact_record(
        repo, direction["artifacts"]["biased_lambda"], label="Armijo lambda direction"
    )
    verify_artifact_record(
        repo, direction["artifacts"]["biased_mu"], label="Armijo mu direction"
    )
    verify_artifact_record(
        repo, manifest["true_receiver_artifact"], label="Armijo TRUE receiver"
    )
    return {
        "parent_path": parent_path,
        "gradient_path": gradient_path,
        "direction_path": direction_path,
        "direction": direction,
    }


def execute_current_armijo(
    *,
    repo: str | Path,
    paths: IterationPaths,
    armijo_manifest: Mapping[str, Any],
    candidate_provider: CandidateProvider,
    evaluation_provider: EvaluationProvider,
) -> Path:
    """Run or resume the CURRENT trial sequence through injected producers."""

    root = Path(repo).expanduser().resolve()
    parent = paths.identity.parent_iteration
    _validate_armijo_inputs(root, paths, armijo_manifest)
    parameters = ArmijoParameters(**armijo_manifest["parameters"])
    final_path = paths.line_search_root / ARMIJO_SUMMARY_NAME
    if final_path.is_file():
        existing = _json(final_path)
        if existing.get("input_signature_sha256") != armijo_manifest[
            "input_signature_sha256"
        ]:
            raise ValueError("existing Armijo summary conflicts")
        require_result(
            existing,
            armijo_search_result(
                parent, accepted=bool(existing.get("accepted"))
            ),
            label="existing Armijo summary",
        )
        require_identity(existing, paths, label="existing Armijo summary")
        if existing.get("accepted"):
            accepted_path = verify_artifact_record(
                root, existing["accepted_trial"], label="existing accepted trial"
            )
            accepted_trial = _json(accepted_path)
            if accepted_trial.get("accepted") is not True:
                raise ValueError("existing Armijo accepted trial is rejected")
            trial_index = int(accepted_trial.get("trial_index", -1))
            require_result(
                accepted_trial,
                armijo_trial_result(parent, trial_index, accepted=True),
                label="existing accepted Armijo trial",
            )
            require_identity(
                accepted_trial, paths, label="existing accepted Armijo trial"
            )
        return final_path

    trials = []
    accepted_record = None
    for trial_index, alpha in parameters.schedule():
        candidate_path = candidate_provider(trial_index, alpha)
        candidate = _json(candidate_path)
        require_result(
            candidate,
            candidate_generated_result(parent, trial_index),
            label="candidate",
        )
        if float(candidate["alpha"]) != float(alpha):
            raise ValueError("candidate alpha differs from Armijo schedule")
        trial_dir = (
            paths.line_search_root / "trials" / candidate_path.parent.name
        )
        trial_path = trial_dir / TRIAL_SUMMARY_NAME
        if trial_path.is_file():
            trial = _json(trial_path)
            require_result(
                trial,
                armijo_trial_result(
                    parent, trial_index, accepted=bool(trial.get("accepted"))
                ),
                label="resumed Armijo trial",
            )
            require_identity(trial, paths, label="resumed Armijo trial")
            if trial["candidate_summary"] != artifact_record(
                candidate_path, repo=root
            ):
                raise ValueError("resumed trial candidate SHA mismatch")
            if (
                float(trial["parent_objective"])
                != float(armijo_manifest["parent_objective"])
                or float(trial["slope"]) != float(armijo_manifest["slope"])
                or trial["registered_gradient"]
                != armijo_manifest["gradient_artifact"]
                or trial["durable_direction"]
                != armijo_manifest["direction_artifact"]
            ):
                raise ValueError("resumed trial provenance differs from Armijo input")
        else:
            evaluation = evaluation_provider(candidate_path, trial_dir)
            trial_path = persist_armijo_trial(
                repo=root,
                paths=paths,
                armijo_manifest=armijo_manifest,
                candidate_summary_path=candidate_path,
                evaluation=evaluation,
                trial_directory=trial_dir,
            )
            trial = _json(trial_path)
        record = artifact_record(trial_path, repo=root)
        trials.append(record)
        if trial["accepted"] is True:
            accepted_record = record
            break

    accepted = accepted_record is not None
    payload = {
        "schema_version": SCHEMA_VERSION,
        "result": armijo_search_result(parent, accepted=accepted),
        "run_id": paths.identity.run_id,
        "parent_iteration": parent,
        "child_iteration": paths.identity.child_iteration,
        "transition": paths.identity.transition_id,
        "accepted": accepted,
        "accepted_trial": accepted_record,
        "trials": trials,
        "input_signature_sha256": armijo_manifest["input_signature_sha256"],
    }
    atomic_json(final_path, payload)
    return final_path


def promote_current_accepted_trial(
    *,
    repo: str | Path,
    paths: IterationPaths,
    material_config: Mapping[str, Any],
    armijo_summary_record: Mapping[str, Any],
) -> Path:
    """Promote only a durable accepted trial with byte-identical materials."""

    root = Path(repo).expanduser().resolve()
    parent = paths.identity.parent_iteration
    child = paths.identity.child_iteration
    armijo_path = verify_artifact_record(
        root,
        armijo_summary_record,
        label="Armijo summary",
        expected_path=paths.line_search_root / ARMIJO_SUMMARY_NAME,
    )
    armijo = _json(armijo_path)
    require_result(
        armijo, armijo_search_result(parent, accepted=True), label="Armijo summary"
    )
    require_identity(armijo, paths, label="Armijo summary")
    if armijo.get("accepted") is not True or not armijo.get("accepted_trial"):
        raise ValueError("Armijo summary has no accepted trial")
    trial_path = verify_artifact_record(
        root, armijo["accepted_trial"], label="accepted Armijo trial"
    )
    trial = _json(trial_path)
    index = int(trial["trial_index"])
    require_result(
        trial,
        armijo_trial_result(parent, index, accepted=True),
        label="accepted Armijo trial",
    )
    require_identity(trial, paths, label="accepted Armijo trial")
    if trial.get("accepted") is not True:
        raise ValueError("rejected trial cannot be promoted")
    candidate_path = verify_artifact_record(
        root, trial["candidate_summary"], label="accepted candidate summary"
    )
    candidate = _json(candidate_path)
    require_result(
        candidate,
        candidate_generated_result(parent, index),
        label="accepted candidate",
    )
    require_identity(candidate, paths, label="accepted candidate")
    if trial.get("parent_accepted_model") != candidate.get(
        "accepted_parent_summary"
    ):
        raise ValueError("accepted trial/candidate parent provenance mismatch")
    if trial.get("durable_direction") != candidate.get("direction_summary"):
        raise ValueError("accepted trial/candidate direction provenance mismatch")
    if candidate["candidate_material_signature_sha256"] != trial[
        "candidate_material_signature_sha256"
    ]:
        raise ValueError("accepted trial candidate SHA mismatch")
    source_dir = candidate_path.parent
    candidate_state_path = verify_artifact_record(
        root, candidate["candidate_state"], label="candidate state"
    )
    if (
        paths.child_state.exists()
        and sha256_file(paths.child_state) != sha256_file(candidate_state_path)
    ):
        raise ValueError("existing child state conflicts")
    for component, record in candidate["candidate_material"].items():
        verify_artifact_record(
            root,
            record,
            label=f"accepted candidate {component}",
            expected_path=source_dir
            / str(material_config["directory"])
            / str(material_config["files"][component]),
        )
    input_payload = {
        "armijo_summary": dict(armijo_summary_record),
        "accepted_trial": dict(armijo["accepted_trial"]),
        "candidate_summary": dict(trial["candidate_summary"]),
        "candidate_material_signature_sha256": candidate[
            "candidate_material_signature_sha256"
        ],
    }
    input_signature = canonical_sha256(input_payload)
    accepted_summary_path = paths.child_accepted / "accepted_summary.json"
    if paths.child_accepted.exists():
        existing = _json(accepted_summary_path)
        if (
            existing.get("result") != accepted_model_result(child)
            or existing.get("promotion_input_signature_sha256") != input_signature
        ):
            raise ValueError("existing accepted child conflicts")
        existing_hashes = {}
        for component, record in existing["material"].items():
            verify_artifact_record(
                root,
                record,
                label=f"existing accepted child {component}",
                expected_path=(
                    paths.child_accepted
                    / str(material_config["directory"])
                    / str(material_config["files"][component])
                ),
            )
            if str(record["sha256"]) != str(
                candidate["candidate_material"][component]["sha256"]
            ):
                raise ValueError("existing accepted child differs from candidate")
            existing_hashes[component] = str(record["sha256"])
        if material_signature(existing_hashes) != candidate[
            "candidate_material_signature_sha256"
        ]:
            raise ValueError("existing accepted child material signature mismatch")
        if (
            not paths.child_state.is_file()
            or sha256_file(paths.child_state) != sha256_file(candidate_state_path)
        ):
            raise ValueError("existing accepted child state is incomplete")
        return accepted_summary_path

    paths.child_accepted.parent.mkdir(parents=True, exist_ok=True)
    temporary = paths.child_accepted.with_name(
        f".{paths.child_accepted.name}.tmp.{os.getpid()}"
    )
    if temporary.exists():
        raise FileExistsError(temporary)
    try:
        shutil.copytree(source_dir, temporary)
        (temporary / CANDIDATE_SUMMARY_NAME).unlink(missing_ok=True)
        material_records = {}
        material_hashes = {}
        for component in ("kappa", "mu", "density"):
            temporary_material = (
                temporary
                / str(material_config["directory"])
                / str(material_config["files"][component])
            )
            final_material = (
                paths.child_accepted
                / str(material_config["directory"])
                / str(material_config["files"][component])
            )
            record = _record_for_final(
                temporary_material, final_material, repo=root
            )
            if record["sha256"] != candidate["candidate_material"][component][
                "sha256"
            ]:
                raise ValueError("promoted child material differs from candidate")
            material_records[component] = record
            material_hashes[str(material_config["files"][component])] = record[
                "sha256"
            ]
        summary = {
            "schema_version": SCHEMA_VERSION,
            "result": accepted_model_result(child),
            "promotion_result": promotion_result(parent, child),
            "run": paths.identity.run_id,
            "run_id": paths.identity.run_id,
            "iter": child,
            "parent_iteration": parent,
            "child_iteration": child,
            "transition": paths.identity.transition_id,
            "accepted_alpha": float(trial["alpha"]),
            "objective": {
                "parent": float(trial["parent_objective"]),
                "accepted": float(trial["candidate_objective"]),
            },
            "parent_accepted_model": dict(trial["parent_accepted_model"]),
            "registered_parent_gradient": dict(trial["registered_gradient"]),
            "direction": dict(trial["durable_direction"]),
            "candidate": dict(trial["candidate_summary"]),
            "candidate_objective_trial": dict(armijo["accepted_trial"]),
            "armijo_summary": dict(armijo_summary_record),
            "accepted_external_receiver": dict(trial["candidate_receiver"]),
            "true_external_receiver": dict(trial["true_receiver"]),
            "external_receiver_sha256": str(
                trial["candidate_receiver"]["sha256"]
            ),
            "true_external_sha256": str(trial["true_receiver"]["sha256"]),
            "material": material_records,
            "material_sha256": material_hashes,
            "candidate_material_signature_sha256": str(
                candidate["candidate_material_signature_sha256"]
            ),
            "promotion_input_signature_sha256": input_signature,
        }
        atomic_json(temporary / "accepted_summary.json", summary)
        temporary.replace(paths.child_accepted)

        paths.child_state.parent.mkdir(parents=True, exist_ok=True)
        if paths.child_state.exists():
            if sha256_file(paths.child_state) != sha256_file(candidate_state_path):
                raise ValueError("existing child state conflicts")
        else:
            state_tmp = paths.child_state.with_name(paths.child_state.name + ".tmp")
            shutil.copy2(candidate_state_path, state_tmp)
            os.replace(state_tmp, paths.child_state)
    except Exception:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise
    return accepted_summary_path
