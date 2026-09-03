#!/usr/bin/env python3
"""Preflight and run one generic exact-reverse material-covector production."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Mapping

import numpy as np


REPOSITORY_IMPORT_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_IMPORT_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_IMPORT_ROOT))

from scripts.exact_adjoint.certify_exact_adjoint_with_fixed_dt_fd import (
    trapezoid_weights,
)
from scripts.exact_adjoint.s43_external_forward import (
    ExternalForwardDriver,
    sha256_arrays,
    sha256_file,
)
from scripts.exact_adjoint.s43_external_reverse_core import (
    atomic_json,
    retained_checkpoint_map,
)
from scripts.fathi_benchmark import (
    run_certified_external_exact_reverse as certified_reverse,
)
from scripts.fathi_benchmark.iteration_context import build_iteration_paths
from scripts.fathi_benchmark.runtime_paths import (
    iteration_runtime_paths,
    resolve_path,
)
from scripts.fathi_benchmark.current_pipeline_contracts import (
    accepted_model_result,
    exact_reverse_result,
    retained_primal_result,
)


GRADIENT_NAMES = (
    "solid_lambda",
    "solid_mu",
    "pml_lambda",
    "pml_mu",
)
SOURCE_EQUIVALENCE = {
    "reverse_delegate": (
        "scripts.fathi_benchmark.run_certified_external_exact_reverse.run_reverse"
    ),
    "material_vjp": (
        f"{certified_reverse.material_vjp.__module__}."
        f"{certified_reverse.material_vjp.__name__}"
    ),
    "adjoint_step": (
        f"{certified_reverse.adjoint_step.__module__}."
        f"{certified_reverse.adjoint_step.__name__}"
    ),
    "ensure_replay_cache": (
        f"{certified_reverse.ensure_replay_cache.__module__}."
        f"{certified_reverse.ensure_replay_cache.__name__}"
    ),
    "load_replay_state": (
        f"{certified_reverse.load_replay_state.__module__}."
        f"{certified_reverse.load_replay_state.__name__}"
    ),
    "cleanup_replay_cache": (
        f"{certified_reverse.cleanup_replay_cache.__module__}."
        f"{certified_reverse.cleanup_replay_cache.__name__}"
    ),
    "receiver_seed": (
        "trapezoid_weights * (current_external_receiver - true_external_receiver)"
    ),
    "reverse_order": "sample_count-1 down to 0",
}


def _preflight_result(iteration: int) -> str:
    return f"PASS_ITER{int(iteration):03d}_EXACT_REVERSE_PRODUCTION_PREFLIGHT"


def _reverse_result(iteration: int) -> str:
    return exact_reverse_result(iteration)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    _require(isinstance(value, dict), f"JSON payload must be an object: {path}")
    return value


def _resolve(repo: Path, value: str | Path) -> Path:
    return resolve_path(value, base=repo).resolve()


def _recorded_path(repo: Path, value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (repo / path).resolve()


def _below(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
    except ValueError:
        return False
    return True


def _file(path: Path) -> dict[str, Any]:
    _require(path.is_file(), f"missing frozen input: {path}")
    return {
        "path": str(path),
        "resolved_path": str(path.resolve()),
        "sha256": sha256_file(path),
    }


def _directory(path: Path) -> dict[str, Any]:
    root = path.resolve()
    _require(root.is_dir(), f"missing frozen input directory: {path}")
    rows = []
    digest = hashlib.sha256()
    for item in sorted(value for value in root.rglob("*") if value.is_file()):
        relative = item.relative_to(root).as_posix()
        item_hash = sha256_file(item)
        rows.append({"relative_path": relative, "sha256": item_hash})
        digest.update(f"{item_hash}  {relative}\n".encode("utf-8"))
    return {
        "path": str(path),
        "resolved_path": str(root),
        "file_count": len(rows),
        "content_signature_sha256": digest.hexdigest(),
        "files": rows,
    }


def _trace(path: Path, shape: tuple[int, int, int]) -> np.ndarray:
    value = np.load(path, mmap_mode="r")
    _require(value.dtype == np.float64, f"trace dtype is not float64: {path}")
    _require(value.shape == shape, f"trace shape mismatch: {path}: {value.shape}")
    _require(np.all(np.isfinite(value)), f"trace is non-finite: {path}")
    return value


def _driver_hashes(driver: ExternalForwardDriver) -> dict[str, Any]:
    result = {
        key: _directory(Path(driver.paths[key]))
        for key in ("topology", "coefficients", "coupled_mass", "receiver")
    }
    result.update(
        {
            key: _file(Path(driver.paths[key]))
            for key in ("config", "gll", "weights", "stf")
        }
    )
    if driver.source_coordinates_path is not None:
        result["source_coordinates"] = _file(driver.source_coordinates_path)
    if driver.source_amplitudes_path is not None:
        result["source_amplitudes"] = _file(driver.source_amplitudes_path)
    result["receiver_operator_sha256"] = sha256_arrays(
        driver.receiver_nodes, driver.receiver_weights
    )
    return result


def _retained_positions(sample_count: int) -> list[int]:
    result = list(range(0, int(sample_count), 50))
    if not result or result[-1] != int(sample_count):
        result.append(int(sample_count))
    return result


def _signature(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def build_runtime(args: argparse.Namespace) -> dict[str, Any]:
    repo = Path(args.repo).expanduser().resolve()
    config_path = _resolve(repo, args.config)
    config = _json(config_path)
    run_id = str(config["benchmark_name"])
    runtime_paths = iteration_runtime_paths(config, args.iter_k, repo_root=repo)
    engine_path = (
        _resolve(repo, args.engine_config)
        if args.engine_config
        else repo / "configs" / f"{run_id}_iteration_engine.json"
    )
    engine = _json(engine_path)
    paths = build_iteration_paths(
        engine,
        args.iter_k,
        child_iteration=args.iter_k + 1,
        repository_root=repo,
        runtime_root=runtime_paths["runtime_root"],
    )
    _require(paths.identity.run_id == run_id, "run identity differs across configs")
    _require(
        paths.transition_root == Path(runtime_paths["transition_root"]),
        "transition path differs across configs",
    )
    _require(
        paths.parent_accepted == Path(runtime_paths["parent_workspace"]),
        "parent path differs across configs",
    )

    primal_root = paths.exact_reverse / "primal_forward"
    primal_summary_path = (
        _resolve(repo, args.primal_summary)
        if args.primal_summary
        else primal_root / "summary.json"
    )
    primal = _json(primal_summary_path)
    current_path = (
        _resolve(repo, args.current_trace)
        if args.current_trace
        else _recorded_path(repo, primal["current_external_receiver"]["path"])
    )
    true_path = (
        _resolve(repo, args.true_trace)
        if args.true_trace
        else _recorded_path(repo, primal["true_external_receiver"]["path"])
    )
    retained_dir = (
        _resolve(repo, args.retained_primal_dir)
        if args.retained_primal_dir
        else _recorded_path(repo, primal["retained_primal"]["directory"])
    )
    output_dir = (
        _resolve(repo, args.output_dir)
        if args.output_dir
        else paths.exact_reverse / "production_reverse"
    )
    _require(
        output_dir == paths.exact_reverse / "production_reverse",
        "output is not the canonical production_reverse directory",
    )
    _require(
        current_path == primal_root / "current_external_receiver.npy",
        "current trace is not transition-local retained-primal output",
    )
    _require(
        retained_dir == primal_root / "checkpoint" / "current_primal_retained",
        "retained-primal directory is not transition-local",
    )
    mutable = (paths.parent_accepted, current_path, retained_dir, output_dir)
    historical = str(engine.get("historical_run_id", ""))
    _require(all(run_id in str(path) for path in mutable), "mutable path left CURRENT")
    _require(
        not historical or all(historical not in str(path) for path in mutable),
        "mutable path entered historical namespace",
    )
    _require(_below(output_dir, paths.transition_root), "output escapes transition")

    expected_identity = (
        run_id,
        int(args.iter_k),
        int(args.iter_k) + 1,
        paths.identity.transition_id,
    )
    actual_identity = (
        str(primal.get("run_id", "")),
        int(primal.get("parent_iteration", -1)),
        int(primal.get("child_iteration", -1)),
        str(primal.get("transition", "")),
    )
    _require(actual_identity == expected_identity, "primal summary identity mismatch")
    _require(
        primal.get("result") == retained_primal_result(args.iter_k),
        "primal result contract mismatch",
    )

    accepted_summary_path = (
        _resolve(repo, args.accepted_summary)
        if args.accepted_summary
        else paths.parent_accepted / "accepted_summary.json"
    )
    accepted = _json(accepted_summary_path)
    _require(int(accepted.get("iter", -1)) == args.iter_k, "accepted iter mismatch")
    _require(str(accepted.get("run", "")) == run_id, "accepted run mismatch")
    _require(
        accepted.get("result") == accepted_model_result(args.iter_k),
        "accepted parent result contract mismatch",
    )
    trial_record = accepted.get("candidate_objective_trial")
    trial_path = (
        _recorded_path(repo, trial_record["path"])
        if isinstance(trial_record, Mapping)
        else _recorded_path(repo, accepted["external_armijo_trial"])
    )
    trial = _json(trial_path)
    _require(trial.get("accepted") is True, "Armijo trial is not accepted")
    if args.iter_k > 0:
        prior = iteration_runtime_paths(config, args.iter_k - 1, repo_root=repo)
        _require(
            _below(trial_path, Path(prior["transition_root"])),
            "accepted trial is outside predecessor transition",
        )
    accepted_receiver = accepted.get("accepted_external_receiver")
    accepted_trace = (
        _resolve(repo, args.accepted_trace)
        if args.accepted_trace
        else (
            _recorded_path(repo, accepted_receiver["path"])
            if isinstance(accepted_receiver, Mapping)
            else _recorded_path(repo, trial["candidate_external_receiver"])
        )
    )

    material_dir = paths.parent_accepted / "mat" / "h5"
    material_files = {
        component: material_dir / str(engine["material"]["files"][component])
        for component in ("kappa", "mu", "density")
    }
    material_hashes = {name: sha256_file(path) for name, path in material_files.items()}
    for component, actual in material_hashes.items():
        _require(
            actual == str(primal["material_sha256"][component]),
            f"{component} differs from primal summary",
        )
        _require(
            actual == str(accepted["material_sha256"][material_files[component].name]),
            f"{component} differs from accepted summary",
        )
    _require(
        _recorded_path(repo, primal["material_dir"]) == material_dir,
        "primal material is not accepted parent material",
    )

    reference_path = (
        _resolve(repo, args.reference_manifest)
        if args.reference_manifest
        else (
            _recorded_path(repo, primal["reference_manifest"])
            if primal.get("reference_manifest")
            else (repo / "results" / run_id / "certified_external_reference.json").resolve()
        )
    )
    _require(reference_path.is_file(), f"missing certified reference: {reference_path}")
    recorded_reference_sha = primal.get("reference_manifest_sha256")
    if recorded_reference_sha:
        _require(
            sha256_file(reference_path) == str(recorded_reference_sha),
            "certified reference differs from retained primal",
        )

    if args.driver_root:
        # Explicit compatibility override for archived/current-T052 certification
        # reproductions. New CURRENT iterations must use the run-level certified
        # reference and repository root instead of a transition-local compat_repo.
        driver = ExternalForwardDriver(
            _resolve(repo, args.driver_root),
            run_id,
            material_dir,
            batch_size=args.batch_size,
        )
    else:
        driver = ExternalForwardDriver(
            repo,
            run_id,
            material_dir,
            batch_size=args.batch_size,
            reference_manifest=reference_path,
        )
    _require(
        driver.signature == str(primal["driver_signature_sha256"]),
        "driver signature differs from retained primal",
    )
    forward = config["forward_operator"]
    sample_count = int(forward["expected_sample_count"])
    source_count = int(forward["source_count"])
    receiver_count = int(forward["physical_receiver_count"])
    configured_dt = float(forward["effective_dt_s"])
    _require(abs(driver.dt - configured_dt) <= 1.0e-18, "dt mismatch")
    _require(len(driver.source_nodes) == source_count, "source count mismatch")
    _require(driver.receiver_count == receiver_count, "receiver count mismatch")
    _require(
        np.isclose(
            np.sum(driver.source_amplitudes),
            float(forward["assembled_peak_force_n"]),
            rtol=1.0e-14,
            atol=1.0e-8,
        ),
        "assembled source amplitude mismatch",
    )

    shape = (sample_count, receiver_count, 3)
    current_hash = sha256_file(current_path)
    true_hash = sha256_file(true_path)
    accepted_hash = sha256_file(accepted_trace)
    _require(
        current_hash == str(primal["current_external_receiver"]["sha256"]),
        "current trace hash differs from primal summary",
    )
    _require(
        primal["current_external_receiver"].get(
            "bitwise_equal_to_accepted_parent",
            primal["current_external_receiver"].get(
                "bitwise_equal_to_accepted_alpha1"
            ),
        )
        is True,
        "primal summary accepted-trace bitwise gate is false",
    )
    _require(
        current_hash == accepted_hash
        == str(accepted["external_receiver_sha256"])
        == str(
            trial.get(
                "candidate_external_sha256",
                trial.get("candidate_receiver", {}).get("sha256", ""),
            )
        ),
        "current trace is not bitwise equal to accepted trace",
    )
    _require(
        true_hash == str(primal["true_external_receiver"]["sha256"])
        == str(accepted["true_external_sha256"])
        == str(
            trial.get(
                "true_external_sha256",
                trial.get("true_receiver", {}).get("sha256", ""),
            )
        ),
        "TRUE trace hash differs from frozen provenance",
    )
    _require(
        _recorded_path(repo, primal["current_external_receiver"]["path"])
        == current_path,
        "current trace path differs from primal summary",
    )
    _require(
        _recorded_path(repo, primal["true_external_receiver"]["path"])
        == true_path,
        "TRUE trace path differs from primal summary",
    )

    current = _trace(current_path, shape)
    truth = _trace(true_path, shape)
    residual = np.asarray(current - truth, dtype=np.float64)
    primal_objective = primal["objective"]

    objective_dt = float(
        primal_objective.get("dt", driver.dt)
    )

    _require(
        objective_dt == float(driver.dt),
        "parent-forward objective dt differs from certified driver dt",
    )

    objective_weights = trapezoid_weights(
        np.arange(sample_count, dtype=np.float64) * objective_dt
    )

    objective = 0.5 * float(
        np.sum(objective_weights[:, None, None] * residual * residual)
    )
    recorded_j = primal_objective.get(
        "J_external", primal_objective.get("J1")
    )
    accepted_j = primal_objective.get(
        "accepted_J", primal_objective.get("accepted_J1")
    )
    _require(
        objective == float(recorded_j)
        == float(accepted_j)
        and primal_objective.get("bitwise_equal") is True,
        "parent objective is not reproduced bitwise",
    )

    receiver_nodes = np.asarray(driver.receiver_nodes, dtype=np.int64)
    receiver_weights = np.asarray(driver.receiver_weights, dtype=np.float64)
    _require(receiver_nodes.shape == receiver_weights.shape, "receiver shape mismatch")
    _require(
        np.allclose(np.sum(receiver_weights, axis=1), 1.0, rtol=0.0, atol=1.0e-14),
        "receiver interpolation rows do not sum to one",
    )
    checkpoints, positions = retained_checkpoint_map(
        retained_dir, sample_count, driver.signature
    )
    _require(
        len(positions) >= 2
        and positions[0] == 0
        and positions[-1] == sample_count
        and all(
            left < right
            for left, right in zip(positions[:-1], positions[1:])
        ),
        "invalid retained checkpoint positions",
    )

    recorded_positions = [
        int(value)
        for value in primal["retained_primal"]["positions"]
    ]

    _require(
        recorded_positions == positions[1:],
        "retained positions differ from primal summary",
    )
    _require(
        _recorded_path(repo, primal["retained_primal"]["directory"])
        == retained_dir,
        "retained directory differs from primal summary",
    )

    input_hashes = {
        "runtime_config": _file(config_path),
        "iteration_engine_config": _file(engine_path),
        "primal_forward_summary": _file(primal_summary_path),
        "accepted_parent_summary": _file(accepted_summary_path),
        "accepted_trial_summary": _file(trial_path),
        "current_external_receiver": _file(current_path),
        "accepted_external_receiver": _file(accepted_trace),
        "true_external_receiver": _file(true_path),
        "parent_material": {
            name: _file(path) for name, path in material_files.items()
        },
        "retained_primal": {
            str(position): _file(path) for position, path in checkpoints.items()
        },
        "driver_assets": _driver_hashes(driver),
    }
    signature_payload = {
        "run_id": run_id,
        "parent_iteration": int(args.iter_k),
        "child_iteration": int(args.iter_k) + 1,
        "transition": paths.identity.transition_id,
        "driver_signature_sha256": driver.signature,
        "input_hashes": input_hashes,
        "residual_sign": "current_external_receiver - true_external_receiver",
        "weighting": "fixed-dt trapezoidal",
    }
    signature = _signature(signature_payload)
    return {
        "repo": repo,
        "config": config,
        "iteration_paths": runtime_paths,
        "identity": paths.identity,
        "output_dir": output_dir,
        "driver": driver,
        "residual": residual,
        "objective_weights": objective_weights,
        "objective": objective,
        "objective_relative": 0.0,
        "receiver_nodes": receiver_nodes,
        "receiver_weights": receiver_weights,
        "checkpoints": checkpoints,
        "coarse_positions": positions,
        "signature": signature,
        "signature_payload": signature_payload,
        "input_hashes": input_hashes,
        "reference_path": primal_summary_path,
        "parent_summary_path": primal_summary_path,
        "parent_summary": {"objective": {"J_external": objective}},
        "gradient_names": GRADIENT_NAMES,
        "pass_result": _reverse_result(args.iter_k),
        "completion_gate_profile": "material_covector",
        "source_equivalence": SOURCE_EQUIVALENCE,
    }


def preflight(runtime: Mapping[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    identity = runtime["identity"]
    checks = {
        "current_trace_hash_match": True,
        "current_trace_bitwise_equal_to_accepted": True,
        "true_trace_hash_match": True,
        "parent_material_hash_match": True,
        "trace_shape_from_config": True,
        "retained_positions_exact": True,
        "retained_checkpoint_metadata_and_signature_valid": True,
        "receiver_operator_rows_sum_to_one": True,
        "source_count_matches_config": True,
        "receiver_count_matches_config": True,
        "dt_matches_config": True,
        "assembled_source_amplitude_matches_config": True,
        "output_is_current_transition_local": True,
        "mutable_paths_exclude_historical_namespace": True,
        "residual_is_current_minus_true": True,
        "fixed_dt_trapezoidal_weighting": True,
        "certified_reverse_delegate": True,
        "no_gpu_capteur_trace": True,
        "no_forward_or_reverse_executed": True,
    }
    payload = {
        "schema_version": 1,
        "result": _preflight_result(identity.parent_iteration),
        "run_id": identity.run_id,
        "parent_iteration": identity.parent_iteration,
        "child_iteration": identity.child_iteration,
        "parent": identity.parent_tag,
        "child": identity.child_tag,
        "transition": identity.transition_id,
        "production_signature_sha256": runtime["signature"],
        "checks": checks,
        "contract": {
            "dt": float(runtime["config"]["forward_operator"]["effective_dt_s"]),
            "sample_count": int(runtime["residual"].shape[0]),
            "source_count": int(len(runtime["driver"].source_nodes)),
            "receiver_count": int(runtime["receiver_nodes"].shape[0]),
            "component_count": int(runtime["residual"].shape[2]),
            "objective_J": float(runtime["objective"]),
            "residual": "current_external_receiver - true_external_receiver",
            "quadrature": "fixed-dt trapezoidal",
            "receiver_adjoint_injection": "exact physical interpolation transpose",
            "replay_stride": int(args.replay_stride),
            "reverse_checkpoint_interval": int(args.reverse_checkpoint_interval),
        },
        "coarse_checkpoint_positions": runtime["coarse_positions"],
        "source_equivalence": SOURCE_EQUIVALENCE,
        "input_hashes": runtime["input_hashes"],
        "exact_reverse_runs": 0,
        "external_forward_runs": 0,
        "sem3d_runs": 0,
    }
    _require(all(checks.values()), "production exact-reverse preflight failed")
    output = runtime["output_dir"] / "preflight_summary.json"
    if output.is_file():
        previous = _json(output)
        _require(
            previous.get("production_signature_sha256") == runtime["signature"],
            "refusing to overwrite mismatched preflight",
        )
    atomic_json(output, payload)
    return payload


def run_reverse(runtime: Mapping[str, Any], args: argparse.Namespace) -> None:
    """Delegate the heavy recurrence to the existing certified implementation."""

    preflight_path = runtime["output_dir"] / "preflight_summary.json"
    _require(preflight_path.is_file(), "production preflight must precede reverse")
    preflight_payload = _json(preflight_path)
    _require(
        preflight_payload.get("result")
        == _preflight_result(runtime["identity"].parent_iteration)
        and preflight_payload.get("production_signature_sha256")
        == runtime["signature"],
        "production preflight does not match reverse inputs",
    )
    certified_reverse.run_reverse(
        runtime,
        args.replay_stride,
        args.reverse_checkpoint_interval,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", default=".")
    parser.add_argument("--config", required=True)
    parser.add_argument("--engine-config")
    parser.add_argument("--iter-k", type=int, required=True)
    parser.add_argument("--current-trace")
    parser.add_argument("--true-trace")
    parser.add_argument("--accepted-trace")
    parser.add_argument("--retained-primal-dir")
    parser.add_argument("--primal-summary")
    parser.add_argument("--accepted-summary")
    parser.add_argument("--reference-manifest")
    parser.add_argument("--driver-root")
    parser.add_argument("--output-dir")
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--replay-stride", type=int, default=10)
    parser.add_argument("--reverse-checkpoint-interval", type=int, default=50)
    parser.add_argument("--action", choices=("preflight", "reverse"), required=True)
    args = parser.parse_args()
    if args.iter_k < 0:
        parser.error("--iter-k must be nonnegative")
    if min(
        args.batch_size,
        args.replay_stride,
        args.reverse_checkpoint_interval,
    ) < 1:
        parser.error("batch, replay, and checkpoint values must be positive")
    runtime = build_runtime(args)
    if args.action == "preflight":
        print(json.dumps(preflight(runtime, args), indent=2))
    else:
        run_reverse(runtime, args)


if __name__ == "__main__":
    main()
