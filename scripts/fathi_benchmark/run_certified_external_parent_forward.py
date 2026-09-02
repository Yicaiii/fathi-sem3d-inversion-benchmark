"""Run the certified external physical forward for an iteration parent model."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path

import numpy as np

from scripts.exact_adjoint.certify_exact_adjoint_with_fixed_dt_fd import (
    trapezoid_weights,
)
from scripts.exact_adjoint.s43_external_forward import (
    ExternalForwardDriver,
    load_certified_reference,
    run_external_forward,
    sha256_arrays,
    sha256_file,
)
from scripts.fathi_benchmark.runtime_paths import (
    iteration_runtime_paths,
    resolve_path,
)
from scripts.fathi_benchmark.current_pipeline_contracts import (
    accepted_model_result,
    artifact_record,
    retained_primal_result,
)
from scripts.fathi_benchmark.iteration_context import build_iteration_paths
from scripts.fathi_benchmark.path_consistency import (
    validate_path_config_consistency,
)


HISTORICAL_PASS_RESULT = "PASS_CERTIFIED_PARENT_EXTERNAL_FORWARD"


def current_parent_forward_contract(paths) -> dict:
    """Return the exact schema/path contract without executing a forward."""

    iteration = paths.identity.parent_iteration
    return {
        "schema_version": 1,
        "result": retained_primal_result(iteration),
        "run_id": paths.identity.run_id,
        "parent_iteration": iteration,
        "child_iteration": paths.identity.child_iteration,
        "transition": paths.identity.transition_id,
        "output_path": str(paths.exact_reverse / "primal_forward"),
        "current_receiver_filename": "current_external_receiver.npy",
        "summary_filename": "summary.json",
        "required_summary_fields": [
            "material_sha256",
            "driver_signature_sha256",
            "forward_run_signature_sha256",
            "current_external_receiver",
            "true_external_receiver",
            "objective",
            "retained_primal",
            "immutable_operator_identity",
        ],
    }


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def manifest_asset(repo: Path, value: str) -> Path:
    path = Path(value).expanduser()
    return (path if path.is_absolute() else repo / path).resolve()


def expected_parent_objective(
    repo: Path,
    reference: dict,
    runtime: dict,
    iteration: int,
) -> tuple[float, Path, str]:
    if int(iteration) == 0:
        path = manifest_asset(
            repo, reference["certification_assets"]["stage5n_summary"]
        )
        payload = json.loads(path.read_text(encoding="utf-8"))
        return (
            float(payload["objective"]["J_external"]),
            path,
            str(payload["result"]),
        )

    path = Path(runtime["parent_workspace"]) / "accepted_summary.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("result") != accepted_model_result(iteration):
        raise RuntimeError(f"parent accepted summary is not certified PASS: {path}")
    if int(payload["iter"]) != int(iteration):
        raise RuntimeError("parent accepted-summary iteration mismatch")
    return float(payload["objective"]["accepted"]), path.resolve(), payload["result"]


def load_external(path: Path, expected_shape: tuple[int, int, int]) -> np.ndarray:
    value = np.load(path)
    if value.dtype != np.float64 or value.shape != expected_shape:
        raise RuntimeError(
            f"external receiver contract mismatch: {path}: "
            f"dtype={value.dtype}, shape={value.shape}, expected={expected_shape}"
        )
    if not np.all(np.isfinite(value)):
        raise RuntimeError(f"non-finite external receiver array: {path}")
    return np.asarray(value, dtype=np.float64)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", default=".")
    parser.add_argument("--config", required=True)
    parser.add_argument("--iter-k", type=int, required=True)
    parser.add_argument("--reference-manifest")
    parser.add_argument("--output-dir")
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--checkpoint-interval", type=int, default=100)
    args = parser.parse_args()

    if args.iter_k < 0:
        parser.error("--iter-k must be nonnegative")
    if args.batch_size < 1 or args.checkpoint_interval < 1:
        parser.error("batch and checkpoint intervals must be positive")

    repo = Path(args.repo).expanduser().resolve()
    config_path = resolve_path(args.config, base=repo)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    runtime = iteration_runtime_paths(config, args.iter_k, repo_root=repo)
    run = str(config["benchmark_name"])
    engine_path = (
        repo / "configs" / f"{run}_iteration_engine.json"
    ).resolve()
    engine = json.loads(engine_path.read_text(encoding="utf-8"))
    validate_path_config_consistency(
        config,
        engine,
        repository_root=repo,
    )
    paths = build_iteration_paths(
        engine,
        args.iter_k,
        child_iteration=args.iter_k + 1,
        repository_root=repo,
        runtime_root=runtime["runtime_root"],
    )
    if paths.transition_root != Path(runtime["transition_root"]):
        raise RuntimeError("runtime/iteration-engine transition path mismatch")
    if paths.parent_accepted != Path(runtime["parent_workspace"]):
        raise RuntimeError("runtime/iteration-engine parent path mismatch")
    pass_result = retained_primal_result(args.iter_k)
    reference_path = (
        resolve_path(args.reference_manifest, base=repo)
        if args.reference_manifest
        else repo / "results" / run / "certified_external_reference.json"
    ).resolve()
    _, reference = load_certified_reference(repo, run, reference_path)
    contract = reference["contract"]
    output_dir = (
        resolve_path(args.output_dir, base=repo)
        if args.output_dir
        else paths.exact_reverse / "primal_forward"
    ).resolve()
    if output_dir != (paths.exact_reverse / "primal_forward").resolve():
        raise RuntimeError("CURRENT parent-forward output is not canonical")
    material_dir = Path(runtime["parent_workspace"]) / "mat" / "h5"
    material_files = {
        component: material_dir / str(engine["material"]["files"][component])
        for component in ("kappa", "mu", "density")
    }
    missing = [str(path) for path in material_files.values() if not path.is_file()]
    if missing:
        raise RuntimeError("missing parent material: " + ", ".join(missing))

    initial = iteration_runtime_paths(config, 0, repo_root=repo)
    initial_density = (
        Path(initial["parent_workspace"]) / "mat" / "h5" / "Mat_0_Density.h5"
    )
    if sha256_file(material_files["density"]) != sha256_file(
        initial_density
    ):
        raise RuntimeError("parent density differs from frozen coupled-mass density")

    driver = ExternalForwardDriver(
        repo,
        run,
        material_dir,
        batch_size=args.batch_size,
        reference_manifest=reference_path,
    )
    sample_count = int(contract["sample_count"])
    expected_shape = (
        sample_count,
        int(contract["receiver_count"]),
        int(contract["component_count"]),
    )
    if not math.isclose(
        driver.dt, float(contract["dt"]), rel_tol=0.0, abs_tol=1.0e-18
    ):
        raise RuntimeError("driver dt differs from certified reference")
    if driver.receiver_count != expected_shape[1]:
        raise RuntimeError("driver receiver count differs from certified reference")
    if sha256_file(driver.paths["stf"]) != reference["immutable_input_assets"][
        "reference_stf_sha256"
    ]:
        raise RuntimeError("reference STF hash mismatch")
    if sha256_file(driver.paths["true_external"]) != reference["hashes"][
        "true_external_sha256"
    ]:
        raise RuntimeError("true external hash mismatch")
    receiver_nodes_path = Path(driver.paths["receiver"]) / "receiver_nodes.npy"
    receiver_weights_path = Path(driver.paths["receiver"]) / "receiver_weights.npy"
    if (
        sha256_file(receiver_nodes_path)
        != reference["hashes"]["receiver_nodes_sha256"]
        or sha256_file(receiver_weights_path)
        != reference["hashes"]["receiver_weights_sha256"]
    ):
        raise RuntimeError("physical receiver operator hash mismatch")
    receiver_hash = sha256_arrays(driver.receiver_nodes, driver.receiver_weights)

    expected_j, expected_j_source, expected_j_result = expected_parent_objective(
        repo, reference, runtime, args.iter_k
    )
    material_hashes = {
        name: sha256_file(path) for name, path in material_files.items()
    }
    signature_payload = {
        "schema_version": 1,
        "iteration": int(args.iter_k),
        "transition": runtime["transition"],
        "reference_manifest_sha256": sha256_file(reference_path),
        "driver_signature_sha256": driver.signature,
        "material_sha256": material_hashes,
        "expected_parent_objective": expected_j,
        "sample_count": sample_count,
        "receiver_operator_sha256": receiver_hash,
    }
    signature = hashlib.sha256(
        json.dumps(signature_payload, sort_keys=True).encode("utf-8")
    ).hexdigest()

    summary_path = output_dir / "summary.json"
    current_path = output_dir / "current_external_receiver.npy"
    if summary_path.is_file():
        existing = json.loads(summary_path.read_text(encoding="utf-8"))
        if (
            existing.get("result") == pass_result
            and existing.get("input_signature_sha256") == signature
            and current_path.is_file()
            and existing["files"]["current_external_sha256"]
            == sha256_file(current_path)
        ):
            print(f"RESULT = {pass_result}")
            print(f"OUTPUT = {output_dir}")
            print("IDEMPOTENT_REUSE = true")
            return
        raise RuntimeError(f"refusing non-matching existing parent forward: {output_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / "checkpoint" / "current_latest.npz"
    retained_dir = output_dir / "checkpoint" / "current_primal_retained"
    run_summary = run_external_forward(
        driver,
        sample_count,
        {"primal": current_path},
        checkpoint_path,
        checkpoint_interval=args.checkpoint_interval,
        retained_primal_dir=retained_dir,
    )

    current = load_external(current_path, expected_shape)
    truth = load_external(Path(driver.paths["true_external"]), expected_shape)
    residual = current - truth
    time_grid = np.arange(sample_count, dtype=np.float64) * driver.dt
    weights = trapezoid_weights(time_grid)
    objective = 0.5 * float(
        np.sum(weights[:, None, None] * residual * residual)
    )
    relative = abs(objective - expected_j) / max(
        abs(expected_j), np.finfo(np.float64).tiny
    )
    retained_last = retained_dir / f"primal_{sample_count:06d}.npz"
    accepted_receiver_hash = None
    if int(args.iter_k) > 0:
        accepted_payload = json.loads(expected_j_source.read_text(encoding="utf-8"))
        accepted_receiver = accepted_payload.get("accepted_external_receiver")
        if isinstance(accepted_receiver, dict):
            accepted_receiver_path = manifest_asset(
                repo, str(accepted_receiver["path"])
            )
        else:
            trial_path = manifest_asset(
                repo, str(accepted_payload["external_armijo_trial"])
            )
            trial = json.loads(trial_path.read_text(encoding="utf-8"))
            accepted_receiver_path = manifest_asset(
                repo, str(trial["candidate_external_receiver"])
            )
        accepted_receiver_hash = sha256_file(accepted_receiver_path)
    current_receiver_hash = sha256_file(current_path)
    gates = {
        "reference_manifest_pass": True,
        "reference_true_external_hash": True,
        "reference_receiver_operator": True,
        "fixed_dt": True,
        "sample_shape": True,
        "density_matches_frozen_coupled_mass": True,
        "retained_endpoint_present": retained_last.is_file(),
        "objective_bitwise_equal_to_accepted": objective == expected_j,
        "current_receiver_bitwise_equal_to_accepted": (
            accepted_receiver_hash is None
            or accepted_receiver_hash == current_receiver_hash
        ),
        "all_finite": bool(np.all(np.isfinite(residual))),
    }
    if not all(gates.values()):
        failed = [name for name, passed in gates.items() if not passed]
        raise RuntimeError("parent-forward gates failed: " + ", ".join(failed))

    summary = {
        "schema_version": 1,
        "result": pass_result,
        "run_id": run,
        "parent_iteration": int(args.iter_k),
        "child_iteration": int(args.iter_k) + 1,
        "iteration": int(args.iter_k),
        "transition": runtime["transition"],
        "input_signature_sha256": signature,
        "reference_manifest": str(reference_path),
        "reference_manifest_sha256": sha256_file(reference_path),
        "material_dir": str(material_dir.resolve()),
        "material_sha256": material_hashes,
        "runtime_config": artifact_record(config_path, repo=repo),
        "iteration_engine_config": artifact_record(engine_path, repo=repo),
        "accepted_parent_summary": (
            None
            if int(args.iter_k) == 0
            else artifact_record(expected_j_source, repo=repo)
        ),
        "immutable_operator_identity": {
            "reference_manifest_sha256": sha256_file(reference_path),
            "driver_signature_sha256": driver.signature,
            "receiver_operator_sha256": receiver_hash,
        },
        "objective": {
            "J_external": objective,
            "accepted_J": expected_j,
            "bitwise_equal": objective == expected_j,
            "relative_error": relative,
            "expected_source": str(expected_j_source),
            "expected_source_result": expected_j_result,
            "residual_sign": "current_external - true_external",
            "time_weighting": "native fixed-dt trapezoidal quadrature",
            "sample_count": sample_count,
            "receiver_count": expected_shape[1],
            "component_count": expected_shape[2],
            "dt": driver.dt,
            "residual_l2": float(np.linalg.norm(residual.reshape(-1))),
        },
        "external_forward": run_summary,
        "driver_signature_sha256": driver.signature,
        "forward_run_signature_sha256": str(
            run_summary.get("signature_sha256", signature)
        ),
        "current_external_receiver": {
            **artifact_record(current_path, repo=repo),
            "bitwise_equal_to_accepted_parent": gates[
                "current_receiver_bitwise_equal_to_accepted"
            ],
        },
        "true_external_receiver": {
            **artifact_record(driver.paths["true_external"], repo=repo),
            "rerun": False,
        },
        "retained_primal": {
            "directory": str(retained_dir),
            "positions": [
                int(path.stem.split("_")[-1])
                for path in sorted(retained_dir.glob("primal_*.npz"))
            ],
            "sha256": {
                path.name: sha256_file(path)
                for path in sorted(retained_dir.glob("primal_*.npz"))
            },
            "latest_resume_checkpoint": str(checkpoint_path),
            "latest_resume_checkpoint_sha256": sha256_file(checkpoint_path),
        },
        "receiver_operator_sha256": receiver_hash,
        "gates": gates,
        "sem3d_runs": 0,
        "full_external_forwards": 1,
    }
    atomic_json(summary_path, summary)
    print(f"RESULT = {pass_result}")
    print(f"J_EXTERNAL = {objective:.17e}")
    print(f"OBJECTIVE_RELATIVE_ERROR = {relative:.17e}")
    print(f"OUTPUT = {output_dir}")


if __name__ == "__main__":
    main()
