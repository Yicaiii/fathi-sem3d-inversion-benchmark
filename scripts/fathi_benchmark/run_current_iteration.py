"""Resume one complete CURRENT inversion transition from parent iteration k.

The only required iteration input is ``--parent-iteration``.  Every numerical
stage is idempotent and reuses its canonical durable artifact when already
certified.  Frozen mathematics are delegated to the existing certified forward,
exact-reverse, bridge, Mtilde, L-BFGS/Eq.25, Armijo, and promotion primitives.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping

from scripts.exact_adjoint.s43_external_forward import load_certified_reference, sha256_file
from scripts.fathi_benchmark.build_current_certified_external_reference import (
    build_reference,
    persist_reference,
)
from scripts.fathi_benchmark.current_pipeline_artifacts import (
    execute_current_armijo,
    generate_raw_alpha_candidate,
    promote_current_accepted_trial,
)
from scripts.fathi_benchmark.current_pipeline_contracts import (
    accepted_model_result,
    artifact_record,
    atomic_json,
    exact_reverse_result,
    gradient_bridge_result,
    optimizer_direction_result,
    registered_gradient_result,
    retained_primal_result,
)
from scripts.fathi_benchmark.external_armijo import ArmijoParameters
from scripts.fathi_benchmark.generic_iteration_runner import GenericIterationRunner
from scripts.fathi_benchmark.iteration_context import build_iteration_paths
from scripts.fathi_benchmark.lbfgs_history import (
    HISTORY_OUTCOME_ACCEPTED,
    HISTORY_OUTCOME_REJECTED,
)
from scripts.fathi_benchmark.runtime_paths import iteration_runtime_paths, resolve_path


DEFAULT_RUNTIME = "configs/fathi_s43_repro_p20_t052_runtime.json"
DEFAULT_ENGINE = "configs/fathi_s43_repro_p20_t052_iteration_engine.json"
STOPS = (
    "reference",
    "parent-forward",
    "reverse",
    "gradient",
    "history",
    "direction",
    "armijo-input",
    "armijo",
    "promote",
)


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"JSON payload must be an object: {path}")
    return value


def _pass(path: Path, result: str) -> bool:
    if not path.is_file():
        return False
    try:
        return _json(path).get("result") == result
    except Exception:
        return False


def _run(command: list[str], *, cwd: Path) -> None:
    print("COMMAND =", " ".join(command), flush=True)
    completed = subprocess.run(command, cwd=str(cwd), check=False)
    if completed.returncode != 0:
        raise RuntimeError(
            f"subprocess failed with rc={completed.returncode}: {' '.join(command)}"
        )


def _path_value(repo: Path, path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(repo))
    except ValueError:
        return str(resolved)


def _material_spec(
    repo: Path,
    material_dir: Path,
    material_config: Mapping[str, Any],
    ordering: Mapping[str, Any],
) -> dict[str, Any]:
    files = material_config["files"]
    hashes = {
        component: sha256_file(material_dir / str(files[component]))
        for component in ("kappa", "mu", "density")
    }
    return {
        "material_dir": _path_value(repo, material_dir),
        "material_sha256": hashes,
        "active_h5_indices": dict(ordering["active_h5_indices"]),
        "active_indices": dict(ordering["active_indices"]),
        "coordinates": dict(ordering["coordinates"]),
    }


def _canonical_mtilde_record(repo: Path, runtime: Mapping[str, Any]) -> dict[str, Any]:
    path = resolve_path(str(runtime["mtilde_matrix_path"]), base=repo)
    return artifact_record(path, repo=repo)


def _normalized_gradient(
    gradient: Mapping[str, Any],
    canonical_mtilde: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "lambda": dict(gradient["lambda"]),
        "mu": dict(gradient["mu"]),
        "active_indices": dict(gradient["active_indices"]),
        "coordinates": dict(gradient["coordinates"]),
        "mtilde": dict(canonical_mtilde),
    }


def _history_outcome_records(repo: Path, history_root: Path, parent: int) -> list[dict[str, Any]]:
    records = []
    if not history_root.is_dir():
        return records
    for directory in sorted(path for path in history_root.iterdir() if path.is_dir()):
        outcome = directory / "curvature_outcome.json"
        if not outcome.is_file():
            continue
        payload = _json(outcome)
        if int(payload.get("to_iteration", -1)) <= parent:
            records.append(artifact_record(outcome, repo=repo))
    return records


def _ensure_reference(
    *,
    repo: Path,
    runtime_path: Path,
    runtime: Mapping[str, Any],
    paths,
    reference_path: Path,
) -> Path:
    run = str(runtime["benchmark_name"])
    required_hashes = {"receiver_nodes_sha256", "receiver_weights_sha256"}
    needs_build = True
    if reference_path.is_file():
        try:
            _, current = load_certified_reference(repo, run, reference_path)
            immutable = current.get("immutable_input_assets", {})
            hashes = current.get("hashes", {})
            needs_build = not (
                immutable.get("runtime_config")
                and required_hashes.issubset(hashes)
            )
        except Exception:
            needs_build = True
    if not needs_build:
        print(f"REFERENCE = REUSE {reference_path}")
        return reference_path

    reverse_candidates = [
        paths.exact_reverse / "production_reverse" / "summary.json",
        *sorted(
            paths.results_run_root.glob(
                "iter_*_to_iter_*/exact_reverse/production_reverse/summary.json"
            )
        ),
    ]
    reverse_path = None
    for path in reverse_candidates:
        if not path.is_file():
            continue
        try:
            payload = _json(path)
            iteration = int(payload.get("iteration", payload.get("parent_iteration", -1)))
        except Exception:
            continue
        if iteration >= 0 and payload.get("result") == exact_reverse_result(iteration):
            reverse_path = path
            break
    if reverse_path is None:
        raise RuntimeError(
            "CURRENT certified reference is absent/incomplete and no certified exact-reverse provenance exists"
        )
    manifest = build_reference(
        repo=repo,
        runtime_config_path=runtime_path,
        reverse_summary_path=reverse_path,
    )
    action = persist_reference(reference_path, manifest)
    load_certified_reference(repo, run, reference_path)
    print(f"REFERENCE = {action} {reference_path}")
    return reference_path


def _ensure_history(
    *,
    repo: Path,
    runtime: Mapping[str, Any],
    engine: Mapping[str, Any],
    runner: GenericIterationRunner,
    current_gradient_path: Path,
) -> Path | None:
    parent = runner.paths.identity.parent_iteration
    if parent == 0:
        return None
    current_gradient = _json(current_gradient_path)
    status = runner.newest_history_preflight(current_gradient)
    if status.get("status") in {HISTORY_OUTCOME_ACCEPTED, HISTORY_OUTCOME_REJECTED}:
        outcome = (
            runner.paths.optimizer_history
            / f"iter_{parent - 1:03d}_to_iter_{parent:03d}"
            / "curvature_outcome.json"
        )
        print(f"HISTORY = REUSE {status['status']} {outcome}")
        return outcome

    canonical_mtilde = _canonical_mtilde_record(repo, runtime)
    current_spec = _normalized_gradient(current_gradient, canonical_mtilde)
    if parent == 1:
        raise RuntimeError(
            "the one-time iter0->iter1 bootstrap curvature outcome is missing; "
            "do not reconstruct it inside the generic CURRENT runner"
        )
    previous_paths = build_iteration_paths(
        engine,
        parent - 1,
        child_iteration=parent,
        repository_root=repo,
        runtime_root=runner.runtime_root,
    )
    previous_transition = previous_paths.gradient_root / "registered_gradient.json"
    if not _pass(previous_transition, registered_gradient_result(parent - 1)):
        raise RuntimeError(f"previous registered gradient is missing: {previous_transition}")
    previous_spec = _normalized_gradient(_json(previous_transition), canonical_mtilde)

    # The physical models use the same canonical active ordering as the current
    # corrected gradient.  build_history_pair independently verifies identity.
    ordering = {
        "active_h5_indices": current_gradient["active_h5_indices"],
        "active_indices": current_gradient["active_indices"],
        "coordinates": current_gradient["coordinates"],
    }
    material_cfg = engine["material"]
    previous_paths = build_iteration_paths(
        engine,
        parent - 1,
        child_iteration=parent,
        repository_root=repo,
        runtime_root=runner.runtime_root,
    )
    model0_dir = previous_paths.parent_accepted / str(material_cfg["directory"])
    model1_dir = runner.paths.parent_accepted / str(material_cfg["directory"])
    request = {
        "parent_iteration": parent - 1,
        "child_iteration": parent,
        "parent_model": _material_spec(repo, model0_dir, material_cfg, ordering),
        "child_model": _material_spec(repo, model1_dir, material_cfg, ordering),
        "parent_gradient": previous_spec,
        "child_gradient": current_spec,
    }
    request_path = runner.paths.optimizer_root / (
        f"history_request_iter_{parent - 1:03d}_to_iter_{parent:03d}.json"
    )
    atomic_json(request_path, request)
    preflight = runner.history_preflight(request)
    if preflight.get("status") != "READY_TO_BUILD_HISTORY":
        raise RuntimeError(f"history preflight blocked: {preflight}")
    result = runner.build_real_history_pair(request)
    outcome_dir = runner.checkpoint_history_outcome(result)
    outcome = outcome_dir / "curvature_outcome.json"
    print(
        "HISTORY = BUILT "
        f"status={'ACCEPTED' if result.audit.accepted else 'REJECTED'} "
        f"sMy={result.audit.s_m_y:.17e} threshold={result.audit.threshold:.17e}"
    )
    return outcome


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", default=".")
    parser.add_argument("--parent-iteration", "--k", dest="parent_iteration", type=int, required=True)
    parser.add_argument("--runtime-config", default=DEFAULT_RUNTIME)
    parser.add_argument("--engine-config", default=DEFAULT_ENGINE)
    parser.add_argument("--reference-manifest")
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--checkpoint-interval", type=int, default=100)
    parser.add_argument("--replay-stride", type=int, default=10)
    parser.add_argument("--reverse-checkpoint-interval", type=int, default=50)
    parser.add_argument("--stop-after", choices=STOPS, default="promote")
    args = parser.parse_args()
    if args.parent_iteration < 1:
        parser.error(
            "the unified CURRENT runner starts at accepted iter_001; "
            "--parent-iteration must be >= 1"
        )

    repo = Path(args.repo).expanduser().resolve()
    runtime_path = resolve_path(args.runtime_config, base=repo)
    engine_path = resolve_path(args.engine_config, base=repo)
    runtime = _json(runtime_path)
    engine = _json(engine_path)
    run = str(engine["run_id"])
    parent = int(args.parent_iteration)
    if str(runtime.get("benchmark_name")) != run:
        raise RuntimeError("runtime/engine run identity mismatch")
    runtime_paths = iteration_runtime_paths(runtime, parent, repo_root=repo)
    paths = build_iteration_paths(
        engine,
        parent,
        child_iteration=parent + 1,
        repository_root=repo,
        runtime_root=runtime_paths["runtime_root"],
    )
    runner = GenericIterationRunner.from_config_files(
        run_id=run,
        parent_iteration=parent,
        child_iteration=parent + 1,
        repository_root=repo,
        runtime_config_path=runtime_path,
        engine_config_path=engine_path,
    )
    reference_path = (
        resolve_path(args.reference_manifest, base=repo)
        if args.reference_manifest
        else (paths.results_run_root / "certified_external_reference.json").resolve()
    )

    print("=" * 76)
    print(f" CURRENT ITERATION {parent:03d} -> {parent + 1:03d}")
    print("=" * 76)

    _ensure_reference(
        repo=repo,
        runtime_path=runtime_path,
        runtime=runtime,
        paths=paths,
        reference_path=reference_path,
    )
    if args.stop_after == "reference":
        return

    primal_summary = paths.exact_reverse / "primal_forward" / "summary.json"
    if _pass(primal_summary, retained_primal_result(parent)):
        print(f"PARENT_FORWARD = REUSE {primal_summary}")
    else:
        _run(
            [
                sys.executable,
                str(repo / "scripts/fathi_benchmark/run_certified_external_parent_forward.py"),
                "--repo", str(repo),
                "--config", str(runtime_path),
                "--iter-k", str(parent),
                "--reference-manifest", str(reference_path),
                "--batch-size", str(args.batch_size),
                "--checkpoint-interval", str(args.checkpoint_interval),
            ],
            cwd=repo,
        )
    if args.stop_after == "parent-forward":
        return

    reverse_summary = paths.exact_reverse / "production_reverse" / "summary.json"
    if _pass(reverse_summary, exact_reverse_result(parent)):
        print(f"EXACT_REVERSE = REUSE {reverse_summary}")
    else:
        common = [
            sys.executable,
            str(repo / "scripts/fathi_benchmark/run_exact_reverse_gradient_generic.py"),
            "--repo", str(repo),
            "--config", str(runtime_path),
            "--engine-config", str(engine_path),
            "--iter-k", str(parent),
            "--reference-manifest", str(reference_path),
            "--batch-size", str(args.batch_size),
            "--replay-stride", str(args.replay_stride),
            "--reverse-checkpoint-interval", str(args.reverse_checkpoint_interval),
        ]
        _run(common + ["--action", "preflight"], cwd=repo)
        _run(common + ["--action", "reverse"], cwd=repo)
    if args.stop_after == "reverse":
        return

    gradient_summary = paths.gradient_root / "summary.json"
    if _pass(gradient_summary, gradient_bridge_result(parent)):
        print(f"GRADIENT_BRIDGE = REUSE {gradient_summary}")
    else:
        _run(
            [
                sys.executable,
                str(repo / "scripts/fathi_benchmark/bridge_certified_external_gradient.py"),
                "--repo", str(repo),
                "--config", str(runtime_path),
                "--iteration", str(parent),
                "--reference-manifest", str(reference_path),
            ],
            cwd=repo,
        )

    registered = paths.gradient_root / "registered_gradient.json"
    if _pass(registered, registered_gradient_result(parent)):
        print(f"REGISTERED_GRADIENT = REUSE {registered}")
    else:
        _run(
            [
                sys.executable,
                str(repo / "scripts/fathi_benchmark/register_certified_gradient.py"),
                "--repo", str(repo),
                "--config", str(runtime_path),
                "--engine-config", str(engine_path),
                "--iteration", str(parent),
            ],
            cwd=repo,
        )
    if args.stop_after == "gradient":
        return

    _ensure_history(
        repo=repo,
        runtime=runtime,
        engine=engine,
        runner=runner,
        current_gradient_path=registered,
    )
    if args.stop_after == "history":
        return

    direction_summary = (
        paths.optimizer_root
        / f"iter_{parent:03d}_lbfgs_eq25_direction"
        / "direction_summary.json"
    )
    if _pass(direction_summary, optimizer_direction_result(parent)):
        print(f"DIRECTION = REUSE {direction_summary}")
    else:
        accepted_summary = paths.parent_accepted / "accepted_summary.json"
        if parent > 0 and not _pass(accepted_summary, accepted_model_result(parent)):
            raise RuntimeError(f"accepted parent summary is missing: {accepted_summary}")
        direction_request = {
            "run_id": run,
            "parent_iteration": parent,
            "child_iteration": parent + 1,
            "transition": paths.identity.transition_id,
            "registered_gradient_manifest": artifact_record(registered, repo=repo),
            "accepted_parent_summary": artifact_record(accepted_summary, repo=repo),
            "history_outcomes": _history_outcome_records(
                repo, paths.optimizer_history, parent
            ),
        }
        request_path = paths.optimizer_root / "direction_request.json"
        atomic_json(request_path, direction_request)
        result = runner.compute_optimizer_direction(direction_request)
        direction_summary = runner.persist_optimizer_direction(direction_request, result)
        print(f"DIRECTION = BUILT {direction_summary}")
    if args.stop_after == "direction":
        return

    armijo_input = paths.line_search_root / "armijo_input.json"
    if _pass(armijo_input, f"PASS_ITER{parent:03d}_EXTERNAL_ARMIJO_READY"):
        print(f"ARMIJO_INPUT = REUSE {armijo_input}")
    else:
        accepted_summary = paths.parent_accepted / "accepted_summary.json"
        parent_payload = _json(accepted_summary)
        parent_objective = float(parent_payload["objective"]["accepted"])
        reference = _json(reference_path)
        true_path = resolve_path(reference["certification_assets"]["true_external"], base=repo)
        direction = _json(direction_summary)
        request = {
            "run_id": run,
            "parent_iteration": parent,
            "child_iteration": parent + 1,
            "transition": paths.identity.transition_id,
            "parent_objective": parent_objective,
            "slope": float(direction["mtilde_slope"]),
            "parent_accepted_artifact": artifact_record(accepted_summary, repo=repo),
            "gradient_artifact": artifact_record(registered, repo=repo),
            "direction_artifact": artifact_record(direction_summary, repo=repo),
            "true_receiver_artifact": artifact_record(true_path, repo=repo),
        }
        request_path = paths.optimizer_root / "armijo_manifest_request.json"
        atomic_json(request_path, request)
        payload = runner.prepare_external_armijo(request)
        atomic_json(armijo_input, payload)
        print(f"ARMIJO_INPUT = BUILT {armijo_input}")
    if args.stop_after == "armijo-input":
        return

    armijo_summary = paths.line_search_root / "armijo_summary.json"
    if armijo_summary.is_file() and _json(armijo_summary).get("accepted") is True:
        print(f"ARMIJO = REUSE ACCEPTED {armijo_summary}")
    else:
        armijo = _json(armijo_input)
        parameters = ArmijoParameters(**armijo["parameters"])

        def candidate_provider(trial_index: int, alpha: float) -> Path:
            return generate_raw_alpha_candidate(
                repo=repo,
                paths=paths,
                material_config=engine["material"],
                accepted_parent_record=armijo["parent_accepted_artifact"],
                direction_record=armijo["direction_artifact"],
                parameters=parameters,
                trial_index=trial_index,
                alpha=alpha,
            )

        from scripts.fathi_benchmark.current_pipeline_artifacts import evaluate_candidate_external

        def evaluation_provider(candidate: Path, trial_dir: Path):
            return evaluate_candidate_external(
                repo=repo,
                paths=paths,
                runtime_config=runtime,
                reference_manifest=reference_path,
                candidate_summary_path=candidate,
                trial_directory=trial_dir,
                batch_size=args.batch_size,
                checkpoint_interval=args.checkpoint_interval,
            )

        armijo_summary = execute_current_armijo(
            repo=repo,
            paths=paths,
            armijo_manifest=armijo,
            candidate_provider=candidate_provider,
            evaluation_provider=evaluation_provider,
        )
        print(f"ARMIJO = COMPLETED {armijo_summary}")
    armijo_payload = _json(armijo_summary)
    if armijo_payload.get("accepted") is not True:
        raise RuntimeError("Armijo completed without an accepted trial")
    if args.stop_after == "armijo":
        return

    accepted_child = paths.child_accepted / "accepted_summary.json"
    if _pass(accepted_child, accepted_model_result(parent + 1)) and paths.child_state.is_file():
        print(f"PROMOTION = REUSE {accepted_child}")
    else:
        accepted_child = promote_current_accepted_trial(
            repo=repo,
            paths=paths,
            material_config=engine["material"],
            armijo_summary_record=artifact_record(armijo_summary, repo=repo),
        )
        print(f"PROMOTION = BUILT {accepted_child}")

    print("=" * 76)
    print(f"RESULT = PASS_CURRENT_ITERATION_{parent:03d}_TO_{parent + 1:03d}_CLOSED")
    print(f"ACCEPTED_CHILD = {accepted_child}")
    print(f"CHILD_STATE = {paths.child_state}")
    print("=" * 76)


if __name__ == "__main__":
    main()
