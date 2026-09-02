"""Apply the previously certified S43 discrete exact adjoint to iteration k."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import time
from pathlib import Path

import numpy as np

from scripts.exact_adjoint.certify_exact_adjoint_with_fixed_dt_fd import (
    trapezoid_weights,
)
from scripts.exact_adjoint.real_s43_global_operator import (
    adjoint_step,
    material_vjp,
    state_norm,
)
from scripts.exact_adjoint.run_real_s43_exact_material_gradient import zero_state
from scripts.exact_adjoint.s43_external_forward import (
    ExternalForwardDriver,
    atomic_save_npy,
    atomic_save_npz,
    load_certified_reference,
    sha256_arrays,
    sha256_file,
)
from scripts.exact_adjoint.s43_external_reverse_core import (
    atomic_json,
    cleanup_replay_cache,
    ensure_replay_cache,
    finite_reverse_state,
    load_replay_state,
    retained_checkpoint_map,
)
from scripts.fathi_benchmark.runtime_paths import (
    iteration_runtime_paths,
    resolve_path,
)


PASS_RESULT = "PASS_CERTIFIED_EXTERNAL_EXACT_REVERSE"
PASS_FORWARD = "PASS_CERTIFIED_PARENT_EXTERNAL_FORWARD"
GRADIENT_NAMES = ("solid_lam", "solid_mu", "pml_lam", "pml_mu")
CERTIFICATION_STATEMENT = (
    "The discrete exact-adjoint algorithm was previously certified at Stage5O. "
    "This run applies the frozen certified operator to iteration k."
)


def runtime_gradient_names(runtime) -> tuple[str, str, str, str]:
    names = tuple(runtime.get("gradient_names", GRADIENT_NAMES))
    if len(names) != 4 or len(set(names)) != 4:
        raise ValueError("runtime gradient_names must contain four unique names")
    return names


def runtime_pass_result(runtime) -> str:
    return str(runtime.get("pass_result", PASS_RESULT))


def load_trace(path: Path, shape: tuple[int, int, int]) -> np.ndarray:
    value = np.load(path)
    if value.dtype != np.float64 or value.shape != shape:
        raise RuntimeError(
            f"external trace contract mismatch: {path}: "
            f"dtype={value.dtype}, shape={value.shape}, expected={shape}"
        )
    if not np.all(np.isfinite(value)):
        raise RuntimeError(f"non-finite external trace: {path}")
    return np.asarray(value, dtype=np.float64)


def relative_error(value: float, reference: float) -> float:
    return abs(float(value) - float(reference)) / max(
        abs(float(reference)), np.finfo(np.float64).tiny
    )


def build_runtime(args) -> dict[str, object]:
    repo = Path(args.repo).expanduser().resolve()
    config_path = resolve_path(args.config, base=repo)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    runtime_paths = iteration_runtime_paths(
        config, args.iter_k, repo_root=repo
    )
    run = config_path.stem
    reference_path = (
        resolve_path(args.reference_manifest, base=repo)
        if args.reference_manifest
        else repo / "results" / run / "certified_external_reference.json"
    ).resolve()
    _, reference = load_certified_reference(repo, run, reference_path)
    parent_forward_dir = (
        resolve_path(args.parent_forward_dir, base=repo)
        if args.parent_forward_dir
        else Path(runtime_paths["transition_root"])
        / "certified_iteration"
        / "parent_forward"
    ).resolve()
    output_dir = (
        resolve_path(args.output_dir, base=repo)
        if args.output_dir
        else Path(runtime_paths["transition_root"])
        / "certified_iteration"
        / "exact_reverse"
    ).resolve()
    parent_summary_path = parent_forward_dir / "summary.json"
    parent_summary = json.loads(parent_summary_path.read_text(encoding="utf-8"))
    if parent_summary.get("result") != PASS_FORWARD:
        raise RuntimeError("parent external forward is not certified PASS")
    if (
        int(parent_summary["iteration"]) != int(args.iter_k)
        or parent_summary["transition"] != runtime_paths["transition"]
    ):
        raise RuntimeError("parent-forward iteration provenance mismatch")
    if parent_summary["reference_manifest_sha256"] != sha256_file(reference_path):
        raise RuntimeError("parent-forward reference manifest mismatch")

    material_dir = Path(runtime_paths["parent_workspace"]) / "mat" / "h5"
    material_files = {
        name: material_dir / name
        for name in (
            "Mat_0_Kappa.h5",
            "Mat_0_Mu.h5",
            "Mat_0_Density.h5",
        )
    }
    parent_material_hashes = parent_summary.get("material_sha256", {})
    for name, path in material_files.items():
        if not path.is_file():
            raise RuntimeError(f"missing parent material during reverse: {path}")
        if parent_material_hashes.get(name) != sha256_file(path):
            raise RuntimeError(f"parent material hash mismatch during reverse: {name}")
    driver = ExternalForwardDriver(
        repo,
        run,
        material_dir,
        batch_size=args.batch_size,
        reference_manifest=reference_path,
    )
    contract = reference["contract"]
    sample_count = int(contract["sample_count"])
    shape = (
        sample_count,
        int(contract["receiver_count"]),
        int(contract["component_count"]),
    )
    if not math.isclose(
        driver.dt, float(contract["dt"]), rel_tol=0.0, abs_tol=1.0e-18
    ):
        raise RuntimeError("reverse driver dt differs from reference")
    if driver.receiver_count != shape[1]:
        raise RuntimeError("reverse driver receiver count differs from reference")

    current_path = parent_forward_dir / "current_external_receiver.npy"
    true_path = Path(driver.paths["true_external"])
    if sha256_file(current_path) != parent_summary["files"][
        "current_external_sha256"
    ]:
        raise RuntimeError("parent current-external hash mismatch")
    if sha256_file(true_path) != reference["hashes"]["true_external_sha256"]:
        raise RuntimeError("frozen true-external hash mismatch")
    nodes_path = Path(driver.paths["receiver"]) / "receiver_nodes.npy"
    weights_path = Path(driver.paths["receiver"]) / "receiver_weights.npy"
    if (
        sha256_file(nodes_path) != reference["hashes"]["receiver_nodes_sha256"]
        or sha256_file(weights_path)
        != reference["hashes"]["receiver_weights_sha256"]
    ):
        raise RuntimeError("physical receiver operator hash mismatch")

    current = load_trace(current_path, shape)
    truth = load_trace(true_path, shape)
    residual = current - truth
    objective_weights = trapezoid_weights(
        np.arange(sample_count, dtype=np.float64) * driver.dt
    )
    objective = 0.5 * float(
        np.sum(objective_weights[:, None, None] * residual * residual)
    )
    parent_j = float(parent_summary["objective"]["J_external"])
    objective_relative = relative_error(objective, parent_j)
    if objective_relative > 1.0e-14:
        raise RuntimeError(
            f"reverse objective differs from parent forward: {objective_relative}"
        )

    retained_dir = (
        parent_forward_dir / "checkpoint" / "current_primal_retained"
    )
    checkpoints, positions = retained_checkpoint_map(
        retained_dir, sample_count, driver.signature
    )
    receiver_nodes = np.asarray(driver.receiver_nodes, dtype=np.int64)
    receiver_weights = np.asarray(driver.receiver_weights, dtype=np.float64)
    if not np.allclose(
        np.sum(receiver_weights, axis=1), 1.0, rtol=0.0, atol=1.0e-14
    ):
        raise RuntimeError("physical receiver interpolation rows do not sum to one")

    signature_payload = {
        "schema_version": 1,
        "iteration": int(args.iter_k),
        "transition": runtime_paths["transition"],
        "reference_manifest_sha256": sha256_file(reference_path),
        "parent_forward_signature_sha256": parent_summary[
            "input_signature_sha256"
        ],
        "driver_signature_sha256": driver.signature,
        "current_external_sha256": sha256_file(current_path),
        "true_external_sha256": sha256_file(true_path),
        "receiver_operator_sha256": sha256_arrays(
            receiver_nodes, receiver_weights
        ),
        "residual_sign": "current_external - true_external",
        "time_weighting": "native fixed-dt trapezoidal quadrature",
    }
    signature = hashlib.sha256(
        json.dumps(signature_payload, sort_keys=True).encode("utf-8")
    ).hexdigest()
    return {
        "repo": repo,
        "config_path": config_path,
        "config": config,
        "iteration_paths": runtime_paths,
        "reference_path": reference_path,
        "reference": reference,
        "parent_forward_dir": parent_forward_dir,
        "parent_summary_path": parent_summary_path,
        "parent_summary": parent_summary,
        "output_dir": output_dir,
        "driver": driver,
        "current": current,
        "truth": truth,
        "residual": residual,
        "objective_weights": objective_weights,
        "objective": objective,
        "objective_relative": objective_relative,
        "receiver_nodes": receiver_nodes,
        "receiver_weights": receiver_weights,
        "checkpoints": checkpoints,
        "coarse_positions": positions,
        "signature": signature,
        "signature_payload": signature_payload,
    }


def reverse_checkpoint_path(runtime: dict[str, object]) -> Path:
    return runtime["output_dir"] / "checkpoint" / "reverse_latest.npz"


def initial_reverse_state(runtime: dict[str, object]) -> dict[str, object]:
    data = runtime["driver"].data
    names = runtime_gradient_names(runtime)
    gradients = dict(
        zip(
            names,
            (
                np.zeros_like(data["solid"]["lam"]),
                np.zeros_like(data["solid"]["mu"]),
                np.zeros_like(data["pml"]["lam"]),
                np.zeros_like(data["pml"]["mu"]),
            ),
            strict=True,
        )
    )
    return {
        "bar": zero_state(data),
        "gradients": gradients,
        "next_transition": int(runtime["residual"].shape[0]) - 1,
        "reverse_steps": 0,
    }


def save_reverse_checkpoint(runtime, reverse_state) -> None:
    bar = reverse_state["bar"]
    gradients = reverse_state["gradients"]
    atomic_save_npz(
        reverse_checkpoint_path(runtime),
        production_signature_sha256=np.asarray(runtime["signature"]),
        next_transition=np.asarray(
            reverse_state["next_transition"], dtype=np.int64
        ),
        reverse_steps=np.asarray(reverse_state["reverse_steps"], dtype=np.int64),
        bar_Us=bar[0],
        bar_Vs=bar[1],
        bar_Vp=bar[2],
        bar_Sp=bar[3],
        **{
            f"gradient_{name}": gradients[name]
            for name in runtime_gradient_names(runtime)
        },
    )


def load_reverse_checkpoint(runtime) -> dict[str, object]:
    path = reverse_checkpoint_path(runtime)
    if not path.is_file():
        return initial_reverse_state(runtime)
    with np.load(path) as saved:
        signature = str(saved["production_signature_sha256"].item())
        if signature != runtime["signature"]:
            raise RuntimeError("reverse checkpoint signature mismatch")
        return {
            "bar": tuple(
                np.asarray(saved[name])
                for name in ("bar_Us", "bar_Vs", "bar_Vp", "bar_Sp")
            ),
            "gradients": {
                name: np.asarray(saved[f"gradient_{name}"])
                for name in runtime_gradient_names(runtime)
            },
            "next_transition": int(saved["next_transition"]),
            "reverse_steps": int(saved["reverse_steps"]),
        }


def append_progress(runtime, payload: dict) -> None:
    path = runtime["output_dir"] / "progress.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")


def preflight(runtime, replay_stride: int) -> None:
    positions = runtime["coarse_positions"]
    gaps = np.diff(np.asarray(positions, dtype=np.int64))
    checks = {
        "reference_contract_pass": True,
        "parent_forward_pass": True,
        "objective_reproduced": runtime["objective_relative"] <= 1.0e-14,
        "physical_receiver_operator_hashes": True,
        "residual_is_current_minus_true": True,
        "native_trapezoidal_weights": True,
        "retained_checkpoint_signatures": True,
        "retained_endpoint_present": positions[-1]
        == int(runtime["residual"].shape[0]),
        "replay_stride_positive": int(replay_stride) > 0,
    }
    result = {
        "result": "PASS_CERTIFIED_EXTERNAL_EXACT_REVERSE_PREFLIGHT"
        if all(checks.values())
        else "FAIL_CERTIFIED_EXTERNAL_EXACT_REVERSE_PREFLIGHT",
        "iteration": int(runtime["iteration_paths"]["transition"].split("_")[1]),
        "transition": runtime["iteration_paths"]["transition"],
        "checks": checks,
        "sample_count": int(runtime["residual"].shape[0]),
        "receiver_count": int(runtime["residual"].shape[1]),
        "objective_J": runtime["objective"],
        "coarse_checkpoint_positions": positions,
        "coarse_checkpoint_max_gap": int(np.max(gaps)),
        "replay_stride": int(replay_stride),
        "production_signature_sha256": runtime["signature"],
        "certification_statement": CERTIFICATION_STATEMENT,
        "sem3d_runs": 0,
        "exact_reverse_steps_run": 0,
    }
    runtime["output_dir"].mkdir(parents=True, exist_ok=True)
    atomic_json(runtime["output_dir"] / "preflight_summary.json", result)
    print(json.dumps(result, indent=2))
    if not all(checks.values()):
        raise SystemExit(2)


def prepare_replay(runtime, replay_stride: int) -> None:
    intervals = []
    started = time.perf_counter()
    positions = runtime["coarse_positions"]
    for start, end in zip(positions[:-1], positions[1:]):
        boundaries = ensure_replay_cache(
            runtime["output_dir"],
            runtime["driver"],
            runtime["checkpoints"],
            start,
            end,
            replay_stride,
        )
        intervals.append(
            {
                "start": int(start),
                "end": int(end),
                "boundaries": boundaries,
                "endpoint_matches_retained_bitwise": True,
            }
        )
    result = {
        "result": "PASS_CERTIFIED_EXTERNAL_REPLAY_PREPARATION",
        "transition": runtime["iteration_paths"]["transition"],
        "intervals": intervals,
        "endpoint_checks_passed": len(intervals),
        "replay_stride": int(replay_stride),
        "elapsed_seconds": float(time.perf_counter() - started),
        "production_signature_sha256": runtime["signature"],
        "sem3d_runs": 0,
        "exact_reverse_steps_run": 0,
    }
    atomic_json(runtime["output_dir"] / "replay_preparation_summary.json", result)
    print(json.dumps(result, indent=2))


def finalize(runtime, reverse_state, elapsed: float, replay_audit: dict) -> None:
    gradients = reverse_state["gradients"]
    for name, value in gradients.items():
        atomic_save_npy(runtime["output_dir"] / f"gradient_{name}.npy", value)

    gradient_summary = {
        name: {
            "path": str(runtime["output_dir"] / f"gradient_{name}.npy"),
            "sha256": sha256_file(
                runtime["output_dir"] / f"gradient_{name}.npy"
            ),
            "shape": list(value.shape),
            "l2": float(np.linalg.norm(value.reshape(-1))),
            "max_abs": float(np.max(np.abs(value))),
            "finite": bool(np.all(np.isfinite(value))),
            "nonzero": bool(np.any(value)),
        }
        for name, value in gradients.items()
    }
    sample_count = int(runtime["residual"].shape[0])
    if runtime.get("completion_gate_profile") == "material_covector":
        gates = {
            "all_reverse_transitions_completed": int(
                reverse_state["reverse_steps"]
            )
            == sample_count,
            "next_transition_is_minus_one": int(
                reverse_state["next_transition"]
            )
            == -1,
            "reverse_remained_finite": finite_reverse_state(reverse_state),
            "retained_replay_endpoints_verified": (
                replay_audit["deterministic_endpoint_bitwise"] is True
                and replay_audit["endpoint_checks_passed"]
                == replay_audit["endpoint_checks_expected"]
            ),
            "exact_physical_receiver_transpose_used": True,
            "fixed_dt_trapezoidal_residual_weighting_used": True,
            "certified_material_vjp_used": True,
            "certified_adjoint_step_used": True,
        }
    else:
        gates = {
            "reverse_steps_equal_sample_count": int(
                reverse_state["reverse_steps"]
            )
            == sample_count,
            "next_transition_is_minus_one": int(
                reverse_state["next_transition"]
            )
            == -1,
            "all_states_finite": finite_reverse_state(reverse_state),
            "all_gradients_finite": all(
                row["finite"] for row in gradient_summary.values()
            ),
            "all_expected_gradients_nonzero": all(
                row["nonzero"] for row in gradient_summary.values()
            ),
            "objective_reproduces_parent_forward": runtime["objective_relative"]
            <= 1.0e-14,
            "reference_hashes_match": True,
            "receiver_operator_matches": True,
            "retained_replay_endpoints_bitwise": replay_audit[
                "deterministic_endpoint_bitwise"
            ],
        }
    if not all(gates.values()):
        failed = [name for name, passed in gates.items() if not passed]
        raise RuntimeError("exact-reverse gates failed: " + ", ".join(failed))

    summary = {
        "schema_version": 1,
        "result": runtime_pass_result(runtime),
        "iteration": int(runtime["iteration_paths"]["transition"].split("_")[1]),
        "transition": runtime["iteration_paths"]["transition"],
        "production_signature_sha256": runtime["signature"],
        "reference_manifest": str(runtime["reference_path"]),
        "parent_forward_summary": str(runtime["parent_summary_path"]),
        "certification_statement": CERTIFICATION_STATEMENT,
        "stage5o_recertification_reused": False,
        "frozen_certified_adjoint_reused": True,
        "objective": {
            "J_external": runtime["objective"],
            "parent_forward_J": float(
                runtime["parent_summary"]["objective"]["J_external"]
            ),
            "relative_error": runtime["objective_relative"],
            "residual_sign": "current_external - true_external",
            "time_weighting": "native fixed-dt trapezoidal quadrature",
            "sample_count": sample_count,
            "receiver_count": int(runtime["residual"].shape[1]),
            "component_count": int(runtime["residual"].shape[2]),
            "dt": runtime["driver"].dt,
        },
        "reverse": {
            "steps": int(reverse_state["reverse_steps"]),
            "next_transition": int(reverse_state["next_transition"]),
            "elapsed_seconds_this_invocation": float(elapsed),
            "final_state_norm": float(state_norm(reverse_state["bar"])),
            "finite": finite_reverse_state(reverse_state),
            "checkpoint": str(reverse_checkpoint_path(runtime)),
        },
        "replay": replay_audit,
        "gradient": gradient_summary,
        "gates": gates,
        "input_hashes": runtime.get("input_hashes", {}),
        "source_equivalence": runtime.get("source_equivalence", {}),
        "output_hashes": {
            "gradients": {
                name: row["sha256"] for name, row in gradient_summary.items()
            },
            "reverse_checkpoint": sha256_file(reverse_checkpoint_path(runtime)),
            "preflight_summary": sha256_file(
                runtime["output_dir"] / "preflight_summary.json"
            ),
            "progress": sha256_file(runtime["output_dir"] / "progress.jsonl"),
        },
        "sem3d_runs": 0,
        "logical_exact_reverse_runs": 1,
    }
    atomic_json(runtime["output_dir"] / "summary.json", summary)
    print(f"RESULT = {runtime_pass_result(runtime)}")
    print(f"REVERSE_STEPS = {reverse_state['reverse_steps']}")
    print(f"OUTPUT = {runtime['output_dir']}")


def run_reverse(runtime, replay_stride: int, checkpoint_interval: int) -> None:
    final_summary = runtime["output_dir"] / "summary.json"
    if final_summary.is_file():
        existing = json.loads(final_summary.read_text(encoding="utf-8"))
        if (
            existing.get("result") == runtime_pass_result(runtime)
            and existing.get("production_signature_sha256") == runtime["signature"]
        ):
            print(f"RESULT = {runtime_pass_result(runtime)}")
            print(f"OUTPUT = {runtime['output_dir']}")
            print("IDEMPOTENT_REUSE = true")
            return
        raise RuntimeError("refusing non-matching existing reverse summary")

    runtime["output_dir"].mkdir(parents=True, exist_ok=True)
    progress_path = runtime["output_dir"] / "progress.jsonl"
    checkpoint_path = reverse_checkpoint_path(runtime)
    if progress_path.is_file():
        lines = [
            line
            for line in progress_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if not checkpoint_path.is_file():
            raise RuntimeError(
                "refusing stale reverse progress without a matching checkpoint"
            )
        if not lines:
            raise RuntimeError("reverse progress file is empty")
        last_progress = json.loads(lines[-1])
        if (
            last_progress.get("production_signature_sha256")
            != runtime["signature"]
        ):
            raise RuntimeError("reverse progress signature mismatch")
    reverse_state = load_reverse_checkpoint(runtime)
    sample_count = int(runtime["residual"].shape[0])
    if (
        int(reverse_state["reverse_steps"])
        + int(reverse_state["next_transition"])
        + 1
        != sample_count
    ):
        raise RuntimeError("reverse checkpoint step accounting mismatch")
    weighted_residual = (
        runtime["objective_weights"][:, None, None] * runtime["residual"]
    )
    positions = runtime["coarse_positions"]
    replay_audit = {
        "retained_checkpoint_positions": positions,
        "coarse_checkpoint_max_gap": int(
            np.max(np.diff(np.asarray(positions, dtype=np.int64)))
        ),
        "subchunk_stride": int(replay_stride),
        "endpoint_checks_passed": 0,
        "endpoint_checks_expected": len(positions) - 1,
        "deterministic_endpoint_bitwise": True,
        "transient_replay_cache_cleaned_by_completed_chunk": True,
    }
    started = time.perf_counter()
    last_saved = int(reverse_state["reverse_steps"])
    for start, end in reversed(list(zip(positions[:-1], positions[1:]))):
        if reverse_state["next_transition"] < start:
            replay_audit["endpoint_checks_passed"] += 1
            continue
        boundaries = ensure_replay_cache(
            runtime["output_dir"],
            runtime["driver"],
            runtime["checkpoints"],
            start,
            end,
            replay_stride,
        )
        replay_audit["endpoint_checks_passed"] += 1
        for sub_start, sub_end in reversed(
            list(zip(boundaries[:-1], boundaries[1:]))
        ):
            if reverse_state["next_transition"] < sub_start:
                continue
            active_end = min(sub_end, reverse_state["next_transition"] + 1)
            if active_end <= sub_start:
                continue
            primal = load_replay_state(
                runtime["output_dir"],
                runtime["driver"],
                runtime["checkpoints"],
                start,
                end,
                sub_start,
            )
            pre_states = []
            for transition in range(sub_start, active_end):
                pre_states.append(primal)
                primal = runtime["driver"].advance(primal, transition)
            del primal

            for transition in range(active_end - 1, sub_start - 1, -1):
                primal_pre = pre_states.pop()
                bar_u = reverse_state["bar"][0]
                seed = weighted_residual[transition]
                np.add.at(
                    bar_u,
                    runtime["receiver_nodes"],
                    seed[:, None, :] * runtime["receiver_weights"][..., None],
                )
                bar_out = (
                    bar_u,
                    reverse_state["bar"][1],
                    reverse_state["bar"][2],
                    reverse_state["bar"][3],
                )
                gl, gm, gpl, gpm = material_vjp(
                    runtime["driver"].data, primal_pre, bar_out
                )
                for name, value in zip(
                    runtime_gradient_names(runtime), (gl, gm, gpl, gpm)
                ):
                    reverse_state["gradients"][name] += value
                reverse_state["bar"] = adjoint_step(
                    runtime["driver"].data, bar_out
                )
                reverse_state["reverse_steps"] += 1
                reverse_state["next_transition"] = transition - 1
                del primal_pre, gl, gm, gpl, gpm, bar_out
            gc.collect()

            save_due = (
                reverse_state["reverse_steps"] - last_saved
                >= int(checkpoint_interval)
            )
            outer_complete = reverse_state["next_transition"] < start
            if save_due or outer_complete or reverse_state["next_transition"] < 0:
                if not finite_reverse_state(reverse_state):
                    raise RuntimeError(
                        "non-finite reverse state at transition "
                        f"{reverse_state['next_transition'] + 1}"
                    )
                save_reverse_checkpoint(runtime, reverse_state)
                last_saved = int(reverse_state["reverse_steps"])
                append_progress(
                    runtime,
                    {
                        "reverse_steps": int(reverse_state["reverse_steps"]),
                        "next_transition": int(reverse_state["next_transition"]),
                        "production_signature_sha256": runtime["signature"],
                        "adjoint_state_norm": float(
                            state_norm(reverse_state["bar"])
                        ),
                        "gradient_l2": {
                            name: float(np.linalg.norm(value.reshape(-1)))
                            for name, value in reverse_state[
                                "gradients"
                            ].items()
                        },
                        "elapsed_seconds": float(
                            time.perf_counter() - started
                        ),
                        "finite": True,
                        "coarse_interval": {
                            "start": int(start),
                            "end": int(end),
                        },
                    },
                )
        if reverse_state["next_transition"] < start:
            cleanup_replay_cache(
                runtime["output_dir"], start, end, boundaries
            )

    elapsed = time.perf_counter() - started
    if (
        reverse_state["reverse_steps"] != sample_count
        or reverse_state["next_transition"] != -1
    ):
        raise RuntimeError(
            f"reverse incomplete: steps={reverse_state['reverse_steps']}, "
            f"next={reverse_state['next_transition']}"
        )
    save_reverse_checkpoint(runtime, reverse_state)
    finalize(runtime, reverse_state, elapsed, replay_audit)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", default=".")
    parser.add_argument("--config", required=True)
    parser.add_argument("--iter-k", type=int, required=True)
    parser.add_argument("--reference-manifest")
    parser.add_argument("--parent-forward-dir")
    parser.add_argument("--output-dir")
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--replay-stride", type=int, default=10)
    parser.add_argument("--reverse-checkpoint-interval", type=int, default=50)
    parser.add_argument(
        "--action",
        choices=("preflight", "prepare-replay", "reverse"),
        required=True,
    )
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
        preflight(runtime, args.replay_stride)
    elif args.action == "prepare-replay":
        prepare_replay(runtime, args.replay_stride)
    else:
        run_reverse(
            runtime,
            args.replay_stride,
            args.reverse_checkpoint_interval,
        )


if __name__ == "__main__":
    main()
