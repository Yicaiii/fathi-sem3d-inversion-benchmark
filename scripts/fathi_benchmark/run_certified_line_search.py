"""Run or prepare a resumable certified physical-step backtracking search."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

from scripts.fathi_benchmark.line_search_contract import (
    ACCEPTANCE_POLICIES,
    acceptance_metrics,
    backtracking_steps,
    candidate_name_from_step_mpa,
    decimal_text,
)
from scripts.fathi_benchmark.runtime_paths import (
    iteration_runtime_paths,
    resolve_path,
)


PASS_OBJECTIVE = "PASS_CERTIFIED_EXTERNAL_CANDIDATE_OBJECTIVE"
PASS_PREFLIGHT = "PASS_CERTIFIED_EXTERNAL_CANDIDATE_PREFLIGHT"
PASS_PARENT = "PASS_CERTIFIED_PARENT_EXTERNAL_FORWARD"
PASS_REFERENCE = "PASS_CERTIFIED_EXTERNAL_REFERENCE_CONTRACT"


def read_json(path: Path) -> dict:
    if not path.is_file():
        raise RuntimeError(f"missing JSON: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def run_checked(command: list[str], root: Path) -> None:
    print("COMMAND =", " ".join(command), flush=True)
    environment = os.environ.copy()
    environment["FATHI_BENCHMARK_ROOT"] = str(root)
    completed = subprocess.run(
        command,
        cwd=root,
        env=environment,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"line-search child command failed with code {completed.returncode}"
        )


def optional_path(value: str | None, default: Path, root: Path) -> Path:
    return resolve_path(value, base=root) if value else default.resolve()


def build_runtime(args) -> dict[str, object]:
    root = Path(args.repo).expanduser().resolve()
    config_path = resolve_path(args.config, base=root)
    config = read_json(config_path)
    runtime_paths = iteration_runtime_paths(
        config,
        args.iter_k,
        repo_root=root,
    )
    transition_root = Path(runtime_paths["transition_root"])
    certified_root = transition_root / "certified_iteration"
    run = config_path.stem
    reference_path = optional_path(
        args.reference_manifest,
        root / "results" / run / "certified_external_reference.json",
        root,
    )
    reference = read_json(reference_path)
    if reference.get("result") != PASS_REFERENCE:
        raise RuntimeError("certified external reference is not PASS")

    parent_forward_dir = optional_path(
        args.parent_forward_dir,
        certified_root / "parent_forward",
        root,
    )
    parent_summary_path = parent_forward_dir / "summary.json"
    parent_summary = read_json(parent_summary_path)
    if parent_summary.get("result") != PASS_PARENT:
        raise RuntimeError("certified parent external forward is not PASS")
    if (
        int(parent_summary.get("iteration", -1)) != int(args.iter_k)
        or parent_summary.get("transition") != runtime_paths["transition"]
    ):
        raise RuntimeError("parent-forward line-search provenance mismatch")
    parent_objective = float(parent_summary["objective"]["J_external"])
    if not math.isfinite(parent_objective) or parent_objective < 0:
        raise RuntimeError("invalid parent objective")

    search_direction_dir = optional_path(
        args.search_direction_dir,
        certified_root / "search_direction",
        root,
    )
    search_summary_path = search_direction_dir / "search_direction_summary.json"
    search_summary = read_json(search_summary_path)
    if (
        search_summary.get("result") != "PASS"
        or int(search_summary.get("iteration", -1)) != int(args.iter_k)
        or search_summary.get("descent_direction") is not True
    ):
        raise RuntimeError("certified search direction is not a PASS descent")

    p_lambda = np.asarray(
        np.load(search_direction_dir / "p_lambda.npy"),
        dtype=np.float64,
    )
    p_mu = np.asarray(
        np.load(search_direction_dir / "p_mu.npy"),
        dtype=np.float64,
    )
    grad_lambda = np.asarray(
        np.load(search_direction_dir / "grad_lambda.npy"),
        dtype=np.float64,
    )
    grad_mu = np.asarray(
        np.load(search_direction_dir / "grad_mu.npy"),
        dtype=np.float64,
    )
    if not (
        p_lambda.shape == p_mu.shape == grad_lambda.shape == grad_mu.shape
        and p_lambda.ndim == 1
        and p_lambda.size > 0
    ):
        raise RuntimeError("line-search gradient/direction shape mismatch")
    if not all(
        np.all(np.isfinite(value))
        for value in (p_lambda, p_mu, grad_lambda, grad_mu)
    ):
        raise RuntimeError("line-search gradient/direction contains non-finite values")

    direction_scale = max(
        float(np.max(np.abs(p_lambda))),
        float(np.max(np.abs(p_mu))),
    )
    computed_g_dot_p = float(
        np.dot(grad_lambda, p_lambda) + np.dot(grad_mu, p_mu)
    )
    summary_g_dot_p = float(search_summary["g_dot_p"])
    dot_relative_error = abs(computed_g_dot_p - summary_g_dot_p) / max(
        abs(summary_g_dot_p),
        np.finfo(np.float64).tiny,
    )
    if dot_relative_error > 1.0e-14 or computed_g_dot_p >= 0:
        raise RuntimeError(
            "search-direction g_dot_p normalization contract mismatch: "
            f"relative_error={dot_relative_error:.17e}"
        )

    line_config = config.get("line_search", {})
    if not isinstance(line_config, dict):
        raise RuntimeError("line_search configuration must be a dictionary")
    configured_steps = line_config.get("steps_mpa", [])
    initial_step = (
        args.initial_step_mpa
        if args.initial_step_mpa is not None
        else line_config.get(
            "initial_step_mpa",
            configured_steps[0] if configured_steps else None,
        )
    )
    if initial_step is None:
        raise RuntimeError("missing initial_step_mpa")
    rho = args.rho if args.rho is not None else line_config.get("rho", 0.5)
    max_attempts = (
        args.max_attempts
        if args.max_attempts is not None
        else int(line_config.get("max_attempts", 8))
    )
    min_step = (
        args.min_step_mpa
        if args.min_step_mpa is not None
        else line_config.get("min_step_mpa", "0.00078125")
    )
    policy = (
        args.acceptance_policy
        if args.acceptance_policy is not None
        else line_config.get("acceptance_policy", "strict_descent")
    )
    if policy not in ACCEPTANCE_POLICIES:
        raise RuntimeError(f"unsupported acceptance policy: {policy}")
    armijo_c1 = (
        args.armijo_c1
        if args.armijo_c1 is not None
        else float(line_config.get("armijo_c1", 1.0e-4))
    )
    steps = backtracking_steps(initial_step, rho, max_attempts, min_step)

    candidate_root = optional_path(
        args.candidate_root,
        certified_root / "candidates",
        root,
    )
    objective_root = optional_path(
        args.candidate_objective_root,
        certified_root / "candidate_external_objectives",
        root,
    )
    output_dir = optional_path(
        args.output_dir,
        certified_root / "line_search",
        root,
    )
    signature_payload = {
        "schema_version": 1,
        "iteration": int(args.iter_k),
        "transition": runtime_paths["transition"],
        "config_sha256": sha256_file(config_path),
        "reference_manifest_sha256": sha256_file(reference_path),
        "parent_forward_summary_sha256": sha256_file(parent_summary_path),
        "search_direction_summary_sha256": sha256_file(search_summary_path),
        "initial_step_mpa": decimal_text(initial_step),
        "rho": decimal_text(rho),
        "max_attempts": int(max_attempts),
        "min_step_mpa": decimal_text(min_step),
        "acceptance_policy": policy,
        "armijo_c1": float(armijo_c1),
    }
    signature = hashlib.sha256(
        json.dumps(signature_payload, sort_keys=True).encode("utf-8")
    ).hexdigest()
    summary_path = output_dir / "line_search_summary.json"
    if summary_path.is_file():
        existing = read_json(summary_path)
        if existing.get("input_signature_sha256") != signature:
            raise RuntimeError(
                "existing line-search summary has different inputs; "
                "use a separate --output-dir"
            )

    return {
        "root": root,
        "config_path": config_path,
        "runtime_paths": runtime_paths,
        "reference_path": reference_path,
        "parent_forward_dir": parent_forward_dir,
        "parent_summary_path": parent_summary_path,
        "parent_objective": parent_objective,
        "search_direction_dir": search_direction_dir,
        "search_summary_path": search_summary_path,
        "search_summary": search_summary,
        "direction_scale": direction_scale,
        "g_dot_p": computed_g_dot_p,
        "g_dot_p_relative_error": dot_relative_error,
        "steps": steps,
        "initial_step": initial_step,
        "rho": rho,
        "max_attempts": int(max_attempts),
        "min_step": min_step,
        "policy": policy,
        "armijo_c1": float(armijo_c1),
        "candidate_root": candidate_root,
        "objective_root": objective_root,
        "output_dir": output_dir,
        "summary_path": summary_path,
        "signature": signature,
        "signature_payload": signature_payload,
        "batch_size": int(args.batch_size),
        "checkpoint_interval": int(args.checkpoint_interval),
    }


def candidate_paths(runtime, step) -> tuple[str, Path, Path]:
    candidate = candidate_name_from_step_mpa(step)
    candidate_dir = runtime["candidate_root"] / candidate
    objective_dir = runtime["objective_root"] / candidate
    return candidate, candidate_dir, objective_dir


def ensure_candidate_and_audit(runtime, step) -> tuple[str, bool, Path]:
    candidate, candidate_dir, _ = candidate_paths(runtime, step)
    generated = False
    if not candidate_dir.is_dir():
        generation_report = (
            runtime["output_dir"] / "candidate_generation" / candidate
        )
        run_checked(
            [
                sys.executable,
                "-m",
                "scripts.iteration_engine.generate_candidates_from_mtilde_gradient",
                "--iter-k",
                str(runtime["signature_payload"]["iteration"]),
                "--steps-mpa",
                decimal_text(step),
                "--search-direction-dir",
                str(runtime["search_direction_dir"]),
                "--candidate-root",
                str(runtime["candidate_root"]),
                "--report-dir",
                str(generation_report),
                "--config",
                str(runtime["config_path"]),
            ],
            runtime["root"],
        )
        generated = True

    audit_dir = runtime["output_dir"] / "candidate_audits" / candidate
    run_checked(
        [
            sys.executable,
            "-m",
            "scripts.iteration_engine.audit_candidates_generic",
            "--iter-k",
            str(runtime["signature_payload"]["iteration"]),
            "--steps-mpa",
            decimal_text(step),
            "--candidates",
            candidate,
            "--search-direction-dir",
            str(runtime["search_direction_dir"]),
            "--candidate-root",
            str(runtime["candidate_root"]),
            "--report-dir",
            str(audit_dir),
            "--config",
            str(runtime["config_path"]),
        ],
        runtime["root"],
    )
    audit_path = (
        audit_dir
        / f"{runtime['runtime_paths']['transition']}_candidate_audit.json"
    )
    audit = read_json(audit_path)
    if (
        audit.get("result") != "PASS"
        or audit.get("candidate_count") != 1
        or audit["records"][0].get("candidate") != candidate
        or audit["records"][0].get("ok") is not True
    ):
        raise RuntimeError(f"single-candidate audit is not PASS: {audit_path}")
    return candidate, generated, audit_path


def valid_preflight(runtime, candidate: str, path: Path) -> bool:
    if not path.is_file():
        return False
    payload = read_json(path)
    return bool(
        payload.get("result") == PASS_PREFLIGHT
        and payload.get("candidate") == candidate
        and int(payload.get("provenance", {}).get("iteration", -1))
        == int(runtime["signature_payload"]["iteration"])
        and payload.get("provenance", {}).get("transition")
        == runtime["runtime_paths"]["transition"]
        and payload.get("provenance", {}).get("reference_manifest_sha256")
        == sha256_file(runtime["reference_path"])
    )


def ensure_preflight(runtime, candidate: str, objective_dir: Path) -> tuple[Path, bool]:
    path = objective_dir / "preflight_summary.json"
    if valid_preflight(runtime, candidate, path):
        return path, True
    run_checked(
        [
            sys.executable,
            "-m",
            "scripts.fathi_benchmark.evaluate_certified_external_candidate_objective",
            "--mode",
            "candidate-preflight",
            "--repo",
            str(runtime["root"]),
            "--config",
            str(runtime["config_path"]),
            "--iter-k",
            str(runtime["signature_payload"]["iteration"]),
            "--candidate",
            candidate,
            "--candidate-root",
            str(runtime["candidate_root"]),
            "--reference-manifest",
            str(runtime["reference_path"]),
            "--output-dir",
            str(runtime["objective_root"]),
            "--batch-size",
            str(runtime["batch_size"]),
        ],
        runtime["root"],
    )
    if not valid_preflight(runtime, candidate, path):
        raise RuntimeError(f"candidate preflight is not PASS: {path}")
    return path, False


def load_reusable_objective(runtime, candidate: str, candidate_dir: Path, path: Path):
    if not path.is_file():
        return None
    payload = read_json(path)
    provenance = payload.get("provenance", {})
    material = payload.get("material", {})
    if (
        payload.get("result") != PASS_OBJECTIVE
        or payload.get("mode") != "candidate-forward"
        or payload.get("candidate") != candidate
        or int(provenance.get("iteration", -1))
        != int(runtime["signature_payload"]["iteration"])
        or provenance.get("transition") != runtime["runtime_paths"]["transition"]
        or provenance.get("reference_manifest_sha256")
        != sha256_file(runtime["reference_path"])
        or Path(material.get("material_dir", "")).resolve()
        != (candidate_dir / "mat" / "h5").resolve()
    ):
        raise RuntimeError(f"existing candidate objective provenance mismatch: {path}")
    for name, expected_hash in material.get("material_sha256", {}).items():
        material_path = candidate_dir / "mat" / "h5" / name
        if not material_path.is_file() or sha256_file(material_path) != expected_hash:
            raise RuntimeError(f"candidate material changed after objective: {name}")
    objective = float(payload["objective"]["objective"])
    if not math.isfinite(objective):
        raise RuntimeError("existing candidate objective is non-finite")
    current_path = Path(payload["objective"]["current_external"])
    if (
        not current_path.is_file()
        or sha256_file(current_path)
        != payload["objective"]["current_external_sha256"]
    ):
        raise RuntimeError("candidate external trace hash mismatch")
    return objective


def run_candidate_forward(runtime, candidate: str) -> None:
    run_checked(
        [
            sys.executable,
            "-m",
            "scripts.fathi_benchmark.evaluate_certified_external_candidate_objective",
            "--mode",
            "candidate-forward",
            "--repo",
            str(runtime["root"]),
            "--config",
            str(runtime["config_path"]),
            "--iter-k",
            str(runtime["signature_payload"]["iteration"]),
            "--candidate",
            candidate,
            "--candidate-root",
            str(runtime["candidate_root"]),
            "--reference-manifest",
            str(runtime["reference_path"]),
            "--output-dir",
            str(runtime["objective_root"]),
            "--batch-size",
            str(runtime["batch_size"]),
            "--checkpoint-interval",
            str(runtime["checkpoint_interval"]),
        ],
        runtime["root"],
    )


def summary_payload(
    runtime,
    attempts,
    *,
    result: str,
    next_step=None,
    accepted_attempt=None,
    reused_objectives: int,
    forwards_run: int,
) -> dict:
    return {
        "schema_version": 1,
        "result": result,
        "iteration": int(runtime["signature_payload"]["iteration"]),
        "transition": runtime["runtime_paths"]["transition"],
        "input_signature_sha256": runtime["signature"],
        "acceptance_policy": runtime["policy"],
        "parent_objective": runtime["parent_objective"],
        "initial_step_mpa": float(runtime["steps"][0]),
        "rho": float(runtime["rho"]),
        "max_attempts": runtime["max_attempts"],
        "min_step_mpa": float(runtime["min_step"]),
        "armijo_c1": (
            runtime["armijo_c1"] if runtime["policy"] == "armijo" else None
        ),
        "attempts": attempts,
        "next_step_mpa": None if next_step is None else float(next_step),
        "accepted_candidate": (
            None if accepted_attempt is None else accepted_attempt["candidate"]
        ),
        "accepted_step_mpa": (
            None if accepted_attempt is None else accepted_attempt["step_mpa"]
        ),
        "accepted_objective": (
            None if accepted_attempt is None else accepted_attempt["objective"]
        ),
        "promotion_ready": accepted_attempt is not None,
        "promotion_performed": False,
        "existing_pass_objectives_reused": int(reused_objectives),
        "candidate_external_forwards_this_invocation": int(forwards_run),
        "sem3d_runs": 0,
        "normalization": {
            "name": "joint_maxabs",
            "direction_scale": runtime["direction_scale"],
            "g_dot_p": runtime["g_dot_p"],
            "g_dot_p_summary_relative_error": runtime["g_dot_p_relative_error"],
            "candidate_update": (
                "m_candidate = m_parent + step_pa * p / direction_scale"
            ),
            "armijo_directional_term": (
                "(step_pa / direction_scale) * g_dot_p"
            ),
            "physical_step_is_not_raw_lbfgs_alpha": True,
        },
        "provenance": {
            "config": str(runtime["config_path"]),
            "reference_manifest": str(runtime["reference_path"]),
            "parent_forward_summary": str(runtime["parent_summary_path"]),
            "search_direction_summary": str(runtime["search_summary_path"]),
            "candidate_root": str(runtime["candidate_root"]),
            "candidate_objective_root": str(runtime["objective_root"]),
        },
    }


def run_line_search(runtime, action: str, run_expensive: bool) -> dict:
    if action == "run" and not run_expensive:
        raise RuntimeError("line-search action run requires --run-expensive")
    runtime["output_dir"].mkdir(parents=True, exist_ok=True)
    attempts = []
    reused_objectives = 0
    forwards_run = 0

    for step_index, step in enumerate(runtime["steps"]):
        candidate, generated, audit_path = ensure_candidate_and_audit(runtime, step)
        _, candidate_dir, objective_dir = candidate_paths(runtime, step)
        preflight_path, preflight_reused = ensure_preflight(
            runtime,
            candidate,
            objective_dir,
        )
        objective_path = objective_dir / "summary.json"
        objective = load_reusable_objective(
            runtime,
            candidate,
            candidate_dir,
            objective_path,
        )
        objective_reused = objective is not None

        if objective is None and action == "next":
            step_pa = float(step * 1_000_000)
            perturbation_multiplier = step_pa / runtime["direction_scale"]
            attempt = {
                "attempt": step_index + 1,
                "step_mpa": float(step),
                "step_pa": step_pa,
                "candidate": candidate,
                "status": "READY_FOR_EXTERNAL_FORWARD",
                "objective": None,
                "accepted": None,
                "actual_perturbation_multiplier": perturbation_multiplier,
                "directional_linear_prediction": (
                    perturbation_multiplier * runtime["g_dot_p"]
                ),
                "candidate_generated_this_invocation": generated,
                "candidate_audit": "PASS",
                "candidate_audit_path": str(audit_path),
                "candidate_preflight": "PASS",
                "candidate_preflight_path": str(preflight_path),
                "candidate_preflight_reused": preflight_reused,
                "objective_reused": False,
            }
            attempts.append(attempt)
            payload = summary_payload(
                runtime,
                attempts,
                result="PASS_CERTIFIED_LINE_SEARCH_NEXT_READY",
                next_step=step,
                reused_objectives=reused_objectives,
                forwards_run=forwards_run,
            )
            atomic_json(runtime["summary_path"], payload)
            return payload

        if objective is None:
            run_candidate_forward(runtime, candidate)
            forwards_run += 1
            objective = load_reusable_objective(
                runtime,
                candidate,
                candidate_dir,
                objective_path,
            )
            if objective is None:
                raise RuntimeError("candidate forward did not produce a PASS objective")
        else:
            reused_objectives += 1

        metrics = acceptance_metrics(
            policy=runtime["policy"],
            parent_objective=runtime["parent_objective"],
            candidate_objective=objective,
            step_mpa=step,
            direction_scale=runtime["direction_scale"],
            g_dot_p=runtime["g_dot_p"],
            armijo_c1=runtime["armijo_c1"],
        )
        attempt = {
            "attempt": step_index + 1,
            "step_mpa": float(step),
            "step_pa": metrics["step_pa"],
            "candidate": candidate,
            "status": "EVALUATED_REUSED" if objective_reused else "EVALUATED",
            "objective": objective,
            "accepted": metrics["accepted"],
            "acceptance_rhs": metrics["acceptance_rhs"],
            "delta_objective": metrics["delta_objective"],
            "relative_change": metrics["relative_change"],
            "directional_linear_prediction": metrics[
                "directional_linear_prediction"
            ],
            "armijo_c1": metrics["armijo_c1"],
            "armijo_rhs": (
                metrics["armijo_rhs"] if runtime["policy"] == "armijo" else None
            ),
            "actual_perturbation_multiplier": metrics[
                "actual_perturbation_multiplier"
            ],
            "candidate_generated_this_invocation": generated,
            "candidate_audit": "PASS",
            "candidate_audit_path": str(audit_path),
            "candidate_preflight": "PASS",
            "candidate_preflight_path": str(preflight_path),
            "candidate_preflight_reused": preflight_reused,
            "objective_reused": objective_reused,
            "objective_summary": str(objective_path),
            "reason": (
                "acceptance condition satisfied"
                if metrics["accepted"]
                else (
                    "not strict descent"
                    if runtime["policy"] == "strict_descent"
                    else "Armijo condition not satisfied"
                )
            ),
        }
        attempts.append(attempt)
        if metrics["accepted"]:
            payload = summary_payload(
                runtime,
                attempts,
                result="PASS_CERTIFIED_LINE_SEARCH_ACCEPTED",
                accepted_attempt=attempt,
                reused_objectives=reused_objectives,
                forwards_run=forwards_run,
            )
            atomic_json(runtime["summary_path"], payload)
            return payload

        next_step = (
            runtime["steps"][step_index + 1]
            if step_index + 1 < len(runtime["steps"])
            else None
        )
        payload = summary_payload(
            runtime,
            attempts,
            result="IN_PROGRESS_CERTIFIED_LINE_SEARCH",
            next_step=next_step,
            reused_objectives=reused_objectives,
            forwards_run=forwards_run,
        )
        atomic_json(runtime["summary_path"], payload)

    payload = summary_payload(
        runtime,
        attempts,
        result="STOP_CERTIFIED_LINE_SEARCH_EXHAUSTED",
        reused_objectives=reused_objectives,
        forwards_run=forwards_run,
    )
    atomic_json(runtime["summary_path"], payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", default=".")
    parser.add_argument("--config", required=True)
    parser.add_argument("--iter-k", type=int, required=True)
    parser.add_argument("--action", choices=("next", "run"), required=True)
    parser.add_argument("--reference-manifest")
    parser.add_argument("--parent-forward-dir")
    parser.add_argument("--search-direction-dir")
    parser.add_argument("--candidate-root")
    parser.add_argument("--candidate-objective-root")
    parser.add_argument("--output-dir")
    parser.add_argument("--initial-step-mpa")
    parser.add_argument("--rho")
    parser.add_argument("--max-attempts", type=int)
    parser.add_argument("--min-step-mpa")
    parser.add_argument("--acceptance-policy", choices=ACCEPTANCE_POLICIES)
    parser.add_argument("--armijo-c1", type=float)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--checkpoint-interval", type=int, default=100)
    parser.add_argument("--run-expensive", action="store_true")
    args = parser.parse_args()
    if args.iter_k < 0:
        parser.error("--iter-k must be nonnegative")
    if args.batch_size < 1 or args.checkpoint_interval < 1:
        parser.error("batch and checkpoint intervals must be positive")

    runtime = build_runtime(args)
    payload = run_line_search(runtime, args.action, args.run_expensive)
    print(f"RESULT = {payload['result']}")
    print(f"SUMMARY = {runtime['summary_path']}")
    if payload["next_step_mpa"] is not None:
        print(f"NEXT_STEP_MPA = {payload['next_step_mpa']}")
    if payload["accepted_candidate"] is not None:
        print(f"ACCEPTED_CANDIDATE = {payload['accepted_candidate']}")
    if payload["result"] == "STOP_CERTIFIED_LINE_SEARCH_EXHAUSTED":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
