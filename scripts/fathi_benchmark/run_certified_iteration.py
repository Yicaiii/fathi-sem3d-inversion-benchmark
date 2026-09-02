"""HISTORICAL_ONLY runner for the pre-CURRENT certified Fathi workflow."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

from scripts.fathi_benchmark.line_search_contract import (
    candidate_name_from_step_mpa,
)
from scripts.fathi_benchmark.runtime_paths import iteration_runtime_paths, resolve_path
from scripts.fathi_benchmark.current_pipeline_contracts import CURRENT_RUN_ID


BLOCKED_CURRENT_RESULT = "BLOCKED_HISTORICAL_ITERATION_RUNNER_FOR_CURRENT"


def guard_historical_runner(run_id: str) -> None:
    """Refuse CURRENT before any historical child process can be dispatched."""

    if str(run_id) == CURRENT_RUN_ID:
        raise RuntimeError(BLOCKED_CURRENT_RESULT)


STAGES = (
    "status",
    "context",
    "parent-forward",
    "reverse-preflight",
    "prepare-replay",
    "reverse",
    "gradient-bridge",
    "search-direction",
    "register-gradient",
    "candidates-audit-inputs",
    "candidates-generate",
    "candidates-audit",
    "candidate-generate",
    "candidate-audit",
    "candidate-preflight",
    "candidate-forward",
    "line-search-next",
    "line-search",
    "promotion-audit",
    "promote",
)
EXPENSIVE_STAGES = {
    "parent-forward",
    "prepare-replay",
    "reverse",
    "candidate-forward",
    "line-search",
}


def run_command(command: list[str], root: Path) -> None:
    print("COMMAND =", " ".join(str(value) for value in command))
    environment = os.environ.copy()
    environment["FATHI_BENCHMARK_ROOT"] = str(root)
    completed = subprocess.run(
        command, cwd=root, env=environment, check=False
    )
    if completed.returncode != 0:
        raise SystemExit(completed.returncode)


def resolved(value: str | None, default: Path, root: Path) -> Path:
    return resolve_path(value, base=root) if value else default.resolve()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--iter-k", type=int, required=True)
    parser.add_argument("--stage", choices=STAGES, required=True)
    parser.add_argument("--repo")
    parser.add_argument("--reference-manifest")
    parser.add_argument("--parent-forward-dir")
    parser.add_argument("--reverse-dir")
    parser.add_argument("--gradient-dir")
    parser.add_argument("--search-direction-dir")
    parser.add_argument("--candidate")
    parser.add_argument("--candidate-root")
    parser.add_argument("--candidate-objective-root")
    parser.add_argument("--output-dir")
    parser.add_argument("--source")
    parser.add_argument("--step-mpa")
    parser.add_argument("--initial-step-mpa")
    parser.add_argument("--rho")
    parser.add_argument("--max-attempts", type=int)
    parser.add_argument("--min-step-mpa")
    parser.add_argument(
        "--acceptance-policy",
        choices=("strict_descent", "armijo"),
    )
    parser.add_argument("--armijo-c1", type=float)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--checkpoint-interval", type=int, default=100)
    parser.add_argument("--replay-stride", type=int, default=10)
    parser.add_argument("--reverse-checkpoint-interval", type=int, default=50)
    parser.add_argument("--run-expensive", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if args.iter_k < 0:
        parser.error("--iter-k must be nonnegative")
    if args.stage in EXPENSIVE_STAGES and not args.run_expensive:
        raise SystemExit(
            f"Stage {args.stage} is expensive. Re-run with --run-expensive "
            "after reviewing resumability and monitoring paths."
        )

    root = (
        Path(args.repo).expanduser().resolve()
        if args.repo
        else Path(os.environ.get("FATHI_BENCHMARK_ROOT", ".")).expanduser().resolve()
    )
    config_path = resolve_path(args.config, base=root)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    guard_historical_runner(str(config.get("benchmark_name", "")))
    runtime = iteration_runtime_paths(config, args.iter_k, repo_root=root)
    run = config_path.stem
    transition_root = Path(runtime["transition_root"])
    certified_root = transition_root / "certified_iteration"
    reference = resolved(
        args.reference_manifest,
        root / "results" / run / "certified_external_reference.json",
        root,
    )
    registry = root / "results" / run / "certified_gradient_registry.json"
    parent_forward_dir = resolved(
        args.parent_forward_dir, certified_root / "parent_forward", root
    )
    reverse_dir = resolved(
        args.reverse_dir, certified_root / "exact_reverse", root
    )
    bridge_dir = certified_root / "optimizer_bridge"
    solve_dir = resolved(
        args.gradient_dir, bridge_dir / "mtilde_solve", root
    )
    search_dir = resolved(
        args.search_direction_dir, certified_root / "search_direction", root
    )
    candidate_root = resolved(
        args.candidate_root, certified_root / "candidates", root
    )
    objective_root = resolved(
        args.candidate_objective_root,
        certified_root / "candidate_external_objectives",
        root,
    )
    report_dir = certified_root / "task4_reports"
    line_search_dir = certified_root / "line_search"
    selected_candidate = args.candidate
    if args.step_mpa is not None:
        step_candidate = candidate_name_from_step_mpa(args.step_mpa)
        if selected_candidate is not None and selected_candidate != step_candidate:
            raise SystemExit(
                "--candidate does not match the deterministic --step-mpa name: "
                f"{step_candidate}"
            )
        selected_candidate = step_candidate

    if args.stage == "status":
        artifacts = {
            "context": transition_root
            / f"{runtime['transition']}_iteration_context.json",
            "parent_forward": parent_forward_dir / "summary.json",
            "reverse_preflight": reverse_dir / "preflight_summary.json",
            "reverse": reverse_dir / "summary.json",
            "optimizer_bridge": bridge_dir / "summary.json",
            "search_direction": search_dir / "search_direction_summary.json",
            "line_search": line_search_dir / "line_search_summary.json",
            "gradient_registry": registry,
            "accepted_next": Path(runtime["accepted_dir"])
            / "accepted_summary.json",
        }
        print(
            json.dumps(
                {
                    "run": run,
                    "iter_k": int(args.iter_k),
                    "transition": runtime["transition"],
                    "parent_workspace": str(runtime["parent_workspace"]),
                    "reference_manifest": str(reference),
                    "artifacts": {
                        name: {"path": str(path), "exists": path.is_file()}
                        for name, path in artifacts.items()
                    },
                    "available_stages": list(STAGES),
                    "expensive_stages_require_flag": sorted(EXPENSIVE_STAGES),
                },
                indent=2,
            )
        )
        return

    if args.stage == "context":
        command = [
            sys.executable,
            "-m",
            "scripts.fathi_benchmark.create_iteration_context_generic",
            "--iter-k",
            str(args.iter_k),
            "--config",
            str(config_path),
            "--parent-forward-contract",
            "certified_external",
            "--write",
        ]
        if args.force:
            command.append("--overwrite")
        run_command(command, root)
        return

    if args.stage == "parent-forward":
        run_command(
            [
                sys.executable,
                "-m",
                "scripts.fathi_benchmark.run_certified_external_parent_forward",
                "--repo",
                str(root),
                "--config",
                str(config_path),
                "--iter-k",
                str(args.iter_k),
                "--reference-manifest",
                str(reference),
                "--output-dir",
                str(resolved(args.output_dir, parent_forward_dir, root)),
                "--batch-size",
                str(args.batch_size),
                "--checkpoint-interval",
                str(args.checkpoint_interval),
            ],
            root,
        )
        return

    if args.stage in {"reverse-preflight", "prepare-replay", "reverse"}:
        action = {
            "reverse-preflight": "preflight",
            "prepare-replay": "prepare-replay",
            "reverse": "reverse",
        }[args.stage]
        run_command(
            [
                sys.executable,
                "-m",
                "scripts.fathi_benchmark.run_certified_external_exact_reverse",
                "--repo",
                str(root),
                "--config",
                str(config_path),
                "--iter-k",
                str(args.iter_k),
                "--reference-manifest",
                str(reference),
                "--parent-forward-dir",
                str(parent_forward_dir),
                "--output-dir",
                str(resolved(args.output_dir, reverse_dir, root)),
                "--batch-size",
                str(args.batch_size),
                "--replay-stride",
                str(args.replay_stride),
                "--reverse-checkpoint-interval",
                str(args.reverse_checkpoint_interval),
                "--action",
                action,
            ],
            root,
        )
        return

    if args.stage == "gradient-bridge":
        run_command(
            [
                sys.executable,
                "-m",
                "scripts.fathi_benchmark.bridge_certified_external_gradient",
                "--repo",
                str(root),
                "--config",
                str(config_path),
                "--iteration",
                str(args.iter_k),
                "--reference-manifest",
                str(reference),
                "--reverse-dir",
                str(reverse_dir),
                "--out-dir",
                str(resolved(args.output_dir, bridge_dir, root)),
                "--batch-size",
                str(max(args.batch_size, 1)),
            ],
            root,
        )
        return

    if args.stage == "search-direction":
        run_command(
            [
                sys.executable,
                "-m",
                "scripts.fathi_benchmark.compute_search_direction",
                "--iteration",
                str(args.iter_k),
                "--gradient-dir",
                str(solve_dir),
                "--out-dir",
                str(resolved(args.output_dir, search_dir, root)),
                "--config",
                str(config_path),
                "--history-gradient-registry",
                str(registry),
            ],
            root,
        )
        return

    if args.stage == "register-gradient":
        run_command(
            [
                sys.executable,
                "-m",
                "scripts.fathi_benchmark.register_certified_gradient",
                "--repo",
                str(root),
                "--registry",
                str(registry),
                "--iteration",
                str(args.iter_k),
                "--gradient-dir",
                str(search_dir),
                "--source",
                args.source or "certified_iteration/search_direction",
                "--historical-legacy",
            ],
            root,
        )
        return

    task4_common = [
        "--iter-k",
        str(args.iter_k),
        "--config",
        str(config_path),
        "--search-direction-dir",
        str(search_dir),
        "--report-dir",
        str(report_dir),
    ]
    if args.stage == "candidates-audit-inputs":
        run_command(
            [
                sys.executable,
                "-m",
                "scripts.iteration_engine.audit_candidate_inputs",
                *task4_common,
            ],
            root,
        )
        return
    if args.stage == "candidates-generate":
        command = [
            sys.executable,
            "-m",
            "scripts.iteration_engine.generate_candidates_from_mtilde_gradient",
            *task4_common,
            "--candidate-root",
            str(candidate_root),
        ]
        if args.force:
            command.append("--force")
        run_command(command, root)
        return
    if args.stage == "candidates-audit":
        run_command(
            [
                sys.executable,
                "-m",
                "scripts.iteration_engine.audit_candidates_generic",
                *task4_common,
                "--candidate-root",
                str(candidate_root),
            ],
            root,
        )
        return

    if args.stage == "candidate-generate":
        if args.step_mpa is None or selected_candidate is None:
            raise SystemExit("--step-mpa is required")
        run_command(
            [
                sys.executable,
                "-m",
                "scripts.iteration_engine.generate_candidates_from_mtilde_gradient",
                "--iter-k",
                str(args.iter_k),
                "--config",
                str(config_path),
                "--search-direction-dir",
                str(search_dir),
                "--candidate-root",
                str(candidate_root),
                "--steps-mpa",
                args.step_mpa,
                "--report-dir",
                str(line_search_dir / "candidate_generation" / selected_candidate),
            ],
            root,
        )
        return

    if args.stage == "candidate-audit":
        if args.step_mpa is None or selected_candidate is None:
            raise SystemExit("--step-mpa is required")
        run_command(
            [
                sys.executable,
                "-m",
                "scripts.iteration_engine.audit_candidates_generic",
                "--iter-k",
                str(args.iter_k),
                "--config",
                str(config_path),
                "--search-direction-dir",
                str(search_dir),
                "--candidate-root",
                str(candidate_root),
                "--candidates",
                selected_candidate,
                "--steps-mpa",
                args.step_mpa,
                "--report-dir",
                str(line_search_dir / "candidate_audits" / selected_candidate),
            ],
            root,
        )
        return

    if args.stage in {"candidate-preflight", "candidate-forward"}:
        if not selected_candidate:
            raise SystemExit("--candidate or --step-mpa is required")
        mode = (
            "candidate-preflight"
            if args.stage == "candidate-preflight"
            else "candidate-forward"
        )
        run_command(
            [
                sys.executable,
                "-m",
                "scripts.fathi_benchmark.evaluate_certified_external_candidate_objective",
                "--mode",
                mode,
                "--repo",
                str(root),
                "--config",
                str(config_path),
                "--iter-k",
                str(args.iter_k),
                "--candidate",
                selected_candidate,
                "--candidate-root",
                str(candidate_root),
                "--reference-manifest",
                str(reference),
                "--output-dir",
                str(objective_root),
                "--batch-size",
                str(args.batch_size),
                "--checkpoint-interval",
                str(args.checkpoint_interval),
            ],
            root,
        )
        return

    if args.stage in {"line-search-next", "line-search"}:
        command = [
            sys.executable,
            "-m",
            "scripts.fathi_benchmark.run_certified_line_search",
            "--repo",
            str(root),
            "--config",
            str(config_path),
            "--iter-k",
            str(args.iter_k),
            "--action",
            "next" if args.stage == "line-search-next" else "run",
            "--reference-manifest",
            str(reference),
            "--parent-forward-dir",
            str(parent_forward_dir),
            "--search-direction-dir",
            str(search_dir),
            "--candidate-root",
            str(candidate_root),
            "--candidate-objective-root",
            str(objective_root),
            "--output-dir",
            str(resolved(args.output_dir, line_search_dir, root)),
            "--batch-size",
            str(args.batch_size),
            "--checkpoint-interval",
            str(args.checkpoint_interval),
        ]
        optional_values = (
            ("--initial-step-mpa", args.initial_step_mpa),
            ("--rho", args.rho),
            ("--max-attempts", args.max_attempts),
            ("--min-step-mpa", args.min_step_mpa),
            ("--acceptance-policy", args.acceptance_policy),
            ("--armijo-c1", args.armijo_c1),
        )
        for option, value in optional_values:
            if value is not None:
                command.extend((option, str(value)))
        if args.stage == "line-search":
            command.append("--run-expensive")
        run_command(command, root)
        return

    if args.stage in {"promotion-audit", "promote"}:
        if not selected_candidate:
            raise SystemExit("--candidate or --step-mpa is required")
        command = [
            sys.executable,
            "-m",
            "scripts.fathi_benchmark.promote_certified_external_candidate",
            "--mode",
            "audit" if args.stage == "promotion-audit" else "promote",
            "--repo",
            str(root),
            "--config",
            str(config_path),
            "--iter-k",
            str(args.iter_k),
            "--candidate",
            selected_candidate,
            "--candidate-root",
            str(candidate_root),
            "--candidate-objective-root",
            str(objective_root),
            "--reference-manifest",
            str(reference),
        ]
        if args.force:
            command.append("--force")
        run_command(command, root)
        return

    raise AssertionError(args.stage)


if __name__ == "__main__":
    main()
