from __future__ import annotations

from datetime import datetime
from pathlib import Path
import argparse
import json

try:
    from scripts.fathi_benchmark.runtime_paths import (
        context_path_value,
        repository_root,
        resolve_path,
        runtime_resolve_path,
        runtime_root,
        sem3d_executable,
    )
except ModuleNotFoundError:
    from runtime_paths import (
        context_path_value,
        repository_root,
        resolve_path,
        runtime_resolve_path,
        runtime_root,
        sem3d_executable,
    )


ROOT = repository_root()
RUNTIME = runtime_root(ROOT)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--iter-k", type=int, required=True)
    parser.add_argument(
        "--config",
        default="benchmark_fathi_strict/config/benchmark_config.json",
    )
    parser.add_argument("--write", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    config_path = resolve_path(args.config, base=ROOT)
    config = json.loads(config_path.read_text(encoding="utf-8"))

    k = int(args.iter_k)
    kp1 = k + 1
    transition = f"iter_{k:03d}_to_iter_{kp1:03d}"

    run_data_root = runtime_resolve_path(
        config["run_data_root"], repo_root=ROOT
    )
    run_result_root = runtime_resolve_path(
        config["run_result_root"], repo_root=ROOT
    )
    state_dir = runtime_resolve_path(
        config["state_dir"], repo_root=ROOT
    )
    observed_traces = runtime_resolve_path(
        config["true_observed_traces_dir"], repo_root=ROOT
    )

    parent_state_value = config.get("parent_state_path")
    parent_state = (
        runtime_resolve_path(parent_state_value, repo_root=ROOT)
        if parent_state_value
        else state_dir / f"iter_{k:03d}_state_v2_corrected.npz"
    )
    next_state = state_dir / f"iter_{kp1:03d}_state_v2_corrected.npz"
    output_iter_root = run_data_root / f"iter_{kp1:03d}"
    transition_root = run_result_root / transition
    parent_accepted_value = config.get("parent_accepted_dir")
    parent_accepted = (
        runtime_resolve_path(parent_accepted_value, repo_root=ROOT)
        if parent_accepted_value
        else run_data_root / f"iter_{k:03d}" / "accepted"
    )
    accepted_next = output_iter_root / "accepted"

    strict_workspace_name = config.get(
        "strict_forward_workspace_name",
        "strict_full_forward_000",
    )
    adjoint_dir_name = config.get(
        "adjoint_batches_dir_name",
        "adjoint_full_grid_batches",
    )
    strict_workspace = (
        output_iter_root
        / "forward_dudx_mgcap_full_batches"
        / strict_workspace_name
    )
    residual_dir = transition_root / "residual_sources"
    adjoint_dir = output_iter_root / adjoint_dir_name
    rhs_dir = transition_root / "component_rhs"
    mtilde_dir = transition_root / "mtilde_solve"

    matrix_value = config.get(
        "mtilde_matrix_path",
        (
            "results/audit_teacher_feedback/"
            "iter005_mass_matrix/"
            "Mtilde_q1_consistent_sparse.npz"
        ),
    )
    matrix_path = runtime_resolve_path(
        matrix_value, repo_root=ROOT
    )
    profile_value = config.get(
        "benchmark_profile_config",
        "configs/fathi_reduced_3x3_12p5.json",
    )
    profile_path = runtime_resolve_path(
        profile_value,
        repo_root=ROOT,
    )

    context = {
        "created": datetime.now().isoformat(),
        "project_root": context_path_value(ROOT),
        "runtime_root": context_path_value(RUNTIME),
        "config_path": context_path_value(config_path),
        "benchmark_profile_config": context_path_value(profile_path),
        "iter_k": k,
        "iter_kp1": kp1,
        "transition": transition,
        "parent_state": context_path_value(parent_state),
        "next_state": context_path_value(next_state),
        "parent_accepted_dir": context_path_value(parent_accepted),
        "accepted_dir": context_path_value(accepted_next),
        "true_observed_traces_dir": context_path_value(observed_traces),
        "output_iter_root": context_path_value(output_iter_root),
        "transition_result_root": context_path_value(transition_root),
        "strict_forward_workspace": context_path_value(strict_workspace),
        "strict_forward_traces": context_path_value(
            strict_workspace / "traces"
        ),
        "residual_dir": context_path_value(residual_dir),
        "residual_h5": context_path_value(
            residual_dir / "454B_strict_residual_timeseries.h5"
        ),
        "residual_summary_txt": context_path_value(
            residual_dir
            / "454B_strict_residual_timeseries_summary.txt"
        ),
        "adjoint_batches_dir": context_path_value(adjoint_dir),
        "rhs_manifest_dir": context_path_value(
            transition_root / "rhs_manifests"
        ),
        "component_rhs_dir": context_path_value(rhs_dir),
        "mtilde_dir": context_path_value(mtilde_dir),
        "mtilde_matrix_path": context_path_value(matrix_path),
        "candidates_dir": context_path_value(
            transition_root / "candidates"
        ),
        "candidate_forward_workspaces_dir": context_path_value(
            output_iter_root / "candidate_forward_workspaces"
        ),
        "candidate_misfit_dir": context_path_value(
            transition_root / "candidate_misfits"
        ),
        "tv_regularization_dir": context_path_value(
            transition_root / "tv_regularization"
        ),
        "sem3d_exe": context_path_value(
            sem3d_executable(config, repo_root=ROOT)
        ),
        "mpi_cores": int(config.get("mpi_cores", 12)),
        "line_search_candidates": config.get(
            "line_search", {}
        ).get(
            "amplitudes_MPa",
            config.get(
                "line_search_amplitudes_mpa",
                [0.10, 0.25, 0.50, 1.00],
            ),
        ),
        "material_shape": config.get(
            "material_shape", [41, 33, 33]
        ),
        "regularization": config.get(
            "regularization",
            config.get("tv_extension", {}),
        ),
        # Current child-script aliases.
        "root": context_path_value(ROOT),
        "work_root": context_path_value(transition_root),
        "true_observed_traces": context_path_value(observed_traces),
        "input_state": context_path_value(parent_state),
        "output_state": context_path_value(next_state),
        "input_accepted_dir": context_path_value(parent_accepted),
        "output_accepted_dir": context_path_value(accepted_next),
        "output_forward_batches_dir": context_path_value(
            output_iter_root / "forward_dudx_mgcap_full_batches"
        ),
        "output_adjoint_batches_dir": context_path_value(adjoint_dir),
        "output_candidate_runs_dir": context_path_value(
            output_iter_root / "candidate_forward_workspaces"
        ),
        "line_search_dir": context_path_value(
            transition_root / "candidate_misfits"
        ),
        "mtilde_solve_dir": context_path_value(mtilde_dir),
        "from_tag": f"iter_{k:03d}",
        "to_tag": f"iter_{kp1:03d}",
        "role": "generic_iteration_context",
    }

    checks = {
        "parent_state_exists": parent_state.exists(),
        "parent_accepted_dir_exists": parent_accepted.exists(),
        "true_observed_traces_dir_exists": observed_traces.exists(),
        "sem3d_executable_exists": Path(context["sem3d_exe"]).is_file(),
    }
    context["preflight_checks"] = checks
    context["preflight_ok"] = all(checks.values())

    transition_root.mkdir(parents=True, exist_ok=True)
    out_json = transition_root / f"{transition}_iteration_context.json"
    out_txt = transition_root / f"{transition}_iteration_context.txt"

    lines = [
        "Generic Fathi iteration context",
        "===============================",
        "",
        f"transition = {transition}",
        f"repository_root = {ROOT}",
        f"runtime_root = {RUNTIME}",
        "",
        "Preflight checks:",
    ]
    for key, value in checks.items():
        lines.append(f"  {key} = {value}")
    lines.extend(
        [
            "",
            f"context = {out_json}",
            (
                "RESULT = PASS"
                if context["preflight_ok"]
                else "RESULT = CHECK_NEEDED"
            ),
        ]
    )

    if args.write:
        if out_json.exists() and not args.overwrite:
            lines.extend(
                [
                    "",
                    "WRITE_SKIPPED: context already exists.",
                    "Use --overwrite only intentionally.",
                ]
            )
        else:
            out_json.write_text(
                json.dumps(context, indent=2),
                encoding="utf-8",
            )
            out_txt.write_text(
                "\n".join(lines) + "\n",
                encoding="utf-8",
            )

    print("\n".join(lines))


if __name__ == "__main__":
    main()
