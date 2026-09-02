"""Fail-fast consistency gate for the two CURRENT path configurations."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, Mapping

from scripts.fathi_benchmark.iteration_context import build_iteration_paths
from scripts.fathi_benchmark.runtime_paths import (
    iteration_runtime_paths,
    runtime_root,
)


def validate_path_config_consistency(
    runtime_config: Mapping[str, Any],
    engine_config: Mapping[str, Any],
    *,
    repository_root: str | Path,
    parent_iterations: Iterable[int] = (0, 1, 9),
) -> dict[str, Any]:
    """Assert equivalent mutable routes resolve identically.

    The legacy runtime layout and the new engine routes intentionally name
    different gradient/optimizer/line-search subtrees.  This gate compares only
    artifacts both schemas claim to describe: iteration roots, accepted
    workspaces, transition roots, and parent/child state files.
    """

    repository = Path(repository_root).expanduser().resolve()
    runtime_run = str(runtime_config.get("benchmark_name", ""))
    engine_run = str(engine_config.get("run_id", ""))
    if not runtime_run or runtime_run != engine_run:
        raise ValueError(
            f"run mismatch between runtime and engine configs: "
            f"{runtime_run!r} != {engine_run!r}"
        )
    selected_runtime_root = runtime_root(repository)
    checks = []
    for parent in parent_iterations:
        legacy = iteration_runtime_paths(
            runtime_config, int(parent), repo_root=repository
        )
        current = build_iteration_paths(
            engine_config,
            int(parent),
            repository_root=repository,
            runtime_root=selected_runtime_root,
        )
        pairs = {
            "parent_iteration_root": (
                legacy["iter_k_root"], current.parent_iteration_root
            ),
            "child_iteration_root": (
                legacy["iter_kp1_root"], current.child_iteration_root
            ),
            "parent_accepted": (
                legacy["parent_workspace"], current.parent_accepted
            ),
            "child_accepted": (legacy["accepted_dir"], current.child_accepted),
            "transition_root": (
                legacy["transition_root"], current.transition_root
            ),
            "parent_state": (legacy["parent_state"], current.parent_state),
            "child_state": (legacy["child_state"], current.child_state),
        }
        mismatches = {
            name: {"runtime_layout": str(left), "iteration_engine": str(right)}
            for name, (left, right) in pairs.items()
            if Path(left).resolve() != Path(right).resolve()
        }
        if mismatches:
            raise ValueError(
                f"path config drift for parent {parent}: {mismatches}"
            )
        checks.append(
            {
                "parent_iteration": int(parent),
                "child_iteration": int(parent) + 1,
                "transition": current.identity.transition_id,
                "paths": {
                    name: str(Path(left).resolve())
                    for name, (left, _) in pairs.items()
                },
                "pass": True,
            }
        )
    return {
        "result": "PASS_CURRENT_PATH_CONFIG_CONSISTENCY",
        "run_id": engine_run,
        "checked_parent_iterations": [
            item["parent_iteration"] for item in checks
        ],
        "checks": checks,
        "simulation_runs": 0,
    }
