from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping
import os

from scripts.fathi_benchmark.iteration_context import IterationIdentity


def repository_root(anchor: str | Path = __file__) -> Path:
    """Return the repository root.

    Priority:
    1. FATHI_BENCHMARK_ROOT environment variable.
    2. Automatic discovery from the current file location.
    """
    configured = os.environ.get("FATHI_BENCHMARK_ROOT")

    if configured:
        root = Path(configured).expanduser().resolve()
        if not root.exists():
            raise RuntimeError(
                "FATHI_BENCHMARK_ROOT does not exist: "
                f"{root}"
            )
        return root

    start = Path(anchor).expanduser().resolve()
    candidates = [start.parent, *start.parents]

    for candidate in candidates:
        if (
            (candidate / "README.md").is_file()
            and (candidate / "scripts").is_dir()
        ):
            return candidate

    raise RuntimeError(
        "Cannot discover the repository root. "
        "Set FATHI_BENCHMARK_ROOT explicitly."
    )


def runtime_root(
    repo_root: Path | None = None,
) -> Path:
    """Return the heavy-data runtime root.

    FATHI_RUNTIME_ROOT may point outside the Git repository.
    During the compatibility phase, the repository root remains
    the fallback so that existing relative paths still work.
    """
    configured = os.environ.get("FATHI_RUNTIME_ROOT")

    if configured:
        return Path(configured).expanduser().resolve()

    return repo_root or repository_root()


def resolve_path(
    value: str | Path,
    *,
    base: Path | None = None,
) -> Path:
    """Resolve an absolute path or a path relative to a selected base."""
    path = Path(value).expanduser()

    if path.is_absolute():
        return path.resolve()

    selected_base = base or repository_root()
    return (selected_base / path).resolve()


def sem3d_executable(
    config: Mapping[str, Any] | None = None,
    *,
    repo_root: Path | None = None,
) -> Path:
    """Resolve SEM3D from SEM3D_EXE or, temporarily, from a config file."""
    configured = os.environ.get("SEM3D_EXE")

    if configured:
        return resolve_path(
            configured,
            base=repo_root or repository_root(),
        )

    config_value = None if config is None else config.get("sem3d_exe")

    if config_value:
        return resolve_path(
            config_value,
            base=repo_root or repository_root(),
        )

    raise RuntimeError(
        "SEM3D executable is not configured. "
        "Set SEM3D_EXE or provide sem3d_exe in the benchmark config."
    )


def sem3d_mpirun(
    config: Mapping[str, Any] | None = None,
    *,
    repo_root: Path | None = None,
) -> Path:
    configured = os.environ.get("SEM3D_MPIRUN")

    if configured:
        return resolve_path(
            configured,
            base=repo_root or repository_root(),
        )

    config_value = (
        None if config is None
        else config.get("sem3d_mpirun")
    )

    if config_value:
        return resolve_path(
            config_value,
            base=repo_root or repository_root(),
        )

    raise RuntimeError(
        "SEM3D MPI launcher is not configured. "
        "Set SEM3D_MPIRUN or provide sem3d_mpirun "
        "in the benchmark config."
    )

def runtime_resolve_path(
    value: str | Path,
    *,
    repo_root: Path | None = None,
    prefer_existing_legacy: bool = True,
) -> Path:
    # Resolve heavy runtime paths below FATHI_RUNTIME_ROOT.
    repository = repo_root or repository_root()
    path = Path(value).expanduser()
    if path.is_absolute():
        return path.resolve()

    runtime_candidate = (runtime_root(repository) / path).resolve()
    legacy_candidate = (repository / path).resolve()

    if (
        prefer_existing_legacy
        and not runtime_candidate.exists()
        and legacy_candidate.exists()
    ):
        return legacy_candidate

    return runtime_candidate


def context_path_value(path: str | Path) -> str:
    # Context files use absolute paths so an external runtime root is unambiguous.
    return str(Path(path).expanduser().resolve())


def iteration_runtime_paths(
    config: Mapping[str, Any],
    iter_k: int,
    *,
    repo_root: Path | None = None,
) -> dict[str, Any]:
    repository = repo_root or repository_root()

    layout = config.get("runtime_layout")
    if not isinstance(layout, Mapping):
        raise RuntimeError(
            "Missing runtime_layout in benchmark config."
        )

    required = [
        "transition_result_pattern",
        "iteration_pattern",
        "initial_parent_workspace",
        "true_observed_workspace",
        "search_direction_subdir",
        "mtilde_subdir",
        "candidates_subdir",
        "candidate_forward_subdir",
        "accepted_subdir",
        "state_pattern",
        "trace_subdir",
    ]

    missing = [
        key
        for key in required
        if not layout.get(key)
    ]

    if missing:
        raise RuntimeError(
            "Missing runtime_layout keys: "
            + ", ".join(missing)
        )

    identity = IterationIdentity.from_parent(
        str(config.get("benchmark_name", "unnamed_run")), int(iter_k)
    )
    k = identity.parent_iteration
    kp1 = identity.child_iteration
    transition = identity.transition_id

    def runtime_template(key: str, *, iteration=None) -> Path:
        value = str(layout[key])

        rendered = value.format(
            **identity.format_values(
                iteration=k if iteration is None else int(iteration)
            )
        )

        return runtime_resolve_path(
            rendered,
            repo_root=repository,
            prefer_existing_legacy=False,
        )

    transition_root = runtime_template(
        "transition_result_pattern"
    )

    iter_k_root = runtime_template(
        "iteration_pattern",
        iteration=k,
    )

    iter_kp1_root = runtime_template(
        "iteration_pattern",
        iteration=kp1,
    )

    if k == 0:
        parent_workspace = runtime_template(
            "initial_parent_workspace"
        )
    else:
        parent_workspace = (
            iter_k_root
            / str(layout["accepted_subdir"])
        ).resolve()

    true_workspace = runtime_template(
        "true_observed_workspace"
    )

    state_out = runtime_template(
        "state_pattern",
        iteration=kp1,
    )

    return {
        "run_id": identity.run_id,
        "parent_iteration": k,
        "child_iteration": kp1,
        "parent_tag": identity.parent_tag,
        "child_tag": identity.child_tag,
        "runtime_root": runtime_root(repository),
        "transition": transition,
        "transition_root": transition_root,
        "iter_k_root": iter_k_root,
        "iter_kp1_root": iter_kp1_root,
        "parent_workspace": parent_workspace,
        "true_workspace": true_workspace,
        "true_trace_dir": (
            true_workspace
            / str(layout["trace_subdir"])
        ).resolve(),
        "search_direction_dir": (
            transition_root
            / str(layout["search_direction_subdir"])
        ).resolve(),
        "mtilde_dir": (
            transition_root
            / str(layout["mtilde_subdir"])
        ).resolve(),
        "candidate_root": (
            transition_root
            / str(layout["candidates_subdir"])
        ).resolve(),
        "candidate_forward_root": (
            iter_kp1_root
            / str(layout["candidate_forward_subdir"])
        ).resolve(),
        "accepted_dir": (
            iter_kp1_root
            / str(layout["accepted_subdir"])
        ).resolve(),
        "state_out": state_out,
        "parent_state": runtime_template(
            "state_pattern",
            iteration=k,
        ),
        "child_state": state_out,
    }
