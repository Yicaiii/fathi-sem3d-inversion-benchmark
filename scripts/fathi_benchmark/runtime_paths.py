from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping
import os


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
