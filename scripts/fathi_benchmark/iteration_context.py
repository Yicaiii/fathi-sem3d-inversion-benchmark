"""Stable naming and path contract for one CURRENT benchmark transition.

This module is deliberately free of solver imports.  It can therefore be used
for dry-run validation, orchestration, tests, and post-run promotion without
starting SEM3D or the certified external forward.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Mapping


HISTORICAL_REUSE_MARKER = "HISTORICAL_CERTIFIED_ASSET_REUSE"


def _require_nonempty(value: object, *, name: str) -> str:
    text = str(value).strip()
    if not text:
        raise ValueError(f"{name} must be non-empty")
    return text


def _iteration_index(value: object, *, name: str) -> int:
    index = int(value)
    if index < 0:
        raise ValueError(f"{name} must be non-negative")
    return index


@dataclass(frozen=True)
class IterationIdentity:
    """Canonical names for a consecutive parent/child transition."""

    run_id: str
    parent_iteration: int
    child_iteration: int

    def __post_init__(self) -> None:
        run_id = _require_nonempty(self.run_id, name="run_id")
        parent = _iteration_index(
            self.parent_iteration, name="parent_iteration"
        )
        child = _iteration_index(self.child_iteration, name="child_iteration")
        if child != parent + 1:
            raise ValueError(
                "child_iteration must equal parent_iteration + 1: "
                f"{parent} -> {child}"
            )
        object.__setattr__(self, "run_id", run_id)
        object.__setattr__(self, "parent_iteration", parent)
        object.__setattr__(self, "child_iteration", child)

    @classmethod
    def from_parent(
        cls, run_id: str, parent_iteration: int
    ) -> "IterationIdentity":
        parent = _iteration_index(parent_iteration, name="parent_iteration")
        return cls(run_id, parent, parent + 1)

    @staticmethod
    def iteration_tag(iteration: int) -> str:
        return f"iter_{_iteration_index(iteration, name='iteration'):03d}"

    @property
    def parent_tag(self) -> str:
        return self.iteration_tag(self.parent_iteration)

    @property
    def child_tag(self) -> str:
        return self.iteration_tag(self.child_iteration)

    @property
    def transition_id(self) -> str:
        return f"{self.parent_tag}_to_{self.child_tag}"

    def format_values(self, *, iteration: int | None = None) -> dict[str, Any]:
        selected = self.parent_iteration if iteration is None else int(iteration)
        return {
            "run_id": self.run_id,
            "parent_iteration": self.parent_iteration,
            "child_iteration": self.child_iteration,
            "iter_k": self.parent_iteration,
            "iter_kp1": self.child_iteration,
            "iteration": selected,
            "iteration_tag": self.iteration_tag(selected),
            "parent_tag": self.parent_tag,
            "child_tag": self.child_tag,
            "transition": self.transition_id,
            "transition_id": self.transition_id,
        }


def _resolve_template(
    root: Path,
    template: str,
    identity: IterationIdentity,
    *,
    iteration: int | None = None,
) -> Path:
    rendered = template.format(**identity.format_values(iteration=iteration))
    value = Path(rendered).expanduser()
    return value.resolve() if value.is_absolute() else (root / value).resolve()


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


@dataclass(frozen=True)
class IterationPaths:
    """All mutable CURRENT paths for one transition."""

    identity: IterationIdentity
    repository_root: Path
    runtime_root: Path
    data_run_root: Path
    results_run_root: Path
    parent_iteration_root: Path
    child_iteration_root: Path
    parent_accepted: Path
    child_accepted: Path
    child_working: Path
    transition_root: Path
    parent_state: Path
    child_state: Path
    exact_reverse: Path
    gradient_root: Path
    material_covector: Path
    control_transpose: Path
    mtilde_solve: Path
    optimizer_root: Path
    optimizer_history: Path
    parent_optimizer_metadata_state: Path
    child_optimizer_metadata_state: Path
    line_search_root: Path
    candidate_root: Path

    def validate_current_namespace(self, historical_run_id: str | None) -> None:
        data_paths = (
            self.parent_iteration_root,
            self.child_iteration_root,
            self.parent_accepted,
            self.child_accepted,
            self.child_working,
        )
        result_paths = (
            self.transition_root,
            self.parent_state,
            self.child_state,
            self.exact_reverse,
            self.gradient_root,
            self.material_covector,
            self.control_transpose,
            self.mtilde_solve,
            self.optimizer_root,
            self.optimizer_history,
            self.parent_optimizer_metadata_state,
            self.child_optimizer_metadata_state,
            self.line_search_root,
            self.candidate_root,
        )
        for path in data_paths:
            if not _is_relative_to(path, self.data_run_root):
                raise ValueError(f"mutable data path escapes CURRENT run: {path}")
        for path in result_paths:
            if not _is_relative_to(path, self.results_run_root):
                raise ValueError(f"mutable result path escapes CURRENT run: {path}")
        if historical_run_id:
            historical = _require_nonempty(
                historical_run_id, name="historical_run_id"
            )
            for path in (*data_paths, *result_paths):
                if historical in str(path):
                    raise ValueError(
                        "mutable CURRENT path resolves into historical namespace: "
                        f"{path}"
                    )

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.identity.run_id,
            "parent_iteration": self.identity.parent_iteration,
            "child_iteration": self.identity.child_iteration,
            "parent": self.identity.parent_tag,
            "child": self.identity.child_tag,
            "transition": self.identity.transition_id,
            "paths": {
                name: str(getattr(self, name))
                for name in (
                    "data_run_root",
                    "results_run_root",
                    "parent_iteration_root",
                    "child_iteration_root",
                    "parent_accepted",
                    "child_accepted",
                    "child_working",
                    "transition_root",
                    "parent_state",
                    "child_state",
                    "exact_reverse",
                    "gradient_root",
                    "material_covector",
                    "control_transpose",
                    "mtilde_solve",
                    "optimizer_root",
                    "optimizer_history",
                    "parent_optimizer_metadata_state",
                    "child_optimizer_metadata_state",
                    "line_search_root",
                    "candidate_root",
                )
            },
        }


def load_engine_config(path: str | Path) -> dict[str, Any]:
    source = Path(path).expanduser().resolve()
    value = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"iteration-engine config must be an object: {source}")
    return value


def build_iteration_paths(
    config: Mapping[str, Any],
    parent_iteration: int,
    *,
    child_iteration: int | None = None,
    repository_root: str | Path,
    runtime_root: str | Path | None = None,
) -> IterationPaths:
    """Resolve and validate the canonical CURRENT transition namespace."""

    run_id = _require_nonempty(config.get("run_id"), name="run_id")
    parent = _iteration_index(parent_iteration, name="parent_iteration")
    child = parent + 1 if child_iteration is None else int(child_iteration)
    identity = IterationIdentity(run_id, parent, child)
    repository = Path(repository_root).expanduser().resolve()
    runtime = (
        repository
        if runtime_root is None
        else Path(runtime_root).expanduser().resolve()
    )

    namespace = config.get("namespace")
    routes = config.get("routes")
    if not isinstance(namespace, Mapping) or not isinstance(routes, Mapping):
        raise ValueError("config requires namespace and routes objects")

    required_namespace = (
        "data_run_pattern",
        "results_run_pattern",
        "iteration_pattern",
        "accepted_subdir",
        "state_pattern",
    )
    required_routes = (
        "exact_reverse",
        "gradient_root",
        "material_covector",
        "control_transpose",
        "mtilde_solve",
        "optimizer_root",
        "optimizer_history",
        "optimizer_state_pattern",
        "line_search_root",
        "candidate_subdir",
    )
    missing = [
        key for key in required_namespace if not namespace.get(key)
    ] + [key for key in required_routes if not routes.get(key)]
    if missing:
        raise ValueError("missing iteration-engine config keys: " + ", ".join(missing))

    data_run = _resolve_template(
        runtime, str(namespace["data_run_pattern"]), identity
    )
    results_run = _resolve_template(
        runtime, str(namespace["results_run_pattern"]), identity
    )

    def under(base: Path, value: object) -> Path:
        candidate = Path(str(value))
        if candidate.is_absolute():
            raise ValueError(f"route fragments must be relative: {candidate}")
        return (base / candidate).resolve()

    parent_root = under(
        data_run,
        str(namespace["iteration_pattern"]).format(
            **identity.format_values(iteration=parent)
        ),
    )
    child_root = under(
        data_run,
        str(namespace["iteration_pattern"]).format(
            **identity.format_values(iteration=child)
        ),
    )
    transition_root = under(
        results_run,
        str(namespace.get("transition_pattern", "{transition_id}")).format(
            **identity.format_values()
        ),
    )
    parent_state = under(
        results_run,
        str(namespace["state_pattern"]).format(
            **identity.format_values(iteration=parent)
        ),
    )
    child_state = under(
        results_run,
        str(namespace["state_pattern"]).format(
            **identity.format_values(iteration=child)
        ),
    )
    accepted_subdir = str(namespace["accepted_subdir"])
    gradient_root = under(transition_root, routes["gradient_root"])
    line_search_root = under(transition_root, routes["line_search_root"])
    optimizer_history = under(results_run, routes["optimizer_history"])
    parent_optimizer_metadata_state = under(
        optimizer_history,
        str(routes["optimizer_state_pattern"]).format(
            **identity.format_values(iteration=parent)
        ),
    )
    child_optimizer_metadata_state = under(
        optimizer_history,
        str(routes["optimizer_state_pattern"]).format(
            **identity.format_values(iteration=child)
        ),
    )

    paths = IterationPaths(
        identity=identity,
        repository_root=repository,
        runtime_root=runtime,
        data_run_root=data_run,
        results_run_root=results_run,
        parent_iteration_root=parent_root,
        child_iteration_root=child_root,
        parent_accepted=under(parent_root, accepted_subdir),
        child_accepted=under(child_root, accepted_subdir),
        child_working=child_root,
        transition_root=transition_root,
        parent_state=parent_state,
        child_state=child_state,
        exact_reverse=under(transition_root, routes["exact_reverse"]),
        gradient_root=gradient_root,
        material_covector=under(gradient_root, routes["material_covector"]),
        control_transpose=under(gradient_root, routes["control_transpose"]),
        mtilde_solve=under(gradient_root, routes["mtilde_solve"]),
        optimizer_root=under(transition_root, routes["optimizer_root"]),
        optimizer_history=optimizer_history,
        parent_optimizer_metadata_state=parent_optimizer_metadata_state,
        child_optimizer_metadata_state=child_optimizer_metadata_state,
        line_search_root=line_search_root,
        candidate_root=under(line_search_root, routes["candidate_subdir"]),
    )
    for numerical, metadata in (
        (paths.parent_state, paths.parent_optimizer_metadata_state),
        (paths.child_state, paths.child_optimizer_metadata_state),
    ):
        if numerical == metadata:
            raise ValueError("numerical state and optimizer metadata paths collide")
        if numerical.suffix != ".npz" or metadata.suffix != ".json":
            raise ValueError("state extensions violate NPZ/JSON separation contract")
    paths.validate_current_namespace(config.get("historical_run_id"))
    return paths


def validate_historical_asset_reference(reference: Mapping[str, Any]) -> None:
    """Require an explicit marker before an immutable historical asset is used."""

    if reference.get("classification") != HISTORICAL_REUSE_MARKER:
        raise ValueError(
            "historical asset reference lacks " + HISTORICAL_REUSE_MARKER
        )
    if reference.get("mutable", True) is not False:
        raise ValueError("historical certified asset must be immutable")
    if not reference.get("sha256"):
        raise ValueError("historical certified asset requires a SHA-256 digest")


def main() -> None:
    parser = argparse.ArgumentParser(description="Dry-run CURRENT iteration paths")
    parser.add_argument("--config", required=True)
    parser.add_argument("--repo", required=True)
    parser.add_argument("--runtime-root")
    parser.add_argument("--parent", type=int, required=True)
    parser.add_argument("--child", type=int)
    args = parser.parse_args()
    config = load_engine_config(args.config)
    paths = build_iteration_paths(
        config,
        args.parent,
        child_iteration=args.child,
        repository_root=args.repo,
        runtime_root=args.runtime_root,
    )
    print(json.dumps(paths.to_dict(), indent=2))


if __name__ == "__main__":
    main()
