"""Iteration-generic Armijo contracts shared by external-forward runners.

The functions here perform routing and acceptance logic only.  They do not run
SEM3D or the external physical forward.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Any, Iterator, Mapping

from scripts.fathi_benchmark.iteration_context import IterationPaths
from scripts.fathi_benchmark.current_pipeline_contracts import (
    armijo_ready_result,
    canonical_sha256,
)


@dataclass(frozen=True)
class ArmijoParameters:
    c1: float
    rho: float
    alpha0: float
    maximum_backtracks: int

    def __post_init__(self) -> None:
        if not math.isfinite(self.c1) or not 0.0 < self.c1 < 1.0:
            raise ValueError("c1 must be in (0, 1)")
        if not math.isfinite(self.rho) or not 0.0 < self.rho < 1.0:
            raise ValueError("rho must be in (0, 1)")
        if not math.isfinite(self.alpha0) or self.alpha0 <= 0.0:
            raise ValueError("alpha0 must be finite and positive")
        if int(self.maximum_backtracks) < 0:
            raise ValueError("maximum_backtracks must be non-negative")

    @classmethod
    def from_config(cls, config: Mapping[str, Any]) -> "ArmijoParameters":
        value = config.get("external_armijo")
        if not isinstance(value, Mapping):
            raise ValueError("iteration-engine config requires external_armijo")
        return cls(
            c1=float(value["c1"]),
            rho=float(value["rho"]),
            alpha0=float(value["alpha0"]),
            maximum_backtracks=int(value["maximum_backtracks"]),
        )

    def schedule(self) -> Iterator[tuple[int, float]]:
        for trial_index in range(self.maximum_backtracks + 1):
            yield trial_index, self.alpha0 * self.rho**trial_index


def candidate_label(trial_index: int, alpha: float) -> str:
    """Return a deterministic trial label without encoding an iteration."""

    index = int(trial_index)
    value = float(alpha)
    if index < 0:
        raise ValueError("trial_index must be non-negative")
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError("alpha must be finite and positive")
    token = format(value, ".17g").replace("-", "m").replace(".", "p")
    return f"trial_{index:03d}_alpha_{token}"


def armijo_decision(
    *,
    parent_objective: float,
    candidate_objective: float,
    slope: float,
    alpha: float,
    c1: float,
) -> dict[str, float | bool]:
    """Apply the frozen strict-descent plus Armijo acceptance contract."""

    values = (parent_objective, candidate_objective, slope, alpha, c1)
    if not all(math.isfinite(float(value)) for value in values):
        raise ValueError("Armijo inputs must be finite")
    if parent_objective < 0.0 or candidate_objective < 0.0:
        raise ValueError("objectives must be non-negative")
    if slope >= 0.0:
        raise ValueError("search direction slope must be negative")
    if alpha <= 0.0 or not 0.0 < c1 < 1.0:
        raise ValueError("invalid alpha or c1")
    rhs = float(parent_objective + c1 * alpha * slope)
    strict_descent = bool(candidate_objective < parent_objective)
    armijo = bool(candidate_objective <= rhs)
    return {
        "armijo_rhs": rhs,
        "strict_descent": strict_descent,
        "armijo": armijo,
        "accepted": bool(strict_descent and armijo),
    }


def candidate_namespace(
    paths: IterationPaths, trial_index: int, alpha: float
) -> Path:
    return (paths.candidate_root / candidate_label(trial_index, alpha)).resolve()


def external_armijo_manifest(
    *,
    paths: IterationPaths,
    parent_objective: float,
    slope: float,
    parent_accepted_artifact: Mapping[str, Any],
    gradient_artifact: Mapping[str, Any],
    direction_artifact: Mapping[str, Any],
    true_receiver_artifact: Mapping[str, Any],
    parameters: ArmijoParameters,
) -> dict[str, Any]:
    """Build a portable line-search input contract from context/manifests."""

    for name, value in (
        ("parent_accepted_artifact", parent_accepted_artifact),
        ("gradient_artifact", gradient_artifact),
        ("direction_artifact", direction_artifact),
        ("true_receiver_artifact", true_receiver_artifact),
    ):
        if not value or not value.get("path") or not value.get("sha256"):
            raise ValueError(f"{name} requires path and sha256")
    if not math.isfinite(parent_objective) or parent_objective < 0.0:
        raise ValueError("parent objective must be finite and non-negative")
    if not math.isfinite(slope) or slope >= 0.0:
        raise ValueError("slope must be finite and negative")
    payload = {
        "schema_version": 1,
        "result": armijo_ready_result(paths.identity.parent_iteration),
        "run_id": paths.identity.run_id,
        "parent_iteration": paths.identity.parent_iteration,
        "child_iteration": paths.identity.child_iteration,
        "transition": paths.identity.transition_id,
        "parent_objective": float(parent_objective),
        "slope": float(slope),
        "parent_accepted_artifact": dict(parent_accepted_artifact),
        "gradient_artifact": dict(gradient_artifact),
        "direction_artifact": dict(direction_artifact),
        "true_receiver_artifact": dict(true_receiver_artifact),
        "candidate_formula": "parent + alpha * biased physical direction",
        "normalization": "none",
        "acceptance": (
            "J_candidate < J_parent and "
            "J_candidate <= J_parent + c1*alpha*slope"
        ),
        "parameters": {
            "c1": parameters.c1,
            "rho": parameters.rho,
            "alpha0": parameters.alpha0,
            "maximum_backtracks": parameters.maximum_backtracks,
        },
        "line_search_root": str(paths.line_search_root),
        "candidate_root": str(paths.candidate_root),
        "candidate_labels": [
            candidate_label(index, alpha)
            for index, alpha in parameters.schedule()
        ],
    }
    payload["input_signature_sha256"] = canonical_sha256(payload)
    return payload
