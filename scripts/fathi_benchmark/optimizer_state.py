"""Serializable, iteration-generic physical optimizer state contracts."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence


@dataclass(frozen=True)
class FixedReproductionScaling:
    m_ref_pa: float
    J_ref: float
    J_ref_iteration: int
    provenance: str

    def __post_init__(self) -> None:
        if not math.isfinite(self.m_ref_pa) or self.m_ref_pa <= 0.0:
            raise ValueError("m_ref_pa must be finite and positive")
        if not math.isfinite(self.J_ref) or self.J_ref <= 0.0:
            raise ValueError("J_ref must be finite and positive")
        if int(self.J_ref_iteration) < 0:
            raise ValueError("J_ref_iteration must be non-negative")
        if not str(self.provenance).strip():
            raise ValueError("fixed scaling provenance is required")

    @property
    def gamma0(self) -> float:
        return float(self.m_ref_pa**2 / self.J_ref)

    def to_dict(self) -> dict[str, Any]:
        return {
            "m_ref_pa": self.m_ref_pa,
            "J_ref": self.J_ref,
            "J_ref_iteration": int(self.J_ref_iteration),
            "J_ref_policy": "frozen; never renormalize from a later objective",
            "gamma0_formula": "m_ref_pa**2 / J_ref",
            "gamma0": self.gamma0,
            "H0": "gamma0 * I",
            "provenance": self.provenance,
        }


def scaling_from_config(config: Mapping[str, Any]) -> FixedReproductionScaling:
    optimizer = config.get("optimizer")
    if not isinstance(optimizer, Mapping):
        raise ValueError("iteration-engine config requires optimizer metadata")
    scaling = optimizer.get("fixed_reproduction_scaling")
    if not isinstance(scaling, Mapping):
        raise ValueError("optimizer requires fixed_reproduction_scaling")
    return FixedReproductionScaling(
        m_ref_pa=float(scaling["m_ref_pa"]),
        J_ref=float(scaling["J_ref"]),
        J_ref_iteration=int(scaling["J_ref_iteration"]),
        provenance=str(scaling["provenance"]),
    )


@dataclass(frozen=True)
class CurvatureArtifact:
    """Immutable references to one accepted physical-coordinate s/y pair."""

    from_iteration: int
    to_iteration: int
    s_lambda: str
    s_mu: str
    y_lambda: str
    y_mu: str
    sMy: float
    accepted: bool
    provenance: str

    def __post_init__(self) -> None:
        if int(self.to_iteration) != int(self.from_iteration) + 1:
            raise ValueError("curvature artifact must connect consecutive iterations")
        for name in ("s_lambda", "s_mu", "y_lambda", "y_mu", "provenance"):
            if not str(getattr(self, name)).strip():
                raise ValueError(f"{name} is required")
        if not math.isfinite(float(self.sMy)):
            raise ValueError("sMy must be finite")
        if self.accepted and self.sMy <= 0.0:
            raise ValueError("accepted curvature artifact requires positive sMy")

    def to_dict(self) -> dict[str, Any]:
        return {
            "from_iteration": int(self.from_iteration),
            "to_iteration": int(self.to_iteration),
            "s_lambda": self.s_lambda,
            "s_mu": self.s_mu,
            "y_lambda": self.y_lambda,
            "y_mu": self.y_mu,
            "sMy": float(self.sMy),
            "accepted": bool(self.accepted),
            "provenance": self.provenance,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "CurvatureArtifact":
        return cls(
            from_iteration=int(value["from_iteration"]),
            to_iteration=int(value["to_iteration"]),
            s_lambda=str(value["s_lambda"]),
            s_mu=str(value["s_mu"]),
            y_lambda=str(value["y_lambda"]),
            y_mu=str(value["y_mu"]),
            sMy=float(value["sMy"]),
            accepted=bool(value["accepted"]),
            provenance=str(value["provenance"]),
        )


@dataclass(frozen=True)
class OptimizerIterationState:
    run_id: str
    iteration: int
    accepted_objective: float
    accepted_model_provenance: Mapping[str, Any]
    gradient_provenance: Mapping[str, Any]
    fixed_scaling: FixedReproductionScaling
    lambda_bias_iteration: int
    accepted_alpha: float
    memory_limit: int = 15
    history: Sequence[CurvatureArtifact] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        if not str(self.run_id).strip():
            raise ValueError("run_id is required")
        if int(self.iteration) < 0:
            raise ValueError("iteration must be non-negative")
        if not math.isfinite(self.accepted_objective) or self.accepted_objective < 0.0:
            raise ValueError("accepted_objective must be finite and non-negative")
        if not self.accepted_model_provenance:
            raise ValueError("accepted model provenance is required")
        if not self.gradient_provenance:
            raise ValueError("gradient provenance is required")
        if int(self.lambda_bias_iteration) != int(self.iteration):
            raise ValueError("lambda bias index must equal the state iteration")
        if not math.isfinite(self.accepted_alpha) or self.accepted_alpha <= 0.0:
            raise ValueError("accepted_alpha must be finite and positive")
        if int(self.memory_limit) <= 0:
            raise ValueError("memory_limit must be positive")
        if len(self.history) > int(self.memory_limit):
            raise ValueError("history exceeds the configured memory limit")

    def accepted_history(self) -> tuple[CurvatureArtifact, ...]:
        """Return only admissible pairs, newest ``memory_limit`` entries."""

        values = tuple(item for item in self.history if item.accepted)
        return values[-int(self.memory_limit) :]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "run_id": self.run_id,
            "iteration": int(self.iteration),
            "accepted_objective": float(self.accepted_objective),
            "accepted_model_provenance": dict(self.accepted_model_provenance),
            "gradient_provenance": dict(self.gradient_provenance),
            "lbfgs": {
                "memory_limit": int(self.memory_limit),
                "metric": "Mtilde",
                "metric_inner_product": "a.T @ Mtilde @ b",
                "history": [item.to_dict() for item in self.history],
                "accepted_history_count": len(self.accepted_history()),
            },
            "fixed_reproduction_scaling": self.fixed_scaling.to_dict(),
            "lambda_bias": {
                "iteration": int(self.lambda_bias_iteration),
                "norm": "paper Eq.25 Euclidean L2",
            },
            "accepted_alpha": float(self.accepted_alpha),
        }

    def write(self, path: str | Path) -> None:
        destination = self._metadata_json_path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_name(destination.name + ".tmp")
        temporary.write_text(
            json.dumps(self.to_dict(), indent=2) + "\n", encoding="utf-8"
        )
        temporary.replace(destination)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "OptimizerIterationState":
        lbfgs = value["lbfgs"]
        scaling = value["fixed_reproduction_scaling"]
        bias = value["lambda_bias"]
        return cls(
            run_id=str(value["run_id"]),
            iteration=int(value["iteration"]),
            accepted_objective=float(value["accepted_objective"]),
            accepted_model_provenance=dict(value["accepted_model_provenance"]),
            gradient_provenance=dict(value["gradient_provenance"]),
            fixed_scaling=FixedReproductionScaling(
                m_ref_pa=float(scaling["m_ref_pa"]),
                J_ref=float(scaling["J_ref"]),
                J_ref_iteration=int(scaling["J_ref_iteration"]),
                provenance=str(scaling["provenance"]),
            ),
            lambda_bias_iteration=int(bias["iteration"]),
            accepted_alpha=float(value["accepted_alpha"]),
            memory_limit=int(lbfgs["memory_limit"]),
            history=tuple(
                CurvatureArtifact.from_dict(item)
                for item in lbfgs.get("history", [])
            ),
        )

    @classmethod
    def read(cls, path: str | Path) -> "OptimizerIterationState":
        source = cls._metadata_json_path(path)
        return cls.from_dict(json.loads(source.read_text(encoding="utf-8")))

    @staticmethod
    def _metadata_json_path(path: str | Path) -> Path:
        value = Path(path).expanduser().resolve()
        if value.suffix.lower() != ".json":
            raise ValueError(
                "OptimizerIterationState requires a separate .json metadata path"
            )
        return value
