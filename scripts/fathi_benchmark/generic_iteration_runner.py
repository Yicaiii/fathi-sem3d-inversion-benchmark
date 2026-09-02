"""Small generic orchestration layer for a future CURRENT transition.

This module wires validated manifests and dynamic paths to the existing
certified optimizer and Armijo primitives.  It does not implement or launch a
forward, reverse, gradient, or line-search simulation.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from scripts.fathi_benchmark.external_armijo import (
    ArmijoParameters,
    external_armijo_manifest,
)
from scripts.fathi_benchmark.immutable_assets import (
    validate_immutable_asset_manifest,
)
from scripts.fathi_benchmark.iteration_context import (
    IterationPaths,
    build_iteration_paths,
)
from scripts.fathi_benchmark.lbfgs_history import (
    BLOCKED_WAITING_FOR_HISTORY_AUDIT,
    HISTORY_OUTCOME_ACCEPTED,
    HISTORY_OUTCOME_REJECTED,
    HistoryBuildBlocked,
    HistoryBuildResult,
    NO_HISTORY_REQUIRED,
    build_history_pair,
    history_build_status,
    load_curvature_outcome,
    load_gradient_artifact,
    load_persisted_history,
    persist_accepted_history_pair,
    persist_curvature_outcome,
    require_newest_curvature_outcome,
)
from scripts.fathi_benchmark.optimizer_state import scaling_from_config
from scripts.fathi_benchmark.path_consistency import (
    validate_path_config_consistency,
)
from scripts.fathi_benchmark.runtime_paths import runtime_root
from scripts.fathi_benchmark.physical_space_optimizer import (
    CurvatureAudit,
    VectorPair,
    apply_lambda_bias_euclidean,
    joint_mtilde_inner,
    lambda_bias_weight,
    physical_lbfgs_direction,
)
from scripts.fathi_benchmark.current_pipeline_contracts import (
    verify_artifact_record,
)
from scripts.fathi_benchmark.current_pipeline_artifacts import (
    persist_optimizer_direction,
)


def _read_json(path: str | Path) -> dict[str, Any]:
    source = Path(path).expanduser().resolve()
    value = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON manifest must be an object: {source}")
    return value


def _check_identity(
    manifest: Mapping[str, Any], paths: IterationPaths, *, name: str
) -> None:
    expected = paths.identity
    actual = (
        str(manifest.get("run_id", "")),
        int(manifest.get("parent_iteration", -1)),
        int(manifest.get("child_iteration", -1)),
        str(manifest.get("transition", "")),
    )
    wanted = (
        expected.run_id,
        expected.parent_iteration,
        expected.child_iteration,
        expected.transition_id,
    )
    if actual != wanted:
        raise ValueError(f"{name} transition identity mismatch: {actual} != {wanted}")


@dataclass(frozen=True)
class OptimizerDirectionResult:
    raw_direction: VectorPair
    biased_direction: VectorPair
    history_audits: Sequence[CurvatureAudit]
    h0_or_history_scale: float
    lambda_bias_weight: float
    slope: float


class GenericIterationRunner:
    """Manifest-driven orchestration without solver side effects."""

    def __init__(
        self,
        *,
        run_id: str,
        parent_iteration: int,
        child_iteration: int,
        repository_root: str | Path,
        runtime_config: Mapping[str, Any],
        engine_config: Mapping[str, Any],
        verify_historical_asset_bytes: bool = True,
    ) -> None:
        if str(engine_config.get("run_id")) != str(run_id):
            raise ValueError("requested run_id differs from iteration-engine config")
        if str(runtime_config.get("benchmark_name")) != str(run_id):
            raise ValueError("requested run_id differs from runtime config")
        self.repository_root = Path(repository_root).expanduser().resolve()
        self.runtime_root = runtime_root(self.repository_root)
        self.runtime_config = dict(runtime_config)
        self.engine_config = dict(engine_config)
        self.paths = build_iteration_paths(
            engine_config,
            int(parent_iteration),
            child_iteration=int(child_iteration),
            repository_root=self.repository_root,
            runtime_root=self.runtime_root,
        )
        self.path_consistency = validate_path_config_consistency(
            runtime_config,
            engine_config,
            repository_root=self.repository_root,
        )
        immutable = engine_config.get("immutable_operator_assets")
        if not isinstance(immutable, Mapping) or not immutable.get("manifest"):
            raise ValueError("iteration-engine config lacks immutable asset manifest")
        manifest_path = Path(str(immutable["manifest"]))
        if not manifest_path.is_absolute():
            manifest_path = self.repository_root / manifest_path
        self.immutable_assets = validate_immutable_asset_manifest(
            manifest_path,
            expected_source_run=str(engine_config["historical_run_id"]),
            repository_root=self.repository_root,
            runtime_root=self.runtime_root,
            verify_bytes=verify_historical_asset_bytes,
        )

    @classmethod
    def from_config_files(
        cls,
        *,
        run_id: str,
        parent_iteration: int,
        child_iteration: int,
        repository_root: str | Path,
        runtime_config_path: str | Path,
        engine_config_path: str | Path,
        verify_historical_asset_bytes: bool = True,
    ) -> "GenericIterationRunner":
        return cls(
            run_id=run_id,
            parent_iteration=parent_iteration,
            child_iteration=child_iteration,
            repository_root=repository_root,
            runtime_config=_read_json(runtime_config_path),
            engine_config=_read_json(engine_config_path),
            verify_historical_asset_bytes=verify_historical_asset_bytes,
        )

    def compute_optimizer_direction(
        self,
        optimizer_manifest: Mapping[str, Any],
        history_results: Sequence[HistoryBuildResult] | None = None,
    ) -> OptimizerDirectionResult:
        """Call the existing physical Mtilde L-BFGS and Eq.25 primitives."""

        _check_identity(optimizer_manifest, self.paths, name="optimizer manifest")
        parent = self.paths.identity.parent_iteration
        gradient_record = optimizer_manifest.get("registered_gradient_manifest")
        if isinstance(gradient_record, Mapping):
            gradient_path = verify_artifact_record(
                self.repository_root,
                gradient_record,
                label="registered gradient manifest",
                expected_path=self.paths.gradient_root / "registered_gradient.json",
            )
            gradient_spec = _read_json(gradient_path)
        elif (
            optimizer_manifest.get("contract_classification")
            == "SYNTHETIC_TEST_FIXTURE"
        ):
            gradient_spec = optimizer_manifest.get("gradient")
        else:
            gradient_spec = None
        if parent > 0:
            gradient_status = history_build_status(
                {
                    "child_iteration": parent,
                    "child_gradient": gradient_spec,
                },
                self.repository_root,
            )
            if gradient_status["status"] != "READY_TO_BUILD_HISTORY":
                raise HistoryBuildBlocked(
                    str(gradient_status["status"]),
                    str(gradient_status["reason"]),
                )
        gradient, _, mtilde, _ = load_gradient_artifact(
            self.repository_root,
            gradient_spec,
            name="current_corrected_gradient",
        )
        optimizer = self.engine_config["optimizer"]
        memory = int(optimizer["memory_limit"])
        newest_outcome = None
        if parent == 0:
            # There is no predecessor pair at the first parent. The certified
            # H0 route is therefore the only legitimate history state.
            history = []
        else:
            # In-memory values cannot bypass the durable outcome gate. An
            # absent newest outcome is distinct from an audited rejection.
            newest_outcome = require_newest_curvature_outcome(
                self.paths.optimizer_history,
                parent_iteration=parent,
                expected_active_indices_sha256=str(
                    gradient_spec["active_indices"]["sha256"]
                ),
                expected_coordinates_sha256=str(
                    gradient_spec["coordinates"]["sha256"]
                ),
                expected_mtilde_sha256=str(gradient_spec["mtilde"]["sha256"]),
            )
            history = load_persisted_history(
                self.paths.optimizer_history,
                parent_iteration=parent,
                memory_limit=memory,
                expected_active_indices_sha256=str(
                    gradient_spec["active_indices"]["sha256"]
                ),
                expected_coordinates_sha256=str(
                    gradient_spec["coordinates"]["sha256"]
                ),
                expected_mtilde_sha256=str(gradient_spec["mtilde"]["sha256"]),
            )
        scaling = scaling_from_config(self.engine_config)
        raw, audits, scale = physical_lbfgs_direction(
            gradient,
            history,
            mtilde,
            gamma0=scaling.gamma0,
            memory=memory,
            curvature_relative_tolerance=float(
                optimizer["curvature_relative_tolerance"]
            ),
        )
        if (
            newest_outcome is not None
            and newest_outcome["status"] == HISTORY_OUTCOME_ACCEPTED
            and (not audits or not audits[-1].accepted)
        ):
            raise ValueError(
                "persisted newest ACCEPTED curvature pair was not admitted by L-BFGS"
            )
        bias_weight = lambda_bias_weight(self.paths.identity.parent_iteration)
        biased = apply_lambda_bias_euclidean(raw, weight=bias_weight)
        slope = joint_mtilde_inner(gradient, biased, mtilde)
        if not np.isfinite(slope) or slope >= 0.0:
            raise ValueError("generic physical direction is not Mtilde descent")
        return OptimizerDirectionResult(
            raw_direction=raw,
            biased_direction=biased,
            history_audits=audits,
            h0_or_history_scale=scale,
            lambda_bias_weight=bias_weight,
            slope=float(slope),
        )

    def persist_optimizer_direction(
        self,
        optimizer_manifest: Mapping[str, Any],
        result: OptimizerDirectionResult,
    ) -> Path:
        """Persist the existing physical L-BFGS/Eq.25 output without recomputing."""

        return persist_optimizer_direction(
            repo=self.repository_root,
            paths=self.paths,
            material_config=self.engine_config["material"],
            optimizer_manifest=optimizer_manifest,
            direction_result=result,
        )

    def parent_lambda_bias_weight(self) -> float:
        """Return paper Eq.25's weight for the dynamic parent iteration."""

        return lambda_bias_weight(self.paths.identity.parent_iteration)

    def build_real_history_pair(
        self, history_request: Mapping[str, Any]
    ) -> HistoryBuildResult:
        """Wire real accepted model/gradient artifacts to the history builder."""

        return build_history_pair(
            history_request,
            repo=self.repository_root,
            material_config=self.engine_config["material"],
            curvature_relative_tolerance=float(
                self.engine_config["optimizer"]["curvature_relative_tolerance"]
            ),
        )

    def checkpoint_accepted_history_pair(
        self, result: HistoryBuildResult
    ) -> Path:
        return persist_accepted_history_pair(
            result, history_root=self.paths.optimizer_history
        )

    def checkpoint_history_outcome(self, result: HistoryBuildResult) -> Path:
        """Durably record either an accepted or rejected completed audit."""

        return persist_curvature_outcome(
            result, history_root=self.paths.optimizer_history
        )

    def restore_optimizer_history(
        self, gradient_artifact: Mapping[str, Any]
    ) -> list[tuple[VectorPair, VectorPair]]:
        return load_persisted_history(
            self.paths.optimizer_history,
            parent_iteration=self.paths.identity.parent_iteration,
            memory_limit=int(self.engine_config["optimizer"]["memory_limit"]),
            expected_active_indices_sha256=str(
                gradient_artifact["active_indices"]["sha256"]
            ),
            expected_coordinates_sha256=str(
                gradient_artifact["coordinates"]["sha256"]
            ),
            expected_mtilde_sha256=str(gradient_artifact["mtilde"]["sha256"]),
        )

    def history_preflight(
        self, history_request: Mapping[str, Any]
    ) -> dict[str, Any]:
        return history_build_status(history_request, self.repository_root)

    def newest_history_preflight(
        self, current_parent_gradient: Mapping[str, Any] | None
    ) -> dict[str, Any]:
        """Resolve the newest history dependency without negative iterations."""

        parent = self.paths.identity.parent_iteration
        if parent == 0:
            return {
                "status": NO_HISTORY_REQUIRED,
                "history_pair_created": False,
                "requested_pair": None,
            }
        status = history_build_status(
            {
                "parent_iteration": parent - 1,
                "child_iteration": parent,
                "child_gradient": current_parent_gradient,
            },
            self.repository_root,
        )
        status["requested_pair"] = {
            "from_iteration": parent - 1,
            "to_iteration": parent,
            "s": (
                f"accepted_model_{parent} - accepted_model_{parent - 1}"
            ),
            "y": (
                f"corrected_gradient_{parent} - corrected_gradient_{parent - 1}"
            ),
        }
        if status["status"] != "READY_TO_BUILD_HISTORY":
            return status
        gradient = current_parent_gradient
        outcome = load_curvature_outcome(
            self.paths.optimizer_history,
            from_iteration=parent - 1,
            to_iteration=parent,
            expected_active_indices_sha256=str(
                gradient["active_indices"]["sha256"]
            ),
            expected_coordinates_sha256=str(
                gradient["coordinates"]["sha256"]
            ),
            expected_mtilde_sha256=str(gradient["mtilde"]["sha256"]),
        )
        if outcome is None:
            status.update(
                {
                    "status": BLOCKED_WAITING_FOR_HISTORY_AUDIT,
                    "reason": "current gradient exists but newest curvature outcome is absent",
                    "history_pair_created": False,
                }
            )
            return status
        status.update(
            {
                "status": str(outcome["status"]),
                "history_pair_created": (
                    outcome["status"] == HISTORY_OUTCOME_ACCEPTED
                ),
                "curvature_outcome": {
                    "status": outcome["status"],
                    "sMy": outcome["sMy"],
                    "curvature_threshold": outcome["curvature_threshold"],
                    "curvature_reason": outcome["curvature_reason"],
                },
                "history_outcome_validated": True,
            }
        )
        return status

    def optimization_structurally_runnable(
        self, history_status: Mapping[str, Any]
    ) -> bool:
        path_ready = (
            self.path_consistency.get("result")
            == "PASS_CURRENT_PATH_CONFIG_CONSISTENCY"
        )
        assets_ready = self.immutable_assets.get("result") == (
            "PASS_PORTABLE_HISTORICAL_IMMUTABLE_OPERATOR_ASSET_MANIFEST"
        )
        status = history_status.get("status")
        history_ready = status == NO_HISTORY_REQUIRED or (
            status in {HISTORY_OUTCOME_ACCEPTED, HISTORY_OUTCOME_REJECTED}
            and history_status.get("history_outcome_validated") is True
        )
        return bool(path_ready and assets_ready and history_ready)

    def prepare_external_armijo(
        self, line_search_inputs: Mapping[str, Any]
    ) -> dict[str, Any]:
        """Build the dynamic external Armijo contract from input manifests."""

        _check_identity(line_search_inputs, self.paths, name="Armijo inputs")
        return external_armijo_manifest(
            paths=self.paths,
            parent_objective=float(line_search_inputs["parent_objective"]),
            slope=float(line_search_inputs["slope"]),
            parent_accepted_artifact=line_search_inputs[
                "parent_accepted_artifact"
            ],
            gradient_artifact=line_search_inputs["gradient_artifact"],
            direction_artifact=line_search_inputs["direction_artifact"],
            true_receiver_artifact=line_search_inputs["true_receiver_artifact"],
            parameters=ArmijoParameters.from_config(self.engine_config),
        )

    def dry_run_summary(self) -> dict[str, Any]:
        return {
            "result": "PASS_GENERIC_ITERATION_ORCHESTRATION_DRY_RUN",
            "run_id": self.paths.identity.run_id,
            "parent_iteration": self.paths.identity.parent_iteration,
            "child_iteration": self.paths.identity.child_iteration,
            "transition": self.paths.identity.transition_id,
            "paths": self.paths.to_dict()["paths"],
            "path_config_consistency": self.path_consistency,
            "immutable_asset_manifest": {
                "result": self.immutable_assets["result"],
                "classification": self.immutable_assets["classification"],
                "asset_ids": [
                    item["asset_id"] for item in self.immutable_assets["assets"]
                ],
            },
            "simulation_runs": 0,
            "sem3d_runs": 0,
        }
