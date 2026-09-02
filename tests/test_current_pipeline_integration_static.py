from __future__ import annotations

import ast
import inspect
import json
from pathlib import Path
import unittest

from scripts.fathi_benchmark import bridge_certified_external_gradient as bridge
from scripts.fathi_benchmark import external_armijo
from scripts.fathi_benchmark import generic_iteration_runner
from scripts.fathi_benchmark import lbfgs_history
from scripts.fathi_benchmark import physical_space_optimizer
from scripts.fathi_benchmark import run_exact_reverse_gradient_generic as reverse
from scripts.fathi_benchmark.iteration_context import build_iteration_paths


REPOSITORY = Path(__file__).resolve().parents[1]
CURRENT_CORE = (
    "runtime_paths.py",
    "iteration_context.py",
    "generic_iteration_runner.py",
    "run_exact_reverse_gradient_generic.py",
    "run_certified_external_exact_reverse.py",
    "bridge_certified_external_gradient.py",
    "certified_gradient_bridge_utils.py",
    "optimizer_state.py",
    "lbfgs_history.py",
    "physical_space_optimizer.py",
    "external_armijo.py",
    "current_pipeline_contracts.py",
    "current_pipeline_artifacts.py",
    "register_certified_gradient.py",
    "run_current_pipeline.py",
    "immutable_assets.py",
    "path_consistency.py",
)
PROHIBITED_CURRENT_DEPENDENCIES = (
    "bridge_stage5o_certified_gradient",
    "run_current_t052_",
    "finalize_current_t052_",
    "424B_compute_rhs_component_from_traces",
    "compute_search_direction",
    "prepare_gpu_adjoint_full",
    "run_gpu_adjoint_task",
    "solve_gpu_mtilde_gradient",
)


def _source(name: str) -> str:
    return (REPOSITORY / "scripts" / "fathi_benchmark" / name).read_text(
        encoding="utf-8"
    )


class CurrentPipelineStaticIntegrationTest(unittest.TestCase):
    def test_current_core_dependency_closure_excludes_historical_routes(self):
        for name in CURRENT_CORE:
            source = _source(name)
            with self.subTest(module=name):
                for dependency in PROHIBITED_CURRENT_DEPENDENCIES:
                    self.assertNotIn(dependency, source)

    def test_cpu_only_current_contract_modules_have_no_subprocess_launch(self):
        cpu_only = (
            "iteration_context.py",
            "generic_iteration_runner.py",
            "bridge_certified_external_gradient.py",
            "certified_gradient_bridge_utils.py",
            "optimizer_state.py",
            "lbfgs_history.py",
            "physical_space_optimizer.py",
            "external_armijo.py",
            "current_pipeline_contracts.py",
            "current_pipeline_artifacts.py",
            "register_certified_gradient.py",
            "run_current_pipeline.py",
        )
        for name in cpu_only:
            tree = ast.parse(_source(name))
            imported = {
                alias.name
                for node in ast.walk(tree)
                if isinstance(node, ast.Import)
                for alias in node.names
            }
            with self.subTest(module=name):
                self.assertNotIn("subprocess", imported)
                self.assertNotIn("mpi4py", imported)

    def test_future_iteration_paths_are_dynamic_for_required_parents(self):
        engine = json.loads(
            (
                REPOSITORY
                / "configs"
                / "fathi_s43_repro_p20_t052_iteration_engine.json"
            ).read_text(encoding="utf-8")
        )
        for parent in (1, 2, 7):
            paths = build_iteration_paths(
                engine,
                parent,
                repository_root=REPOSITORY,
                runtime_root=REPOSITORY,
            )
            with self.subTest(parent=parent):
                self.assertEqual(paths.identity.parent_iteration, parent)
                self.assertEqual(paths.identity.child_iteration, parent + 1)
                self.assertEqual(
                    paths.identity.transition_id,
                    f"iter_{parent:03d}_to_iter_{parent + 1:03d}",
                )
                self.assertIn(paths.identity.transition_id, str(paths.gradient_root))
                self.assertIn(paths.identity.transition_id, str(paths.candidate_root))

    def test_reverse_to_bridge_contract_is_exact(self):
        for iteration in (1, 2, 7):
            self.assertEqual(
                bridge.current_reverse_result(iteration),
                reverse._reverse_result(iteration),
            )
        self.assertEqual(bridge.CURRENT_GRADIENT_NAMES, reverse.GRADIENT_NAMES)

    def test_physical_metric_and_eq25_contracts_remain_distinct(self):
        optimizer_source = inspect.getsource(physical_space_optimizer)
        runner_source = inspect.getsource(
            generic_iteration_runner.GenericIterationRunner.compute_optimizer_direction
        )
        self.assertIn("a @ (mtilde @ b)", optimizer_source)
        self.assertIn("np.linalg.norm(lam)", optimizer_source)
        self.assertIn("np.linalg.norm(mu)", optimizer_source)
        self.assertIn("parent_iteration", runner_source)
        self.assertIn("apply_lambda_bias_euclidean", runner_source)
        self.assertNotIn("joint_maxabs", runner_source)
        self.assertNotIn("m_ref_pa", runner_source)

    def test_missing_iter001_gradient_remains_a_blocking_state(self):
        status = lbfgs_history.history_build_status(
            {"child_iteration": 1, "child_gradient": None}, REPOSITORY
        )
        self.assertEqual(
            status["status"], "BLOCKED_WAITING_FOR_ITER001_GRADIENT"
        )
        self.assertFalse(status["history_pair_created"])
        self.assertEqual(external_armijo.ArmijoParameters(1e-4, 0.5, 1.0, 12).c1, 1e-4)


if __name__ == "__main__":
    unittest.main()
