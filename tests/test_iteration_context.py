import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from scripts.fathi_benchmark.external_armijo import (
    armijo_decision,
    candidate_label,
    candidate_namespace,
)
from scripts.fathi_benchmark.iteration_context import (
    IterationIdentity,
    build_iteration_paths,
)
from scripts.fathi_benchmark.optimizer_state import (
    CurvatureArtifact,
    OptimizerIterationState,
    scaling_from_config,
)
from scripts.fathi_benchmark.physical_space_optimizer import (
    physical_curvature_pair,
)


class IterationContextTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.repo = Path(__file__).resolve().parents[1]
        cls.config_path = (
            cls.repo
            / "configs"
            / "fathi_s43_repro_p20_t052_iteration_engine.json"
        )
        cls.config = json.loads(cls.config_path.read_text(encoding="utf-8"))

    def resolve(self, parent, child):
        return build_iteration_paths(
            self.config,
            parent,
            child_iteration=child,
            repository_root=self.repo,
        )

    def test_required_transition_names(self):
        expected = (
            (0, 1, "iter_000", "iter_001", "iter_000_to_iter_001"),
            (1, 2, "iter_001", "iter_002", "iter_001_to_iter_002"),
            (9, 10, "iter_009", "iter_010", "iter_009_to_iter_010"),
        )
        for parent, child, parent_tag, child_tag, transition in expected:
            with self.subTest(parent=parent, child=child):
                paths = self.resolve(parent, child)
                self.assertEqual(paths.identity.parent_tag, parent_tag)
                self.assertEqual(paths.identity.child_tag, child_tag)
                self.assertEqual(paths.identity.transition_id, transition)
                self.assertEqual(paths.parent_iteration_root.name, parent_tag)
                self.assertEqual(paths.child_iteration_root.name, child_tag)
                self.assertEqual(paths.transition_root.name, transition)
                self.assertEqual(paths.parent_state.name, f"{parent_tag}_state.npz")
                self.assertEqual(paths.child_state.name, f"{child_tag}_state.npz")
                self.assertEqual(
                    paths.parent_optimizer_metadata_state.name,
                    f"{parent_tag}_optimizer_state.json",
                )
                self.assertEqual(
                    paths.child_optimizer_metadata_state.name,
                    f"{child_tag}_optimizer_state.json",
                )
                self.assertNotEqual(
                    paths.parent_state, paths.parent_optimizer_metadata_state
                )

    def test_mandatory_parent_one_dry_run_is_current_only(self):
        paths = self.resolve(1, 2)
        payload = json.dumps(paths.to_dict(), sort_keys=True)
        self.assertIn("fathi_s43_repro_p20_t052", payload)
        self.assertNotIn(self.config["historical_run_id"], payload)
        self.assertEqual(paths.gradient_root.parent, paths.transition_root)
        self.assertEqual(paths.candidate_root.parent, paths.line_search_root)

    def test_nonconsecutive_child_is_rejected(self):
        with self.assertRaises(ValueError):
            IterationIdentity("run", 1, 3)

    def test_candidate_namespace_and_acceptance_are_iteration_generic(self):
        paths = self.resolve(9, 10)
        self.assertEqual(candidate_label(0, 1.0), "trial_000_alpha_1")
        self.assertEqual(
            candidate_namespace(paths, 2, 0.25),
            paths.candidate_root / "trial_002_alpha_0p25",
        )
        decision = armijo_decision(
            parent_objective=10.0,
            candidate_objective=9.0,
            slope=-2.0,
            alpha=1.0,
            c1=1.0e-4,
        )
        self.assertTrue(decision["accepted"])

    def test_fixed_scaling_and_nonempty_history_are_serializable(self):
        scaling = scaling_from_config(self.config)
        history = CurvatureArtifact(
            from_iteration=0,
            to_iteration=1,
            s_lambda="s_lambda.npy",
            s_mu="s_mu.npy",
            y_lambda="y_lambda.npy",
            y_mu="y_mu.npy",
            sMy=2.0,
            accepted=True,
            provenance="accepted physical/control history",
        )
        state = OptimizerIterationState(
            run_id=self.config["run_id"],
            iteration=1,
            accepted_objective=3.0,
            accepted_model_provenance={"path": "accepted", "sha256": "model"},
            gradient_provenance={"path": "gradient", "sha256": "gradient"},
            fixed_scaling=scaling,
            lambda_bias_iteration=1,
            accepted_alpha=1.0,
            memory_limit=15,
            history=(history,),
        )
        payload = state.to_dict()
        self.assertEqual(payload["lbfgs"]["accepted_history_count"], 1)
        self.assertEqual(payload["fixed_reproduction_scaling"]["J_ref"], scaling.J_ref)
        self.assertNotEqual(payload["accepted_objective"], scaling.J_ref)
        restored = OptimizerIterationState.from_dict(payload)
        self.assertEqual(restored.to_dict(), payload)
        with tempfile.TemporaryDirectory() as directory:
            metadata_path = Path(directory) / "iter_optimizer_state.json"
            state.write(metadata_path)
            self.assertEqual(
                OptimizerIterationState.read(metadata_path).to_dict(), payload
            )
            with self.assertRaisesRegex(ValueError, "separate .json"):
                state.write(Path(directory) / "numerical_state.npz")

    def test_curvature_pair_comes_from_actual_accepted_history(self):
        model0 = (np.array([1.0, 2.0]), np.array([3.0, 4.0]))
        model1 = (np.array([1.5, 2.25]), np.array([2.0, 5.0]))
        grad0 = (np.array([0.2, 0.3]), np.array([0.5, 0.7]))
        grad1 = (np.array([0.6, 0.1]), np.array([0.9, 1.2]))
        s_pair, y_pair = physical_curvature_pair(model0, model1, grad0, grad1)
        np.testing.assert_array_equal(s_pair[0], model1[0] - model0[0])
        np.testing.assert_array_equal(s_pair[1], model1[1] - model0[1])
        np.testing.assert_array_equal(y_pair[0], grad1[0] - grad0[0])
        np.testing.assert_array_equal(y_pair[1], grad1[1] - grad0[1])


if __name__ == "__main__":
    unittest.main()
