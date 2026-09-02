import copy
from dataclasses import replace
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import h5py
import numpy as np
from scipy.sparse import csr_matrix, save_npz

from scripts.fathi_benchmark.generic_iteration_runner import (
    GenericIterationRunner,
)
from scripts.fathi_benchmark.iteration_context import build_iteration_paths
from scripts.fathi_benchmark.lbfgs_history import (
    BLOCKED_WAITING_FOR_HISTORY_AUDIT,
    HISTORY_OUTCOME_ACCEPTED,
    HISTORY_OUTCOME_REJECTED,
    HistoryBuildBlocked,
    NO_HISTORY_REQUIRED,
    accepted_history_pairs,
    load_curvature_outcome,
    load_persisted_history,
    persist_accepted_history_pair,
    persist_curvature_outcome,
    sha256_file,
    waiting_for_gradient_status,
)
from scripts.fathi_benchmark.optimizer_state import (
    OptimizerIterationState,
    scaling_from_config,
)
from scripts.fathi_benchmark.path_consistency import (
    validate_path_config_consistency,
)


class IterationEngineIntegrationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.repo = Path(__file__).resolve().parents[1]
        cls.runtime_path = (
            cls.repo / "configs" / "fathi_s43_repro_p20_t052_runtime.json"
        )
        cls.engine_path = (
            cls.repo
            / "configs"
            / "fathi_s43_repro_p20_t052_iteration_engine.json"
        )
        cls.runtime = json.loads(cls.runtime_path.read_text(encoding="utf-8"))
        cls.engine = json.loads(cls.engine_path.read_text(encoding="utf-8"))
        cls.runner = GenericIterationRunner.from_config_files(
            run_id=cls.engine["run_id"],
            parent_iteration=1,
            child_iteration=2,
            repository_root=cls.repo,
            runtime_config_path=cls.runtime_path,
            engine_config_path=cls.engine_path,
        )

    def test_generic_runner_resolves_current_paths_only(self):
        paths = self.runner.paths.to_dict()
        mutable = json.dumps(paths["paths"], sort_keys=True)
        self.assertIn(self.engine["run_id"], mutable)
        self.assertNotIn(self.engine["historical_run_id"], mutable)
        self.assertEqual(paths["parent"], "iter_001")
        self.assertEqual(paths["child"], "iter_002")
        self.assertEqual(paths["transition"], "iter_001_to_iter_002")

    def test_armijo_parent_objective_is_dynamic_not_frozen_scaling_reference(self):
        dynamic_parent_objective = 7.25
        payload = self.runner.prepare_external_armijo(
            {
                "run_id": self.engine["run_id"],
                "parent_iteration": 1,
                "child_iteration": 2,
                "parent_objective": dynamic_parent_objective,
                "slope": -0.5,
                "gradient_artifact": {"path": "g", "sha256": "g-hash"},
                "direction_artifact": {"path": "p", "sha256": "p-hash"},
                "true_receiver_artifact": {
                    "path": "truth",
                    "sha256": "truth-hash",
                },
            }
        )
        scaling = scaling_from_config(self.engine)
        self.assertEqual(payload["parent_objective"], dynamic_parent_objective)
        self.assertNotEqual(payload["parent_objective"], scaling.J_ref)
        self.assertEqual(scaling.J_ref_iteration, 0)
        self.assertEqual(
            scaling.J_ref,
            self.engine["optimizer"]["fixed_reproduction_scaling"]["J_ref"],
        )

    def test_lambda_bias_uses_dynamic_parent_iteration(self):
        self.assertEqual(self.runner.parent_lambda_bias_weight(), 0.98)

    def test_parent_zero_legitimately_uses_h0_without_history(self):
        parent0 = GenericIterationRunner.from_config_files(
            run_id=self.engine["run_id"],
            parent_iteration=0,
            child_iteration=1,
            repository_root=self.repo,
            runtime_config_path=self.runtime_path,
            engine_config_path=self.engine_path,
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            active = self._write_array(root, "optimizer_active.npy", [10, 20])
            coords = self._write_array(
                root,
                "optimizer_coords.npy",
                [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
            )
            matrix_path = root / "optimizer_mtilde.npz"
            save_npz(matrix_path, csr_matrix(np.eye(2)))
            result = parent0.compute_optimizer_direction(
                {
                    "run_id": self.engine["run_id"],
                    "parent_iteration": 0,
                    "child_iteration": 1,
                    "gradient": {
                        "lambda": self._write_array(
                            root, "optimizer_g_lambda.npy", [1.0, 0.5]
                        ),
                        "mu": self._write_array(
                            root, "optimizer_g_mu.npy", [0.25, 1.0]
                        ),
                        "active_indices": active,
                        "coordinates": coords,
                        "mtilde": {
                            "path": str(matrix_path),
                            "sha256": sha256_file(matrix_path),
                        },
                    },
                },
                [],
            )
            self.assertEqual(result.lambda_bias_weight, 1.0)
            self.assertEqual(result.history_audits, [])
            self.assertEqual(
                result.h0_or_history_scale, scaling_from_config(self.engine).gamma0
            )
            self.assertLess(result.slope, 0.0)

    def test_history_refuses_to_invent_child_gradient(self):
        status = self.runner.newest_history_preflight(
            current_parent_gradient=None
        )
        self.assertEqual(status["status"], waiting_for_gradient_status(1))
        self.assertFalse(status["history_pair_created"])
        with self.assertRaises(HistoryBuildBlocked) as caught:
            self.runner.compute_optimizer_direction(
                {
                    "run_id": self.engine["run_id"],
                    "parent_iteration": 1,
                    "child_iteration": 2,
                }
            )
        self.assertEqual(caught.exception.status, waiting_for_gradient_status(1))

    def test_parent_zero_and_parent_two_history_boundaries_are_dynamic(self):
        parent0 = GenericIterationRunner.from_config_files(
            run_id=self.engine["run_id"],
            parent_iteration=0,
            child_iteration=1,
            repository_root=self.repo,
            runtime_config_path=self.runtime_path,
            engine_config_path=self.engine_path,
        )
        status0 = parent0.newest_history_preflight(current_parent_gradient=None)
        self.assertEqual(status0["status"], NO_HISTORY_REQUIRED)
        self.assertIsNone(status0["requested_pair"])

        parent2 = GenericIterationRunner.from_config_files(
            run_id=self.engine["run_id"],
            parent_iteration=2,
            child_iteration=3,
            repository_root=self.repo,
            runtime_config_path=self.runtime_path,
            engine_config_path=self.engine_path,
        )
        status2 = parent2.newest_history_preflight(current_parent_gradient=None)
        self.assertEqual(status2["status"], waiting_for_gradient_status(2))
        self.assertEqual(status2["requested_pair"]["from_iteration"], 1)
        self.assertEqual(status2["requested_pair"]["to_iteration"], 2)

    def test_numerical_npz_and_optimizer_metadata_json_are_separate(self):
        numerical = self.runner.paths.parent_state
        metadata = self.runner.paths.parent_optimizer_metadata_state
        before = sha256_file(numerical)
        self.assertEqual(numerical.suffix, ".npz")
        self.assertEqual(metadata.suffix, ".json")
        self.assertNotEqual(numerical, metadata)
        with self.assertRaisesRegex(ValueError, "separate .json"):
            OptimizerIterationState.read(numerical)
        self.assertEqual(sha256_file(numerical), before)

    def test_path_config_cross_check_passes_for_required_parents(self):
        result = validate_path_config_consistency(
            self.runtime,
            self.engine,
            repository_root=self.repo,
            parent_iterations=(0, 1, 9),
        )
        self.assertEqual(result["result"], "PASS_CURRENT_PATH_CONFIG_CONSISTENCY")
        self.assertEqual(result["checked_parent_iterations"], [0, 1, 9])

    def test_path_config_drift_fails_before_execution(self):
        bad_runtime = copy.deepcopy(self.runtime)
        bad_runtime["runtime_layout"]["state_pattern"] = (
            "results/{run_id}/states/drifted_{iteration_tag}.npz"
        )
        with self.assertRaisesRegex(ValueError, "path config drift"):
            validate_path_config_consistency(
                bad_runtime,
                self.engine,
                repository_root=self.repo,
                parent_iterations=(1,),
            )

    def test_historical_mutable_route_is_rejected(self):
        bad = copy.deepcopy(self.engine)
        bad["namespace"]["data_run_pattern"] = (
            "data/reproduction/" + self.engine["historical_run_id"]
        )
        with self.assertRaisesRegex(ValueError, "historical namespace"):
            build_iteration_paths(
                bad,
                1,
                repository_root=self.repo,
            )

    def test_historical_asset_manifest_is_complete_and_immutable(self):
        manifest = self.runner.immutable_assets
        self.assertEqual(
            {item["asset_id"] for item in manifest["assets"]},
            {
                "exact_spatial_operator",
                "real_s43_compact_topology",
                "strict_full_grid_station_geometry",
            },
        )
        self.assertTrue(all(item["mutable"] is False for item in manifest["assets"]))
        self.assertTrue(
            all(
                item["classification"] == "HISTORICAL_CERTIFIED_ASSET_REUSE"
                for item in manifest["assets"]
            )
        )
        portable_path = self.repo / self.engine["immutable_operator_assets"]["manifest"]
        self.assertNotIn(
            "/home/crellamaybe", portable_path.read_text(encoding="utf-8")
        )

    def test_generic_execution_has_no_completed_runner_special_case(self):
        files = (
            "generic_iteration_runner.py",
            "lbfgs_history.py",
            "external_armijo.py",
            "path_consistency.py",
        )
        for filename in files:
            text = (
                self.repo / "scripts" / "fathi_benchmark" / filename
            ).read_text(encoding="utf-8").lower()
            self.assertNotIn("current_t052", text, filename)

    def _write_array(self, root, name, value):
        path = root / name
        np.save(path, np.asarray(value))
        return {"path": str(path), "sha256": sha256_file(path)}

    def _write_material(self, root, name, lam, mu):
        material = root / name
        material.mkdir()
        files = self.engine["material"]["files"]
        values = {
            "kappa": np.asarray(lam) + (2.0 / 3.0) * np.asarray(mu),
            "mu": np.asarray(mu),
            "density": np.full_like(np.asarray(mu), 2000.0),
        }
        hashes = {}
        for component, value in values.items():
            path = material / files[component]
            with h5py.File(path, "w") as handle:
                handle.create_dataset("samples", data=value)
            hashes[component] = sha256_file(path)
        return material, hashes

    def _history_fixture(self, root):
        active_h5 = self._write_array(root, "active_h5.npy", [0, 1])
        active = self._write_array(root, "active.npy", [10, 20])
        coords = self._write_array(
            root, "coords.npy", [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]
        )
        model0, hashes0 = self._write_material(
            root, "model0", [1.0, 2.0], [3.0, 4.0]
        )
        model1, hashes1 = self._write_material(
            root, "model1", [2.0, 3.0], [4.0, 5.0]
        )
        mtilde_path = root / "mtilde.npz"
        save_npz(mtilde_path, csr_matrix(np.eye(2)))
        mtilde = {"path": str(mtilde_path), "sha256": sha256_file(mtilde_path)}

        def model_spec(path, hashes):
            return {
                "material_dir": str(path),
                "material_sha256": hashes,
                "active_h5_indices": active_h5,
                "active_indices": active,
                "coordinates": coords,
            }

        def gradient_spec(prefix, lam, mu):
            return {
                "lambda": self._write_array(root, prefix + "_lambda.npy", lam),
                "mu": self._write_array(root, prefix + "_mu.npy", mu),
                "active_indices": active,
                "coordinates": coords,
                "mtilde": mtilde,
            }

        return {
            "parent_iteration": 0,
            "child_iteration": 1,
            "parent_model": model_spec(model0, hashes0),
            "child_model": model_spec(model1, hashes1),
            "parent_gradient": gradient_spec("g0", [0.0, 0.0], [0.0, 0.0]),
            "child_gradient": gradient_spec("g1", [1.0, 1.0], [1.0, 1.0]),
        }

    def _parent1_runner_with_history_root(self, history_root):
        runner = GenericIterationRunner.from_config_files(
            run_id=self.engine["run_id"],
            parent_iteration=1,
            child_iteration=2,
            repository_root=self.repo,
            runtime_config_path=self.runtime_path,
            engine_config_path=self.engine_path,
        )
        runner.paths = replace(runner.paths, optimizer_history=Path(history_root))
        return runner

    def _parent1_optimizer_manifest(self, request):
        return {
            "run_id": self.engine["run_id"],
            "parent_iteration": 1,
            "child_iteration": 2,
            "gradient": request["child_gradient"],
        }

    def test_parent_one_gradient_without_outcome_blocks_optimizer(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            request = self._history_fixture(root)
            runner = self._parent1_runner_with_history_root(root / "history")
            status = runner.newest_history_preflight(request["child_gradient"])
            self.assertEqual(status["status"], BLOCKED_WAITING_FOR_HISTORY_AUDIT)
            self.assertFalse(runner.optimization_structurally_runnable(status))
            with self.assertRaises(HistoryBuildBlocked) as caught:
                runner.compute_optimizer_direction(
                    self._parent1_optimizer_manifest(request), []
                )
            self.assertEqual(
                caught.exception.status, BLOCKED_WAITING_FOR_HISTORY_AUDIT
            )

    def test_parent_one_accepted_outcome_requires_and_uses_persisted_pair(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            request = self._history_fixture(root)
            accepted = self.runner.build_real_history_pair(request)
            self.assertTrue(accepted.audit.accepted)
            history_root = root / "history"
            persist_curvature_outcome(accepted, history_root=history_root)
            runner = self._parent1_runner_with_history_root(history_root)
            status = runner.newest_history_preflight(request["child_gradient"])
            self.assertEqual(status["status"], HISTORY_OUTCOME_ACCEPTED)
            self.assertTrue(runner.optimization_structurally_runnable(status))
            result = runner.compute_optimizer_direction(
                self._parent1_optimizer_manifest(request)
            )
            self.assertEqual(len(result.history_audits), 1)
            self.assertTrue(result.history_audits[0].accepted)
            self.assertEqual(result.h0_or_history_scale, 1.0)
            self.assertNotEqual(
                result.h0_or_history_scale, scaling_from_config(self.engine).gamma0
            )

    def test_parent_one_rejected_outcome_is_explicit_and_not_used(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            request = self._history_fixture(root)
            for component in ("lambda", "mu"):
                artifact = request["child_gradient"][component]
                path = Path(artifact["path"])
                np.save(path, np.array([-1.0, -1.0]))
                artifact["sha256"] = sha256_file(path)
            rejected = self.runner.build_real_history_pair(request)
            self.assertFalse(rejected.audit.accepted)
            history_root = root / "history"
            checkpoint = persist_curvature_outcome(
                rejected, history_root=history_root
            )
            self.assertFalse((checkpoint / "curvature_pair.json").exists())
            self.assertFalse((checkpoint / "s_lambda.npy").exists())
            outcome = load_curvature_outcome(
                history_root,
                from_iteration=0,
                to_iteration=1,
                expected_active_indices_sha256=request["child_gradient"][
                    "active_indices"
                ]["sha256"],
                expected_coordinates_sha256=request["child_gradient"][
                    "coordinates"
                ]["sha256"],
                expected_mtilde_sha256=request["child_gradient"]["mtilde"][
                    "sha256"
                ],
            )
            self.assertEqual(outcome["status"], HISTORY_OUTCOME_REJECTED)
            runner = self._parent1_runner_with_history_root(history_root)
            status = runner.newest_history_preflight(request["child_gradient"])
            self.assertEqual(status["status"], HISTORY_OUTCOME_REJECTED)
            self.assertTrue(runner.optimization_structurally_runnable(status))
            result = runner.compute_optimizer_direction(
                self._parent1_optimizer_manifest(request)
            )
            self.assertEqual(result.history_audits, [])
            self.assertEqual(
                result.h0_or_history_scale, scaling_from_config(self.engine).gamma0
            )

    def test_ready_to_build_history_is_not_optimization_runnable(self):
        self.assertFalse(
            self.runner.optimization_structurally_runnable(
                {"status": "READY_TO_BUILD_HISTORY"}
            )
        )
        self.assertFalse(
            self.runner.optimization_structurally_runnable(
                {"status": HISTORY_OUTCOME_ACCEPTED}
            )
        )

    def test_real_history_requires_canonical_active_order(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            request = self._history_fixture(root)
            valid = self.runner.build_real_history_pair(request)
            self.assertTrue(valid.audit.accepted)
            self.assertEqual(len(accepted_history_pairs([valid], memory_limit=15)), 1)

            reversed_path = root / "g1_active_reversed.npy"
            np.save(reversed_path, np.array([20, 10], dtype=np.int64))
            request["child_gradient"]["active_indices"] = {
                "path": str(reversed_path),
                "sha256": sha256_file(reversed_path),
            }
            with self.assertRaisesRegex(ValueError, "active index identity"):
                self.runner.build_real_history_pair(request)

    def test_accepted_history_checkpoint_resume_and_hash_gates(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            request = self._history_fixture(root)
            valid = self.runner.build_real_history_pair(request)
            history_root = root / "history"
            checkpoint = persist_accepted_history_pair(
                valid, history_root=history_root
            )
            hashes = {
                "active": valid.provenance["parent_gradient"][
                    "active_indices_sha256"
                ],
                "coords": valid.provenance["parent_gradient"][
                    "coordinates_sha256"
                ],
                "mtilde": valid.provenance["parent_gradient"]["mtilde_sha256"],
            }
            restored = load_persisted_history(
                history_root,
                parent_iteration=1,
                memory_limit=15,
                expected_active_indices_sha256=hashes["active"],
                expected_coordinates_sha256=hashes["coords"],
                expected_mtilde_sha256=hashes["mtilde"],
            )
            self.assertEqual(len(restored), 1)
            np.testing.assert_array_equal(restored[0][0][0], valid.s_pair[0])

            code = (
                "from scripts.fathi_benchmark.lbfgs_history import "
                "load_persisted_history; import sys; "
                "v=load_persisted_history(sys.argv[1], parent_iteration=1, "
                "memory_limit=15, expected_active_indices_sha256=sys.argv[2], "
                "expected_coordinates_sha256=sys.argv[3], "
                "expected_mtilde_sha256=sys.argv[4]); print(len(v))"
            )
            fresh = subprocess.run(
                [
                    sys.executable,
                    "-c",
                    code,
                    str(history_root),
                    hashes["active"],
                    hashes["coords"],
                    hashes["mtilde"],
                ],
                cwd=self.repo,
                check=True,
                capture_output=True,
                text=True,
            )
            self.assertEqual(fresh.stdout.strip(), "1")

            with self.assertRaisesRegex(ValueError, "mtilde_sha256 mismatch"):
                load_persisted_history(
                    history_root,
                    parent_iteration=1,
                    memory_limit=15,
                    expected_active_indices_sha256=hashes["active"],
                    expected_coordinates_sha256=hashes["coords"],
                    expected_mtilde_sha256="wrong",
                )

            outcome_path = checkpoint / "curvature_outcome.json"
            original_outcome = outcome_path.read_text(encoding="utf-8")
            corrupted_outcome = json.loads(original_outcome)
            corrupted_outcome["mtilde_sha256"] = "corrupted"
            outcome_path.write_text(
                json.dumps(corrupted_outcome, indent=2) + "\n", encoding="utf-8"
            )
            with self.assertRaisesRegex(ValueError, "outcome mtilde_sha256 mismatch"):
                load_persisted_history(
                    history_root,
                    parent_iteration=1,
                    memory_limit=15,
                    expected_active_indices_sha256=hashes["active"],
                    expected_coordinates_sha256=hashes["coords"],
                    expected_mtilde_sha256=hashes["mtilde"],
                )
            outcome_path.write_text(original_outcome, encoding="utf-8")

            array_path = checkpoint / "s_lambda.npy"
            np.save(array_path, np.array([99.0, 99.0]))
            with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
                load_persisted_history(
                    history_root,
                    parent_iteration=1,
                    memory_limit=15,
                    expected_active_indices_sha256=hashes["active"],
                    expected_coordinates_sha256=hashes["coords"],
                    expected_mtilde_sha256=hashes["mtilde"],
                )

    def test_only_accepted_pairs_persist_and_memory_truncates_to_fifteen(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            valid = self.runner.build_real_history_pair(self._history_fixture(root))
            rejected = replace(
                valid,
                audit=replace(valid.audit, accepted=False, reason="test_rejected"),
            )
            history_root = root / "history"
            with self.assertRaisesRegex(ValueError, "rejected curvature pair"):
                persist_accepted_history_pair(rejected, history_root=history_root)

            for index in range(17):
                value = float(index + 1)
                pair = replace(
                    valid,
                    from_iteration=index,
                    to_iteration=index + 1,
                    s_pair=(
                        np.full(2, value),
                        np.full(2, value),
                    ),
                )
                persist_accepted_history_pair(pair, history_root=history_root)
            provenance = valid.provenance["parent_gradient"]
            restored = load_persisted_history(
                history_root,
                parent_iteration=17,
                memory_limit=15,
                expected_active_indices_sha256=provenance[
                    "active_indices_sha256"
                ],
                expected_coordinates_sha256=provenance["coordinates_sha256"],
                expected_mtilde_sha256=provenance["mtilde_sha256"],
            )
            self.assertEqual(len(restored), 15)
            self.assertEqual(restored[0][0][0][0], 3.0)
            self.assertEqual(restored[-1][0][0][0], 17.0)


if __name__ == "__main__":
    unittest.main()
