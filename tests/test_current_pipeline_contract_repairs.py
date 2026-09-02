from __future__ import annotations

import copy
import json
from pathlib import Path
from types import SimpleNamespace
import sys
import tempfile
import unittest
from unittest import mock

import h5py
import numpy as np
from scipy.sparse import csr_matrix, save_npz

from scripts.fathi_benchmark.current_pipeline_artifacts import (
    CandidateEvaluation,
    execute_current_armijo,
    generate_raw_alpha_candidate,
    persist_armijo_trial,
    persist_optimizer_direction,
    promote_current_accepted_trial,
)
from scripts.fathi_benchmark.current_pipeline_contracts import (
    CURRENT_RUN_ID,
    accepted_model_result,
    armijo_search_result,
    artifact_record,
    candidate_generated_result,
    exact_reverse_result,
    gradient_bridge_result,
    optimizer_direction_result,
    registered_gradient_result,
    require_result,
    retained_primal_result,
    sha256_file,
)
from scripts.fathi_benchmark.external_armijo import (
    ArmijoParameters,
    external_armijo_manifest,
)
from scripts.fathi_benchmark.iteration_context import build_iteration_paths
from scripts.fathi_benchmark.lbfgs_history import load_gradient_artifact
from scripts.fathi_benchmark.physical_space_optimizer import (
    apply_lambda_bias_euclidean,
    joint_mtilde_inner,
)
from scripts.fathi_benchmark.register_certified_gradient import (
    register_current_gradient,
)
from scripts.fathi_benchmark.run_certified_external_parent_forward import (
    current_parent_forward_contract,
)
from scripts.fathi_benchmark import run_certified_iteration


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")


class CurrentContractRepairTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.repo = Path(self.temporary.name).resolve()
        self.engine = {
            "run_id": CURRENT_RUN_ID,
            "historical_run_id": "historical_run",
            "namespace": {
                "data_run_pattern": "data/reproduction/{run_id}",
                "results_run_pattern": "results/{run_id}",
                "iteration_pattern": "iterations/{iteration_tag}",
                "accepted_subdir": "accepted",
                "transition_pattern": "{transition_id}",
                "state_pattern": "states/{iteration_tag}_state.npz",
            },
            "routes": {
                "exact_reverse": "exact_reverse",
                "gradient_root": "corrected_gradient",
                "material_covector": "material_covector",
                "control_transpose": "control_interpolation_transpose",
                "mtilde_solve": "mtilde_solve",
                "optimizer_root": "physical_optimizer",
                "optimizer_history": "optimizer_history",
                "optimizer_state_pattern": "{iteration_tag}_optimizer_state.json",
                "line_search_root": "external_armijo",
                "candidate_subdir": "candidates",
            },
            "material": {
                "directory": "mat/h5",
                "dataset": "samples",
                "files": {
                    "kappa": "Mat_0_Kappa.h5",
                    "mu": "Mat_0_Mu.h5",
                    "density": "Mat_0_Density.h5",
                },
            },
        }
        self.paths = build_iteration_paths(
            self.engine, 0, repository_root=self.repo, runtime_root=self.repo
        )
        self._make_parent()
        self._make_gradient_stage()
        self.gradient_manifest_path = register_current_gradient(
            repo=self.repo, paths=self.paths
        )
        self.gradient_record = artifact_record(
            self.gradient_manifest_path, repo=self.repo
        )
        self.parent_record = artifact_record(
            self.paths.parent_accepted / "accepted_summary.json", repo=self.repo
        )
        self.direction_summary_path = self._make_direction()
        self.direction_record = artifact_record(
            self.direction_summary_path, repo=self.repo
        )
        self.direction_slope = float(
            json.loads(self.direction_summary_path.read_text())["mtilde_slope"]
        )
        self.true_path = self.repo / "immutable" / "true.npy"
        self.true_path.parent.mkdir(parents=True)
        np.save(self.true_path, np.zeros((2, 1, 1), dtype=np.float64))
        self.true_record = artifact_record(self.true_path, repo=self.repo)
        self.armijo = external_armijo_manifest(
            paths=self.paths,
            parent_objective=10.0,
            slope=self.direction_slope,
            parent_accepted_artifact=self.parent_record,
            gradient_artifact=self.gradient_record,
            direction_artifact=self.direction_record,
            true_receiver_artifact=self.true_record,
            parameters=ArmijoParameters(1.0e-4, 0.5, 1.0, 2),
        )

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def _make_parent(self) -> None:
        material = self.paths.parent_accepted / "mat" / "h5"
        material.mkdir(parents=True)
        lam = np.array([10.0, 20.0])
        mu = np.array([30.0, 40.0])
        values = {
            "kappa": lam + (2.0 / 3.0) * mu,
            "mu": mu,
            "density": np.array([2000.0, 2000.0]),
        }
        hashes = {}
        for component, value in values.items():
            path = material / self.engine["material"]["files"][component]
            with h5py.File(path, "w") as handle:
                handle.create_dataset("samples", data=value)
            hashes[path.name] = sha256_file(path)
        write_json(
            self.paths.parent_accepted / "accepted_summary.json",
            {
                "schema_version": 1,
                "result": accepted_model_result(0),
                "run": CURRENT_RUN_ID,
                "iter": 0,
                "objective": {"accepted": 10.0},
                "material_sha256": hashes,
            },
        )

    def _identity(self) -> dict:
        return {
            "run_id": CURRENT_RUN_ID,
            "parent_iteration": 0,
            "child_iteration": 1,
            "transition": "iter_000_to_iter_001",
        }

    def _make_gradient_stage(self) -> None:
        reverse_dir = self.paths.exact_reverse / "production_reverse"
        reverse_dir.mkdir(parents=True)
        reverse_summary = reverse_dir / "summary.json"
        write_json(
            reverse_summary,
            {
                "schema_version": 1,
                "result": exact_reverse_result(0),
                "iteration": 0,
                "transition": "iter_000_to_iter_001",
            },
        )
        self.paths.mtilde_solve.mkdir(parents=True)
        active = np.array([10, 20], dtype=np.int64)
        active_h5 = np.array([0, 1], dtype=np.int64)
        coords = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
        np.save(self.paths.gradient_root / "mtilde_active_full_indices.npy", active)
        np.save(self.paths.gradient_root / "active_h5_indices.npy", active_h5)
        np.save(self.paths.gradient_root / "mtilde_active_coords.npy", coords)
        np.save(self.paths.mtilde_solve / "gradient_coords.npy", coords)
        np.save(self.paths.mtilde_solve / "Mtilde_interior_indices.npy", active_h5)
        np.save(self.paths.mtilde_solve / "g_lambda.npy", np.array([1.0, 2.0]))
        np.save(self.paths.mtilde_solve / "g_mu.npy", np.array([3.0, 4.0]))
        save_npz(
            self.paths.mtilde_solve / "Mtilde_interior_sparse.npz",
            csr_matrix(np.eye(2)),
        )
        bridge_result = gradient_bridge_result(0)
        write_json(
            self.paths.mtilde_solve / "mtilde_gradient_summary.json",
            {
                "schema_version": 1,
                "result": bridge_result,
                **self._identity(),
            },
        )
        write_json(
            self.paths.gradient_root / "summary.json",
            {
                "schema_version": 1,
                "result": bridge_result,
                **self._identity(),
                "reverse_source": str(reverse_dir),
                "reverse_result": exact_reverse_result(0),
                "provenance": {
                    "input_sha256": {
                        "reverse_summary": sha256_file(reverse_summary)
                    }
                },
            },
        )

    def _make_direction(self) -> Path:
        optimizer_manifest = {
            **self._identity(),
            "registered_gradient_manifest": self.gradient_record,
            "accepted_parent_summary": self.parent_record,
            "history_outcomes": [],
        }
        raw = (
            np.array([-1.0, -2.0]),
            np.array([-3.0, -4.0]),
        )
        biased = apply_lambda_bias_euclidean(raw, weight=1.0)
        gradient = (np.array([1.0, 2.0]), np.array([3.0, 4.0]))
        result = SimpleNamespace(
            raw_direction=raw,
            biased_direction=biased,
            history_audits=[],
            h0_or_history_scale=1.0,
            lambda_bias_weight=1.0,
            slope=joint_mtilde_inner(gradient, biased, csr_matrix(np.eye(2))),
        )
        return persist_optimizer_direction(
            repo=self.repo,
            paths=self.paths,
            material_config=self.engine["material"],
            optimizer_manifest=optimizer_manifest,
            direction_result=result,
        )

    def _candidate(self, trial: int, alpha: float) -> Path:
        return generate_raw_alpha_candidate(
            repo=self.repo,
            paths=self.paths,
            material_config=self.engine["material"],
            accepted_parent_record=self.parent_record,
            direction_record=self.direction_record,
            parameters=ArmijoParameters(1.0e-4, 0.5, 1.0, 2),
            trial_index=trial,
            alpha=alpha,
        )

    def _evaluation(
        self, candidate_path: Path, trial_dir: Path, objective: float
    ) -> CandidateEvaluation:
        candidate = json.loads(candidate_path.read_text())
        receiver = trial_dir / "candidate_external_receiver.npy"
        receiver.parent.mkdir(parents=True, exist_ok=True)
        np.save(receiver, np.full((2, 1, 1), objective))
        return CandidateEvaluation(
            candidate_material_signature_sha256=candidate[
                "candidate_material_signature_sha256"
            ],
            current_receiver=artifact_record(receiver, repo=self.repo),
            true_receiver=self.true_record,
            objective=objective,
            sample_count=2,
            receiver_count=1,
            component_count=1,
            dt=0.5,
        )

    def test_d1_exact_results_reject_wrong_pass_stage(self):
        require_result(
            {"result": retained_primal_result(2)},
            retained_primal_result(2),
            label="primal",
        )
        with self.assertRaisesRegex(ValueError, "result mismatch"):
            require_result(
                {"result": "PASS_UNRELATED_STAGE"},
                retained_primal_result(2),
                label="primal",
            )
        parent = json.loads(
            (self.paths.parent_accepted / "accepted_summary.json").read_text()
        )
        parent["result"] = "PASS_UNRELATED_ACCEPTED_STAGE"
        write_json(
            self.paths.parent_accepted / "accepted_summary.json", parent
        )
        with self.assertRaisesRegex(ValueError, "accepted parent result"):
            generate_raw_alpha_candidate(
                repo=self.repo,
                paths=self.paths,
                material_config=self.engine["material"],
                accepted_parent_record=artifact_record(
                    self.paths.parent_accepted / "accepted_summary.json",
                    repo=self.repo,
                ),
                direction_record=self.direction_record,
                parameters=ArmijoParameters(1.0e-4, 0.5, 1.0, 2),
                trial_index=0,
                alpha=1.0,
            )

    def test_d2_parent_forward_contract_is_canonical_and_dynamic(self):
        for parent in (1, 2):
            paths = build_iteration_paths(
                self.engine,
                parent,
                repository_root=self.repo,
                runtime_root=self.repo,
            )
            contract = current_parent_forward_contract(paths)
            self.assertEqual(
                Path(contract["output_path"]),
                paths.exact_reverse / "primal_forward",
            )
            self.assertEqual(contract["result"], retained_primal_result(parent))
            self.assertEqual(
                contract["current_receiver_filename"],
                "current_external_receiver.npy",
            )

    def test_d3_bridge_manifest_registers_without_aliases_and_loads(self):
        manifest = json.loads(self.gradient_manifest_path.read_text())
        self.assertEqual(manifest["result"], registered_gradient_result(0))
        self.assertEqual(Path(manifest["lambda"]["path"]).name, "g_lambda.npy")
        self.assertEqual(Path(manifest["mu"]["path"]).name, "g_mu.npy")
        self.assertEqual(
            Path(manifest["coordinates"]["path"]).name, "gradient_coords.npy"
        )
        self.assertFalse((self.paths.mtilde_solve / "grad_lambda.npy").exists())
        self.assertFalse(
            (self.paths.mtilde_solve / "search_direction_summary.json").exists()
        )
        pair, ordering, matrix, provenance = load_gradient_artifact(
            self.repo, manifest, name="registered"
        )
        self.assertEqual(pair[0].shape, (2,))
        self.assertEqual(ordering.active_h5_indices.tolist(), [0, 1])
        self.assertEqual(matrix.shape, (2, 2))
        self.assertEqual(
            provenance["registered_result"], registered_gradient_result(0)
        )
        for key in (
            "source_reverse_summary",
            "source_bridge_summary",
            "source_mtilde_summary",
            "parent_accepted_model",
            "active_indices",
            "active_h5_indices",
            "mtilde",
        ):
            self.assertIn(key, manifest)

    def test_d3_coordinate_index_and_mtilde_hash_mismatch_rejected(self):
        manifest = json.loads(self.gradient_manifest_path.read_text())
        for key in ("coordinates", "active_indices", "mtilde"):
            bad = copy.deepcopy(manifest)
            bad[key]["sha256"] = "0" * 64
            with self.subTest(key=key), self.assertRaisesRegex(
                ValueError, "SHA-256 mismatch"
            ):
                load_gradient_artifact(self.repo, bad, name="bad")

    def test_d4_direction_is_durable_physical_and_fully_bound(self):
        summary = json.loads(self.direction_summary_path.read_text())
        self.assertEqual(summary["result"], optimizer_direction_result(0))
        self.assertEqual(summary["units"], "physical Pa")
        self.assertEqual(summary["normalization"], "none")
        self.assertEqual(summary["mtilde_slope"], self.direction_slope)
        self.assertEqual(
            summary["registered_gradient_manifest"], self.gradient_record
        )
        for key in (
            "raw_lambda",
            "raw_mu",
            "biased_lambda",
            "biased_mu",
            "coordinates",
        ):
            self.assertEqual(len(summary["artifacts"][key]["sha256"]), 64)

    def test_d4_wrong_parent_history_is_rejected(self):
        history = self.repo / "wrong_history.json"
        write_json(
            history,
            {
                "from_iteration": 0,
                "to_iteration": 1,
                "status": "ACCEPTED",
            },
        )
        request = {
            **self._identity(),
            "registered_gradient_manifest": self.gradient_record,
            "accepted_parent_summary": self.parent_record,
            "history_outcomes": [artifact_record(history, repo=self.repo)],
        }
        result = SimpleNamespace(
            raw_direction=(np.array([-1.0, -2.0]), np.array([-3.0, -4.0])),
            biased_direction=(np.array([-1.0, -2.0]), np.array([-3.0, -4.0])),
            history_audits=[],
            h0_or_history_scale=1.0,
            lambda_bias_weight=1.0,
            slope=-30.0,
        )
        with self.assertRaisesRegex(
            ValueError, "history outcome iteration|history outcome .* mismatch|iter0 H0"
        ):
            persist_optimizer_direction(
                repo=self.repo,
                paths=self.paths,
                material_config=self.engine["material"],
                optimizer_manifest=request,
                direction_result=result,
            )

    def test_d4_wrong_parent_gradient_is_rejected(self):
        manifest = json.loads(self.gradient_manifest_path.read_text())
        manifest["iteration"] = 1
        manifest["parent_iteration"] = 1
        manifest["child_iteration"] = 2
        manifest["transition"] = "iter_001_to_iter_002"
        manifest["result"] = registered_gradient_result(1)
        write_json(self.gradient_manifest_path, manifest)
        request = {
            **self._identity(),
            "registered_gradient_manifest": artifact_record(
                self.gradient_manifest_path, repo=self.repo
            ),
            "accepted_parent_summary": self.parent_record,
            "history_outcomes": [],
        }
        result = SimpleNamespace(
            raw_direction=(np.array([-1.0, -2.0]), np.array([-3.0, -4.0])),
            biased_direction=(np.array([-1.0, -2.0]), np.array([-3.0, -4.0])),
            history_audits=[],
            h0_or_history_scale=1.0,
            lambda_bias_weight=1.0,
            slope=-30.0,
        )
        with self.assertRaisesRegex(ValueError, "result mismatch"):
            persist_optimizer_direction(
                repo=self.repo,
                paths=self.paths,
                material_config=self.engine["material"],
                optimizer_manifest=request,
                direction_result=result,
            )

    def test_d5_candidate_is_raw_alpha_update_for_one_and_half(self):
        biased = apply_lambda_bias_euclidean(
            (np.array([-1.0, -2.0]), np.array([-3.0, -4.0])),
            weight=1.0,
        )
        expected = {
            (0, 1.0): (
                np.array([10.0, 20.0]) + biased[0],
                np.array([30.0, 40.0]) + biased[1],
            ),
            (1, 0.5): (
                np.array([10.0, 20.0]) + 0.5 * biased[0],
                np.array([30.0, 40.0]) + 0.5 * biased[1],
            ),
        }
        for (trial, alpha), (lam_expected, mu_expected) in expected.items():
            path = self._candidate(trial, alpha)
            summary = json.loads(path.read_text())
            self.assertEqual(
                summary["result"], candidate_generated_result(0, trial)
            )
            self.assertEqual(summary["normalization"], "none")
            material = path.parent / "mat" / "h5"
            with h5py.File(material / "Mat_0_Kappa.h5") as kh, h5py.File(
                material / "Mat_0_Mu.h5"
            ) as mh:
                kappa = np.asarray(kh["samples"])
                mu = np.asarray(mh["samples"])
            lam = kappa - (2.0 / 3.0) * mu
            np.testing.assert_allclose(lam, lam_expected)
            np.testing.assert_allclose(mu, mu_expected)
            self.assertEqual(len(summary["direction_summary"]["sha256"]), 64)
            self.assertEqual(summary["alpha"], alpha)

    def test_d5_mapping_identity_mismatch_is_rejected(self):
        summary = json.loads(self.direction_summary_path.read_text())
        wrong = self.repo / "wrong_h5.npy"
        np.save(wrong, np.array([1, 0], dtype=np.int64))
        summary["active_h5_indices"] = artifact_record(wrong, repo=self.repo)
        write_json(self.direction_summary_path, summary)
        changed_direction = artifact_record(
            self.direction_summary_path, repo=self.repo
        )
        with self.assertRaisesRegex(ValueError, "differs from registered"):
            generate_raw_alpha_candidate(
                repo=self.repo,
                paths=self.paths,
                material_config=self.engine["material"],
                accepted_parent_record=self.parent_record,
                direction_record=changed_direction,
                parameters=ArmijoParameters(1.0e-4, 0.5, 1.0, 2),
                trial_index=0,
                alpha=1.0,
            )

    def test_d6_sha_parent_and_rejected_trial_gates(self):
        candidate = self._candidate(0, 1.0)
        trial_dir = (
            self.paths.line_search_root / "trials" / candidate.parent.name
        )
        evaluation = self._evaluation(candidate, trial_dir, 11.0)
        bad_candidate = copy.copy(evaluation)
        object.__setattr__(
            bad_candidate, "candidate_material_signature_sha256", "0" * 64
        )
        with self.assertRaisesRegex(ValueError, "candidate SHA"):
            persist_armijo_trial(
                repo=self.repo,
                paths=self.paths,
                armijo_manifest=self.armijo,
                candidate_summary_path=candidate,
                evaluation=bad_candidate,
                trial_directory=trial_dir,
            )
        bad_trace = copy.copy(evaluation)
        object.__setattr__(
            bad_trace,
            "current_receiver",
            {**evaluation.current_receiver, "sha256": "0" * 64},
        )
        with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
            persist_armijo_trial(
                repo=self.repo,
                paths=self.paths,
                armijo_manifest=self.armijo,
                candidate_summary_path=candidate,
                evaluation=bad_trace,
                trial_directory=trial_dir,
            )
        wrong_j = copy.deepcopy(self.armijo)
        wrong_j["parent_objective"] = 9.0
        with self.assertRaisesRegex(ValueError, "accepted J_k"):
            persist_armijo_trial(
                repo=self.repo,
                paths=self.paths,
                armijo_manifest=wrong_j,
                candidate_summary_path=candidate,
                evaluation=evaluation,
                trial_directory=trial_dir,
            )
        trial = persist_armijo_trial(
            repo=self.repo,
            paths=self.paths,
            armijo_manifest=self.armijo,
            candidate_summary_path=candidate,
            evaluation=evaluation,
            trial_directory=trial_dir,
        )
        payload = json.loads(trial.read_text())
        self.assertFalse(payload["accepted"])
        self.assertTrue(payload["result"].startswith("REJECTED_"))
        self.assertEqual(
            payload["objective_result"],
            "PASS_ITER000_ALPHA_000_EXTERNAL_OBJECTIVE",
        )

        wrong_slope = copy.deepcopy(self.armijo)
        wrong_slope["slope"] = -29.0
        with self.assertRaisesRegex(ValueError, "slope differs"):
            persist_armijo_trial(
                repo=self.repo,
                paths=self.paths,
                armijo_manifest=wrong_slope,
                candidate_summary_path=candidate,
                evaluation=evaluation,
                trial_directory=trial_dir,
            )

    def test_d6_armijo_resume_is_idempotent_and_preserves_rejection(self):
        calls = []

        def candidate_provider(index: int, alpha: float) -> Path:
            return self._candidate(index, alpha)

        def evaluation_provider(candidate: Path, trial_dir: Path):
            index = int(json.loads(candidate.read_text())["trial_index"])
            calls.append(index)
            return self._evaluation(
                candidate, trial_dir, 11.0 if index == 0 else 9.0
            )

        first = execute_current_armijo(
            repo=self.repo,
            paths=self.paths,
            armijo_manifest=self.armijo,
            candidate_provider=candidate_provider,
            evaluation_provider=evaluation_provider,
        )
        self.assertEqual(calls, [0, 1])
        final = json.loads(first.read_text())
        self.assertEqual(final["result"], armijo_search_result(0, accepted=True))
        trial0 = json.loads(
            (
                self.paths.line_search_root
                / "trials"
                / self._candidate(0, 1.0).parent.name
                / "trial_summary.json"
            ).read_text()
        )
        self.assertFalse(trial0["accepted"])
        calls.clear()
        second = execute_current_armijo(
            repo=self.repo,
            paths=self.paths,
            armijo_manifest=self.armijo,
            candidate_provider=candidate_provider,
            evaluation_provider=evaluation_provider,
        )
        self.assertEqual(second, first)
        self.assertEqual(calls, [])

    def _accepted_line_search(self) -> Path:
        def candidate_provider(index: int, alpha: float) -> Path:
            return self._candidate(index, alpha)

        def evaluation_provider(candidate: Path, trial_dir: Path):
            return self._evaluation(candidate, trial_dir, 9.0)

        return execute_current_armijo(
            repo=self.repo,
            paths=self.paths,
            armijo_manifest=self.armijo,
            candidate_provider=candidate_provider,
            evaluation_provider=evaluation_provider,
        )

    def test_d7_only_accepted_trial_promotes_hash_identically_and_idempotently(self):
        line = self._accepted_line_search()
        record = artifact_record(line, repo=self.repo)
        accepted = promote_current_accepted_trial(
            repo=self.repo,
            paths=self.paths,
            material_config=self.engine["material"],
            armijo_summary_record=record,
        )
        summary = json.loads(accepted.read_text())
        self.assertEqual(summary["result"], accepted_model_result(1))
        for component, candidate_record in json.loads(
            (self._candidate(0, 1.0)).read_text()
        )["candidate_material"].items():
            self.assertEqual(
                summary["material"][component]["sha256"],
                candidate_record["sha256"],
            )
        self.assertEqual(
            promote_current_accepted_trial(
                repo=self.repo,
                paths=self.paths,
                material_config=self.engine["material"],
                armijo_summary_record=record,
            ),
            accepted,
        )
        self.assertTrue(self.paths.child_state.is_file())
        summary["promotion_input_signature_sha256"] = "conflict"
        write_json(accepted, summary)
        with self.assertRaisesRegex(ValueError, "accepted child conflicts"):
            promote_current_accepted_trial(
                repo=self.repo,
                paths=self.paths,
                material_config=self.engine["material"],
                armijo_summary_record=record,
            )

    def test_d7_rejected_and_conflicting_child_are_rejected(self):
        fail_summary = self.paths.line_search_root / "armijo_summary.json"
        write_json(
            fail_summary,
            {
                "schema_version": 1,
                "result": armijo_search_result(0, accepted=False),
                **self._identity(),
                "accepted": False,
                "accepted_trial": None,
            },
        )
        with self.assertRaisesRegex(ValueError, "result mismatch"):
            promote_current_accepted_trial(
                repo=self.repo,
                paths=self.paths,
                material_config=self.engine["material"],
                armijo_summary_record=artifact_record(
                    fail_summary, repo=self.repo
                ),
            )

    def test_d8_current_historical_runner_blocks_before_subprocess(self):
        config = self.repo / "current.json"
        write_json(config, {"benchmark_name": CURRENT_RUN_ID})
        argv = [
            "run_certified_iteration",
            "--config",
            str(config),
            "--iter-k",
            "1",
            "--stage",
            "status",
            "--repo",
            str(self.repo),
        ]
        with mock.patch.object(sys, "argv", argv), mock.patch(
            "scripts.fathi_benchmark.run_certified_iteration.subprocess.run"
        ) as launched, self.assertRaisesRegex(
            RuntimeError, "BLOCKED_HISTORICAL_ITERATION_RUNNER_FOR_CURRENT"
        ):
            run_certified_iteration.main()
        launched.assert_not_called()
        run_certified_iteration.guard_historical_runner("historical_reproduction")

    def test_future_paths_and_cpu_modules_are_current_only(self):
        for parent in (1, 2, 7):
            paths = build_iteration_paths(
                self.engine,
                parent,
                repository_root=self.repo,
                runtime_root=self.repo,
            )
            self.assertEqual(
                paths.identity.transition_id,
                f"iter_{parent:03d}_to_iter_{parent + 1:03d}",
            )
            self.assertIn(CURRENT_RUN_ID, str(paths.transition_root))
        for filename in (
            "register_certified_gradient.py",
            "current_pipeline_artifacts.py",
            "external_armijo.py",
            "run_current_pipeline.py",
        ):
            source = (
                Path(__file__).resolve().parents[1]
                / "scripts"
                / "fathi_benchmark"
                / filename
            ).read_text(encoding="utf-8")
            self.assertNotIn("import subprocess", source)
            self.assertNotIn("mpi4py", source)
        candidate_source = (
            Path(__file__).resolve().parents[1]
            / "scripts"
            / "fathi_benchmark"
            / "current_pipeline_artifacts.py"
        ).read_text(encoding="utf-8")
        self.assertNotIn("joint_maxabs", candidate_source)


if __name__ == "__main__":
    unittest.main()
