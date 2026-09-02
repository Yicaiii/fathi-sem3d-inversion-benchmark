from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import h5py
import numpy as np

from scripts.fathi_benchmark import bridge_certified_external_gradient as bridge
from scripts.fathi_benchmark.iteration_context import build_iteration_paths
from scripts.fathi_benchmark.current_pipeline_contracts import (
    accepted_model_result,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _record(path: Path, *, sha256: str | None = None) -> dict[str, str]:
    return {
        "path": str(path),
        "resolved_path": str(path.resolve()),
        "sha256": sha256 or _sha256(path),
    }


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True), encoding="utf-8")


class CurrentBridgeContractFixture:
    def __init__(self, root: Path) -> None:
        self.repo = root.resolve()
        self.run_id = "current_test_run"
        self.iteration = 1
        self.config = {
            "benchmark_name": self.run_id,
            "forward_operator": {"expected_sample_count": 1040},
        }
        self.engine = {
            "run_id": self.run_id,
            "historical_run_id": "historical_test_run",
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
                "directory": "custom/material_store",
                "dataset": "current_samples",
                "files": {
                    "kappa": "bulk_current.h5",
                    "mu": "shear_current.h5",
                    "density": "rho_current.h5",
                },
            },
        }
        self.config_path = self.repo / "configs" / "runtime.json"
        self.engine_path = self.repo / "configs" / "engine.json"
        _write_json(self.config_path, self.config)
        _write_json(self.engine_path, self.engine)
        self.paths = build_iteration_paths(
            self.engine,
            self.iteration,
            repository_root=self.repo,
            runtime_root=self.repo,
        )

        self.reverse_dir = self.paths.exact_reverse / "production_reverse"
        self.reverse_dir.mkdir(parents=True)
        self.gradient_paths = bridge.current_gradient_paths(self.reverse_dir)
        for name, path in self.gradient_paths.items():
            path.write_bytes(("gradient:" + name).encode("ascii"))

        self.material_shape = (2, 3)
        self.material_dir = (
            self.paths.parent_accepted / self.engine["material"]["directory"]
        )
        self.material_dir.mkdir(parents=True)
        self.material_paths = {}
        for component, filename in self.engine["material"]["files"].items():
            path = self.material_dir / filename
            with h5py.File(path, "w") as handle:
                handle.create_dataset(
                    self.engine["material"]["dataset"],
                    data=np.full(self.material_shape, len(component), dtype=np.float64),
                )
            self.material_paths[component] = path
        self.accepted_summary = self.paths.parent_accepted / "accepted_summary.json"
        _write_json(self.accepted_summary, {"result": "PASS", "iter": 1})

        self.primal_root = self.paths.exact_reverse / "primal_forward"
        self.current_trace = self.primal_root / "current_external_receiver.npy"
        self.current_trace.parent.mkdir(parents=True)
        self.current_trace.write_bytes(b"current receiver")
        self.true_trace = self.repo / "immutable" / "true_external_receiver.npy"
        self.true_trace.parent.mkdir(parents=True)
        self.true_trace.write_bytes(b"true receiver")
        _write_json(
            self.accepted_summary,
            {
                "result": accepted_model_result(self.iteration),
                "run": self.run_id,
                "iter": self.iteration,
                "external_receiver_sha256": _sha256(self.current_trace),
                "true_external_sha256": _sha256(self.true_trace),
                "material_sha256": {
                    path.name: _sha256(path)
                    for path in self.material_paths.values()
                },
            },
        )
        self.primal_path = self.primal_root / "summary.json"
        self.primal = {
            "result": "PASS_ITER001_ACCEPTED_EXTERNAL_PRIMAL_FOR_G1",
            "run_id": self.run_id,
            "parent_iteration": 1,
            "child_iteration": 2,
            "transition": "iter_001_to_iter_002",
            "material_dir": str(self.material_dir),
            "material_sha256": {
                key: _sha256(path) for key, path in self.material_paths.items()
            },
            "current_external_receiver": {
                "path": str(self.current_trace),
                "sha256": _sha256(self.current_trace),
            },
            "true_external_receiver": {
                "path": str(self.true_trace),
                "sha256": _sha256(self.true_trace),
            },
        }
        _write_json(self.primal_path, self.primal)

        self.operator_dir = self.repo / "immutable" / "exact_spatial_operator"
        self.topology_dir = self.repo / "immutable" / "compact_topology"
        self.operator_dir.mkdir(parents=True)
        self.topology_dir.mkdir(parents=True)
        self.gll = self.operator_dir / "gll_coordinates.npy"
        self.weights = self.operator_dir / "gll_weights.npy"
        self.gll.write_bytes(b"gll")
        self.weights.write_bytes(b"weights")
        (self.topology_dir / "solid.npy").write_bytes(b"topology")
        self.operator_signature = "operator-content-signature"
        self.topology_signature = "topology-content-signature"

        gradient_hashes = {
            key: _sha256(path) for key, path in self.gradient_paths.items()
        }
        self.summary = {
            "result": bridge.current_reverse_result(self.iteration),
            "iteration": self.iteration,
            "transition": self.paths.identity.transition_id,
            "reference_manifest": str(self.primal_path),
            "parent_forward_summary": str(self.primal_path),
            "reverse": {"steps": 1040, "next_transition": -1, "finite": True},
            "gates": {name: True for name in bridge.CURRENT_REVERSE_GATES},
            "gradient": {
                name: {
                    "path": str(path),
                    "sha256": gradient_hashes[name],
                    "finite": True,
                }
                for name, path in self.gradient_paths.items()
            },
            "output_hashes": {"gradients": gradient_hashes},
            "input_hashes": {
                "runtime_config": _record(self.config_path),
                "iteration_engine_config": _record(self.engine_path),
                "primal_forward_summary": _record(self.primal_path),
                "accepted_parent_summary": _record(self.accepted_summary),
                "parent_material": {
                    key: _record(path) for key, path in self.material_paths.items()
                },
                "current_external_receiver": _record(self.current_trace),
                "accepted_external_receiver": _record(self.current_trace),
                "true_external_receiver": _record(self.true_trace),
                "driver_assets": {
                    "config": _record(self.config_path),
                    "gll": _record(self.gll),
                    "weights": _record(self.weights),
                    "topology": {
                        "path": str(self.topology_dir),
                        "resolved_path": str(self.topology_dir),
                        "content_signature_sha256": self.topology_signature,
                    },
                },
            },
        }

    def validate(self, summary: dict | None = None, **overrides):
        arguments = {
            "repo": self.repo,
            "config_path": self.config_path,
            "config": self.config,
            "engine_path": self.engine_path,
            "engine": self.engine,
            "paths": self.paths,
            "iteration": self.iteration,
            "reverse_dir": self.reverse_dir,
            "reverse_summary": self.summary if summary is None else summary,
            "operator_dir": self.operator_dir,
            "topology_dir": self.topology_dir,
            "operator_content_signature": self.operator_signature,
            "topology_content_signature": self.topology_signature,
        }
        arguments.update(overrides)
        return bridge.validate_current_reverse_contract(**arguments)


class CurrentBridgeContractTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.fixture = CurrentBridgeContractFixture(Path(self.temporary.name))

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_01_iter001_current_result_string_is_accepted(self):
        result = self.fixture.validate()
        self.assertEqual(
            result["result_contract"],
            "PASS_ITER001_EXACT_REVERSE_MATERIAL_COVECTOR",
        )

    def test_02_wrong_iteration_specific_result_is_rejected(self):
        summary = copy.deepcopy(self.fixture.summary)
        summary["result"] = "PASS_ITER000_EXACT_REVERSE_MATERIAL_COVECTOR"
        with self.assertRaisesRegex(RuntimeError, "result mismatch"):
            self.fixture.validate(summary)

    def test_03_current_lambda_filenames_are_required(self):
        result = self.fixture.validate()
        self.assertEqual(
            {path.name for path in result["gradient_paths"].values()},
            {
                "gradient_solid_lambda.npy",
                "gradient_solid_mu.npy",
                "gradient_pml_lambda.npy",
                "gradient_pml_mu.npy",
            },
        )

    def test_04_lam_filenames_alone_are_rejected(self):
        for domain in ("solid", "pml"):
            source = self.fixture.reverse_dir / f"gradient_{domain}_lambda.npy"
            source.rename(self.fixture.reverse_dir / f"gradient_{domain}_lam.npy")
        with self.assertRaisesRegex(RuntimeError, "missing CURRENT gradient file"):
            self.fixture.validate()

    def test_05_gradient_sha_mismatch_is_rejected(self):
        self.fixture.gradient_paths["solid_lambda"].write_bytes(b"corrupt")
        with self.assertRaisesRegex(RuntimeError, "output SHA-256 mismatch"):
            self.fixture.validate()

    def test_06_reverse_iteration_mismatch_is_rejected(self):
        summary = copy.deepcopy(self.fixture.summary)
        summary["iteration"] = 0
        with self.assertRaisesRegex(RuntimeError, "iteration mismatch"):
            self.fixture.validate(summary)

    def test_07_reverse_transition_mismatch_is_rejected(self):
        summary = copy.deepcopy(self.fixture.summary)
        summary["transition"] = "iter_000_to_iter_001"
        with self.assertRaisesRegex(RuntimeError, "transition mismatch"):
            self.fixture.validate(summary)

    def test_08_reverse_step_count_mismatch_is_rejected(self):
        summary = copy.deepcopy(self.fixture.summary)
        summary["reverse"]["steps"] = 1039
        with self.assertRaisesRegex(RuntimeError, "step count mismatch"):
            self.fixture.validate(summary)

    def test_09_next_transition_mismatch_is_rejected(self):
        summary = copy.deepcopy(self.fixture.summary)
        summary["reverse"]["next_transition"] = 0
        with self.assertRaisesRegex(RuntimeError, "transition -1"):
            self.fixture.validate(summary)

    def test_10_nonfinite_reverse_is_rejected(self):
        summary = copy.deepcopy(self.fixture.summary)
        summary["reverse"]["finite"] = False
        with self.assertRaisesRegex(RuntimeError, "finite gate"):
            self.fixture.validate(summary)

    def test_11_canonical_primal_forward_provenance_passes(self):
        result = self.fixture.validate()
        self.assertEqual(result["canonical_primal_summary"], self.fixture.primal_path)

    def test_12_wrong_primal_forward_path_or_hash_fails(self):
        wrong_path = copy.deepcopy(self.fixture.summary)
        wrong_path["reference_manifest"] = str(self.fixture.repo / "wrong.json")
        with self.subTest("path"), self.assertRaisesRegex(RuntimeError, "canonical"):
            self.fixture.validate(wrong_path)
        wrong_hash = copy.deepcopy(self.fixture.summary)
        wrong_hash["input_hashes"]["primal_forward_summary"]["sha256"] = "0" * 64
        with self.subTest("hash"), self.assertRaisesRegex(RuntimeError, "SHA-256"):
            self.fixture.validate(wrong_hash)

    def test_13_operator_or_topology_identity_mismatch_fails(self):
        wrong_operator = copy.deepcopy(self.fixture.summary)
        wrong_operator["input_hashes"]["driver_assets"]["gll"]["sha256"] = "0" * 64
        with self.subTest("operator"), self.assertRaisesRegex(RuntimeError, "GLL"):
            self.fixture.validate(wrong_operator)
        wrong_topology = copy.deepcopy(self.fixture.summary)
        wrong_topology["input_hashes"]["driver_assets"]["topology"][
            "content_signature_sha256"
        ] = "wrong"
        with self.subTest("topology"), self.assertRaisesRegex(
            RuntimeError, "topology identity"
        ):
            self.fixture.validate(wrong_topology)

    def test_14_current_bridge_has_no_stage5o_production_import(self):
        source = Path(bridge.__file__).read_text(encoding="utf-8")
        self.assertNotIn("bridge_stage5o_certified_gradient", source)
        self.assertIn("certified_gradient_bridge_utils", source)

    def test_15_old_trace_correlation_route_is_absent(self):
        source = Path(bridge.__file__).read_text(encoding="utf-8")
        for legacy in (
            "component_rhs",
            "run_task3_gradient_core",
            "project_gpu_kernel_to_q1_rhs",
            "assemble_gpu_rhs_total",
            "Capteur",
        ):
            self.assertNotIn(legacy, source)

    def test_16_current_material_preflight_uses_engine_contract_only(self):
        self.assertNotIn(
            "material_spec", self.fixture.config.get("sem3d_mesh", {})
        )
        result = bridge.resolve_current_parent_material_contract(
            engine=self.fixture.engine,
            parent_workspace=self.fixture.paths.parent_accepted,
            expected_shape=self.fixture.material_shape,
        )
        self.assertEqual(result["directory"], str(self.fixture.material_dir))
        self.assertEqual(result["dataset"], "current_samples")
        self.assertEqual(
            Path(result["files"]["kappa"]["path"]),
            self.fixture.material_paths["kappa"],
        )
        self.assertEqual(
            Path(result["files"]["mu"]["path"]),
            self.fixture.material_paths["mu"],
        )

    def test_17_missing_engine_material_keys_fail_clearly(self):
        for key in ("kappa", "mu"):
            engine = copy.deepcopy(self.fixture.engine)
            del engine["material"]["files"][key]
            with self.subTest(key=key), self.assertRaisesRegex(
                RuntimeError, f"files requires {key}"
            ):
                bridge.resolve_current_parent_material_contract(
                    engine=engine,
                    parent_workspace=self.fixture.paths.parent_accepted,
                    expected_shape=self.fixture.material_shape,
                )
        engine = copy.deepcopy(self.fixture.engine)
        del engine["material"]["dataset"]
        with self.subTest(key="dataset"), self.assertRaisesRegex(
            RuntimeError, "requires dataset"
        ):
            bridge.resolve_current_parent_material_contract(
                engine=engine,
                parent_workspace=self.fixture.paths.parent_accepted,
                expected_shape=self.fixture.material_shape,
            )

    def test_18_missing_directory_dataset_or_wrong_shape_fails(self):
        engine = copy.deepcopy(self.fixture.engine)
        engine["material"]["directory"] = "missing/materials"
        with self.subTest(case="directory"), self.assertRaisesRegex(
            RuntimeError, "material directory is missing"
        ):
            bridge.resolve_current_parent_material_contract(
                engine=engine,
                parent_workspace=self.fixture.paths.parent_accepted,
                expected_shape=self.fixture.material_shape,
            )

        with h5py.File(self.fixture.material_paths["mu"], "w") as handle:
            handle.create_dataset("wrong_dataset", data=np.zeros(self.fixture.material_shape))
        with self.subTest(case="dataset"), self.assertRaisesRegex(
            RuntimeError, "H5 dataset is missing"
        ):
            bridge.resolve_current_parent_material_contract(
                engine=self.fixture.engine,
                parent_workspace=self.fixture.paths.parent_accepted,
                expected_shape=self.fixture.material_shape,
            )

        with h5py.File(self.fixture.material_paths["mu"], "w") as handle:
            handle.create_dataset(
                self.fixture.engine["material"]["dataset"],
                data=np.zeros((1, 1)),
            )
        with self.subTest(case="shape"), self.assertRaisesRegex(
            RuntimeError, "H5 shape mismatch"
        ):
            bridge.resolve_current_parent_material_contract(
                engine=self.fixture.engine,
                parent_workspace=self.fixture.paths.parent_accepted,
                expected_shape=self.fixture.material_shape,
            )

    def test_19_no_legacy_material_spec_or_production_literal_dependency(self):
        source = Path(bridge.__file__).read_text(encoding="utf-8")
        self.assertNotIn("material_spec", source)
        self.assertNotIn("sem3d_mesh", source)
        self.assertNotIn("Mat_0_Kappa.h5", source)
        self.assertNotIn("Mat_0_Mu.h5", source)
        self.assertNotIn('"mat/h5"', source)
        self.assertNotIn('config["material_grid"]["dataset"]', source)


if __name__ == "__main__":
    unittest.main()
