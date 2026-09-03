from __future__ import annotations

import inspect
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from scripts.exact_adjoint.s43_external_forward import common_paths, sha256_file
from scripts.fathi_benchmark import build_current_certified_external_reference as reference_builder
from scripts.fathi_benchmark import run_current_iteration
from scripts.fathi_benchmark import current_pipeline_artifacts
from scripts.fathi_benchmark import run_exact_reverse_gradient_generic as generic_reverse


class CurrentIterationRoutingTest(unittest.TestCase):
    def _touch_reference_assets(self, root: Path, run: str):
        assets = root / "assets"
        topology = assets / "topology"
        coefficients = assets / "coefficients"
        coupled_mass = assets / "coupled_mass"
        receiver = assets / "receiver"
        for directory in (topology, coefficients, coupled_mass, receiver):
            directory.mkdir(parents=True, exist_ok=True)
            (directory / "summary.json").write_text("{}", encoding="utf-8")
        np.save(receiver / "receiver_nodes.npy", np.array([[0, 1]], dtype=np.int64))
        np.save(receiver / "receiver_weights.npy", np.array([[0.5, 0.5]], dtype=np.float64))
        gll = assets / "gll.npy"
        weights = assets / "weights.npy"
        stf = assets / "stf.txt"
        true_external = assets / "true.npy"
        np.save(gll, np.array([-1.0, 1.0]))
        np.save(weights, np.array([1.0, 1.0]))
        stf.write_text("0 0\n", encoding="utf-8")
        np.save(true_external, np.zeros((2, 1, 3), dtype=np.float64))
        return {
            "topology": topology,
            "coefficients": coefficients,
            "coupled_mass": coupled_mass,
            "receiver": receiver,
            "gll": gll,
            "weights": weights,
            "stf": stf,
            "true": true_external,
        }

    def test_current_reference_routes_external_driver_to_runtime_config(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run = "current_run"
            configs = root / "configs"
            configs.mkdir()
            legacy = configs / f"{run}.json"
            legacy.write_text(json.dumps({"legacy": True}), encoding="utf-8")
            runtime = configs / f"{run}_runtime.json"
            runtime.write_text(json.dumps({"domain": {}}), encoding="utf-8")
            assets = self._touch_reference_assets(root, run)
            reference = root / "results" / run / "certified_external_reference.json"
            reference.parent.mkdir(parents=True)
            reference.write_text(
                json.dumps(
                    {
                        "result": "PASS_CERTIFIED_EXTERNAL_REFERENCE_CONTRACT",
                        "run": run,
                        "reference_root": f"results/{run}",
                        "operator_assets": {
                            key: str(value.relative_to(root))
                            for key, value in assets.items()
                            if key in {"topology", "coefficients", "coupled_mass", "gll", "weights", "receiver"}
                        },
                        "certification_assets": {
                            "true_external": str(assets["true"].relative_to(root))
                        },
                        "immutable_input_assets": {
                            "reference_stf": str(assets["stf"].relative_to(root)),
                            "runtime_config": str(runtime.relative_to(root)),
                        },
                    }
                ),
                encoding="utf-8",
            )
            resolved = common_paths(root, run, reference_manifest=reference)
            self.assertEqual(Path(resolved["config"]), runtime.resolve())

    def test_historical_reference_keeps_configs_run_fallback(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run = "historical_run"
            configs = root / "configs"
            configs.mkdir()
            legacy = configs / f"{run}.json"
            legacy.write_text(json.dumps({"legacy": True}), encoding="utf-8")
            assets = self._touch_reference_assets(root, run)
            reference = root / "results" / run / "certified_external_reference.json"
            reference.parent.mkdir(parents=True)
            reference.write_text(
                json.dumps(
                    {
                        "result": "PASS_CERTIFIED_EXTERNAL_REFERENCE_CONTRACT",
                        "run": run,
                        "reference_root": f"results/{run}",
                        "operator_assets": {
                            key: str(value.relative_to(root))
                            for key, value in assets.items()
                            if key in {"topology", "coefficients", "coupled_mass", "gll", "weights", "receiver"}
                        },
                        "certification_assets": {
                            "true_external": str(assets["true"].relative_to(root))
                        },
                        "immutable_input_assets": {
                            "reference_stf": str(assets["stf"].relative_to(root))
                        },
                    }
                ),
                encoding="utf-8",
            )
            resolved = common_paths(root, run, reference_manifest=reference)
            self.assertEqual(Path(resolved["config"]), legacy.resolve())

    def test_reference_builder_includes_future_parent_forward_hashes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run = "current_run"
            (root / "configs").mkdir()
            assets = self._touch_reference_assets(root, run)
            source_xyz = root / "source_xyz.npy"
            source_amp = root / "source_amp.npy"
            np.save(source_xyz, np.array([[0.0, 0.0, 0.0]], dtype=np.float64))
            np.save(source_amp, np.array([12.0], dtype=np.float64))
            runtime = root / "configs" / f"{run}_runtime.json"
            runtime.write_text(
                json.dumps(
                    {
                        "benchmark_name": run,
                        "forward_operator": {
                            "expected_sample_count": 2,
                            "physical_receiver_count": 1,
                            "source_count": 1,
                            "dimension": 3,
                            "effective_dt_s": 0.5,
                            "source_coordinates_path": str(source_xyz.relative_to(root)),
                            "source_amplitudes_path": str(source_amp.relative_to(root)),
                            "source_direction": [0.0, 0.0, 1.0],
                            "assembled_peak_force_n": 12.0,
                        },
                    }
                ),
                encoding="utf-8",
            )
            accepted = root / "accepted_summary.json"
            accepted.write_text(
                json.dumps(
                    {
                        "run": run,
                        "true_external_sha256": sha256_file(assets["true"]),
                    }
                ),
                encoding="utf-8",
            )

            def file_record(path: Path):
                return {
                    "path": str(path),
                    "resolved_path": str(path.resolve()),
                    "sha256": sha256_file(path),
                }

            def dir_record(path: Path):
                return {
                    "path": str(path),
                    "resolved_path": str(path.resolve()),
                    "content_signature_sha256": reference_builder._directory_signature(path),
                }

            reverse = root / "reverse_summary.json"
            reverse.write_text(
                json.dumps(
                    {
                        "result": "PASS_ITER001_EXACT_REVERSE_MATERIAL_COVECTOR",
                        "iteration": 1,
                        "input_hashes": {
                            "accepted_parent_summary": file_record(accepted),
                            "true_external_receiver": file_record(assets["true"]),
                            "driver_assets": {
                                "topology": dir_record(assets["topology"]),
                                "coefficients": dir_record(assets["coefficients"]),
                                "coupled_mass": dir_record(assets["coupled_mass"]),
                                "receiver": dir_record(assets["receiver"]),
                                "gll": file_record(assets["gll"]),
                                "weights": file_record(assets["weights"]),
                                "stf": file_record(assets["stf"]),
                                "config": file_record(runtime),
                            },
                        },
                    }
                ),
                encoding="utf-8",
            )
            manifest = reference_builder.build_reference(
                repo=root,
                runtime_config_path=runtime,
                reverse_summary_path=reverse,
            )
            self.assertEqual(manifest["immutable_input_assets"]["runtime_config"], str(runtime.relative_to(root)))
            self.assertEqual(manifest["hashes"]["receiver_nodes_sha256"], sha256_file(assets["receiver"] / "receiver_nodes.npy"))
            self.assertEqual(manifest["hashes"]["receiver_weights_sha256"], sha256_file(assets["receiver"] / "receiver_weights.npy"))
            self.assertEqual(manifest["contract"]["source_count"], 1)

    def test_generic_reverse_no_longer_defaults_to_compat_repo(self):
        source = inspect.getsource(generic_reverse.build_runtime)
        self.assertIn("reference_manifest", source)
        self.assertNotIn('true_path.parent.parent / "compat_repo"', source)
        module_source = Path(generic_reverse.__file__).read_text(encoding="utf-8")
        self.assertIn("--reference-manifest", module_source)

    def test_candidate_external_forward_uses_primal_receiver_label(self):
        source = inspect.getsource(current_pipeline_artifacts.evaluate_candidate_external)
        self.assertIn('{"primal": current_path}', source)
        self.assertNotIn('{"candidate": current_path}', source)


    def test_unified_runner_is_current_only_and_k_driven(self):
        source = Path(run_current_iteration.__file__).read_text(encoding="utf-8")
        self.assertIn('"--parent-iteration", "--k"', source)
        for legacy in (
            "run_current_t052_external_armijo.py",
            "compute_search_direction.py",
            "solve_gpu_mtilde_gradient.py",
            "424B_compute_rhs_component_from_traces.py",
        ):
            self.assertNotIn(legacy, source)
        self.assertIn("run_certified_external_parent_forward.py", source)
        self.assertIn("run_exact_reverse_gradient_generic.py", source)
        self.assertIn("bridge_certified_external_gradient.py", source)


if __name__ == "__main__":
    unittest.main()
