#!/usr/bin/env python3
"""Regression tests for the standalone SEM3D workspace generator."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import re
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import h5py
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
GENERATOR_PATH = ROOT / "scripts" / "bootstrap" / "generate_sem3d_workspace.py"
CONFIG_PATH = ROOT / "configs" / "fathi_reduced_3x3_12p5.json"


def load_generator_module():
    module_spec = importlib.util.spec_from_file_location(
        "generate_sem3d_workspace",
        GENERATOR_PATH,
    )
    if module_spec is None or module_spec.loader is None:
        raise RuntimeError(f"Cannot import generator: {GENERATOR_PATH}")
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    return module


def parse_sources(path: Path) -> list[dict[str, object]]:
    text = path.read_text(encoding="utf-8")
    blocks = re.findall(r"source\s*\{(.*?)\};", text, flags=re.DOTALL)
    sources: list[dict[str, object]] = []

    for block in blocks:
        def scalar(key: str) -> str:
            match = re.search(
                rf"(?m)^\s*{re.escape(key)}\s*=\s*([^;]+);",
                block,
            )
            if match is None:
                raise AssertionError(f"Missing {key} in source block")
            return match.group(1).strip().strip('"')

        sources.append(
            {
                "coords": tuple(float(value) for value in scalar("coords").split()),
                "type": scalar("type"),
                "direction": tuple(float(value) for value in scalar("dir").split()),
                "func": scalar("func"),
                "time_file": scalar("time_file"),
            }
        )

    return sources


class StandaloneWorkspaceGeneratorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.generator = load_generator_module()
        cls.spec = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))

    def test_plan_builds_both_material_models(self) -> None:
        for model in ("true_layered", "initial_homogeneous"):
            with self.subTest(model=model):
                text_files, fields, stations, stf = self.generator.build_files(
                    self.spec,
                    model=model,
                )
                self.assertEqual(len(text_files), 8)
                self.assertEqual(stations.shape, (225, 3))
                self.assertEqual(stf.shape, (5001, 2))
                self.assertEqual(fields["Kappa"].shape, (41, 33, 33))
                self.assertEqual(fields["Mu"].shape, (41, 33, 33))
                self.assertEqual(fields["Density"].shape, (41, 33, 33))
                for field in fields.values():
                    self.assertTrue(np.isfinite(field).all())

    def test_full_bootstrap_models_save_physical_traces(self) -> None:
        for model in ("true_layered", "initial_homogeneous"):
            with self.subTest(model=model):
                input_spec = self.generator.generate_input_spec(
                    self.spec,
                    model=model,
                )
                sim_time_line = next(
                    line
                    for line in input_spec.splitlines()
                    if line.startswith("sim_time = ")
                )
                sim_time = float(
                    sim_time_line.split("=", 1)[1].strip().rstrip(";")
                )
                self.assertAlmostEqual(sim_time, 0.45, places=14)
                self.assertIn("save_traces = true;", input_spec)

    def test_strict_full_grid_stations_match_canonical_oracle(self) -> None:
        text, stations = self.generator.generate_stations(
            self.spec,
            receiver_role="strict_full_grid",
        )
        self.assertEqual(stations.shape, (38440, 3))
        self.assertEqual(
            tuple(stations[0]),
            (-18.75, -18.75, 0.0),
        )
        self.assertEqual(
            tuple(stations[-1]),
            (18.75, 18.75, -48.75),
        )
        self.assertEqual(
            hashlib.sha256(text.encode("utf-8")).hexdigest(),
            "8af4381e963ad118c1054639c8ee9ac4bb1b9b604f33a09467ea10d0e6191769",
        )

    def test_true_layered_depth_axis_and_values(self) -> None:
        fields = self.generator.material_fields(
            self.spec,
            model="true_layered",
        )
        mu = fields["Mu"]
        kappa = fields["Kappa"]

        self.assertTrue(np.all(mu[0:19] == 125_000_000.0))
        self.assertTrue(np.all(mu[19:31] == 101_250_000.0))
        self.assertTrue(np.all(mu[31:41] == 80_000_000.0))

        expected_kappa = np.asarray(
            [
                125_000_000.0 + (2.0 / 3.0) * 125_000_000.0,
                101_250_000.0 + (2.0 / 3.0) * 101_250_000.0,
                80_000_000.0 + (2.0 / 3.0) * 80_000_000.0,
            ],
            dtype=np.float64,
        )
        self.assertEqual(kappa[0, 0, 0], expected_kappa[0])
        self.assertEqual(kappa[19, 0, 0], expected_kappa[1])
        self.assertEqual(kappa[31, 0, 0], expected_kappa[2])

    def test_initial_model_is_homogeneous_80_mpa(self) -> None:
        fields = self.generator.material_fields(
            self.spec,
            model="initial_homogeneous",
        )
        self.assertTrue(np.all(fields["Mu"] == 80_000_000.0))
        self.assertTrue(
            np.all(
                fields["Kappa"]
                == 80_000_000.0 + (2.0 / 3.0) * 80_000_000.0
            )
        )
        self.assertTrue(np.all(fields["Density"] == 2000.0))

    def test_stf_support_floor_and_peak(self) -> None:
        _, stf = self.generator.generate_stf(self.spec)
        times = stf[:, 0]
        values = stf[:, 1]
        nonzero = np.flatnonzero(values)

        self.assertEqual(nonzero.size, 831)
        self.assertAlmostEqual(times[nonzero[0]], 0.0017, places=15)
        self.assertAlmostEqual(times[nonzero[-1]], 0.0183, places=15)
        self.assertEqual(values[nonzero[0]], 1e-15)
        self.assertEqual(values[nonzero[-1]], 1e-15)
        peak = int(np.argmax(values))
        self.assertAlmostEqual(times[peak], 0.01, places=15)
        self.assertEqual(values[peak], 1.0)

    def test_material_input_uses_canonical_bottom_pml_order(self) -> None:
        lines = self.generator.generate_material_input(self.spec).splitlines()
        bottom = lines[-9:]
        expected_xy = [
            (-20.0, -20.0),
            (0.0, -20.0),
            (20.0, -20.0),
            (-20.0, 0.0),
            (0.0, 0.0),
            (20.0, 0.0),
            (-20.0, 20.0),
            (0.0, 20.0),
            (20.0, 20.0),
        ]
        parsed_xy = []
        for line in bottom:
            tokens = line.split()
            parsed_xy.append((float(tokens[2]), float(tokens[4])))
            self.assertEqual(float(tokens[6]), -50.0)
            self.assertEqual(float(tokens[7]), -1.3)
            self.assertEqual(int(tokens[8]), 2)
        self.assertEqual(parsed_xy, expected_xy)

    def test_cli_writes_clean_and_consistent_workspaces(self) -> None:
        with tempfile.TemporaryDirectory(prefix="fathi_generator_test.") as tmp:
            tmp_root = Path(tmp)
            outputs = {
                "true_layered": tmp_root / "true",
                "initial_homogeneous": tmp_root / "initial",
            }

            for model, output in outputs.items():
                completed = subprocess.run(
                    [
                        sys.executable,
                        str(GENERATOR_PATH),
                        "--config",
                        str(CONFIG_PATH),
                        "--model",
                        model,
                        "--output",
                        str(output),
                        "--write",
                    ],
                    cwd=ROOT,
                    text=True,
                    capture_output=True,
                    check=False,
                )
                self.assertEqual(
                    completed.returncode,
                    0,
                    msg=completed.stdout + "\n" + completed.stderr,
                )
                self.assertIn(
                    "RESULT = PASS_STANDALONE_WORKSPACE_GENERATOR_WRITE",
                    completed.stdout,
                )

            common = [
                "mesh.input",
                "mat.dat",
                "mater.in",
                "material.input",
                "material.spec",
                "stations.txt",
                "gaussian_stf.txt",
            ]
            for relative in common:
                self.assertEqual(
                    (outputs["true_layered"] / relative).read_bytes(),
                    (outputs["initial_homogeneous"] / relative).read_bytes(),
                    msg=f"Common file differs: {relative}",
                )

            self.assertEqual(
                parse_sources(outputs["true_layered"] / "input.spec"),
                parse_sources(outputs["initial_homogeneous"] / "input.spec"),
            )

            runtime_dirs = ("traces", "res", "logs", "prot", "mirror", "sem")
            for output in outputs.values():
                for name in runtime_dirs:
                    self.assertFalse(
                        (output / name).exists(),
                        msg=f"Unexpected runtime directory: {output / name}",
                    )

            expected_attrs = {
                "xMinGlob": np.asarray([-20.0, -20.0, -50.0]),
                "xMaxGlob": np.asarray([20.0, 20.0, 0.0]),
            }
            for output in outputs.values():
                for field_name in ("Kappa", "Mu", "Density"):
                    path = output / "mat" / "h5" / f"Mat_0_{field_name}.h5"
                    self.assertTrue(path.is_file())
                    with h5py.File(path, "r") as h5:
                        dataset = h5["samples"]
                        self.assertEqual(dataset.shape, (41, 33, 33))
                        self.assertEqual(dataset.dtype, np.dtype("float64"))
                        for attr_name, expected in expected_attrs.items():
                            np.testing.assert_array_equal(
                                np.asarray(h5.attrs[attr_name]),
                                expected,
                            )


if __name__ == "__main__":
    unittest.main(verbosity=2)
