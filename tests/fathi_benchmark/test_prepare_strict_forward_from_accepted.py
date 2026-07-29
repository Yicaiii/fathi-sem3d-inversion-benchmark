#!/usr/bin/env python3
"""Tests for deterministic strict-forward preparation."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
CONFIG_SOURCE = ROOT / "configs/fathi_reduced_3x3_12p5.json"
EXPECTED_HASH = "8af4381e963ad118c1054639c8ee9ac4bb1b9b604f33a09467ea10d0e6191769"


class FreshStrictForwardPreparationTests(unittest.TestCase):
    def test_prepares_static_workspace_without_legacy_template(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config_dir = root / "configs"
            config_dir.mkdir(parents=True)
            (config_dir / CONFIG_SOURCE.name).write_text(
                CONFIG_SOURCE.read_text(encoding="utf-8"),
                encoding="utf-8",
            )

            parent = root / "data/iter_000/accepted"
            workspace = root / "data/iter_001/strict_full_forward_000"
            (parent / "mat/h5").mkdir(parents=True)
            (parent / "sem").mkdir(parents=True)

            static_text = {
                "mesh.input": "12\n1\n",
                "mat.dat": "dummy mat.dat\n",
                "mater.in": "dummy mater.in\n",
                "material.input": "dummy material.input\n",
                "material.spec": "dummy material.spec\n",
                "input.spec": (
                    'run_name = "parent";\n'
                    'save_traces = true;\n'
                    'file = "stations.txt";\n'
                ),
                "gaussian_stf.txt": "0 0\n",
                "stations.txt": "0 0 0\n",
            }
            for name, content in static_text.items():
                (parent / name).write_text(content, encoding="utf-8")

            for name in (
                "Mat_0_Kappa.h5",
                "Mat_0_Mu.h5",
                "Mat_0_Density.h5",
            ):
                (parent / "mat/h5" / name).write_bytes(b"material")

            for index in range(12):
                (parent / "sem" / f"mesh4spec.{index:04d}.h5").write_bytes(
                    b"mesh"
                )

            (parent / "traces").mkdir()
            (parent / "traces/capteurs.0000.h5").write_bytes(b"runtime")
            (parent / "logs").mkdir()
            (parent / "logs/solver.stdout").write_text(
                "runtime",
                encoding="utf-8",
            )

            context = {
                "transition": "iter_000_to_iter_001",
                "input_accepted_dir": "data/iter_000/accepted",
                "strict_forward_workspace": (
                    "data/iter_001/strict_full_forward_000"
                ),
                "strict_forward_traces": (
                    "data/iter_001/strict_full_forward_000/traces"
                ),
                "benchmark_profile_config": (
                    "configs/fathi_reduced_3x3_12p5.json"
                ),
            }
            context_path = root / "context.json"
            context_path.write_text(
                json.dumps(context),
                encoding="utf-8",
            )

            env = os.environ.copy()
            env["FATHI_BENCHMARK_ROOT"] = str(root)

            proc = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    (
                        "scripts.fathi_benchmark."
                        "prepare_strict_forward_from_accepted"
                    ),
                    "--context",
                    str(context_path),
                ],
                cwd=ROOT,
                env=env,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )

            self.assertEqual(proc.returncode, 0, proc.stderr)
            self.assertIn("RESULT = PASS", proc.stdout)

            station_bytes = (workspace / "stations.txt").read_bytes()
            self.assertEqual(
                hashlib.sha256(station_bytes).hexdigest(),
                EXPECTED_HASH,
            )
            self.assertEqual(
                len(
                    [
                        line
                        for line in station_bytes.decode("utf-8").splitlines()
                        if line.strip()
                    ]
                ),
                38440,
            )

            self.assertTrue(
                (workspace / "mat/h5/Mat_0_Kappa.h5").is_file()
            )
            self.assertEqual(
                len(list((workspace / "sem").glob("mesh4spec.*.h5"))),
                12,
            )
            self.assertFalse((workspace / "traces").exists())
            self.assertFalse((workspace / "logs").exists())
            self.assertTrue(
                (workspace / "STRICT_FORWARD_PREPARATION.json").is_file()
            )


if __name__ == "__main__":
    unittest.main()
