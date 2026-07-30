from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "bootstrap" / "initialize_iter000_from_bootstrap.py"
SPEC = importlib.util.spec_from_file_location("initialize_iter000", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def traces(directory: Path, positions, values, time=None):
    directory.mkdir(parents=True, exist_ok=True)
    t = np.asarray([0.0, 1.0] if time is None else time, dtype=np.float64)
    with h5py.File(directory / "capteurs.0000.h5", "w") as h5:
        for index, (position, value) in enumerate(zip(positions, values, strict=True)):
            key = f"UU_{index}"
            u = np.zeros((len(t), 3), dtype=np.float64)
            u[:, 0] = value
            h5.create_dataset(key, data=np.column_stack([t, u]))
            h5.create_dataset(key + "_pos", data=np.asarray(position, dtype=np.float64))


def solver_manifest(workspace: Path, smoke=None):
    (workspace / "logs").mkdir(parents=True, exist_ok=True)
    data = {
        "smoke_seconds": smoke,
        "return_code": 0,
        "timed_out": False,
        "effective_input_settings": {"sim_time_s": 0.45, "save_traces": True},
        "output_audit": {"passed": True, "trace_count": 1,
                         "fin_sem": {"exists": True, "value": "1"}},
    }
    (workspace / "logs" / "solver_manifest.json").write_text(json.dumps(data), encoding="utf-8")


def operator_files(workspace: Path, run_name: str):
    workspace.mkdir(parents=True, exist_ok=True)
    (workspace / "stations.txt").write_text("0 0 0\n", encoding="utf-8")
    (workspace / "gaussian_stf.txt").write_text("0 1\n", encoding="utf-8")
    (workspace / "input.spec").write_text(
        f'run_name = "{run_name}";\nsim_time = 4.5000000000000001e-01;\nsave_traces = true;\n',
        encoding="utf-8",
    )


class InitializeIter000Tests(unittest.TestCase):
    def test_known_misfit(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            positions = [(0.0, 0.0, 0.0)]
            traces(root / "true", positions, [0.0])
            traces(root / "initial", positions, [1.0])
            result = MODULE.initial_misfit(root / "true", root / "initial", 1, 1.0, 8)
            self.assertAlmostEqual(result["J"], 0.5)

    def test_receiver_coordinate_mismatch_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            traces(root / "true", [(0.0, 0.0, 0.0)], [0.0])
            traces(root / "initial", [(1.0, 0.0, 0.0)], [0.0])
            with self.assertRaisesRegex(ValueError, "coordinate sets differ"):
                MODULE.initial_misfit(root / "true", root / "initial", 1, 1.0, 8)

    def test_smoke_bootstrap_fails(self):
        manifest = {"status": "passed", "audit_only": False, "smoke_seconds": 0.012,
                    "models": list(MODULE.MODELS), "steps": []}
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "bootstrap_manifest.json"
            with self.assertRaisesRegex(ValueError, "Smoke bootstrap"):
                MODULE.workspaces_from_manifest(manifest, path)

    def test_full_bootstrap_evidence_passes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            true_ws, init_ws = root / "true", root / "initial"
            for workspace, name in ((true_ws, "true"), (init_ws, "initial")):
                operator_files(workspace, name)
                solver_manifest(workspace)
            manifest = {
                "status": "passed", "audit_only": False, "smoke_seconds": None,
                "models": list(MODULE.MODELS),
                "steps": [
                    {"model": "true_layered", "stage": "solve", "workspace": str(true_ws),
                     "passed": True, "return_code": 0},
                    {"model": "initial_homogeneous", "stage": "solve", "workspace": str(init_ws),
                     "passed": True, "return_code": 0},
                ],
            }
            path = root / "bootstrap_manifest.json"
            path.write_text(json.dumps(manifest), encoding="utf-8")
            _, workspaces = MODULE.validate_bootstrap(path, 0.45)
            self.assertEqual(workspaces["true_layered"], true_ws.resolve())

    def test_runtime_ignore_keeps_sem_and_material(self):
        ignored = MODULE.ignore_runtime("", ["traces", "logs", "fin_sem", "output.0",
                                                "sem", "mat", "input.spec"])
        self.assertTrue({"traces", "logs", "fin_sem", "output.0"} <= ignored)
        self.assertFalse({"sem", "mat", "input.spec"} & ignored)

    def test_state_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "state.npz"
            base = np.full((2, 2, 2), 80.0)
            fields = {"lambda": base, "mu": base, "kappa": base + 10, "density": base + 1920}
            MODULE.build_state(path, "data/iter_000/accepted", fields, 1.25)
            with np.load(path) as state:
                self.assertAlmostEqual(float(state["J"]), 1.25)
                self.assertEqual(int(state["iter"]), 0)
                np.testing.assert_allclose(state["mu"], base)


if __name__ == "__main__":
    unittest.main()
