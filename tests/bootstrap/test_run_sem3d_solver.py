from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
RUNNER = REPO_ROOT / "scripts" / "bootstrap" / "run_sem3d_solver.py"
CONFIG = REPO_ROOT / "configs" / "fathi_reduced_3x3_12p5.json"


class RunSem3DSolverTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory(prefix="test_sem3d_solver_")
        self.root = Path(self.temp.name)
        self.workspace = self.root / "workspace"
        self.workspace.mkdir()

        (self.workspace / "input.spec").write_text(
            'run_name = "test";\n'
            'sim_time = 0.45;\n'
            'mesh_file = "mesh4spec";\n'
            'mat_file = "material.input";\n'
            'save_traces = false;\n',
            encoding="utf-8",
        )
        for name in (
            "material.input",
            "material.spec",
            "stations.txt",
            "gaussian_stf.txt",
        ):
            (self.workspace / name).write_text(f"{name}\n", encoding="utf-8")

        h5 = self.workspace / "mat" / "h5"
        h5.mkdir(parents=True)
        for name in ("Mat_0_Kappa.h5", "Mat_0_Mu.h5", "Mat_0_Density.h5"):
            (h5 / name).write_bytes(b"H5")

        sem = self.workspace / "sem"
        sem.mkdir()
        for index in range(12):
            (sem / f"mesh4spec.{index:04d}.h5").write_bytes(b"mesh")

        self.solver = self.make_fake_solver()
        self.mpirun = self.make_fake_mpirun()

    def tearDown(self) -> None:
        self.temp.cleanup()

    def make_fake_mpirun(self) -> Path:
        path = self.root / "fake_mpirun.py"
        path.write_text(
            """#!/usr/bin/env python3
import subprocess
import sys

args = sys.argv[1:]
if len(args) != 3 or args[0] != "-np":
    print(f"unexpected MPI arguments: {args}", file=sys.stderr)
    raise SystemExit(90)
raise SystemExit(subprocess.run([args[2]], check=False).returncode)
""",
            encoding="utf-8",
        )
        path.chmod(0o755)
        return path

    def make_fake_solver(self) -> Path:
        path = self.root / "fake_sem3d.py"
        path.write_text(
            """#!/usr/bin/env python3
import os
from pathlib import Path
import time

workspace = Path.cwd()
text = (workspace / "input.spec").read_text(encoding="utf-8")
(workspace / "seen_input.spec").write_text(text, encoding="utf-8")
mode = os.environ.get("FAKE_SOLVER_MODE", "success")

if mode == "timeout":
    time.sleep(5)
if mode == "nonzero":
    raise SystemExit(7)
if mode != "missing_fin_sem":
    (workspace / "fin_sem").write_text("1\\n", encoding="utf-8")
if mode not in {"no_traces", "missing_fin_sem"}:
    traces = workspace / "traces"
    traces.mkdir(exist_ok=True)
    (traces / "capteurs.0000.h5").write_bytes(b"trace-data")
print("fin du calcul sur processeurs")
""",
            encoding="utf-8",
        )
        path.chmod(0o755)
        return path

    def command(self, *extra: str) -> list[str]:
        return [
            sys.executable,
            str(RUNNER),
            "--config",
            str(CONFIG),
            "--workspace",
            str(self.workspace),
            "--solver",
            str(self.solver),
            "--mpirun",
            str(self.mpirun),
            *extra,
        ]

    def run_command(
        self,
        *extra: str,
        mode: str = "success",
    ) -> subprocess.CompletedProcess[str]:
        env = os.environ.copy()
        env["FATHI_BENCHMARK_ROOT"] = str(REPO_ROOT)
        env["FAKE_SOLVER_MODE"] = mode
        return subprocess.run(
            self.command(*extra),
            text=True,
            capture_output=True,
            env=env,
            check=False,
        )

    def test_plan_only_does_not_change_workspace(self) -> None:
        original = (self.workspace / "input.spec").read_bytes()
        result = self.run_command("--smoke-seconds", "0.012")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("RESULT = PASS_SEM3D_SOLVER_PLAN", result.stdout)
        self.assertEqual((self.workspace / "input.spec").read_bytes(), original)
        self.assertFalse((self.workspace / "fin_sem").exists())

    def test_smoke_execution_patches_then_restores_input_and_audits(self) -> None:
        original = (self.workspace / "input.spec").read_bytes()
        result = self.run_command(
            "--smoke-seconds",
            "0.012",
            "--timeout-seconds",
            "10",
            "--execute",
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("RESULT = PASS_SEM3D_SOLVER_EXECUTION_AND_AUDIT", result.stdout)
        self.assertEqual((self.workspace / "input.spec").read_bytes(), original)

        seen = (self.workspace / "seen_input.spec").read_text(encoding="utf-8")
        self.assertIn("sim_time = 0.012;", seen)
        self.assertIn("save_traces = true;", seen)

        manifest = json.loads(
            (self.workspace / "logs" / "solver_manifest.json").read_text(encoding="utf-8")
        )
        self.assertEqual(manifest["schema_version"], 1)
        self.assertTrue(manifest["input_spec_restored_after_run"])
        self.assertEqual(manifest["original_input_settings"]["sim_time_s"], 0.45)
        self.assertEqual(manifest["effective_input_settings"]["sim_time_s"], 0.012)
        self.assertTrue(manifest["effective_input_settings"]["save_traces"])
        self.assertTrue(manifest["output_audit"]["passed"])

    def test_normal_run_with_save_traces_false_allows_no_traces(self) -> None:
        result = self.run_command("--execute", mode="no_traces")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("trace_count = 0", result.stdout)
        self.assertIn("RESULT = PASS_SEM3D_SOLVER_EXECUTION_AND_AUDIT", result.stdout)

    def test_missing_mesh_partition_is_rejected(self) -> None:
        (self.workspace / "sem" / "mesh4spec.0011.h5").unlink()
        result = self.run_command()
        self.assertEqual(result.returncode, 2)
        self.assertIn("mesh4spec.0011.h5", result.stderr)

    def test_existing_outputs_require_overwrite(self) -> None:
        (self.workspace / "fin_sem").write_text("stale\n", encoding="utf-8")
        refused = self.run_command("--execute")
        self.assertEqual(refused.returncode, 2)
        self.assertIn("Refusing to mix stale and fresh files", refused.stderr)

        replaced = self.run_command("--execute", "--overwrite", mode="no_traces")
        self.assertEqual(replaced.returncode, 0, replaced.stdout + replaced.stderr)
        self.assertEqual((self.workspace / "fin_sem").read_text().strip(), "1")

    def test_nonzero_solver_exit_is_reported(self) -> None:
        result = self.run_command("--execute", mode="nonzero")
        self.assertEqual(result.returncode, 7, result.stdout + result.stderr)
        self.assertIn("RESULT = FAIL_SEM3D_SOLVER_EXECUTION", result.stdout)
        manifest = json.loads(
            (self.workspace / "logs" / "solver_manifest.json").read_text(encoding="utf-8")
        )
        self.assertEqual(manifest["return_code"], 7)
        self.assertFalse(manifest["output_audit"]["passed"])

    def test_timeout_is_reported_and_input_is_restored(self) -> None:
        original = (self.workspace / "input.spec").read_bytes()
        result = self.run_command(
            "--smoke-seconds",
            "0.012",
            "--timeout-seconds",
            "0.2",
            "--execute",
            mode="timeout",
        )
        self.assertEqual(result.returncode, 124, result.stdout + result.stderr)
        self.assertIn("RESULT = FAIL_SEM3D_SOLVER_TIMEOUT", result.stdout)
        self.assertEqual((self.workspace / "input.spec").read_bytes(), original)

    def test_missing_fin_sem_fails_output_audit(self) -> None:
        result = self.run_command(
            "--smoke-seconds",
            "0.012",
            "--execute",
            mode="missing_fin_sem",
        )
        self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
        self.assertIn("RESULT = FAIL_SEM3D_SOLVER_OUTPUT_AUDIT", result.stdout)

    def test_audit_only_uses_manifest_smoke_expectation_after_input_restore(self) -> None:
        executed = self.run_command("--smoke-seconds", "0.012", "--execute")
        self.assertEqual(executed.returncode, 0, executed.stdout + executed.stderr)
        audited = self.run_command("--audit-only")
        self.assertEqual(audited.returncode, 0, audited.stdout + audited.stderr)
        self.assertIn("expect_traces = True", audited.stdout)
        self.assertIn("RESULT = PASS_SEM3D_SOLVER_OUTPUT_AUDIT", audited.stdout)


if __name__ == "__main__":
    unittest.main()
