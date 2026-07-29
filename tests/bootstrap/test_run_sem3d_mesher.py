from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
RUNNER = REPO_ROOT / "scripts" / "bootstrap" / "run_sem3d_mesher.py"
CONFIG = REPO_ROOT / "configs" / "fathi_reduced_3x3_12p5.json"


class RunSem3DMesherTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory(prefix="test_sem3d_mesher_")
        self.root = Path(self.temp.name)
        self.workspace = self.root / "workspace"
        self.workspace.mkdir()
        (self.workspace / "mesh.input").write_text("12\n1\n", encoding="utf-8")
        (self.workspace / "mat.dat").write_text("mat\n", encoding="utf-8")
        (self.workspace / "mater.in").write_text("mater\n", encoding="utf-8")

    def tearDown(self) -> None:
        self.temp.cleanup()

    def command(self, *extra: str) -> list[str]:
        return [
            sys.executable,
            str(RUNNER),
            "--config",
            str(CONFIG),
            "--workspace",
            str(self.workspace),
            *extra,
        ]

    def run_command(self, *extra: str) -> subprocess.CompletedProcess[str]:
        env = os.environ.copy()
        env["FATHI_BENCHMARK_ROOT"] = str(REPO_ROOT)
        return subprocess.run(
            self.command(*extra),
            text=True,
            capture_output=True,
            env=env,
            check=False,
        )

    def fake_mesher(self, partition_count: int = 12, return_code: int = 0) -> Path:
        path = self.root / "fake_mesher.py"
        script = f"""#!/usr/bin/env python3
import pathlib
import sys

workspace = pathlib.Path.cwd()
_ = sys.stdin.read()
for index in range({partition_count}):
    (workspace / f\"mesh4spec.{{index:04d}}.h5\").write_bytes(b\"H5\" + bytes([index]))
for name in (
    \"mesh4spec.elems.xmf\",
    \"mesh4spec.faces.xmf\",
    \"mesh4spec.edges.xmf\",
    \"mesh4spec.mirror.xmf\",
    \"mesh4spec.comms.faces.xmf\",
    \"mesh4spec.comms.edges.xmf\",
):
    (workspace / name).write_text(\"<Xdmf/>\\n\", encoding=\"utf-8\")
(workspace / \"domains.txt\").write_text(\"domains\\n\", encoding=\"utf-8\")
print(\"fake mesher completed\")
raise SystemExit({return_code})
"""
        path.write_text(script, encoding="utf-8")
        path.chmod(0o755)
        return path

    def create_complete_outputs(self, directory: Path, partition_count: int = 12) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        for index in range(partition_count):
            (directory / f"mesh4spec.{index:04d}.h5").write_bytes(b"H5")
        for name in (
            "mesh4spec.elems.xmf",
            "mesh4spec.faces.xmf",
            "mesh4spec.edges.xmf",
            "mesh4spec.mirror.xmf",
            "mesh4spec.comms.faces.xmf",
            "mesh4spec.comms.edges.xmf",
        ):
            (directory / name).write_text("<Xdmf/>\n", encoding="utf-8")
        (directory / "domains.txt").write_text("domains\n", encoding="utf-8")

    def test_plan_only_does_not_create_outputs(self) -> None:
        result = self.run_command("--mesher", str(self.root / "not-needed"))
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("RESULT = PASS_SEM3D_MESHER_PLAN", result.stdout)
        self.assertFalse((self.workspace / "sem").exists())
        self.assertEqual(list(self.workspace.glob("mesh4spec*.h5")), [])

    def test_execute_creates_organized_and_audited_twelve_partitions(self) -> None:
        mesher = self.fake_mesher()
        result = self.run_command(
            "--mesher",
            str(mesher),
            "--timeout-seconds",
            "30",
            "--execute",
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn(
            "RESULT = PASS_SEM3D_MESHER_EXECUTION_AND_AUDIT",
            result.stdout,
        )
        self.assertEqual(len(list((self.workspace / "sem").glob("mesh4spec.*.h5"))), 12)
        self.assertEqual(list(self.workspace.glob("mesh4spec*.h5")), [])
        self.assertTrue((self.workspace / "sem" / "domains.txt").is_file())

        manifest = json.loads(
            (self.workspace / "logs" / "mesher_manifest.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertEqual(manifest["schema_version"], 2)
        self.assertEqual(manifest["solver_mesh_directory"], str(self.workspace / "sem"))
        self.assertTrue(manifest["audit"]["passed"])
        self.assertEqual(manifest["audit"]["partition_count_actual"], 12)
        self.assertGreater(len(manifest["relocated_outputs"]), 12)

    def test_incomplete_partition_set_fails_audit(self) -> None:
        mesher = self.fake_mesher(partition_count=11)
        result = self.run_command(
            "--mesher",
            str(mesher),
            "--execute",
        )
        self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
        self.assertIn("RESULT = FAIL_SEM3D_MESHER_OUTPUT_AUDIT", result.stdout)
        self.assertIn("mesh4spec.0011.h5", result.stdout)

    def test_existing_outputs_require_overwrite(self) -> None:
        sem = self.workspace / "sem"
        sem.mkdir()
        stale = sem / "mesh4spec.0000.h5"
        stale.write_bytes(b"stale")
        mesher = self.fake_mesher()

        refused = self.run_command(
            "--mesher",
            str(mesher),
            "--execute",
        )
        self.assertEqual(refused.returncode, 2)
        self.assertIn("Refusing to mix stale and fresh files", refused.stderr)

        replaced = self.run_command(
            "--mesher",
            str(mesher),
            "--execute",
            "--overwrite",
        )
        self.assertEqual(replaced.returncode, 0, replaced.stdout + replaced.stderr)
        self.assertNotEqual(stale.read_bytes(), b"stale")

    def test_missing_required_input_is_rejected(self) -> None:
        (self.workspace / "mater.in").unlink()
        result = self.run_command()
        self.assertEqual(result.returncode, 2)
        self.assertIn("Required mesher input not found", result.stderr)

    def test_audit_only_accepts_complete_organized_outputs(self) -> None:
        self.create_complete_outputs(self.workspace / "sem")
        audited = self.run_command("--audit-only")
        self.assertEqual(audited.returncode, 0, audited.stdout + audited.stderr)
        self.assertIn("RESULT = PASS_SEM3D_MESHER_OUTPUT_AUDIT", audited.stdout)

    def test_audit_rejects_complete_outputs_left_in_workspace_root(self) -> None:
        self.create_complete_outputs(self.workspace)
        audited = self.run_command("--audit-only")
        self.assertEqual(audited.returncode, 1, audited.stdout + audited.stderr)
        self.assertIn("RESULT = FAIL_SEM3D_MESHER_OUTPUT_AUDIT", audited.stdout)
        self.assertIn("stray_root_outputs", audited.stdout)


if __name__ == "__main__":
    unittest.main()
