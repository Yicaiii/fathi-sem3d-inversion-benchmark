from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


SCRIPT = (
    Path(__file__).resolve().parents[2]
    / "scripts"
    / "bootstrap"
    / "bootstrap_fathi_benchmark.py"
)
SPEC = importlib.util.spec_from_file_location("bootstrap_fathi_benchmark", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class BootstrapFathiBenchmarkTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        (self.root / "configs").mkdir(parents=True)
        (self.root / "scripts" / "bootstrap").mkdir(parents=True)
        self.config = self.root / "configs" / "fathi_reduced_3x3_12p5.json"
        self.config.write_text(
            json.dumps({"name": "fathi_reduced_3x3_12p5"}),
            encoding="utf-8",
        )
        for name in (
            "generate_sem3d_workspace.py",
            "run_sem3d_mesher.py",
            "run_sem3d_solver.py",
        ):
            (self.root / "scripts" / "bootstrap" / name).write_text(
                "#!/usr/bin/env python3\n",
                encoding="utf-8",
            )
        self.output = self.root / "outputs" / "bootstrap"

    def tearDown(self) -> None:
        self.temp.cleanup()

    def args(self, *extra: str) -> list[str]:
        return [
            "--config",
            str(self.config),
            "--output-root",
            str(self.output),
            *extra,
        ]

    def fake_success(self, command: list[str], *, cwd: Path) -> tuple[int, float]:
        del cwd
        if "--output" in command:
            workspace = Path(command[command.index("--output") + 1])
            workspace.mkdir(parents=True, exist_ok=True)
        return 0, 0.01

    @patch.object(MODULE, "repository_root")
    @patch.object(MODULE, "run_command")
    def test_plan_only_does_not_execute_or_create_output(self, run_mock, root_mock) -> None:
        root_mock.return_value = self.root
        result = MODULE.main(self.args())
        self.assertEqual(result, 0)
        run_mock.assert_not_called()
        self.assertFalse(self.output.exists())

    @patch.object(MODULE, "repository_root")
    @patch.object(MODULE, "run_command")
    def test_execute_both_models_runs_ten_steps(self, run_mock, root_mock) -> None:
        root_mock.return_value = self.root
        run_mock.side_effect = self.fake_success
        result = MODULE.main(self.args("--execute"))
        self.assertEqual(result, 0)
        self.assertEqual(run_mock.call_count, 10)
        manifest = json.loads((self.output / "bootstrap_manifest.json").read_text())
        self.assertEqual(manifest["status"], "passed")
        self.assertEqual(len(manifest["steps"]), 10)
        self.assertEqual(
            manifest["models"],
            ["true_layered", "initial_homogeneous"],
        )

    @patch.object(MODULE, "repository_root")
    @patch.object(MODULE, "run_command")
    def test_true_only_runs_five_steps(self, run_mock, root_mock) -> None:
        root_mock.return_value = self.root
        run_mock.side_effect = self.fake_success
        result = MODULE.main(self.args("--model", "true_layered", "--execute"))
        self.assertEqual(result, 0)
        self.assertEqual(run_mock.call_count, 5)
        manifest = json.loads((self.output / "bootstrap_manifest.json").read_text())
        self.assertEqual(manifest["models"], ["true_layered"])

    @patch.object(MODULE, "repository_root")
    @patch.object(MODULE, "run_command")
    def test_smoke_seconds_and_np_are_forwarded_to_solver(self, run_mock, root_mock) -> None:
        root_mock.return_value = self.root
        run_mock.side_effect = self.fake_success
        result = MODULE.main(
            self.args(
                "--model",
                "true_layered",
                "--smoke-seconds",
                "0.012",
                "--np",
                "12",
                "--execute",
            )
        )
        self.assertEqual(result, 0)
        commands = [call.args[0] for call in run_mock.call_args_list]
        solver = next(command for command in commands if "run_sem3d_solver.py" in command[1] and "--execute" in command)
        self.assertIn("--smoke-seconds", solver)
        self.assertEqual(solver[solver.index("--smoke-seconds") + 1], "0.012")
        self.assertEqual(solver[solver.index("--np") + 1], "12")

    @patch.object(MODULE, "repository_root")
    @patch.object(MODULE, "run_command")
    def test_failure_stops_pipeline_and_writes_failed_manifest(self, run_mock, root_mock) -> None:
        root_mock.return_value = self.root
        calls = 0

        def fail_second(command: list[str], *, cwd: Path) -> tuple[int, float]:
            nonlocal calls
            calls += 1
            if "--output" in command:
                Path(command[command.index("--output") + 1]).mkdir(parents=True, exist_ok=True)
            return (7, 0.02) if calls == 2 else (0, 0.01)

        run_mock.side_effect = fail_second
        result = MODULE.main(self.args("--execute"))
        self.assertEqual(result, 7)
        self.assertEqual(run_mock.call_count, 2)
        manifest = json.loads((self.output / "bootstrap_manifest.json").read_text())
        self.assertEqual(manifest["status"], "failed")
        self.assertEqual(len(manifest["steps"]), 2)
        self.assertFalse(manifest["steps"][-1]["passed"])

    @patch.object(MODULE, "repository_root")
    @patch.object(MODULE, "run_command")
    def test_existing_output_requires_overwrite(self, run_mock, root_mock) -> None:
        root_mock.return_value = self.root
        self.output.mkdir(parents=True)
        with self.assertRaises(FileExistsError):
            MODULE.main(self.args("--execute"))
        run_mock.assert_not_called()

    @patch.object(MODULE, "repository_root")
    @patch.object(MODULE, "run_command")
    def test_overwrite_replaces_existing_output(self, run_mock, root_mock) -> None:
        root_mock.return_value = self.root
        self.output.mkdir(parents=True)
        marker = self.output / "stale.txt"
        marker.write_text("stale", encoding="utf-8")
        run_mock.side_effect = self.fake_success
        result = MODULE.main(self.args("--model", "true_layered", "--execute", "--overwrite"))
        self.assertEqual(result, 0)
        self.assertFalse(marker.exists())
        self.assertTrue((self.output / "bootstrap_manifest.json").is_file())

    @patch.object(MODULE, "repository_root")
    @patch.object(MODULE, "run_command")
    def test_audit_only_runs_four_read_only_steps(self, run_mock, root_mock) -> None:
        root_mock.return_value = self.root
        self.output.mkdir(parents=True)
        run_mock.return_value = (0, 0.01)
        result = MODULE.main(self.args("--audit-only"))
        self.assertEqual(result, 0)
        self.assertEqual(run_mock.call_count, 4)
        commands = [call.args[0] for call in run_mock.call_args_list]
        self.assertTrue(all("--audit-only" in command for command in commands))
        self.assertTrue(all("--execute" not in command for command in commands))

    @patch.object(MODULE, "repository_root")
    def test_invalid_argument_combinations_are_rejected(self, root_mock) -> None:
        root_mock.return_value = self.root
        with self.assertRaises(ValueError):
            MODULE.main(self.args("--audit-only", "--execute"))
        with self.assertRaises(ValueError):
            MODULE.main(self.args("--audit-only", "--smoke-seconds", "0.01"))
        with self.assertRaises(ValueError):
            MODULE.main(self.args("--np", "0"))


if __name__ == "__main__":
    unittest.main()
