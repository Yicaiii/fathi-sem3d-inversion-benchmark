"""No-simulation structural dry-run for the generic iteration orchestrator."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


REPOSITORY_IMPORT_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_IMPORT_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_IMPORT_ROOT))

from scripts.fathi_benchmark.generic_iteration_runner import (
    GenericIterationRunner,
)
from scripts.fathi_benchmark.lbfgs_history import waiting_for_gradient_status


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", required=True)
    parser.add_argument("--runtime-config", required=True)
    parser.add_argument("--engine-config", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--parent", type=int, required=True)
    parser.add_argument("--child", type=int, required=True)
    parser.add_argument("--output", required=True)
    gradient = parser.add_mutually_exclusive_group()
    gradient.add_argument("--current-parent-gradient-manifest")
    gradient.add_argument("--current-parent-gradient-status")
    args = parser.parse_args()
    runner = GenericIterationRunner.from_config_files(
        run_id=args.run_id,
        parent_iteration=args.parent,
        child_iteration=args.child,
        repository_root=args.repo,
        runtime_config_path=args.runtime_config,
        engine_config_path=args.engine_config,
    )
    payload = runner.dry_run_summary()
    current_parent_gradient = None
    if args.current_parent_gradient_manifest:
        gradient_path = Path(args.current_parent_gradient_manifest).expanduser()
        if not gradient_path.is_absolute():
            gradient_path = Path(args.repo).expanduser().resolve() / gradient_path
        current_parent_gradient = json.loads(
            gradient_path.resolve().read_text(encoding="utf-8")
        )
        if not isinstance(current_parent_gradient, dict):
            raise ValueError("current-parent gradient manifest must be an object")
        payload["current_parent_gradient_input"] = str(gradient_path.resolve())
        payload["history"] = runner.newest_history_preflight(
            current_parent_gradient=current_parent_gradient
        )
    elif args.current_parent_gradient_status:
        expected = waiting_for_gradient_status(args.parent)
        if args.parent <= 0 or args.current_parent_gradient_status != expected:
            raise ValueError(
                "a status-only gradient input must be the dynamic missing-gradient status"
            )
        payload["current_parent_gradient_input"] = args.current_parent_gradient_status
        payload["history"] = {
            "status": args.current_parent_gradient_status,
            "reason": "current-parent gradient status supplied to structural dry-run",
            "history_pair_created": False,
            "requested_pair": {
                "from_iteration": args.parent - 1,
                "to_iteration": args.parent,
            },
        }
    else:
        payload["current_parent_gradient_input"] = "not supplied"
        payload["history"] = runner.newest_history_preflight(
            current_parent_gradient=current_parent_gradient
        )
    payload["next_optimization_runnable"] = (
        runner.optimization_structurally_runnable(payload["history"])
    )
    output = Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(output)


if __name__ == "__main__":
    main()
