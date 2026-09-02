"""Distinct manifest-driven entry point for the CURRENT iteration pipeline.

The command dispatches directly to CURRENT contracts and never imports the
historical top-level runner. Numerical stages execute only when their explicit
stage is selected; status and contract preparation are side-effect free apart
from the requested manifest write.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from scripts.fathi_benchmark.current_pipeline_artifacts import (
    evaluate_candidate_external,
    execute_current_armijo,
    generate_raw_alpha_candidate,
    promote_current_accepted_trial,
)
from scripts.fathi_benchmark.current_pipeline_contracts import (
    artifact_record,
    atomic_json,
)
from scripts.fathi_benchmark.generic_iteration_runner import GenericIterationRunner
from scripts.fathi_benchmark.external_armijo import ArmijoParameters
from scripts.fathi_benchmark.register_certified_gradient import (
    register_current_gradient,
)


STAGES = (
    "status",
    "register-gradient",
    "direction",
    "armijo-manifest",
    "candidate",
    "armijo",
    "promote",
)


def _json(path: str | Path) -> dict[str, Any]:
    source = Path(path).expanduser().resolve()
    value = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON manifest must be an object: {source}")
    return value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", default=".")
    parser.add_argument("--runtime-config", required=True)
    parser.add_argument("--engine-config", required=True)
    parser.add_argument("--parent-iteration", type=int, required=True)
    parser.add_argument("--stage", choices=STAGES, required=True)
    parser.add_argument("--request")
    args = parser.parse_args()
    repo = Path(args.repo).expanduser().resolve()
    runtime_path = Path(args.runtime_config).expanduser().resolve()
    engine_path = Path(args.engine_config).expanduser().resolve()
    runtime = _json(runtime_path)
    engine = _json(engine_path)
    parent = int(args.parent_iteration)
    runner = GenericIterationRunner.from_config_files(
        run_id=str(engine["run_id"]),
        parent_iteration=parent,
        child_iteration=parent + 1,
        repository_root=repo,
        runtime_config_path=runtime_path,
        engine_config_path=engine_path,
    )
    paths = runner.paths
    if args.stage == "status":
        print(json.dumps(runner.dry_run_summary(), indent=2))
        return
    if not args.request:
        parser.error(f"{args.stage} requires --request")
    request_path = Path(args.request).expanduser().resolve()
    request = _json(request_path)

    if args.stage == "register-gradient":
        output = register_current_gradient(repo=repo, paths=paths)
    elif args.stage == "direction":
        result = runner.compute_optimizer_direction(request)
        output = runner.persist_optimizer_direction(request, result)
    elif args.stage == "armijo-manifest":
        payload = runner.prepare_external_armijo(request)
        output = paths.line_search_root / "armijo_input.json"
        if output.is_file() and _json(output) != payload:
            raise ValueError("existing Armijo input conflicts")
        atomic_json(output, payload)
    elif args.stage == "candidate":
        output = generate_raw_alpha_candidate(
            repo=repo,
            paths=paths,
            material_config=engine["material"],
            accepted_parent_record=request["accepted_parent_record"],
            direction_record=request["direction_record"],
            parameters=ArmijoParameters.from_config(engine),
            trial_index=int(request["trial_index"]),
            alpha=float(request["alpha"]),
        )
    elif args.stage == "armijo":
        armijo_input_path = Path(request["armijo_input"]).expanduser().resolve()
        armijo = _json(armijo_input_path)

        def candidate_provider(trial_index: int, alpha: float) -> Path:
            return generate_raw_alpha_candidate(
                repo=repo,
                paths=paths,
                material_config=engine["material"],
                accepted_parent_record=armijo["parent_accepted_artifact"],
                direction_record=armijo["direction_artifact"],
                parameters=ArmijoParameters(**armijo["parameters"]),
                trial_index=trial_index,
                alpha=alpha,
            )

        def evaluation_provider(candidate: Path, trial_dir: Path):
            return evaluate_candidate_external(
                repo=repo,
                paths=paths,
                runtime_config=runtime,
                reference_manifest=request["reference_manifest"],
                candidate_summary_path=candidate,
                trial_directory=trial_dir,
                batch_size=int(request.get("batch_size", 2048)),
                checkpoint_interval=int(
                    request.get("checkpoint_interval", 100)
                ),
            )

        output = execute_current_armijo(
            repo=repo,
            paths=paths,
            armijo_manifest=armijo,
            candidate_provider=candidate_provider,
            evaluation_provider=evaluation_provider,
        )
    elif args.stage == "promote":
        armijo_record = request.get("armijo_summary_record")
        if armijo_record is None:
            armijo_record = artifact_record(
                paths.line_search_root / "armijo_summary.json", repo=repo
            )
        output = promote_current_accepted_trial(
            repo=repo,
            paths=paths,
            material_config=engine["material"],
            armijo_summary_record=armijo_record,
        )
    else:
        raise AssertionError(args.stage)
    print(f"OUTPUT = {output}")


if __name__ == "__main__":
    main()
