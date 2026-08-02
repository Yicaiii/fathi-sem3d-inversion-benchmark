from __future__ import annotations

from datetime import datetime
from pathlib import Path
import argparse
import json
import subprocess
import sys

from scripts.fathi_benchmark.runtime_paths import repository_root, resolve_path, runtime_resolve_path

ROOT = repository_root()


def context_from_iter(iter_k: int, config_path: Path) -> Path:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    transition = f"iter_{iter_k:03d}_to_iter_{iter_k + 1:03d}"
    result_root = runtime_resolve_path(config["run_result_root"], repo_root=ROOT)
    return result_root / transition / f"{transition}_iteration_context.json"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--context", default=None)
    parser.add_argument("--iter-k", type=int, default=None)
    parser.add_argument("--config", default="benchmark_fathi_strict/config/benchmark_config.json")
    parser.add_argument("--stage", choices=["plan", "regularization", "candidates", "acceptance-plan", "all-light"], default="plan")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--tv-config", default=None)
    parser.add_argument("--label", default=None)
    parser.add_argument("--alpha-lambda", type=float, default=0.0)
    parser.add_argument("--alpha-mu", type=float, default=0.0)
    parser.add_argument("--steps-mpa", nargs="+", type=float, default=[0.10, 0.25, 0.50, 1.00])
    parser.add_argument("--parent-j-data", type=float, default=None)
    parser.add_argument("--candidate-j-data", type=float, default=None)
    parser.add_argument("--parent-tv-json", default=None)
    parser.add_argument("--candidate-tv-json", default=None)
    args = parser.parse_args()

    config_path = resolve_path(args.config, base=ROOT)
    context_path = resolve_path(args.context, base=ROOT) if args.context else context_from_iter(args.iter_k, config_path)
    if not context_path.exists():
        raise SystemExit(f"Missing context: {context_path}")
    context = json.loads(context_path.read_text(encoding="utf-8"))
    transition = context["transition"]
    tv_root = Path(context["tv_regularization_dir"]).resolve()
    tv_config = Path(args.tv_config).expanduser().resolve() if args.tv_config else tv_root / "tv_config.json"
    label = args.label or transition
    python = sys.executable

    commands = {
        "build-config": [python, "-m", "scripts.regularization.build_tv_config_from_context", "--context", str(context_path), "--strict-config", str(config_path), "--alpha-lambda", str(args.alpha_lambda), "--alpha-mu", str(args.alpha_mu), "--output", str(tv_config)],
        "regularization": [python, "-m", "scripts.regularization.run_tv_pipeline", "--config", str(tv_config), "--label", label],
        "candidates": [python, "-m", "scripts.regularization.08_generate_tv_candidates_from_mtilde_gradient", "--config", str(tv_config), "--label", label, "--steps-mpa", *[str(v) for v in args.steps_mpa]],
    }
    if args.execute:
        commands["regularization"].append("--execute")

    acceptance = None
    if all(v is not None for v in [args.parent_j_data, args.candidate_j_data, args.parent_tv_json, args.candidate_tv_json]):
        acceptance = [python, "-m", "scripts.regularization.total_objective", "--parent-j-data", str(args.parent_j_data), "--candidate-j-data", str(args.candidate_j_data), "--parent-tv-json", str(Path(args.parent_tv_json).expanduser().resolve()), "--candidate-tv-json", str(Path(args.candidate_tv_json).expanduser().resolve()), "--alpha-lambda", str(args.alpha_lambda), "--alpha-mu", str(args.alpha_mu), "--output", str(tv_root / "total_objective_acceptance.json")]
    commands["acceptance-plan"] = acceptance

    manifest = {
        "created": datetime.now().isoformat(),
        "transition": transition,
        "context": str(context_path),
        "stage": args.stage,
        "execute": args.execute,
        "tv_config": str(tv_config),
        "label": label,
        "commands": commands,
        "sem3d_launched": False,
        "accepted_state_mutated": False,
        "physical_descent_validated": False,
    }
    tv_root.mkdir(parents=True, exist_ok=True)
    manifest_path = tv_root / f"{transition}_tv_iteration_plan.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print("TV iteration stage")
    print("==================")
    print(f"transition = {transition}")
    print(f"stage = {args.stage}")
    print(f"execute = {args.execute}")
    print(f"tv_config = {tv_config}")
    print("SEM3D launched = False")
    print("accepted state mutated = False")
    for name, command in commands.items():
        print(f"{name} = {' '.join(command) if command else 'NOT_CONFIGURED'}")

    if args.stage == "plan" or not args.execute:
        print(f"manifest = {manifest_path}")
        print("RESULT = PASS_TV_ITERATION_PLAN")
        return

    sequence = [args.stage]
    if args.stage == "all-light":
        sequence = ["build-config", "regularization", "candidates"]
        if acceptance is not None:
            sequence.append("acceptance-plan")
    elif args.stage == "regularization":
        sequence = ["build-config", "regularization"]

    for name in sequence:
        command = commands.get(name)
        if command is None:
            raise SystemExit(f"Stage {name} is not configured")
        print("$ " + " ".join(command))
        subprocess.run(command, cwd=ROOT, check=True)

    print("SEM3D launched = False")
    print("accepted state mutated = False")
    print("RESULT = PASS_TV_ITERATION_LIGHT")


if __name__ == "__main__":
    main()
