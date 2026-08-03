from __future__ import annotations

from datetime import datetime
from pathlib import Path
import argparse
import json
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[2]


def resolve(path: str) -> Path:
    value = Path(path).expanduser()
    return value.resolve() if value.is_absolute() else (ROOT / value).resolve()


def run(command: list[str]) -> None:
    print()
    print("$ " + " ".join(command))
    subprocess.run(command, cwd=ROOT, check=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--label", default=None)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--j-data", type=float, default=None)
    parser.add_argument("--j-data-file", default=None)
    parser.add_argument("--skip-diagnose", action="store_true")
    args = parser.parse_args()

    config_path = resolve(args.config)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    label = args.label or config.get("transition", "tv_parent")

    scripts = ROOT / "scripts/regularization"
    python = sys.executable
    commands = [
        [
            python,
            str(scripts / "02_compute_tv_q1_full_grid.py"),
            "--config",
            str(config_path),
            "--label",
            label,
        ],
        [
            python,
            str(scripts / "04_restrict_tv_rhs_to_active.py"),
            "--config",
            str(config_path),
            "--label",
            label,
        ],
        [
            python,
            str(scripts / "05_assemble_data_tv_rhs.py"),
            "--config",
            str(config_path),
            "--label",
            label,
        ],
        [
            python,
            str(scripts / "06_solve_mtilde_data_tv_rhs.py"),
            "--config",
            str(config_path),
            "--label",
            label,
        ],
    ]

    manifest = {
        "created": datetime.now().isoformat(),
        "config": str(config_path),
        "label": label,
        "execute": args.execute,
        "commands": commands,
        "sem3d_launched": False,
        "accepted_state_mutated": False,
    }
    manifest_path = (
        resolve(config["tv_transition_dir"])
        / f"tv_pipeline_{label}.json"
    )
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    if not args.execute:
        run(commands[0])
        print()
        print(f"manifest = {manifest_path}")
        print("No TV arrays were assembled.")
        print("Use --execute after reviewing the plan.")
        print("RESULT = PASS_TV_PIPELINE_PLAN")
        return

    commands[0].append("--execute")
    for command in commands:
        run(command)

    if not args.skip_diagnose:
        diagnose = [
            python,
            str(scripts / "07_diagnose_tv_weight.py"),
            "--config",
            str(config_path),
            "--label",
            label,
        ]
        if args.j_data is not None:
            diagnose.extend(["--j-data", str(args.j_data)])
        if args.j_data_file is not None:
            diagnose.extend(["--j-data-file", str(resolve(args.j_data_file))])
        run(diagnose)

    print()
    print(f"manifest = {manifest_path}")
    print("SEM3D was not launched.")
    print("Accepted state was not modified.")
    print("RESULT = PASS_TV_PIPELINE")


if __name__ == "__main__":
    main()
