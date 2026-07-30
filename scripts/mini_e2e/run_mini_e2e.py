#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
import sys

STAGES = [
    "preflight",
    "residual",
    "prepare-adjoint",
    "adjoint-x",
    "adjoint-y",
    "adjoint-z",
    "adjoint-all",
    "gradient",
    "candidates",
    "candidate-forward",
    "accept",
    "status",
    "next-forward-prepare",
    "next-forward",
    "next-forward-status",
    "all",
]


def command(module, config, *extra):
    return [sys.executable, "-m", module, "--config", config, *extra]


def run(cmd):
    print("\n" + "=" * 100)
    print(" ".join(cmd))
    print("=" * 100 + "\n", flush=True)
    proc = subprocess.run(cmd)
    if proc.returncode != 0:
        raise SystemExit(proc.returncode)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/fathi_mini_e2e_3600.json")
    parser.add_argument("--stage", choices=STAGES, required=True)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--np", type=int)
    parser.add_argument("--allow-non-descent", action="store_true")
    args = parser.parse_args()
    force = ["--force"] if args.force else []
    np_args = ["--np", str(args.np)] if args.np else []

    commands = {
        "preflight": [command("scripts.mini_e2e.preflight", args.config)],
        "residual": [command("scripts.mini_e2e.build_residual", args.config, *force)],
        "prepare-adjoint": [command("scripts.mini_e2e.prepare_adjoint", args.config, *force)],
        "adjoint-x": [command("scripts.mini_e2e.run_adjoint", args.config, "--component", "x", *np_args, *force)],
        "adjoint-y": [command("scripts.mini_e2e.run_adjoint", args.config, "--component", "y", *np_args, *force)],
        "adjoint-z": [command("scripts.mini_e2e.run_adjoint", args.config, "--component", "z", *np_args, *force)],
        "adjoint-all": [command("scripts.mini_e2e.run_adjoint", args.config, "--component", "all", *np_args, *force)],
        "gradient": [
            command("scripts.mini_e2e.build_manifests", args.config),
            command("scripts.mini_e2e.compute_rhs", args.config, "--component", "x"),
            command("scripts.mini_e2e.compute_rhs", args.config, "--component", "y"),
            command("scripts.mini_e2e.compute_rhs", args.config, "--component", "z"),
            command("scripts.mini_e2e.assemble_rhs", args.config),
            command("scripts.mini_e2e.solve_mtilde", args.config),
        ],
        "candidates": [
            command("scripts.mini_e2e.generate_candidates", args.config, *force),
            command("scripts.mini_e2e.prepare_candidate_workspaces", args.config, *force),
        ],
        "candidate-forward": [command("scripts.mini_e2e.run_candidates", args.config, *np_args, *force)],
        "accept": [
            command(
                "scripts.mini_e2e.misfit_accept",
                args.config,
                *force,
                *(["--allow-non-descent"] if args.allow_non_descent else []),
            )
        ],
        "status": [command("scripts.mini_e2e.status", args.config)],
        "next-forward-prepare": [
            command(
                "scripts.mini_e2e.next_forward",
                args.config,
                "--prepare-only",
                *force,
            )
        ],
        "next-forward": [
            command(
                "scripts.mini_e2e.next_forward",
                args.config,
                *np_args,
                *force,
            )
        ],
        "next-forward-status": [
            command(
                "scripts.mini_e2e.next_forward",
                args.config,
                "--status-only",
            )
        ],
    }
    if args.stage == "all":
        sequence = ["preflight", "residual", "prepare-adjoint", "adjoint-all", "gradient", "candidates", "candidate-forward", "accept", "status"]
        for stage in sequence:
            for cmd in commands[stage]:
                run(cmd)
    else:
        for cmd in commands[args.stage]:
            run(cmd)


if __name__ == "__main__":
    main()
