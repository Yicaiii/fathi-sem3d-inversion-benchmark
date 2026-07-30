#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json

from .common import load_config, output_paths


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/fathi_mini_e2e_3600.json")
    args = parser.parse_args()
    root, cfg, _ = load_config(args.config)
    paths = output_paths(root, cfg)
    checks = {
        "preflight": paths["report_dir"] / "preflight.json",
        "residual": paths["residual_dir"] / "mini_residual_summary.json",
        "adjoint_preparation": paths["report_dir"] / "adjoint_preparation.json",
        "adjoint_x": paths["report_dir"] / "adjoint_run_x.json",
        "adjoint_y": paths["report_dir"] / "adjoint_run_y.json",
        "adjoint_z": paths["report_dir"] / "adjoint_run_z.json",
        "manifests": paths["report_dir"] / "trace_manifests.json",
        "rhs_x": paths["component_rhs_dir"] / "mini_RHS_x_summary.json",
        "rhs_y": paths["component_rhs_dir"] / "mini_RHS_y_summary.json",
        "rhs_z": paths["component_rhs_dir"] / "mini_RHS_z_summary.json",
        "rhs_total": paths["component_rhs_dir"] / "mini_RHS_total_summary.json",
        "mtilde": paths["mtilde_dir"] / "mini_mtilde_summary.json",
        "candidates": paths["report_dir"] / "candidate_generation.json",
        "candidate_workspaces": paths["report_dir"] / "candidate_workspace_preparation.json",
        "candidate_runs": paths["report_dir"] / "candidate_runs_summary.json",
        "selection": paths["misfit_dir"] / "mini_candidate_selection.json",
        "iter001_accepted": paths["accepted_dir"] / "MINI_ITER001_ACCEPTANCE.json",
        "iter001_state": paths["state_dir"] / "iter_001_state_v2_corrected.npz",
    }
    print("MINI E2E STATUS")
    print("===============")
    complete = True
    for name, path in checks.items():
        exists = path.exists()
        result = ""
        if exists and path.suffix == ".json":
            try:
                result = str(json.loads(path.read_text(encoding="utf-8")).get("result", ""))
            except Exception:
                result = "UNREADABLE"
        print(f"{name:24s} exists={str(exists):5s} result={result:20s} path={path}")
        if name in {"iter001_accepted", "iter001_state"} and not exists:
            complete = False
    print()
    print("RESULT = COMPLETE_MINI_ITER000_TO_ITER001" if complete else "RESULT = MINI_E2E_IN_PROGRESS")


if __name__ == "__main__":
    main()
