#!/usr/bin/env python3
from __future__ import annotations

import argparse
from datetime import datetime
import json
import shutil
from pathlib import Path

import h5py
import numpy as np

from .common import (
    copy_static_workspace,
    load_config,
    load_displacement,
    output_paths,
    require,
    trace_position_map,
    write_json,
)


def candidate_j(trace_dir: Path, observed_map, selected):
    synthetic_map = trace_position_map(trace_dir)
    missing = [key for key in selected if key not in synthetic_map]
    require(not missing, f"Candidate missing selected receivers: {missing[:10]}")
    total = 0.0
    records = []
    for key in selected:
        t_obs, u_obs = load_displacement(observed_map[key])
        t_syn, u_syn = load_displacement(synthetic_map[key])
        t0 = max(float(t_obs[0]), float(t_syn[0]))
        t1 = min(float(t_obs[-1]), float(t_syn[-1]))
        mask = (t_obs >= t0) & (t_obs <= t1)
        t_eval = t_obs[mask]
        u_true = u_obs[mask]
        u_sim = np.column_stack([np.interp(t_eval, t_syn, u_syn[:, index]) for index in range(3)])
        residual = u_sim - u_true
        local = 0.5 * float(np.trapezoid(np.sum(residual * residual, axis=1), t_eval))
        total += local
        records.append({"position": key, "local_J": local})
    return total, records


def clone_accepted(workspace: Path, destination: Path) -> None:
    copy_static_workspace(workspace, destination, symlink_sem=True)
    if (workspace / "traces").is_dir():
        shutil.copytree(workspace / "traces", destination / "traces")
    if (workspace / "fin_sem").is_file():
        shutil.copy2(workspace / "fin_sem", destination / "fin_sem")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/fathi_mini_e2e_3600.json")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--allow-non-descent", action="store_true")
    args = parser.parse_args()
    root, cfg, _ = load_config(args.config)
    paths = output_paths(root, cfg)
    residual_summary = json.loads((paths["residual_dir"] / "mini_residual_summary.json").read_text(encoding="utf-8"))
    parent_j = float(residual_summary["parent_J"])
    observed_dir = Path(residual_summary["observed_trace_dir"])
    observed_map = trace_position_map(observed_dir)
    with h5py.File(paths["residual_dir"] / "mini_residual_timeseries.h5", "r") as handle:
        selected = [tuple(float(value) for value in handle[name]["position"][()]) for name in sorted(handle.keys()) if isinstance(handle[name], h5py.Group)]

    paths["misfit_dir"].mkdir(parents=True, exist_ok=True)
    records = []
    for workspace in sorted(paths["candidate_ws_root"].glob("mini_line_search_pos_mtilde_*")):
        require((workspace / "fin_sem").is_file() and (workspace / "fin_sem").read_text(errors="ignore").strip() == "1", f"Candidate not complete: {workspace}")
        value, per_receiver = candidate_j(workspace / "traces", observed_map, selected)
        record = {
            "candidate": workspace.name,
            "workspace": str(workspace),
            "parent_J": parent_j,
            "candidate_J": value,
            "delta_J": value - parent_j,
            "descent": value < parent_j,
            "per_receiver": per_receiver,
        }
        records.append(record)
        write_json(paths["misfit_dir"] / f"{workspace.name}_misfit.json", record)

    require(bool(records), "No candidate results")
    best = min(records, key=lambda item: item["candidate_J"])
    accept = bool(best["descent"] or args.allow_non_descent)
    summary = {
        "created": datetime.now().isoformat(),
        "parent_J": parent_j,
        "records": records,
        "best": best,
        "allow_non_descent": bool(args.allow_non_descent),
        "accepted": accept,
        "result": "PASS_ACCEPTED" if accept else "CHECK_NO_DESCENT",
    }

    if accept:
        accepted = paths["accepted_dir"]
        if accepted.exists() and not args.force:
            raise RuntimeError(f"Accepted output exists: {accepted}; use --force to replace")
        clone_accepted(Path(best["workspace"]), accepted)
        candidate_dir = paths["candidate_dir"] / best["candidate"]
        states = sorted(candidate_dir.glob("*_state_candidate.npz"))
        require(len(states) == 1, f"Candidate state missing: {candidate_dir}")
        paths["state_dir"].mkdir(parents=True, exist_ok=True)
        state_out = paths["state_dir"] / "iter_001_state_v2_corrected.npz"
        if state_out.exists() and not args.force:
            raise RuntimeError(f"State output exists: {state_out}; use --force to replace")
        shutil.copy2(states[0], state_out)
        marker = {
            "created": datetime.now().isoformat(),
            "profile": cfg["name"],
            "candidate": best["candidate"],
            "parent_J": parent_j,
            "candidate_J": best["candidate_J"],
            "delta_J": best["delta_J"],
            "descent": best["descent"],
            "accepted_dir": str(accepted),
            "state": str(state_out),
            "result": "PASS",
        }
        write_json(accepted / "MINI_ITER001_ACCEPTANCE.json", marker)
        summary["accepted_dir"] = str(accepted)
        summary["state_out"] = str(state_out)

    write_json(paths["misfit_dir"] / "mini_candidate_selection.json", summary)
    print(f"parent_J = {parent_j:.16e}")
    for record in records:
        print(f"{record['candidate']}: J={record['candidate_J']:.16e} delta={record['delta_J']:.16e} descent={record['descent']}")
    print(f"best = {best['candidate']}")
    print(f"accepted = {accept}")
    print(f"RESULT = {summary['result']}")
    if not accept:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
