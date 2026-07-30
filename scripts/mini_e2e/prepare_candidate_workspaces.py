#!/usr/bin/env python3
from __future__ import annotations

import argparse
from datetime import datetime
import re
import shutil

import h5py

from .common import (
    copy_static_workspace,
    load_config,
    output_paths,
    replace_run_name,
    require,
    resolve,
    set_dudx,
    write_json,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/fathi_mini_e2e_3600.json")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    root, cfg, _ = load_config(args.config)
    paths = output_paths(root, cfg)
    parent = resolve(root, cfg["parent_accepted_dir"])
    residual_h5 = paths["residual_dir"] / "mini_residual_timeseries.h5"
    require(residual_h5.is_file(), f"Missing residual file: {residual_h5}")

    positions = []
    with h5py.File(residual_h5, "r") as handle:
        for name in sorted(handle.keys()):
            obj = handle[name]
            if isinstance(obj, h5py.Group) and re.fullmatch(r"station_\d{4}", name):
                positions.append(tuple(float(value) for value in obj["position"][()]))
    require(len(positions) == int(cfg["observation_subset"]["expected_count"]), "Bad candidate receiver count")
    station_text = "".join(f"{x:.6f} {y:.6f} {z:.1f}\n" for x, y, z in positions)

    records = []
    for candidate_dir in sorted(paths["candidate_dir"].glob("mini_line_search_pos_mtilde_*")):
        candidate_h5 = candidate_dir / "mat/h5"
        state_files = sorted(candidate_dir.glob("*_state_candidate.npz"))
        require(candidate_h5.is_dir() and len(state_files) == 1, f"Incomplete candidate: {candidate_dir}")
        workspace = paths["candidate_ws_root"] / candidate_dir.name
        if workspace.exists() and not args.force:
            raise RuntimeError(f"Candidate workspace exists: {workspace}; use --force to replace")
        copy_static_workspace(parent, workspace, symlink_sem=True)
        shutil.rmtree(workspace / "mat")
        shutil.copytree(candidate_dir / "mat", workspace / "mat")
        (workspace / "stations.txt").write_text(station_text, encoding="utf-8")
        text = (parent / "input.spec").read_text(encoding="utf-8")
        text = replace_run_name(text, f"mini_candidate_{candidate_dir.name}")
        text = re.sub(r"save_snap\s*=\s*(?:true|false)\s*;", "save_snap = false;", text, count=1)
        text = set_dudx(text, False)
        (workspace / "input.spec").write_text(text, encoding="utf-8")
        marker = {
            "created": datetime.now().isoformat(),
            "candidate": candidate_dir.name,
            "workspace": str(workspace),
            "receiver_count": len(positions),
            "dudx": False,
            "sem_directory_is_symlink": (workspace / "sem").is_symlink(),
            "state": str(state_files[0]),
            "result": "PASS",
        }
        write_json(workspace / "MINI_CANDIDATE_PREPARATION.json", marker)
        records.append(marker)

    require(len(records) == len(cfg["line_search_steps_mpa"]), f"Prepared {len(records)} candidates")
    write_json(paths["report_dir"] / "candidate_workspace_preparation.json", {"records": records, "result": "PASS"})
    for record in records:
        print(f"{record['candidate']}: receivers={record['receiver_count']} workspace={record['workspace']}")
    print("RESULT = PASS_MINI_CANDIDATE_WORKSPACES")


if __name__ == "__main__":
    main()
