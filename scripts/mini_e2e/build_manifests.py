#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from datetime import datetime
from pathlib import Path

import h5py
import numpy as np

from .common import load_config, output_paths, require, resolve, sorted_uu_keys, write_json


def scan(trace_dirs: list[Path], out_csv: Path) -> dict:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    rows = 0
    files = []
    bad = []
    with out_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["batch", "trace_file", "receiver_key", "x", "y", "z", "nsteps", "ncols", "t0", "t1"],
        )
        writer.writeheader()
        for trace_dir in trace_dirs:
            for path in sorted(trace_dir.glob("capteurs.*.h5")):
                files.append(path)
                try:
                    with h5py.File(path, "r") as h5:
                        for key in sorted_uu_keys(h5):
                            pos_name = key + "_pos"
                            if pos_name not in h5:
                                bad.append((str(path), key, "missing_pos"))
                                continue
                            pos = np.asarray(h5[pos_name], dtype=np.float64).reshape(-1)
                            arr = h5[key]
                            if pos.size < 3 or len(arr.shape) != 2:
                                bad.append((str(path), key, f"bad_shape_{arr.shape}"))
                                continue
                            writer.writerow(
                                {
                                    "batch": trace_dir.parent.name,
                                    "trace_file": str(path.resolve()),
                                    "receiver_key": key,
                                    "x": float(pos[0]),
                                    "y": float(pos[1]),
                                    "z": float(pos[2]),
                                    "nsteps": int(arr.shape[0]),
                                    "ncols": int(arr.shape[1]),
                                    "t0": float(arr[0, 0]),
                                    "t1": float(arr[-1, 0]),
                                }
                            )
                            rows += 1
                except Exception as exc:
                    bad.append((str(path), "", repr(exc)))
    return {"csv": str(out_csv), "trace_file_count": len(files), "row_count": rows, "bad": bad}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/fathi_mini_e2e_3600.json")
    args = parser.parse_args()
    root, cfg, _ = load_config(args.config)
    paths = output_paths(root, cfg)
    expected = int(cfg["control_region"]["expected_count"])
    strict = resolve(root, cfg["strict_forward_workspace"])

    results = {
        "forward": scan([strict / "traces"], paths["manifest_dir"] / "forward_control_manifest.csv"),
        "adjoint": {},
    }
    for component in cfg["adjoint"]["components"]:
        trace_dir = paths["adjoint_root"] / component / "batch_000" / "traces"
        results["adjoint"][component] = scan(
            [trace_dir], paths["manifest_dir"] / f"adjoint_{component}_control_manifest.csv"
        )

    checks = [results["forward"]["row_count"] == expected, not results["forward"]["bad"]]
    for component in cfg["adjoint"]["components"]:
        record = results["adjoint"][component]
        checks.extend([record["row_count"] == expected, not record["bad"]])
    require(all(checks), f"Manifest checks failed: {results}")

    payload = {"created": datetime.now().isoformat(), "expected_count": expected, **results, "result": "PASS"}
    report = paths["report_dir"] / "trace_manifests.json"
    write_json(report, payload)
    print("MINI TRACE MANIFESTS")
    print("====================")
    print(f"forward rows = {results['forward']['row_count']}")
    for component in cfg["adjoint"]["components"]:
        print(f"adjoint {component} rows = {results['adjoint'][component]['row_count']}")
    print(f"manifest_dir = {paths['manifest_dir']}")
    print("RESULT = PASS_MINI_TRACE_MANIFESTS")


if __name__ == "__main__":
    main()
