#!/usr/bin/env python3
"""Promote a completed full bootstrap into canonical iter_000 assets.

Read-only by default. With --write, validate the non-smoke true/initial runs,
compute J0 from receiver traces, install observed data, create iter_000/accepted
and its state file, then create the iter_000_to_iter_001 context.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import h5py
import numpy as np

UU_RE = re.compile(r"^UU_\d+$")
MODELS = ("true_layered", "initial_homogeneous")
RUNTIME_NAMES = {"traces", "res", "logs", "prot", "mirror", "fin_sem", "stat.log"}


def root_dir() -> Path:
    default = Path(__file__).resolve().parents[2]
    return Path(os.environ.get("FATHI_BENCHMARK_ROOT", default)).expanduser().resolve()


def resolve(root: Path, value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"JSON file not found: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise TypeError(f"JSON root must be an object: {path}")
    return data


def require(ok: bool, message: str) -> None:
    if not ok:
        raise ValueError(message)


def repo_value(root: Path, path: Path) -> str:
    try:
        return str(path.resolve().relative_to(root.resolve()))
    except ValueError:
        return str(path.resolve())


def workspaces_from_manifest(manifest: dict[str, Any], manifest_path: Path) -> dict[str, Path]:
    require(manifest.get("status") == "passed", "Bootstrap status is not passed")
    require(manifest.get("audit_only") is False, "Audit-only bootstrap cannot initialize iter_000")
    require(manifest.get("smoke_seconds") is None, "Smoke bootstrap cannot initialize iter_000")
    require(set(MODELS) <= set(manifest.get("models", [])), "Both bootstrap models are required")
    steps = manifest.get("steps", [])
    result: dict[str, Path] = {}
    for model in MODELS:
        matches = [s for s in steps if s.get("model") == model and s.get("stage") == "solve"]
        require(len(matches) == 1, f"Expected exactly one solve step for {model}")
        step = matches[0]
        require(step.get("passed") is True and int(step.get("return_code", -1)) == 0,
                f"Bootstrap solve failed for {model}")
        workspace = Path(step["workspace"]).expanduser()
        if not workspace.is_absolute():
            workspace = manifest_path.parent / workspace
        result[model] = workspace.resolve()
    return result


def validate_solver(workspace: Path, expected_time: float, model: str) -> dict[str, Any]:
    require(workspace.is_dir(), f"Workspace missing for {model}: {workspace}")
    data = load_json(workspace / "logs" / "solver_manifest.json")
    require(data.get("smoke_seconds") is None, f"{model} is a smoke solver run")
    require(int(data.get("return_code", -1)) == 0, f"{model} solver return code is not zero")
    require(data.get("timed_out") is False, f"{model} solver timed out")
    effective = data.get("effective_input_settings", {})
    require(effective.get("save_traces") is True, f"{model} did not save traces")
    require(np.isclose(float(effective.get("sim_time_s")), expected_time, rtol=0.0, atol=1e-12),
            f"{model} sim_time is not the full benchmark duration")
    audit = data.get("output_audit", {})
    fin = audit.get("fin_sem", {})
    require(audit.get("passed") is True, f"{model} output audit failed")
    require(int(audit.get("trace_count", 0)) > 0, f"{model} has no trace files")
    require(fin.get("exists") is True and str(fin.get("value")) == "1", f"{model} fin_sem is invalid")
    return data


def normalized_input(path: Path) -> list[str]:
    return [line for line in path.read_text(encoding="utf-8").splitlines()
            if not line.startswith("run_name = ")]


def validate_bootstrap(manifest_path: Path, expected_time: float) -> tuple[dict[str, Any], dict[str, Path]]:
    manifest = load_json(manifest_path)
    workspaces = workspaces_from_manifest(manifest, manifest_path)
    for model, workspace in workspaces.items():
        validate_solver(workspace, expected_time, model)
    true_ws = workspaces["true_layered"]
    init_ws = workspaces["initial_homogeneous"]
    for name in ("stations.txt", "gaussian_stf.txt"):
        require((true_ws / name).read_bytes() == (init_ws / name).read_bytes(),
                f"Forward-operator file differs: {name}")
    require(normalized_input(true_ws / "input.spec") == normalized_input(init_ws / "input.spec"),
            "input.spec differs beyond run_name")
    return manifest, workspaces


def trace_map(directory: Path, decimals: int) -> tuple[dict[tuple[float, float, float], tuple[Path, str]], int]:
    require(directory.is_dir(), f"Trace directory missing: {directory}")
    result: dict[tuple[float, float, float], tuple[Path, str]] = {}
    duplicates = 0
    files = sorted(directory.glob("capteurs*.h5"))
    require(bool(files), f"No capteurs HDF5 files in {directory}")
    for path in files:
        with h5py.File(path, "r") as h5:
            keys = sorted((k for k in h5 if UU_RE.match(k)), key=lambda k: int(k.split("_")[1]))
            for key in keys:
                pos_key = key + "_pos"
                if pos_key not in h5:
                    continue
                pos = np.asarray(h5[pos_key], dtype=np.float64).reshape(-1)
                if pos.size < 3:
                    continue
                xyz = tuple(round(float(v), decimals) for v in pos[:3])
                if xyz in result:
                    duplicates += 1
                else:
                    result[xyz] = (path, key)
    return result, duplicates


def read_trace(entry: tuple[Path, str]) -> tuple[np.ndarray, np.ndarray]:
    path, key = entry
    with h5py.File(path, "r") as h5:
        array = np.asarray(h5[key], dtype=np.float64)
    require(array.ndim == 2 and array.shape[1] >= 4, f"Invalid trace shape: {path}:{key}")
    time, displacement = array[:, 0], array[:, 1:4]
    require(len(time) >= 2 and np.all(np.diff(time) > 0), f"Invalid trace time axis: {path}:{key}")
    require(np.isfinite(time).all() and np.isfinite(displacement).all(), f"Non-finite trace: {path}:{key}")
    return time, displacement


def trapz(values: np.ndarray, time: np.ndarray) -> float:
    return float(np.trapezoid(values, time) if hasattr(np, "trapezoid") else np.trapz(values, time))


def initial_misfit(true_dir: Path, initial_dir: Path, receivers: int,
                   simulation_time: float, decimals: int) -> dict[str, Any]:
    true_map, true_dup = trace_map(true_dir, decimals)
    init_map, init_dup = trace_map(initial_dir, decimals)
    require(true_dup == 0 and init_dup == 0, "Duplicate receiver coordinates found")
    require(len(true_map) == receivers, f"True receiver count mismatch: {len(true_map)} != {receivers}")
    require(len(init_map) == receivers, f"Initial receiver count mismatch: {len(init_map)} != {receivers}")
    require(set(true_map) == set(init_map), "True and initial receiver coordinate sets differ")
    total_j = 0.0
    true_energy = 0.0
    true_ends: list[float] = []
    init_ends: list[float] = []
    for xyz in sorted(true_map):
        tt, ut = read_trace(true_map[xyz])
        ts, us = read_trace(init_map[xyz])
        interp = np.column_stack([np.interp(tt, ts, us[:, j]) for j in range(3)])
        residual = interp - ut
        total_j += 0.5 * trapz(np.sum(residual * residual, axis=1), tt)
        true_energy += 0.5 * trapz(np.sum(ut * ut, axis=1), tt)
        true_ends.append(float(tt[-1]))
        init_ends.append(float(ts[-1]))
    tolerance = max(1e-9, simulation_time * 5e-4)
    require(min(true_ends) >= simulation_time - tolerance, "True traces do not cover full simulation time")
    require(min(init_ends) >= simulation_time - tolerance, "Initial traces do not cover full simulation time")
    require(np.isfinite(total_j) and total_j >= 0.0, f"Invalid J0: {total_j}")
    return {
        "J": float(total_j),
        "relative_J": float(total_j / true_energy) if true_energy > 0 else float("nan"),
        "true_energy": float(true_energy),
        "receiver_count": receivers,
        "true_time_end_min_s": min(true_ends),
        "initial_time_end_min_s": min(init_ends),
    }


def find_dataset(path: Path, shape: tuple[int, int, int]) -> str:
    names: list[str] = []
    with h5py.File(path, "r") as h5:
        h5.visititems(lambda name, obj: names.append(name)
                      if isinstance(obj, h5py.Dataset) and tuple(obj.shape) == shape else None)
    require(bool(names), f"No dataset with shape {shape} in {path}")
    return "samples" if "samples" in names else names[0]


def field(path: Path, shape: tuple[int, int, int]) -> np.ndarray:
    with h5py.File(path, "r") as h5:
        data = np.asarray(h5[find_dataset(path, shape)], dtype=np.float64)
    require(np.isfinite(data).all(), f"Non-finite material field: {path}")
    return data


def initial_fields(workspace: Path, spec: dict[str, Any]) -> dict[str, np.ndarray]:
    shape = tuple(int(v) for v in spec["material_grid"]["shape"])
    h5_dir = workspace / "mat" / "h5"
    mu = field(h5_dir / "Mat_0_Mu.h5", shape)
    kappa = field(h5_dir / "Mat_0_Kappa.h5", shape)
    density = field(h5_dir / "Mat_0_Density.h5", shape)
    model = spec["material_models"]["initial_homogeneous"]
    require(np.allclose(mu, float(model["mu_pa"]), rtol=0, atol=1e-8), "Initial Mu is not homogeneous")
    require(np.allclose(kappa, float(model["kappa_pa"]), rtol=0, atol=1e-8), "Initial Kappa is wrong")
    require(np.allclose(density, float(spec["material_models"]["density_kg_m3"]), rtol=0, atol=1e-8),
            "Initial density is wrong")
    return {"lambda": kappa - (2.0 / 3.0) * mu, "mu": mu, "kappa": kappa, "density": density}


def build_state(path: Path, accepted_value: str, fields: dict[str, np.ndarray], j0: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        **{
            "lambda": fields["lambda"], "lambda_field": fields["lambda"],
            "mu": fields["mu"], "kappa": fields["kappa"], "density": fields["density"],
            "J": np.float64(j0), "parent_J": np.float64(j0), "delta_J": np.float64(0.0),
            "iter_k": np.int64(-1), "iter": np.int64(0),
            "accepted_from": np.array("bootstrap_fathi_reduced_3x3_12p5"),
            "accepted_dir": np.array(accepted_value),
            "transition": np.array("bootstrap_to_iter_000"),
            "descent": np.bool_(True),
            "candidate_misfit_summary": np.array("J0 computed from full bootstrap traces"),
        },
    )


def ignore_runtime(_directory: str, names: list[str]) -> set[str]:
    return {name for name in names if name in RUNTIME_NAMES or name.startswith("output.")}


def remove(path: Path) -> None:
    if path.is_dir() and not path.is_symlink():
        shutil.rmtree(path)
    elif path.exists() or path.is_symlink():
        path.unlink()


def reserve(path: Path, overwrite: bool) -> None:
    if path.exists() or path.is_symlink():
        if not overwrite:
            raise FileExistsError(f"Output exists: {path}\nUse --overwrite only after checking it.")
        remove(path)


def copy_dir(source: Path, target: Path, overwrite: bool, ignore=None) -> None:
    require(source.is_dir(), f"Source directory missing: {source}")
    if source.resolve() == target.resolve():
        return
    reserve(target, overwrite)
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source, target, ignore=ignore)


def output_paths(root: Path, cfg: dict[str, Any]) -> dict[str, Path]:
    data_root = resolve(root, cfg["run_data_root"])
    state_root = resolve(root, cfg["state_dir"])
    result_root = resolve(root, cfg["run_result_root"])
    transition = result_root / "iter_000_to_iter_001"
    return {
        "observed_traces": resolve(root, cfg["true_observed_traces_dir"]),
        "true_material": resolve(root, cfg["true_material_dir"]),
        "accepted": data_root / "iter_000" / "accepted",
        "state": state_root / "iter_000_state_v2_corrected.npz",
        "context_json": transition / "iter_000_to_iter_001_iteration_context.json",
        "context_txt": transition / "iter_000_to_iter_001_iteration_context.txt",
        "report_json": result_root / "bootstrap_to_iter_000" / "iter000_from_bootstrap_manifest.json",
        "report_txt": result_root / "bootstrap_to_iter_000" / "iter000_from_bootstrap_manifest.txt",
    }


def create_context(root: Path, cfg_path: Path, paths: dict[str, Path], overwrite: bool) -> dict[str, Any]:
    script = root / "scripts" / "fathi_benchmark" / "create_iteration_context_generic.py"
    command = [sys.executable, str(script), "--iter-k", "0", "--config", repo_value(root, cfg_path), "--write"]
    if overwrite:
        command.append("--overwrite")
    env = os.environ.copy()
    env["FATHI_BENCHMARK_ROOT"] = str(root)
    completed = subprocess.run(command, cwd=root, env=env, text=True, capture_output=True, check=False)
    require(completed.returncode == 0, "Context creation failed:\n" + completed.stdout + completed.stderr)
    context = load_json(paths["context_json"])
    require(context.get("preflight_ok") is True, "iter_000_to_iter_001 preflight failed")
    return context


def write_report(paths: dict[str, Path], payload: dict[str, Any]) -> None:
    paths["report_json"].parent.mkdir(parents=True, exist_ok=True)
    paths["report_json"].write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    lines = [
        "ITER_000 INITIALIZATION FROM FULL BOOTSTRAP", "==========================================", "",
        f"bootstrap_manifest = {payload['bootstrap_manifest']}",
        f"accepted = {payload['outputs']['accepted']}",
        f"state = {payload['outputs']['state']}",
        f"context = {payload['outputs']['context']}", "",
        f"J0 = {payload['misfit']['J']:.16e}",
        f"relative_J0 = {payload['misfit']['relative_J']:.16e}",
        f"receivers = {payload['misfit']['receiver_count']}", "",
        "RESULT = PASS_ITER000_INITIALIZATION_FROM_BOOTSTRAP",
    ]
    paths["report_txt"].write_text("\n".join(lines) + "\n", encoding="utf-8")


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--bootstrap-root", default="data/bootstrap/fathi_reduced_3x3_12p5")
    p.add_argument("--manifest", default=None)
    p.add_argument("--benchmark-spec", default="configs/fathi_reduced_3x3_12p5.json")
    p.add_argument("--iteration-config", default="benchmark_fathi_strict/config/benchmark_config.json")
    p.add_argument("--round-decimals", type=int, default=8)
    p.add_argument("--write", action="store_true")
    p.add_argument("--overwrite", action="store_true")
    return p


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    root = root_dir()
    spec_path = resolve(root, args.benchmark_spec)
    cfg_path = resolve(root, args.iteration_config)
    bootstrap_root = resolve(root, args.bootstrap_root)
    manifest_path = resolve(root, args.manifest) if args.manifest else bootstrap_root / "bootstrap_manifest.json"
    spec, cfg = load_json(spec_path), load_json(cfg_path)
    simulation_time = float(spec["forward_operator"]["simulation_time_s"])
    receivers = int(spec["receivers"]["physical"]["count"])
    manifest, workspaces = validate_bootstrap(manifest_path, simulation_time)
    misfit = initial_misfit(workspaces["true_layered"] / "traces",
                            workspaces["initial_homogeneous"] / "traces",
                            receivers, simulation_time, args.round_decimals)
    fields = initial_fields(workspaces["initial_homogeneous"], spec)
    paths = output_paths(root, cfg)

    print("INITIALIZE ITER_000 FROM FULL BOOTSTRAP")
    print("=======================================")
    print(f"manifest = {manifest_path}")
    print(f"true workspace = {workspaces['true_layered']}")
    print(f"initial workspace = {workspaces['initial_homogeneous']}")
    print(f"J0 = {misfit['J']:.16e}")
    print(f"relative_J0 = {misfit['relative_J']:.16e}")
    print(f"mode = {'WRITE' if args.write else 'PLAN ONLY'}")
    for key, path in paths.items():
        print(f"{key} = {path}")

    if not args.write:
        print("RESULT = PASS_ITER000_INITIALIZATION_PLAN")
        return 0

    for path in paths.values():
        reserve(path, args.overwrite)
    copy_dir(workspaces["true_layered"] / "traces", paths["observed_traces"], args.overwrite)
    copy_dir(workspaces["true_layered"] / "mat" / "h5", paths["true_material"], args.overwrite)
    copy_dir(workspaces["initial_homogeneous"], paths["accepted"], args.overwrite, ignore_runtime)
    build_state(paths["state"], repo_value(root, paths["accepted"]), fields, misfit["J"])
    context = create_context(root, cfg_path, paths, args.overwrite)
    payload = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "result": "PASS_ITER000_INITIALIZATION_FROM_BOOTSTRAP",
        "bootstrap_manifest": str(manifest_path),
        "bootstrap_status": manifest["status"],
        "workspaces": {k: str(v) for k, v in workspaces.items()},
        "misfit": misfit,
        "outputs": {
            "observed_traces": str(paths["observed_traces"]),
            "true_material": str(paths["true_material"]),
            "accepted": str(paths["accepted"]),
            "state": str(paths["state"]),
            "context": str(paths["context_json"]),
        },
        "context_preflight_ok": context["preflight_ok"],
        "context_preflight_checks": context["preflight_checks"],
    }
    write_report(paths, payload)
    print(paths["report_txt"].read_text(encoding="utf-8"))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileNotFoundError, FileExistsError, TypeError, ValueError, RuntimeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2)
