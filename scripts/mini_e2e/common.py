#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
from pathlib import Path
from typing import Any, Iterable

import h5py
import numpy as np

UU_RE = re.compile(r"^UU_\d+$")
SOURCE_RE = re.compile(r"source\s*\{.*?\}\s*;\s*", re.DOTALL)


def repository_root() -> Path:
    env = os.environ.get("FATHI_BENCHMARK_ROOT")
    if env:
        return Path(env).expanduser().resolve()
    return Path(__file__).resolve().parents[2]


def resolve(root: Path, value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def load_config(path: str | Path) -> tuple[Path, dict[str, Any], Path]:
    root = repository_root()
    cfg_path = resolve(root, path)
    data = json.loads(cfg_path.read_text(encoding="utf-8"))
    name = str(data.get("name", ""))
    if not name.startswith("fathi_mini_e2e"):
        raise ValueError(f"Unexpected mini config: {cfg_path}")
    iteration_context(data)
    return root, data, cfg_path


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sorted_uu_keys(handle: h5py.File) -> list[str]:
    return sorted(
        (key for key in handle.keys() if UU_RE.fullmatch(key)),
        key=lambda key: int(key.split("_")[1]),
    )


def pos_key(pos: Iterable[float], ndigits: int = 8) -> tuple[float, float, float]:
    values = []
    for value in pos:
        number = round(float(value), ndigits)
        if abs(number) < 10 ** (-ndigits):
            number = 0.0
        values.append(number)
    if len(values) != 3:
        raise ValueError(f"Expected 3 coordinates, got {values}")
    return tuple(values)  # type: ignore[return-value]


def trace_position_map(trace_dir: Path) -> dict[tuple[float, float, float], dict[str, Any]]:
    result: dict[tuple[float, float, float], dict[str, Any]] = {}
    files = sorted(trace_dir.glob("capteurs.*.h5"))
    for path in files:
        with h5py.File(path, "r") as handle:
            for key in sorted_uu_keys(handle):
                pos_name = f"{key}_pos"
                if pos_name not in handle:
                    continue
                pos = np.asarray(handle[pos_name], dtype=np.float64).reshape(-1)
                if pos.size < 3:
                    continue
                coordinate = pos_key(pos[:3])
                if coordinate in result:
                    raise RuntimeError(f"Duplicate trace coordinate {coordinate} in {trace_dir}")
                result[coordinate] = {
                    "file": path,
                    "dataset": key,
                    "position": np.asarray(pos[:3], dtype=np.float64),
                    "shape": tuple(handle[key].shape),
                }
    return result


def load_displacement(entry: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    with h5py.File(entry["file"], "r") as handle:
        arr = np.asarray(handle[entry["dataset"]], dtype=np.float64)
    require(arr.ndim == 2 and arr.shape[1] >= 4, f"Bad trace shape: {arr.shape}")
    return arr[:, 0], arr[:, 1:4]



def discover_parent_trace_dir(root: Path, cfg: dict[str, Any]) -> Path:
    expected = int(cfg["observation_subset"]["expected_count"])
    checked: list[str] = []
    for value in cfg.get("parent_predicted_trace_candidates", []):
        candidate = resolve(root, value)
        checked.append(str(candidate))
        if not candidate.is_dir():
            continue
        try:
            parent_map = trace_position_map(candidate)
        except Exception:
            continue
        if len(parent_map) >= expected:
            return candidate
    raise RuntimeError(
        "No compatible parent predicted trace directory found. Checked: "
        + ", ".join(checked)
    )

def discover_observed_dir(root: Path, cfg: dict[str, Any], parent_trace_dir: Path) -> Path:
    parent_map = trace_position_map(parent_trace_dir)
    for value in cfg["true_observed_trace_candidates"]:
        candidate = resolve(root, value)
        if not candidate.is_dir():
            continue
        try:
            observed_map = trace_position_map(candidate)
        except Exception:
            continue
        if len(set(parent_map) & set(observed_map)) >= int(cfg["observation_subset"]["expected_count"]):
            return candidate
    raise RuntimeError(
        "No compatible true-observed trace directory found. Checked: "
        + ", ".join(str(resolve(root, value)) for value in cfg["true_observed_trace_candidates"])
    )


def find_sem3d(root: Path, cfg: dict[str, Any]) -> Path:
    env = os.environ.get("SEM3D_BIN")
    candidates: list[Path] = []
    if env:
        candidates.append(Path(env).expanduser())
    candidates.extend(Path(value).expanduser() for value in cfg["sem3d_executable_candidates"])
    found = shutil.which("sem3d.exe")
    if found:
        candidates.append(Path(found))
    for candidate in candidates:
        path = candidate.resolve()
        if path.is_file() and os.access(path, os.X_OK):
            return path
    raise RuntimeError("sem3d.exe not found; set SEM3D_BIN=/absolute/path/to/sem3d.exe")


def iteration_context(cfg: dict[str, Any]) -> dict[str, Any]:
    parent_iteration = int(cfg.get("parent_iteration", 0))
    next_iteration = int(cfg.get("next_iteration", parent_iteration + 1))
    require(parent_iteration >= 0, "parent_iteration must be non-negative")
    require(
        next_iteration == parent_iteration + 1,
        "next_iteration must equal parent_iteration + 1",
    )
    parent_tag = f"iter_{parent_iteration:03d}"
    next_tag = f"iter_{next_iteration:03d}"
    transition = str(
        cfg.get("transition", f"{parent_tag}_to_{next_tag}")
    )
    return {
        "parent_iteration": parent_iteration,
        "next_iteration": next_iteration,
        "parent_tag": parent_tag,
        "next_tag": next_tag,
        "transition": transition,
        "accepted_marker_name": str(
            cfg.get(
                "accepted_marker_name",
                f"MINI_ITER{next_iteration:03d}_ACCEPTANCE.json",
            )
        ),
        "state_filename": str(
            cfg.get(
                "state_filename",
                f"{next_tag}_state_v2_corrected.npz",
            )
        ),
        "complete_marker": (
            f"COMPLETE_MINI_ITER{parent_iteration:03d}"
            f"_TO_ITER{next_iteration:03d}"
        ),
        "forward_preparation_marker": (
            f"PASS_MINI_ITER{parent_iteration:03d}"
            f"_TO_ITER{next_iteration:03d}_FORWARD_PREPARATION"
        ),
        "forward_handoff_marker": (
            f"PASS_MINI_ITER{parent_iteration:03d}"
            f"_TO_ITER{next_iteration:03d}_FORWARD_HANDOFF"
        ),
    }


def output_paths(root: Path, cfg: dict[str, Any]) -> dict[str, Path]:
    context = iteration_context(cfg)
    data_root = resolve(root, cfg["output_data_root"])
    result_root = resolve(root, cfg["output_result_root"])
    transition_root = result_root / context["transition"]
    next_iter_root = data_root / context["next_tag"]
    state_dir = result_root / "states_corrected"
    accepted_dir = next_iter_root / "accepted"
    return {
        "data_root": data_root,
        "result_root": result_root,
        "transition_root": transition_root,
        "next_iter_root": next_iter_root,
        "residual_dir": transition_root / "residual_sources",
        "adjoint_root": next_iter_root / "adjoint_full_grid_batches",
        "manifest_dir": transition_root / "rhs_manifests",
        "component_rhs_dir": transition_root / "component_rhs",
        "mtilde_dir": transition_root / "mtilde_solve",
        "candidate_dir": transition_root / "candidates",
        "candidate_ws_root": next_iter_root / "candidate_forward_workspaces",
        "misfit_dir": transition_root / "candidate_misfits",
        "accepted_dir": accepted_dir,
        "accepted_marker": accepted_dir / context["accepted_marker_name"],
        "state_dir": state_dir,
        "state_file": state_dir / context["state_filename"],
        "report_dir": transition_root / "reports",
    }


def remove_runtime_outputs(workspace: Path) -> None:
    for name in ("traces", "logs", "res", "prot", "snapshots", "mirror", "fin_sem"):
        path = workspace / name
        if path.is_symlink() or path.is_file():
            path.unlink()
        elif path.is_dir():
            shutil.rmtree(path)


def copy_static_workspace(source: Path, destination: Path, *, symlink_sem: bool = True) -> None:
    if destination.exists() or destination.is_symlink():
        if destination.is_symlink() or destination.is_file():
            destination.unlink()
        else:
            shutil.rmtree(destination)
    destination.mkdir(parents=True)
    for path in source.iterdir():
        if path.name in {"traces", "logs", "res", "prot", "snapshots", "mirror", "fin_sem", "sem", "mat"}:
            continue
        target = destination / path.name
        if path.is_file():
            shutil.copy2(path, target)
        elif path.is_dir():
            shutil.copytree(path, target)
    if symlink_sem:
        (destination / "sem").symlink_to((source / "sem").resolve(), target_is_directory=True)
    else:
        shutil.copytree(source / "sem", destination / "sem")
    shutil.copytree(source / "mat", destination / "mat")


def replace_run_name(text: str, run_name: str) -> str:
    if re.search(r'run_name\s*=\s*"[^"]*"\s*;', text):
        return re.sub(
            r'run_name\s*=\s*"[^"]*"\s*;',
            f'run_name = "{run_name}";',
            text,
            count=1,
        )
    return f'run_name = "{run_name}";\n\n{text}'


def set_dudx(text: str, enabled: bool) -> str:
    value = 1 if enabled else 0
    if re.search(r"dudx\s*=\s*\d+\s*;", text):
        return re.sub(r"dudx\s*=\s*\d+\s*;", f"dudx = {value};", text, count=1)
    match = re.search(r"acc\s*=\s*\d+\s*;", text)
    require(match is not None, "Cannot insert dudx setting: acc line absent")
    return text[: match.end()] + f"\ndudx = {value};" + text[match.end() :]


def replace_sources(text: str, blocks: list[str]) -> str:
    stripped = SOURCE_RE.sub("", text)
    marker = re.search(r"\ntime_scheme\s*\{", stripped)
    require(marker is not None, "time_scheme block not found in input.spec")
    source_text = "\n\n".join(blocks) + "\n\n"
    return stripped[: marker.start() + 1] + source_text + stripped[marker.start() + 1 :]


def source_block(position: Iterable[float], direction: Iterable[float], filename: str) -> str:
    x, y, z = (float(value) for value in position)
    dx, dy, dz = (float(value) for value in direction)
    return "\n".join(
        [
            "source {",
            f"coords = {x:.10g} {y:.10g} {z:.10g};",
            "type = impulse;",
            f"dir = {dx:.1f} {dy:.1f} {dz:.1f};",
            "func = file;",
            f'time_file = "{filename}";',
            "};",
        ]
    )


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
