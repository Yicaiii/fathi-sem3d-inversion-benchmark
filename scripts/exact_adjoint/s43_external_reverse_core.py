"""Iteration-independent deterministic replay helpers for the S43 exact reverse.

The numerical recurrence is owned by :mod:`s43_external_forward`.  This module
only manages retained primal checkpoints and deterministic replay caches; it
contains no Stage5 directional-certification references.
"""

from __future__ import annotations

import gc
import json
import os
from pathlib import Path

import numpy as np

from scripts.exact_adjoint.run_real_s43_exact_material_gradient import zero_state
from scripts.exact_adjoint.s43_external_forward import (
    _checkpoint_values,
    _load_checkpoint,
    atomic_save_npz,
)


def atomic_json(path: str | Path, payload: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def checkpoint_metadata(path: str | Path) -> dict[str, object]:
    with np.load(path) as saved:
        return {
            "completed": int(saved["completed"]),
            "signature_sha256": str(saved["signature_sha256"].item()),
            "labels": [str(value) for value in saved["labels"]],
        }


def retained_checkpoint_map(
    directory: str | Path,
    expected_last: int,
    driver_signature: str,
) -> tuple[dict[int, Path], list[int]]:
    directory = Path(directory).resolve()
    checkpoints: dict[int, Path] = {}
    for path in sorted(directory.glob("primal_*.npz")):
        token = path.stem.removeprefix("primal_")
        if token.isdigit():
            checkpoints[int(token)] = path.resolve()

    expected_last = int(expected_last)
    if expected_last not in checkpoints:
        raise RuntimeError(f"missing retained primal endpoint {expected_last}")

    positions = [0, *sorted(checkpoints)]
    if positions[-1] != expected_last or len(positions) != len(set(positions)):
        raise RuntimeError(f"invalid retained checkpoint positions: {positions}")

    for position, path in checkpoints.items():
        expected = {
            "completed": position,
            "signature_sha256": driver_signature,
            "labels": ["primal"],
        }
        if checkpoint_metadata(path) != expected:
            raise RuntimeError(f"retained checkpoint metadata mismatch: {path}")
    return checkpoints, positions


def coarse_state(driver, checkpoints: dict[int, Path], position: int):
    position = int(position)
    if position == 0:
        return zero_state(driver.data)
    if position not in checkpoints:
        raise RuntimeError(f"missing retained coarse state at {position}")
    completed, states = _load_checkpoint(
        checkpoints[position], driver.signature
    )
    if completed != position or set(states) != {"primal"}:
        raise RuntimeError(f"invalid retained state at {position}")
    return states["primal"]


def replay_cache_directory(output_dir: str | Path, start: int, end: int) -> Path:
    return (
        Path(output_dir)
        / "replay_cache"
        / f"chunk_{int(start):06d}_{int(end):06d}"
    )


def replay_cache_path(
    output_dir: str | Path,
    start: int,
    end: int,
    position: int,
) -> Path:
    return replay_cache_directory(output_dir, start, end) / (
        f"primal_{int(position):06d}.npz"
    )


def replay_manifest_path(output_dir: str | Path, start: int, end: int) -> Path:
    return replay_cache_directory(output_dir, start, end) / "manifest.json"


def sub_boundaries(start: int, end: int, replay_stride: int) -> list[int]:
    start = int(start)
    end = int(end)
    replay_stride = int(replay_stride)
    if replay_stride < 1:
        raise ValueError("replay_stride must be positive")
    if end <= start:
        raise ValueError(f"invalid replay interval [{start}, {end})")
    values = list(range(start, end, replay_stride))
    if not values or values[0] != start:
        values.insert(0, start)
    if values[-1] != end:
        values.append(end)
    return values


def ensure_replay_cache(
    output_dir: str | Path,
    driver,
    checkpoints: dict[int, Path],
    start: int,
    end: int,
    replay_stride: int,
) -> list[int]:
    start = int(start)
    end = int(end)
    boundaries = sub_boundaries(start, end, replay_stride)
    cached_positions = boundaries[1:-1]
    manifest_path = replay_manifest_path(output_dir, start, end)

    if manifest_path.is_file() and all(
        replay_cache_path(output_dir, start, end, position).is_file()
        for position in cached_positions
    ):
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if (
            manifest.get("driver_signature_sha256") == driver.signature
            and manifest.get("endpoint_matches_retained_bitwise") is True
            and manifest.get("boundaries") == boundaries
        ):
            return boundaries

    state = coarse_state(driver, checkpoints, start)
    wanted = set(cached_positions)
    for transition in range(start, end):
        state = driver.advance(state, transition)
        position = transition + 1
        if position not in wanted:
            continue
        path = replay_cache_path(output_dir, start, end, position)
        if path.is_file():
            expected = {
                "completed": position,
                "signature_sha256": driver.signature,
                "labels": ["primal"],
            }
            if checkpoint_metadata(path) != expected:
                raise RuntimeError(f"invalid replay cache metadata: {path}")
        else:
            atomic_save_npz(
                path,
                **_checkpoint_values(
                    position,
                    driver.signature,
                    {"primal": state},
                ),
            )

    endpoint = coarse_state(driver, checkpoints, end)
    endpoint_equal = all(
        np.array_equal(value, reference)
        for value, reference in zip(state, endpoint)
    )
    endpoint_max_abs = max(
        float(np.max(np.abs(value - reference)))
        for value, reference in zip(state, endpoint)
    )
    del state, endpoint
    gc.collect()
    if not endpoint_equal:
        raise RuntimeError(
            f"deterministic replay endpoint mismatch for [{start},{end}): "
            f"max_abs={endpoint_max_abs:.17e}"
        )

    atomic_json(
        manifest_path,
        {
            "start": start,
            "end": end,
            "boundaries": boundaries,
            "driver_signature_sha256": driver.signature,
            "endpoint_matches_retained_bitwise": True,
            "endpoint_max_abs_difference": endpoint_max_abs,
        },
    )
    return boundaries


def load_replay_state(
    output_dir: str | Path,
    driver,
    checkpoints: dict[int, Path],
    outer_start: int,
    outer_end: int,
    position: int,
):
    position = int(position)
    if position == int(outer_start):
        return coarse_state(driver, checkpoints, position)
    path = replay_cache_path(
        output_dir, outer_start, outer_end, position
    )
    completed, states = _load_checkpoint(path, driver.signature)
    if completed != position or set(states) != {"primal"}:
        raise RuntimeError(f"invalid replay state at {position}")
    return states["primal"]


def cleanup_replay_cache(
    output_dir: str | Path,
    start: int,
    end: int,
    boundaries: list[int],
) -> None:
    directory = replay_cache_directory(output_dir, start, end)
    for position in boundaries[1:-1]:
        path = replay_cache_path(output_dir, start, end, position)
        if path.is_file():
            path.unlink()
    manifest = replay_manifest_path(output_dir, start, end)
    if manifest.is_file():
        manifest.unlink()
    if directory.is_dir() and not any(directory.iterdir()):
        directory.rmdir()


def finite_reverse_state(reverse_state: dict[str, object]) -> bool:
    return bool(
        all(np.all(np.isfinite(value)) for value in reverse_state["bar"])
        and all(
            np.all(np.isfinite(value))
            for value in reverse_state["gradients"].values()
        )
    )
