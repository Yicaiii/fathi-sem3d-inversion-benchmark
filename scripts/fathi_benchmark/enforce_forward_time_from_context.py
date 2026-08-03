from __future__ import annotations

from argparse import ArgumentParser
from pathlib import Path
import json
import re

import h5py
import numpy as np


def resolve_path(value: str, root: Path) -> Path:
    path = Path(value).expanduser()

    if not path.is_absolute():
        path = root / path

    return path.resolve()


parser = ArgumentParser()

parser.add_argument(
    "--context",
    required=True,
)

args = parser.parse_args()

root = Path.cwd().resolve()

context_path = resolve_path(
    args.context,
    root,
)

context = json.loads(
    context_path.read_text(encoding="utf-8")
)

config_path = resolve_path(
    context["config_path"],
    root,
)

config = json.loads(
    config_path.read_text(encoding="utf-8")
)

profile_path = resolve_path(
    context["benchmark_profile_config"],
    root,
)

profile = json.loads(
    profile_path.read_text(encoding="utf-8")
)

config_time = config.get("simulation_time_s")

profile_time = (
    profile
    .get("forward_operator", {})
    .get("simulation_time_s")
)

if config_time is None and profile_time is None:
    raise RuntimeError(
        "No simulation_time_s found in config or profile"
    )

target_time = float(
    config_time
    if config_time is not None
    else profile_time
)

if target_time <= 0.0:
    raise RuntimeError(
        f"Invalid simulation time: {target_time}"
    )

if (
    config_time is not None
    and profile_time is not None
    and not np.isclose(
        float(config_time),
        float(profile_time),
        rtol=0.0,
        atol=1e-12,
    )
):
    print(
        "WARNING: config/profile simulation times differ"
    )
    print(f"config_time = {config_time}")
    print(f"profile_time = {profile_time}")
    print(
        "The benchmark config value will be used."
    )

workspace = resolve_path(
    context["strict_forward_workspace"],
    root,
)

input_spec = workspace / "input.spec"

if not input_spec.is_file():
    raise RuntimeError(
        f"Missing input.spec: {input_spec}"
    )

text = input_spec.read_text(
    encoding="utf-8"
)

pattern = re.compile(
    r"(?m)^(\s*sim_time\s*=\s*)[^;]+;"
)

updated, replacement_count = pattern.subn(
    lambda match: (
        f"{match.group(1)}"
        f"{target_time:.16e};"
    ),
    text,
    count=1,
)

if replacement_count != 1:
    raise RuntimeError(
        "Expected exactly one sim_time assignment "
        f"in {input_spec}, found {replacement_count}"
    )

input_spec.write_text(
    updated,
    encoding="utf-8",
)

match = pattern.search(updated)

if match is None:
    raise RuntimeError(
        "Unable to read updated sim_time"
    )

actual_text = match.group(0)

observed_root = resolve_path(
    context["true_observed_traces_dir"],
    root,
)

observed_files = sorted(
    observed_root.glob("*.h5")
)

if not observed_files:
    raise RuntimeError(
        f"No observed HDF5 files in {observed_root}"
    )

maximum_observed_time = -np.inf
trace_dataset_count = 0

for observed_file in observed_files:
    with h5py.File(observed_file, "r") as handle:
        for name, dataset in handle.items():
            if not isinstance(dataset, h5py.Dataset):
                continue

            if not name.startswith("UU_"):
                continue

            if name.endswith("_pos"):
                continue

            if dataset.ndim != 2:
                continue

            if dataset.shape[1] < 1:
                continue

            time_values = np.asarray(
                dataset[:, 0],
                dtype=np.float64,
            )

            if time_values.size == 0:
                raise RuntimeError(
                    f"Empty trace dataset: "
                    f"{observed_file}:{name}"
                )

            if np.any(np.diff(time_values) < 0.0):
                raise RuntimeError(
                    f"Non-monotonic time axis: "
                    f"{observed_file}:{name}"
                )

            maximum_observed_time = max(
                maximum_observed_time,
                float(time_values[-1]),
            )

            trace_dataset_count += 1

if trace_dataset_count == 0:
    raise RuntimeError(
        "No observed trace datasets were found"
    )

time_tolerance = max(
    target_time * 1e-6,
    1e-9,
)

if maximum_observed_time > (
    target_time + time_tolerance
):
    raise RuntimeError(
        "Observed traces exceed configured time: "
        f"{maximum_observed_time} > {target_time}"
    )

preparation_path = (
    workspace
    / "STRICT_FORWARD_PREPARATION.json"
)

if preparation_path.is_file():
    preparation = json.loads(
        preparation_path.read_text(
            encoding="utf-8"
        )
    )

    preparation[
        "configured_simulation_time_s"
    ] = target_time

    preparation[
        "observed_maximum_time_s"
    ] = maximum_observed_time

    preparation[
        "simulation_time_check"
    ] = True

    preparation_path.write_text(
        json.dumps(
            preparation,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

print("Forward time synchronization")
print("============================")
print(f"context = {context_path}")
print(f"config = {config_path}")
print(f"profile = {profile_path}")
print(f"workspace = {workspace}")
print(f"input_spec = {input_spec}")
print(f"configured_time_s = {target_time}")
print(
    "maximum_observed_time_s = "
    f"{maximum_observed_time:.16e}"
)
print(
    "observed_trace_datasets = "
    f"{trace_dataset_count}"
)
print(f"updated_assignment = {actual_text}")
print("SEM3D launched = False")
print("accepted state mutated = False")
print("RESULT = PASS_FORWARD_TIME_SYNC")
