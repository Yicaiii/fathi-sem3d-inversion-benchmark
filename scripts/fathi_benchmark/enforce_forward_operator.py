#!/usr/bin/env python3

from pathlib import Path
from datetime import datetime
import argparse
import json
import os
import re
import sys


ROOT = Path(
    os.environ.get(
        "FATHI_BENCHMARK_ROOT",
        str(Path.home() / "sem3d_fathi_clean"),
    )
).expanduser().resolve()


def resolve(value):
    path = Path(value).expanduser()
    return path if path.is_absolute() else ROOT / path


def source_coordinates(text):
    coordinates = []

    for block in re.findall(
        r"source\s*\{(.*?)\}\s*;",
        text,
        flags=re.DOTALL,
    ):
        match = re.search(
            r"coords\s*=\s*"
            r"([-+0-9.eE]+)\s+"
            r"([-+0-9.eE]+)\s+"
            r"([-+0-9.eE]+)\s*;",
            block,
        )

        if match is None:
            raise RuntimeError(
                f"Unreadable source block:\n{block}"
            )

        coordinates.append(
            tuple(float(value) for value in match.groups())
        )

    return coordinates


def station_count(path):
    return sum(
        bool(line.strip())
        and not line.lstrip().startswith("#")
        for line in path.read_text(
            encoding="utf-8",
            errors="ignore",
        ).splitlines()
    )


parser = argparse.ArgumentParser()
parser.add_argument("--context", required=True)
args = parser.parse_args()

context_path = resolve(args.context)
context = json.loads(
    context_path.read_text(encoding="utf-8")
)

config_path = resolve(context["config_path"])
config = json.loads(
    config_path.read_text(encoding="utf-8")
)

operator = config.get("forward_operator")

if not operator:
    raise RuntimeError(
        "Missing forward_operator in benchmark config"
    )

workspace = resolve(context["strict_forward_workspace"])
parent_input = (
    resolve(context["input_accepted_dir"])
    / "input.spec"
)
station_file = workspace / operator.get(
    "station_file",
    "stations.txt",
)

for required_path in [
    workspace,
    parent_input,
    station_file,
]:
    if not required_path.exists():
        raise RuntimeError(
            f"Missing required path: {required_path}"
        )

# Preserve the source operator from the accepted parent model.
text = parent_input.read_text(encoding="utf-8")

run_name = (
    f'{context["transition"]}_strict_forward_fullgrid'
)

if re.search(
    r'run_name\s*=\s*"[^"]*"\s*;',
    text,
):
    text = re.sub(
        r'run_name\s*=\s*"[^"]*"\s*;',
        f'run_name = "{run_name}";',
        text,
        count=1,
    )
else:
    text = f'run_name = "{run_name}";\n\n{text}'

# Full-grid forward does not need snapshots.
text = re.sub(
    r"save_snap\s*=\s*(?:true|false)\s*;",
    "save_snap = false;",
    text,
    count=1,
)

# Full-grid forward must record displacement gradients.
if re.search(r"dudx\s*=\s*\d+\s*;", text):
    text = re.sub(
        r"dudx\s*=\s*\d+\s*;",
        "dudx = 1;",
        text,
        count=1,
    )
else:
    match = re.search(
        r"acc\s*=\s*\d+\s*;",
        text,
    )

    if match is None:
        raise RuntimeError(
            "Cannot insert dudx=1: acc line absent"
        )

    text = (
        text[:match.end()]
        + "\ndudx = 1;"
        + text[match.end():]
    )

workspace_input = workspace / "input.spec"
workspace_input.write_text(
    text,
    encoding="utf-8",
)

actual_coordinates = source_coordinates(text)

expected_coordinates = {
    tuple(float(value) for value in row)
    for row in operator["source_coordinates_m"]
}

actual_station_count = station_count(station_file)

checks = {
    "source_count": (
        len(actual_coordinates)
        == int(operator["source_count"])
    ),
    "source_coordinates": (
        set(actual_coordinates)
        == expected_coordinates
    ),
    "station_count": (
        actual_station_count
        == int(operator["full_grid_station_count"])
    ),
    "station_file": (
        f'file = "{station_file.name}";'
        in text
    ),
    "dudx": "dudx = 1;" in text,
    "snapshots": "save_snap = false;" in text,
}

ok = all(checks.values())

report = {
    "created": datetime.now().isoformat(),
    "context": str(context_path),
    "config": str(config_path),
    "workspace": str(workspace),
    "source_operator_source": str(parent_input),
    "actual_source_count": len(actual_coordinates),
    "actual_source_coordinates_m": sorted(
        set(actual_coordinates)
    ),
    "actual_station_count": actual_station_count,
    "checks": checks,
    "result": "PASS" if ok else "FAIL",
}

report_dir = (
    ROOT
    / "benchmark_fathi_strict/reports/"
    / "executable_tasks"
)
report_dir.mkdir(parents=True, exist_ok=True)

stem = (
    f'{context["transition"]}_forward_operator'
)

(report_dir / f"{stem}.json").write_text(
    json.dumps(report, indent=2) + "\n",
    encoding="utf-8",
)

print("Forward operator enforcement")
print("============================")
print(f"source operator = {parent_input}")
print(
    f"sources = {len(actual_coordinates)}"
    f"/{operator['source_count']}"
)
print(
    "source coordinates ok = "
    f"{checks['source_coordinates']}"
)
print(
    f"stations = {actual_station_count}"
    f"/{operator['full_grid_station_count']}"
)
print(f"dudx ok = {checks['dudx']}")
print(
    "snapshots disabled = "
    f"{checks['snapshots']}"
)
print("RESULT = PASS" if ok else "RESULT = FAIL")

if not ok:
    sys.exit(1)
