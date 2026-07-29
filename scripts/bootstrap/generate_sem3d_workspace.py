#!/usr/bin/env python3
"""Generate a standalone SEM3D workspace from a benchmark JSON specification.

This generator does not copy a local template. It creates the deterministic
text inputs and material HDF5 fields required by the reduced Fathi-inspired
3x3-source benchmark.

The command is plan-only unless --write is supplied.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
from pathlib import Path
from typing import Any

import h5py
import numpy as np


def repository_root() -> Path:
    """Return the repository root, honoring FATHI_BENCHMARK_ROOT."""
    default_root = Path(__file__).resolve().parents[2]
    return Path(
        os.environ.get("FATHI_BENCHMARK_ROOT", str(default_root))
    ).expanduser().resolve()


def resolve_path(root: Path, value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Configuration file not found: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise TypeError(f"Configuration root must be an object: {path}")
    return data


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def float_text(value: float, decimals: int = 1) -> str:
    return f"{float(value):.{decimals}f}"


def generate_mesh_input(spec: dict[str, Any]) -> str:
    lines = spec["sem3d_mesh"]["mesh_input_lines"]
    require(len(lines) == 2, "mesh_input_lines must contain two lines")
    return "\n".join(str(item) for item in lines) + "\n"


def generate_mat_dat(spec: dict[str, Any]) -> str:
    cfg = spec["sem3d_mesh"]["mat_dat"]
    lines = [
        float_text(cfg["x"]["min_m"]),
        float_text(cfg["x"]["max_m"]),
        str(cfg["x"]["spacing_m"]),
        float_text(cfg["y"]["min_m"]),
        float_text(cfg["y"]["max_m"]),
        str(cfg["y"]["spacing_m"]),
        float_text(cfg["surface_z_m"]),
        str(cfg["layer_count"]),
    ]
    for block in cfg["vertical_blocks"]:
        lines.append(
            f"{float(block['thickness_m']):.1f} "
            f"{int(block['element_count'])}"
        )
    lines.extend(str(item) for item in cfg["validated_trailing_lines"])
    return "\n".join(lines) + "\n"


def material_velocity_line(material: dict[str, Any], prefix: str) -> str:
    return (
        f"{prefix} "
        f"{float(material['cp_m_s']):.6f} "
        f"{float(material['cs_m_s']):.6f} "
        f"{float(material['density_kg_m3']):.6f} "
        f"{float(material['qp']):.6f} "
        f"{float(material['qs']):.6f}"
    )


def generate_mater_in(spec: dict[str, Any]) -> str:
    cfg = spec["sem3d_mesh"]["mater_in"]
    materials = cfg["materials"]
    require(
        int(cfg["material_count"]) == len(materials),
        "mater_in material_count does not match materials",
    )
    lines = [str(len(materials))]
    for material in materials:
        lines.append(
            f"{material['type']} "
            f"{float(material['cp_m_s']):.7f} "
            f"{float(material['cs_m_s']):.7f} "
            f"{float(material['density_kg_m3']):.1f} "
            f"{float(material['qp']):.1f} "
            f"{float(material['qs']):.1f}"
        )
    return "\n".join(lines) + "\n"


def boundary_width(position: float, negative: float, positive: float) -> float:
    if math.isclose(position, negative):
        return -1.2
    if math.isclose(position, positive):
        return 1.2
    return 0.0


def pml_xy_positions() -> list[tuple[float, float]]:
    return [
        (-20.0, -20.0),
        (0.0, -20.0),
        (20.0, -20.0),
        (-20.0, 0.0),
        (20.0, 0.0),
        (-20.0, 20.0),
        (0.0, 20.0),
        (20.0, 20.0),
    ]


def generate_material_input(spec: dict[str, Any]) -> str:
    """Generate the validated 36-material/PML mapping deterministically."""
    materials = spec["sem3d_mesh"]["mater_in"]["materials"]
    require(len(materials) == 3, "Reduced benchmark requires three materials")

    lines: list[str] = ["36"]

    for material in materials:
        lines.append(material_velocity_line(material, "S"))

    # Validated PML material-entry multiplicities: 8, 8 and 17.
    for material, count in zip(materials, (8, 8, 17), strict=True):
        for _ in range(count):
            lines.append(material_velocity_line(material, "P"))

    lines.extend(
        [
            "# PML properties",
            "# npow,Apow,posX,widthX,posY,widthY,posZ,widthZ,mat",
        ]
    )

    for material_id in (0, 1, 2):
        for x_pos, y_pos in pml_xy_positions():
            x_width = boundary_width(x_pos, -20.0, 20.0)
            y_width = boundary_width(y_pos, -20.0, 20.0)
            lines.append(
                f"2 10. "
                f"{x_pos:8.1f} {x_width:8.1f} "
                f"{y_pos:8.1f} {y_width:8.1f} "
                f"{0.0:8.1f} {0.0:8.1f}  {material_id}"
            )

    # Bottom PML is associated with the deepest material.
    for y_pos in (-20.0, 0.0, 20.0):
        for x_pos in (-20.0, 0.0, 20.0):
            x_width = boundary_width(x_pos, -20.0, 20.0)
            y_width = boundary_width(y_pos, -20.0, 20.0)
            lines.append(
                f"2 10. "
                f"{x_pos:8.1f} {x_width:8.1f} "
                f"{y_pos:8.1f} {y_width:8.1f} "
                f"{-50.0:8.1f} {-1.3:8.1f}  2"
            )

    require(len(lines) == 72, f"material.input must have 72 lines, got {len(lines)}")
    return "\n".join(lines) + "\n"


def generate_material_spec(spec: dict[str, Any]) -> str:
    cfg = spec["sem3d_mesh"]["material_spec"]
    lines = [
        "# Definition materiaux",
        "",
        "material 0 {",
        "domain = solid;",
        f"deftype = {cfg['deftype']};",
        f"spacedef = {cfg['spacedef']};",
        f'filename0 = "{cfg["kappa_file"]}";',
        f'filename1 = "{cfg["mu_file"]}";',
        f'filename2 = "{cfg["density_file"]}";',
        "};",
        "",
    ]
    for material_id in cfg["solid_pml_copy_ids"]:
        lines.append(
            f"material {int(material_id)} "
            "{ copy = 0; domain = solidpml;};"
        )
    require(len(lines) == 37, f"material.spec must have 37 lines, got {len(lines)}")
    return "\n".join(lines) + "\n"


def inclusive_axis(minimum: float, maximum: float, spacing: float) -> np.ndarray:
    count_float = (maximum - minimum) / spacing
    count = int(round(count_float))
    require(
        math.isclose(count_float, count, rel_tol=0.0, abs_tol=1e-10),
        "Axis extent is not divisible by spacing",
    )
    return np.linspace(minimum, maximum, count + 1, dtype=np.float64)


def ordered_axis(start: float, end: float, spacing: float) -> np.ndarray:
    """Return an inclusive axis while preserving the requested direction."""
    require(not math.isclose(spacing, 0.0), "Axis spacing must be non-zero")
    count_float = (end - start) / spacing
    count = int(round(count_float))
    require(count >= 0, "Axis spacing has the wrong sign")
    require(
        math.isclose(count_float, count, rel_tol=0.0, abs_tol=1e-10),
        "Axis extent is not divisible by spacing",
    )
    return start + np.arange(count + 1, dtype=np.float64) * spacing


def generate_stations(
    spec: dict[str, Any],
    *,
    receiver_role: str = "physical",
) -> tuple[str, np.ndarray]:
    """Generate either physical receivers or the strict full-grid controls."""
    require(
        receiver_role in {"physical", "strict_full_grid"},
        f"Unsupported receiver role: {receiver_role}",
    )
    cfg = spec["receivers"][receiver_role]

    if receiver_role == "physical":
        x_values = inclusive_axis(
            float(cfg["x_min_m"]),
            float(cfg["x_max_m"]),
            float(cfg["spacing_m"]),
        )
        y_values = inclusive_axis(
            float(cfg["y_min_m"]),
            float(cfg["y_max_m"]),
            float(cfg["spacing_m"]),
        )
        z_value = float(cfg["z_m"])
        rows = np.asarray(
            [(x, y, z_value) for y in y_values for x in x_values],
            dtype=np.float64,
        )
        require(
            rows.shape == (int(cfg["count"]), 3),
            f"Expected {cfg['count']} physical stations, got {rows.shape[0]}",
        )
        text = "".join(
            f"{x:.6f} {y:.6f} {z:.1f}\n"
            for x, y, z in rows
        )
        return text, rows

    x_values = ordered_axis(
        float(cfg["x_start_m"]),
        float(cfg["x_end_m"]),
        float(cfg["x_spacing_m"]),
    )
    y_values = ordered_axis(
        float(cfg["y_start_m"]),
        float(cfg["y_end_m"]),
        float(cfg["y_spacing_m"]),
    )
    z_values = ordered_axis(
        float(cfg["z_start_m"]),
        float(cfg["z_end_m"]),
        float(cfg["z_spacing_m"]),
    )

    expected_shape = tuple(int(value) for value in cfg["shape_zyx"])
    actual_shape = (len(z_values), len(y_values), len(x_values))
    require(
        actual_shape == expected_shape,
        f"Expected strict-grid shape {expected_shape}, got {actual_shape}",
    )

    rows = np.asarray(
        [
            (x, y, z)
            for z in z_values
            for y in y_values
            for x in x_values
        ],
        dtype=np.float64,
    )
    require(
        rows.shape == (int(cfg["count"]), 3),
        f"Expected {cfg['count']} strict stations, got {rows.shape[0]}",
    )

    decimals = int(cfg.get("format_decimals", 10))
    text = "".join(
        f"{x:.{decimals}f} {y:.{decimals}f} {z:.{decimals}f}\n"
        for x, y, z in rows
    )

    expected_hash = cfg.get("sha256")
    if expected_hash:
        actual_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
        require(
            actual_hash == expected_hash,
            "Strict full-grid station SHA256 mismatch: "
            f"expected {expected_hash}, got {actual_hash}",
        )

    return text, rows


def generate_stf(spec: dict[str, Any]) -> tuple[str, np.ndarray]:
    cfg = spec["source_time_function"]
    row_count = int(cfg["row_count"])
    start = float(cfg["start_s"])
    end = float(cfg["end_s"])
    dt = float(cfg["dt_s"])

    times = start + np.arange(row_count, dtype=np.float64) * dt
    require(
        math.isclose(float(times[-1]), end, rel_tol=0.0, abs_tol=1e-14),
        "STF start, end, dt and row_count are inconsistent",
    )

    amplitude = float(cfg["amplitude"])
    center = float(cfg["center_s"])
    sigma = float(cfg["sigma_s"])
    values = amplitude * np.exp(-((times - center) / sigma) ** 2)

    active_start = float(cfg["active_start_s"])
    active_end = float(cfg["active_end_s"])
    positive_floor = float(cfg["positive_floor_absolute"])
    active = (times >= active_start - 1e-15) & (times <= active_end + 1e-15)
    values[~active] = 0.0
    values[active] = np.maximum(values[active], positive_floor)

    data = np.column_stack([times, values])
    lines = "".join(f"{t:.18e} {value:.18e}\n" for t, value in data)

    require(data.shape == (row_count, 2), "Unexpected STF shape")
    require(
        math.isclose(float(np.max(values)), amplitude, abs_tol=1e-14),
        "STF peak does not match configured amplitude",
    )
    require(
        int(np.count_nonzero(values)) == int(cfg["expected_nonzero_count"]),
        "STF nonzero-count mismatch",
    )
    return lines, data


def format_number(value: float) -> str:
    if math.isclose(value, round(value), abs_tol=1e-12):
        return f"{value:.1f}"
    return f"{value:g}"


def source_block(source: list[float], source_cfg: dict[str, Any]) -> list[str]:
    x, y, z = (float(item) for item in source)
    direction = [float(item) for item in source_cfg["source_direction"]]
    return [
        "source {",
        f"coords = {format_number(x)} {format_number(y)} {format_number(z)};",
        f"type = {source_cfg['source_type']};",
        "dir = " + " ".join(format_number(item) for item in direction) + ";",
        "func = file;",
        f'time_file = "{source_cfg["source_time_function_file"]}";',
        "};",
    ]


def generate_input_spec(
    spec: dict[str, Any],
    *,
    model: str,
) -> str:
    workspace_cfg = spec["bootstrap_workspaces"][
        "true_observed" if model == "true_layered" else "initial_iter_000"
    ]
    forward = spec["forward_operator"]
    receiver = spec["receivers"]["physical"]
    scheme = forward["time_scheme"]

    lines = [
        f'run_name = "{workspace_cfg["run_name"]}";',
        "",
        f"sim_time = {float(forward['simulation_time_s']):.16e};",
        "",
        f'mesh_file = "{spec["sem3d_mesh"]["mesh_file_stem"]}";',
        'mat_file = "material.input";',
        "",
        f"dim = {int(forward['dimension'])};",
        f"ngll = {int(forward['ngll'])};",
        "",
        "snapshots {",
        f"save_snap = {'true' if workspace_cfg['save_snapshots'] else 'false'};",
        "snap_interval = 0.02;",
        "select all;",
        "};",
        "",
        f"save_traces = {'true' if workspace_cfg['save_traces'] else 'false'};",
        f"traces_format = {workspace_cfg.get('traces_format', 'hdf5')};",
        "",
        'capteurs "UU" {',
        f"type = {receiver['type']};",
        f'file = "{receiver["file"]}";',
        f"period = {int(receiver['period'])};",
        "};",
        "",
        "prorep = false;",
        "prorep_iter = 1000;",
        "",
    ]

    sources = forward["source_coordinates_m"]
    require(
        len(sources) == int(forward["source_count"]),
        "source_count does not match source_coordinates_m",
    )
    for source in sources:
        lines.extend(source_block(source, forward))
        lines.append("")

    lines.extend(
        [
            "time_scheme {",
            f"accel_scheme = {'true' if scheme['accel_scheme'] else 'false'};",
            f"veloc_scheme = {'true' if scheme['veloc_scheme'] else 'false'};",
            f"alpha = {scheme['alpha']};",
            f"beta = {scheme['beta']};",
            f"gamma = {scheme['gamma']};",
            f"courant = {scheme['courant']};",
            "};",
            "",
            "out_variables {",
            "enP = 0;",
            "enS = 0;",
            "evol = 0;",
            "pre = 0;",
            "dis = 1;",
            "vel = 1;",
            "acc = 1;",
            "edev = 0;",
            "sdev = 0;",
            "edevpl = 0;",
            "};",
        ]
    )
    return "\n".join(lines) + "\n"


def kappa_from_lambda_mu(lambda_pa: float, mu_pa: float) -> float:
    """Compute bulk modulus using the validated floating-point operation order."""
    return float(lambda_pa) + (2.0 / 3.0) * float(mu_pa)


def material_fields(
    spec: dict[str, Any],
    *,
    model: str,
) -> dict[str, np.ndarray]:
    grid = spec["material_grid"]
    shape = tuple(int(value) for value in grid["shape"])
    require(len(shape) == 3, "material_grid.shape must have three dimensions")

    density_value = float(spec["material_models"]["density_kg_m3"])
    density = np.full(shape, density_value, dtype=np.float64)

    if model == "initial_homogeneous":
        cfg = spec["material_models"]["initial_homogeneous"]
        mu_value = float(cfg["mu_pa"])
        kappa_value = kappa_from_lambda_mu(
            float(cfg["lambda_pa"]),
            mu_value,
        )
        mu = np.full(shape, mu_value, dtype=np.float64)
        kappa = np.full(shape, kappa_value, dtype=np.float64)
    elif model == "true_layered":
        domain = spec["domain"]
        z_values = np.linspace(
            float(domain["z_min_m"]),
            float(domain["z_max_m"]),
            shape[0],
            dtype=np.float64,
        )
        mu_z = np.empty(shape[0], dtype=np.float64)
        kappa_z = np.empty(shape[0], dtype=np.float64)
        layers = spec["material_models"]["true_layered"]["layers"]

        for index, z_value in enumerate(z_values):
            selected: dict[str, Any] | None = None
            if z_value < -27.0:
                selected = layers[0]
            elif z_value < -12.0:
                selected = layers[1]
            else:
                selected = layers[2]
            mu_value = float(selected["mu_pa"])
            mu_z[index] = mu_value
            kappa_z[index] = kappa_from_lambda_mu(
                float(selected["lambda_pa"]),
                mu_value,
            )

        mu = np.broadcast_to(mu_z[:, None, None], shape).copy()
        kappa = np.broadcast_to(kappa_z[:, None, None], shape).copy()
    else:
        raise ValueError(f"Unsupported material model: {model}")

    for label, field in {
        "Kappa": kappa,
        "Mu": mu,
        "Density": density,
    }.items():
        require(field.shape == shape, f"{label} shape mismatch")
        require(np.isfinite(field).all(), f"{label} contains non-finite values")

    return {
        "Kappa": kappa,
        "Mu": mu,
        "Density": density,
    }


def write_material_h5(
    directory: Path,
    spec: dict[str, Any],
    fields: dict[str, np.ndarray],
) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    h5_cfg = spec["material_grid"]["hdf5"]
    attrs = h5_cfg["file_attributes"]

    for label, field in fields.items():
        path = directory / f"Mat_0_{label}.h5"
        with h5py.File(path, "w") as h5:
            h5.attrs["xMinGlob"] = np.asarray(
                attrs["xMinGlob"],
                dtype=np.float64,
            )
            h5.attrs["xMaxGlob"] = np.asarray(
                attrs["xMaxGlob"],
                dtype=np.float64,
            )
            h5.create_dataset(
                spec["material_grid"]["dataset"],
                data=np.asarray(field, dtype=np.float64),
            )


def build_files(
    spec: dict[str, Any],
    *,
    model: str,
) -> tuple[dict[str, str], dict[str, np.ndarray], np.ndarray, np.ndarray]:
    stations_text, stations = generate_stations(spec)
    stf_text, stf = generate_stf(spec)
    text_files = {
        "mesh.input": generate_mesh_input(spec),
        "mat.dat": generate_mat_dat(spec),
        "mater.in": generate_mater_in(spec),
        "material.input": generate_material_input(spec),
        "material.spec": generate_material_spec(spec),
        "input.spec": generate_input_spec(spec, model=model),
        "stations.txt": stations_text,
        "gaussian_stf.txt": stf_text,
    }
    fields = material_fields(spec, model=model)
    return text_files, fields, stations, stf


def validate_spec(spec: dict[str, Any]) -> None:
    require(spec.get("schema_version") == 1, "Unsupported schema_version")
    require(
        spec.get("name") == "fathi_reduced_3x3_12p5",
        "Unexpected benchmark specification name",
    )
    require(
        int(spec["forward_operator"]["source_count"]) == 9,
        "Canonical reduced benchmark requires nine sources",
    )
    require(
        int(spec["receivers"]["physical"]["count"]) == 225,
        "Canonical reduced benchmark requires 225 physical receivers",
    )
    require(
        tuple(spec["material_grid"]["shape"]) == (41, 33, 33),
        "Canonical material shape must be (41, 33, 33)",
    )


def print_plan(
    *,
    root: Path,
    config_path: Path,
    output: Path,
    model: str,
    text_files: dict[str, str],
    fields: dict[str, np.ndarray],
    stations: np.ndarray,
    stf: np.ndarray,
) -> None:
    print("STANDALONE SEM3D WORKSPACE GENERATOR")
    print("====================================")
    print()
    print(f"root = {root}")
    print(f"config = {config_path}")
    print(f"model = {model}")
    print(f"output = {output}")
    print()
    print("generated text files:")
    for relative, text in text_files.items():
        print(
            f"  {relative:20s} "
            f"{len(text.encode('utf-8')):10d} bytes "
            f"{len(text.splitlines()):6d} lines"
        )
    print()
    print(f"stations = {stations.shape}")
    print(f"stf = {stf.shape}")
    print(
        "stf peak = "
        f"{float(np.max(stf[:, 1])):.16e} "
        f"at t={float(stf[np.argmax(stf[:, 1]), 0]):.16e}"
    )
    print()
    print("material fields:")
    for label, field in fields.items():
        print(
            f"  {label:8s} "
            f"shape={field.shape} "
            f"dtype={field.dtype} "
            f"min={float(np.min(field)):.16e} "
            f"max={float(np.max(field)):.16e}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Generate a complete reduced Fathi-inspired SEM3D workspace "
            "from a committed JSON specification."
        )
    )
    parser.add_argument(
        "--config",
        default="configs/fathi_reduced_3x3_12p5.json",
        help="Benchmark specification JSON.",
    )
    parser.add_argument(
        "--model",
        choices=("true_layered", "initial_homogeneous"),
        required=True,
        help="Material model to generate.",
    )
    parser.add_argument(
        "--output",
        help=(
            "Output workspace. When omitted, use the canonical directory "
            "defined by bootstrap_workspaces in the specification."
        ),
    )
    parser.add_argument(
        "--write",
        action="store_true",
        help="Write files. Without this flag, only print the plan.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing output directory.",
    )
    args = parser.parse_args()

    root = repository_root()
    config_path = resolve_path(root, args.config)
    spec = load_json(config_path)
    validate_spec(spec)

    workspace_key = (
        "true_observed"
        if args.model == "true_layered"
        else "initial_iter_000"
    )
    configured_output = spec["bootstrap_workspaces"][workspace_key][
        "output_directory"
    ]
    output = resolve_path(root, args.output or configured_output)

    text_files, fields, stations, stf = build_files(
        spec,
        model=args.model,
    )

    print_plan(
        root=root,
        config_path=config_path,
        output=output,
        model=args.model,
        text_files=text_files,
        fields=fields,
        stations=stations,
        stf=stf,
    )

    if not args.write:
        print()
        print("PLAN ONLY: no files were written.")
        print("RESULT = PASS_STANDALONE_WORKSPACE_GENERATOR_PLAN")
        return

    if output.exists():
        if not args.overwrite:
            raise FileExistsError(
                f"Output already exists: {output}\n"
                "Use --overwrite only after checking the existing directory."
            )
        shutil.rmtree(output)

    output.mkdir(parents=True, exist_ok=False)

    try:
        for relative, text in text_files.items():
            path = output / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text, encoding="utf-8", newline="\n")

        write_material_h5(output / "mat" / "h5", spec, fields)

        runtime_names = ("traces", "res", "logs", "prot", "mirror")
        present_runtime = [
            name for name in runtime_names if (output / name).exists()
        ]
        require(
            not present_runtime,
            "Fresh generated workspace unexpectedly contains runtime outputs: "
            + ", ".join(present_runtime),
        )
    except Exception:
        shutil.rmtree(output, ignore_errors=True)
        raise

    print()
    print("Workspace written successfully.")
    print(f"output = {output}")
    print("RESULT = PASS_STANDALONE_WORKSPACE_GENERATOR_WRITE")


if __name__ == "__main__":
    main()
