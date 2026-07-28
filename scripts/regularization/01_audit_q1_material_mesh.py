from __future__ import annotations

from pathlib import Path
import argparse
import json
import re

import numpy as np
from scipy import sparse


ROOT = Path(__file__).resolve().parents[2]


def resolve(path: str) -> Path:
    p = Path(path)
    return p if p.is_absolute() else ROOT / p


def load_config(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Missing configuration file: {path}")

    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def extract_source_evidence(source_path: Path) -> list[str]:
    if not source_path.exists():
        return [f"Source file not found: {source_path}"]

    keywords = (
        "nx",
        "ny",
        "nz",
        "linspace",
        "gauss",
        "quadrature",
        "shape",
        "local",
        "jacob",
        "connect",
        "element",
        "kron",
        "flatten",
        "order",
    )

    lines = source_path.read_text(
        encoding="utf-8",
        errors="ignore",
    ).splitlines()

    evidence = []

    for lineno, line in enumerate(lines, start=1):
        if any(keyword in line.lower() for keyword in keywords):
            evidence.append(f"{lineno:4d}: {line}")

    return evidence


def extract_linspace_definitions(source_path: Path) -> list[str]:
    if not source_path.exists():
        return []

    text = source_path.read_text(
        encoding="utf-8",
        errors="ignore",
    )

    pattern = re.compile(
        r"^\s*([xyz])\s*=\s*np\.linspace\((.*?)\)",
        re.MULTILINE,
    )

    return [
        f"{axis} = np.linspace({arguments})"
        for axis, arguments in pattern.findall(text)
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default=(
            "benchmark_fathi_tv/config/"
            "tv_config_iter008_to_iter009.json"
        ),
    )
    args = parser.parse_args()

    config_path = resolve(args.config)
    config = load_config(config_path)

    output_dir = resolve(config["tv_transition_dir"]) / "mesh_audit"
    output_dir.mkdir(parents=True, exist_ok=True)

    state_path = resolve(config["parent_state"])
    full_mtilde_path = resolve(config["full_mtilde"])
    active_indices_path = resolve(config["active_indices"])
    active_coords_path = resolve(config["active_coords"])

    builder_path = (
        ROOT / "scripts/audit/302_build_q1_consistent_mtilde.py"
    )

    state = np.load(state_path, allow_pickle=True)

    required_state_keys = ["mu", "kappa", "density"]

    for key in required_state_keys:
        if key not in state.files:
            raise KeyError(
                f"Missing key '{key}' in parent state. "
                f"Available keys: {state.files}"
            )

    if "lambda" in state.files:
        lambda_field = np.asarray(state["lambda"], dtype=float)
        lambda_key = "lambda"
    elif "lambda_field" in state.files:
        lambda_field = np.asarray(
            state["lambda_field"],
            dtype=float,
        )
        lambda_key = "lambda_field"
    else:
        kappa = np.asarray(state["kappa"], dtype=float)
        mu = np.asarray(state["mu"], dtype=float)
        lambda_field = kappa - (2.0 / 3.0) * mu
        lambda_key = "reconstructed_from_kappa_mu"

    mu_field = np.asarray(state["mu"], dtype=float)
    kappa_field = np.asarray(state["kappa"], dtype=float)
    density_field = np.asarray(state["density"], dtype=float)

    expected_shape = (
        int(config["mesh"]["nz"]),
        int(config["mesh"]["ny"]),
        int(config["mesh"]["nx"]),
    )

    fields = {
        "lambda": lambda_field,
        "mu": mu_field,
        "kappa": kappa_field,
        "density": density_field,
    }

    for name, field in fields.items():
        if field.shape != expected_shape:
            raise RuntimeError(
                f"{name} shape mismatch: "
                f"expected {expected_shape}, got {field.shape}"
            )

        if not np.all(np.isfinite(field)):
            raise RuntimeError(
                f"{name} contains non-finite values"
            )

    nx = int(config["mesh"]["nx"])
    ny = int(config["mesh"]["ny"])
    nz = int(config["mesh"]["nz"])

    full_node_count = nx * ny * nz
    expected_element_count = (
        (nx - 1) * (ny - 1) * (nz - 1)
    )

    full_mtilde = sparse.load_npz(full_mtilde_path)
    active_indices = np.load(active_indices_path)
    active_coords = np.load(active_coords_path)

    if full_mtilde.shape != (
        full_node_count,
        full_node_count,
    ):
        raise RuntimeError(
            f"Full Mtilde shape mismatch: {full_mtilde.shape}"
        )

    if active_indices.ndim != 1:
        raise RuntimeError(
            f"active_indices must be 1D, got {active_indices.shape}"
        )

    if active_coords.shape != (active_indices.size, 3):
        raise RuntimeError(
            "active_coords and active_indices are inconsistent: "
            f"{active_coords.shape}, {active_indices.shape}"
        )

    if len(np.unique(active_indices)) != active_indices.size:
        raise RuntimeError("Duplicate active indices detected")

    if active_indices.min() < 0:
        raise RuntimeError("Negative active index detected")

    if active_indices.max() >= full_node_count:
        raise RuntimeError(
            "Active index exceeds the full material-grid size"
        )

    lambda_reference = float(np.median(lambda_field))
    mu_reference = float(np.median(mu_field))

    metadata = {
        "config": str(config_path.relative_to(ROOT)),
        "parent_state": str(state_path.relative_to(ROOT)),
        "lambda_source_key": lambda_key,
        "array_shape": list(expected_shape),
        "axis_convention": config["mesh"]["array_order"],
        "flatten_order": config["mesh"]["flatten_order"],
        "fastest_axis": config["mesh"]["fastest_axis"],
        "nx": nx,
        "ny": ny,
        "nz": nz,
        "full_node_count": full_node_count,
        "expected_q1_element_count": expected_element_count,
        "expected_q1_element_shape": [
            nz - 1,
            ny - 1,
            nx - 1,
        ],
        "full_mtilde_shape": list(full_mtilde.shape),
        "full_mtilde_nnz": int(full_mtilde.nnz),
        "active_node_count": int(active_indices.size),
        "inactive_node_count": int(
            full_node_count - active_indices.size
        ),
        "active_index_min": int(active_indices.min()),
        "active_index_max": int(active_indices.max()),
        "active_indices_unique": int(
            len(np.unique(active_indices))
        ),
        "active_coords_shape": list(active_coords.shape),
        "active_coord_min": active_coords.min(axis=0).tolist(),
        "active_coord_max": active_coords.max(axis=0).tolist(),
        "lambda_reference_pa": lambda_reference,
        "mu_reference_pa": mu_reference,
        "lambda_min_pa": float(lambda_field.min()),
        "lambda_max_pa": float(lambda_field.max()),
        "mu_min_pa": float(mu_field.min()),
        "mu_max_pa": float(mu_field.max()),
        "builder_script": str(builder_path.relative_to(ROOT)),
        "linspace_definitions_found": extract_linspace_definitions(
            builder_path
        ),
    }

    metadata_path = output_dir / "q1_material_mesh_metadata.json"
    metadata_path.write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )

    evidence = extract_source_evidence(builder_path)
    evidence_path = output_dir / "q1_builder_source_evidence.txt"
    evidence_path.write_text(
        "\n".join(evidence) + "\n",
        encoding="utf-8",
    )

    report = []

    report.append("Q1 MATERIAL / CONTROL MESH AUDIT")
    report.append("================================")
    report.append("")
    report.append(f"Parent state: {state_path}")
    report.append(f"Lambda source: {lambda_key}")
    report.append("")
    report.append("Full material field")
    report.append("-------------------")
    report.append(f"array shape = {expected_shape}")
    report.append(f"nx, ny, nz = {nx}, {ny}, {nz}")
    report.append(f"full nodes = {full_node_count}")
    report.append(
        "expected Q1 elements = "
        f"{nx - 1} x {ny - 1} x {nz - 1} "
        f"= {expected_element_count}"
    )
    report.append("")
    report.append("DOF ordering")
    report.append("------------")
    report.append(
        f"array convention = {config['mesh']['array_order']}"
    )
    report.append(
        f"flatten order = {config['mesh']['flatten_order']}"
    )
    report.append(
        f"fastest axis = {config['mesh']['fastest_axis']}"
    )
    report.append("")
    report.append("Full Mtilde")
    report.append("-----------")
    report.append(f"shape = {full_mtilde.shape}")
    report.append(f"nnz = {full_mtilde.nnz}")
    report.append("")
    report.append("Active control subset")
    report.append("---------------------")
    report.append(f"active nodes = {active_indices.size}")
    report.append(
        f"inactive nodes = "
        f"{full_node_count - active_indices.size}"
    )
    report.append(
        f"active indices unique = "
        f"{len(np.unique(active_indices))}"
    )
    report.append(f"active coords shape = {active_coords.shape}")
    report.append(
        f"active coordinate min = "
        f"{active_coords.min(axis=0)}"
    )
    report.append(
        f"active coordinate max = "
        f"{active_coords.max(axis=0)}"
    )
    report.append("")
    report.append("Parameter scaling")
    report.append("-----------------")
    report.append(
        f"lambda reference = {lambda_reference:.16e} Pa"
    )
    report.append(
        f"mu reference = {mu_reference:.16e} Pa"
    )
    report.append("")
    report.append("Source-code evidence")
    report.append("--------------------")

    definitions = metadata["linspace_definitions_found"]

    if definitions:
        report.extend(definitions)
    else:
        report.append(
            "No direct x/y/z np.linspace definition was "
            "automatically extracted."
        )

    report.append("")
    report.append("RESULT = PASS_MESH_METADATA_AUDIT")
    report.append("")
    report.append(
        "Important: this verifies dimensions, ordering, "
        "matrix size, active indices and parameter scales. "
        "The exact Q1 shape functions and quadrature still "
        "have to be confirmed from the builder source."
    )

    report_text = "\n".join(report) + "\n"

    report_path = output_dir / "q1_material_mesh_audit.txt"
    report_path.write_text(report_text, encoding="utf-8")

    print(report_text)
    print(f"metadata = {metadata_path}")
    print(f"evidence = {evidence_path}")


if __name__ == "__main__":
    main()
