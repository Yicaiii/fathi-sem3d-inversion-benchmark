"""Map a certified external exact gradient into optimizer and Mtilde spaces."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping

import h5py
import numpy as np
from scipy.sparse import load_npz, save_npz
from scipy.sparse.linalg import spsolve

from scripts.exact_adjoint.audit_real_s43_pml_material_sensitivity import (
    production_sample_xyz,
)
from scripts.exact_adjoint.build_real_s43_coupled_mass import trilinear_sample
from scripts.exact_adjoint.s43_external_forward import (
    common_paths,
    load_certified_reference,
    sha256_file,
)
from scripts.fathi_benchmark.certified_gradient_bridge_utils import (
    array_stats,
    build_solid_row_map,
    configured_path,
    material_grid_coordinates,
    relative_error,
    trilinear_transpose,
)
from scripts.fathi_benchmark.immutable_assets import (
    validate_immutable_asset_manifest,
)
from scripts.fathi_benchmark.iteration_context import (
    IterationPaths,
    build_iteration_paths,
)
from scripts.fathi_benchmark.runtime_paths import (
    iteration_runtime_paths,
    resolve_path,
)
from scripts.fathi_benchmark.current_pipeline_contracts import (
    accepted_model_result,
    exact_reverse_result,
    gradient_bridge_result,
    retained_primal_result,
)


PASS_RESULT = "PASS_CERTIFIED_EXTERNAL_OPTIMIZER_BRIDGE"
HISTORICAL_REVERSE_PASS = (
    "PASS_STAGE5O_EXTERNAL_PHYSICAL_EXACT_ADJOINT_CERTIFICATION"
)
CURRENT_GRADIENT_NAMES = (
    "solid_lambda",
    "solid_mu",
    "pml_lambda",
    "pml_mu",
)
CURRENT_REVERSE_GATES = (
    "all_reverse_transitions_completed",
    "next_transition_is_minus_one",
    "reverse_remained_finite",
    "retained_replay_endpoints_verified",
    "exact_physical_receiver_transpose_used",
    "fixed_dt_trapezoidal_residual_weighting_used",
    "certified_material_vjp_used",
    "certified_adjoint_step_used",
)


def current_reverse_result(iteration: int) -> str:
    return exact_reverse_result(iteration)


def current_gradient_paths(reverse_dir: Path) -> dict[str, Path]:
    return {
        name: reverse_dir / f"gradient_{name}.npy"
        for name in CURRENT_GRADIENT_NAMES
    }


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"JSON manifest must contain an object: {path}")
    return value


def _recorded_path(repo: Path, record: Mapping[str, Any]) -> Path:
    value = record.get("resolved_path") or record.get("path")
    if not value:
        raise RuntimeError("provenance record has no path")
    return resolve_path(str(value), base=repo)


def _manifest_path(repo: Path, value: object) -> Path:
    if not value:
        raise RuntimeError("provenance manifest path is absent")
    return resolve_path(str(value), base=repo)


def _verify_file_record(
    repo: Path,
    record: Mapping[str, Any],
    *,
    label: str,
    expected_path: Path | None = None,
) -> Path:
    path = _recorded_path(repo, record)
    if expected_path is not None and path != expected_path.resolve():
        raise RuntimeError(f"{label} provenance path mismatch")
    if not path.is_file():
        raise RuntimeError(f"{label} provenance file is missing: {path}")
    expected_hash = str(record.get("sha256", ""))
    if not expected_hash or sha256_file(path) != expected_hash:
        raise RuntimeError(f"{label} provenance SHA-256 mismatch")
    return path


def _asset_paths(
    manifest: Mapping[str, Any], *, repo: Path, runtime_root: Path
) -> dict[str, Path]:
    bases = {"repository_root": repo, "runtime_root": runtime_root}
    result: dict[str, Path] = {}
    for asset in manifest["assets"]:
        base_name = str(asset["path_base"])
        if base_name not in bases:
            raise RuntimeError(f"unsupported immutable asset base: {base_name}")
        result[str(asset["asset_id"])] = (
            bases[base_name] / str(asset["source_path"])
        ).resolve()
    return result


def validate_current_reverse_contract(
    *,
    repo: Path,
    config_path: Path,
    config: Mapping[str, Any],
    engine_path: Path,
    engine: Mapping[str, Any],
    paths: IterationPaths,
    iteration: int,
    reverse_dir: Path,
    reverse_summary: Mapping[str, Any],
    operator_dir: Path,
    topology_dir: Path,
    operator_content_signature: str,
    topology_content_signature: str,
) -> dict[str, Any]:
    """Validate only the CURRENT reverse-to-bridge interface and provenance."""

    expected_result = current_reverse_result(iteration)
    if reverse_summary.get("result") != expected_result:
        raise RuntimeError(
            "CURRENT reverse result mismatch: "
            f"{reverse_summary.get('result')} != {expected_result}"
        )
    if int(reverse_summary.get("iteration", -1)) != int(iteration):
        raise RuntimeError("CURRENT reverse iteration mismatch")
    if reverse_summary.get("transition") != paths.identity.transition_id:
        raise RuntimeError("CURRENT reverse transition mismatch")
    expected_samples = int(config["forward_operator"]["expected_sample_count"])
    reverse = reverse_summary.get("reverse", {})
    if int(reverse.get("steps", -1)) != expected_samples:
        raise RuntimeError("CURRENT reverse step count mismatch")
    if int(reverse.get("next_transition", 0)) != -1:
        raise RuntimeError("CURRENT reverse did not finish at transition -1")
    if reverse.get("finite") is not True:
        raise RuntimeError("CURRENT reverse finite gate is false")
    gates = reverse_summary.get("gates", {})
    failed = [name for name in CURRENT_REVERSE_GATES if gates.get(name) is not True]
    if failed:
        raise RuntimeError("CURRENT reverse gates failed: " + ", ".join(failed))

    canonical_reverse = (paths.exact_reverse / "production_reverse").resolve()
    if reverse_dir.resolve() != canonical_reverse:
        raise RuntimeError("CURRENT material covector is outside canonical reverse path")
    gradient_paths = current_gradient_paths(canonical_reverse)
    output_hashes = reverse_summary.get("output_hashes", {}).get("gradients", {})
    gradient_metadata = reverse_summary.get("gradient", {})
    for name, path in gradient_paths.items():
        if not path.is_file():
            raise RuntimeError(f"missing CURRENT gradient file: {path.name}")
        actual = sha256_file(path)
        if actual != str(output_hashes.get(name, "")):
            raise RuntimeError(f"CURRENT gradient output SHA-256 mismatch: {name}")
        metadata = gradient_metadata.get(name, {})
        if actual != str(metadata.get("sha256", "")):
            raise RuntimeError(f"CURRENT gradient metadata SHA-256 mismatch: {name}")
        if metadata.get("finite") is not True:
            raise RuntimeError(f"CURRENT gradient finite gate is false: {name}")
        if _manifest_path(repo, metadata.get("path")) != path.resolve():
            raise RuntimeError(f"CURRENT gradient metadata path mismatch: {name}")

    canonical_primal = (paths.exact_reverse / "primal_forward" / "summary.json").resolve()
    for key in ("reference_manifest", "parent_forward_summary"):
        if _manifest_path(repo, reverse_summary.get(key)) != canonical_primal:
            raise RuntimeError(f"CURRENT reverse {key} is not canonical primal summary")
    input_hashes = reverse_summary.get("input_hashes", {})
    _verify_file_record(
        repo,
        input_hashes.get("primal_forward_summary", {}),
        label="canonical primal-forward summary",
        expected_path=canonical_primal,
    )
    primal = _read_json(canonical_primal)
    primal_identity = (
        primal.get("run_id"),
        int(primal.get("parent_iteration", -1)),
        int(primal.get("child_iteration", -1)),
        primal.get("transition"),
    )
    expected_identity = (
        paths.identity.run_id,
        paths.identity.parent_iteration,
        paths.identity.child_iteration,
        paths.identity.transition_id,
    )
    if (
        primal_identity != expected_identity
        or primal.get("result") != retained_primal_result(iteration)
    ):
        raise RuntimeError("canonical primal-forward identity/result mismatch")

    _verify_file_record(
        repo,
        input_hashes.get("runtime_config", {}),
        label="runtime config",
        expected_path=config_path,
    )
    _verify_file_record(
        repo,
        input_hashes.get("iteration_engine_config", {}),
        label="iteration-engine config",
        expected_path=engine_path,
    )
    driver_assets = input_hashes.get("driver_assets", {})
    _verify_file_record(
        repo,
        driver_assets.get("config", {}),
        label="reverse driver runtime config",
        expected_path=config_path,
    )

    accepted_summary = (paths.parent_accepted / "accepted_summary.json").resolve()
    _verify_file_record(
        repo,
        input_hashes.get("accepted_parent_summary", {}),
        label="accepted parent summary",
        expected_path=accepted_summary,
    )
    accepted = _read_json(accepted_summary)
    if (
        int(accepted.get("iter", -1)) != int(iteration)
        or accepted.get("run") != paths.identity.run_id
        or accepted.get("result") != accepted_model_result(iteration)
    ):
        raise RuntimeError("accepted parent summary identity/result mismatch")
    material_dir = (
        paths.parent_accepted / str(engine["material"]["directory"])
    ).resolve()
    if _manifest_path(repo, primal.get("material_dir")) != material_dir:
        raise RuntimeError("canonical primal material directory mismatch")
    primal_material_hashes = primal.get("material_sha256", {})
    material_records = input_hashes.get("parent_material", {})
    for component in ("kappa", "mu", "density"):
        material_path = (
            material_dir / str(engine["material"]["files"][component])
        ).resolve()
        _verify_file_record(
            repo,
            material_records.get(component, {}),
            label=f"accepted parent {component}",
            expected_path=material_path,
        )
        if sha256_file(material_path) != str(primal_material_hashes.get(component, "")):
            raise RuntimeError(f"primal/accepted material SHA-256 mismatch: {component}")
        if sha256_file(material_path) != str(
            accepted.get("material_sha256", {}).get(material_path.name, "")
        ):
            raise RuntimeError(f"accepted-summary material SHA-256 mismatch: {component}")

    canonical_current = (
        paths.exact_reverse / "primal_forward" / "current_external_receiver.npy"
    ).resolve()
    _verify_file_record(
        repo,
        input_hashes.get("current_external_receiver", {}),
        label="current external receiver",
        expected_path=canonical_current,
    )
    primal_current = primal.get("current_external_receiver", {})
    if (
        _manifest_path(repo, primal_current.get("path")) != canonical_current
        or sha256_file(canonical_current) != str(primal_current.get("sha256", ""))
        or sha256_file(canonical_current)
        != str(accepted.get("external_receiver_sha256", ""))
    ):
        raise RuntimeError("primal/current external receiver provenance mismatch")
    accepted_trace = _verify_file_record(
        repo,
        input_hashes.get("accepted_external_receiver", {}),
        label="accepted external receiver",
    )
    if sha256_file(accepted_trace) != sha256_file(canonical_current):
        raise RuntimeError("current receiver is not the accepted external receiver")
    true_record = input_hashes.get("true_external_receiver", {})
    true_path = _verify_file_record(
        repo, true_record, label="true external receiver"
    )
    primal_true = primal.get("true_external_receiver", {})
    if (
        _manifest_path(repo, primal_true.get("path")) != true_path
        or str(primal_true.get("sha256", "")) != str(true_record.get("sha256", ""))
        or str(true_record.get("sha256", ""))
        != str(accepted.get("true_external_sha256", ""))
    ):
        raise RuntimeError("primal/TRUE external receiver provenance mismatch")

    gll_path = (operator_dir / "gll_coordinates.npy").resolve()
    weights_path = (operator_dir / "gll_weights.npy").resolve()
    _verify_file_record(
        repo, driver_assets.get("gll", {}), label="GLL coordinates", expected_path=gll_path
    )
    _verify_file_record(
        repo,
        driver_assets.get("weights", {}),
        label="GLL weights",
        expected_path=weights_path,
    )
    topology = driver_assets.get("topology", {})
    if _recorded_path(repo, topology) != topology_dir.resolve():
        raise RuntimeError("reverse topology path differs from certified bridge asset")
    if str(topology.get("content_signature_sha256", "")) != str(
        topology_content_signature
    ):
        raise RuntimeError("reverse topology identity mismatch")
    if not operator_content_signature:
        raise RuntimeError("certified exact-spatial-operator identity is absent")

    return {
        "result_contract": expected_result,
        "gradient_paths": gradient_paths,
        "canonical_primal_summary": canonical_primal,
        "material_covector_input": canonical_reverse,
        "optimizer_bridge_output": paths.gradient_root.resolve(),
        "registered_physical_gradient": paths.mtilde_solve.resolve(),
        "operator_content_signature_sha256": operator_content_signature,
        "topology_content_signature_sha256": topology_content_signature,
    }


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def coordinate_index(rows: np.ndarray, decimals: int) -> dict[tuple, int]:
    result = {
        tuple(np.round(row, decimals)): index for index, row in enumerate(rows)
    }
    if len(result) != len(rows):
        raise RuntimeError("coordinate index contains duplicate rows")
    return result


def resolve_current_parent_material_contract(
    *,
    engine: Mapping[str, Any],
    parent_workspace: str | Path,
    expected_shape: tuple[int, ...],
) -> dict[str, Any]:
    """Resolve and preflight CURRENT parent materials from engine.material."""

    material = engine.get("material")
    if not isinstance(material, Mapping):
        raise RuntimeError(
            "CURRENT iteration-engine material contract must be a mapping"
        )
    directory_value = material.get("directory")
    if not isinstance(directory_value, str) or not directory_value.strip():
        raise RuntimeError(
            "CURRENT iteration-engine material contract requires directory"
        )
    files = material.get("files")
    if not isinstance(files, Mapping):
        raise RuntimeError(
            "CURRENT iteration-engine material contract requires files mapping"
        )
    for component in ("kappa", "mu"):
        value = files.get(component)
        if not isinstance(value, str) or not value.strip():
            raise RuntimeError(
                "CURRENT iteration-engine material files requires " + component
            )
    dataset = material.get("dataset")
    if not isinstance(dataset, str) or not dataset.strip():
        raise RuntimeError(
            "CURRENT iteration-engine material contract requires dataset"
        )

    parent = Path(parent_workspace).expanduser().resolve()
    relative_directory = Path(directory_value)
    if relative_directory.is_absolute():
        raise RuntimeError("CURRENT material directory must be parent-relative")
    material_dir = (parent / relative_directory).resolve()
    try:
        material_dir.relative_to(parent)
    except ValueError as exc:
        raise RuntimeError("CURRENT material directory escapes parent workspace") from exc
    if not material_dir.is_dir():
        raise RuntimeError(
            f"CURRENT parent material directory is missing: {material_dir}"
        )

    shape = tuple(int(value) for value in expected_shape)
    if not shape or any(value <= 0 for value in shape):
        raise RuntimeError("CURRENT expected material shape is invalid")
    metadata: dict[str, dict[str, Any]] = {}
    for component in ("kappa", "mu"):
        relative_file = Path(str(files[component]))
        if relative_file.is_absolute():
            raise RuntimeError(
                f"CURRENT {component} material file must be directory-relative"
            )
        path = (material_dir / relative_file).resolve()
        try:
            path.relative_to(material_dir)
        except ValueError as exc:
            raise RuntimeError(
                f"CURRENT {component} material file escapes material directory"
            ) from exc
        if not path.is_file():
            raise RuntimeError(
                f"CURRENT parent {component} H5 file is missing: {path}"
            )
        with h5py.File(path, "r") as handle:
            if dataset not in handle:
                raise RuntimeError(
                    f"CURRENT parent {component} H5 dataset is missing: "
                    f"{dataset} in {path}"
                )
            actual_shape = tuple(int(value) for value in handle[dataset].shape)
            if actual_shape != shape:
                raise RuntimeError(
                    f"CURRENT parent {component} H5 shape mismatch: "
                    f"{actual_shape} != {shape}"
                )
            metadata[component] = {
                "path": str(path),
                "dataset": dataset,
                "shape": list(actual_shape),
                "dtype": str(handle[dataset].dtype),
            }
    return {
        "directory": str(material_dir),
        "dataset": dataset,
        "files": metadata,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", default=".")
    parser.add_argument("--config", required=True)
    parser.add_argument("--iteration", type=int, required=True)
    parser.add_argument("--reference-manifest")
    parser.add_argument("--reverse-dir")
    parser.add_argument("--out-dir")
    parser.add_argument("--batch-size", type=int, default=131072)
    args = parser.parse_args()
    if args.iteration < 0:
        parser.error("--iteration must be nonnegative")
    if args.batch_size < 1:
        parser.error("--batch-size must be positive")

    repo = Path(args.repo).expanduser().resolve()
    config_path = resolve_path(args.config, base=repo)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    runtime = iteration_runtime_paths(
        config, args.iteration, repo_root=repo
    )
    run = str(config["benchmark_name"])
    engine_path = (repo / "configs" / f"{run}_iteration_engine.json").resolve()
    engine = _read_json(engine_path)
    paths = build_iteration_paths(
        engine,
        args.iteration,
        child_iteration=args.iteration + 1,
        repository_root=repo,
        runtime_root=Path(runtime["runtime_root"]),
    )
    if paths.transition_root != Path(runtime["transition_root"]):
        raise RuntimeError("runtime/iteration-engine transition path mismatch")
    if paths.parent_accepted != Path(runtime["parent_workspace"]):
        raise RuntimeError("runtime/iteration-engine parent path mismatch")
    legacy_reference_path = (
        resolve_path(args.reference_manifest, base=repo)
        if args.reference_manifest
        else repo / "results" / run / "certified_external_reference.json"
    ).resolve()
    reverse_dir = (
        resolve_path(args.reverse_dir, base=repo)
        if args.reverse_dir
        else paths.exact_reverse / "production_reverse"
    ).resolve()
    out_dir = (
        resolve_path(args.out_dir, base=repo)
        if args.out_dir
        else paths.gradient_root
    ).resolve()

    reverse_summary_path = reverse_dir / "summary.json"
    reverse_summary = json.loads(
        reverse_summary_path.read_text(encoding="utf-8")
    )
    reverse_result = reverse_summary.get("result")
    current_contract = None
    if reverse_result == current_reverse_result(args.iteration):
        asset_manifest_path = resolve_path(
            str(engine["immutable_operator_assets"]["manifest"]), base=repo
        )
        asset_manifest = validate_immutable_asset_manifest(
            asset_manifest_path,
            expected_source_run=str(engine["historical_run_id"]),
            repository_root=repo,
            runtime_root=Path(runtime["runtime_root"]),
            verify_bytes=True,
        )
        asset_paths = _asset_paths(
            asset_manifest, repo=repo, runtime_root=Path(runtime["runtime_root"])
        )
        asset_records = {
            str(item["asset_id"]): item for item in asset_manifest["assets"]
        }
        operator_dir = asset_paths["exact_spatial_operator"]
        topology_dir = asset_paths["real_s43_compact_topology"]
        current_contract = validate_current_reverse_contract(
            repo=repo,
            config_path=config_path,
            config=config,
            engine_path=engine_path,
            engine=engine,
            paths=paths,
            iteration=args.iteration,
            reverse_dir=reverse_dir,
            reverse_summary=reverse_summary,
            operator_dir=operator_dir,
            topology_dir=topology_dir,
            operator_content_signature=str(
                asset_records["exact_spatial_operator"][
                    "content_signature_sha256"
                ]
            ),
            topology_content_signature=str(
                asset_records["real_s43_compact_topology"][
                    "content_signature_sha256"
                ]
            ),
        )
        if out_dir != paths.gradient_root.resolve():
            raise RuntimeError("CURRENT optimizer bridge output is not canonical")
        current_files = current_contract["gradient_paths"]
        gradient_paths = {
            "solid_lam": current_files["solid_lambda"],
            "solid_mu": current_files["solid_mu"],
            "pml_lam": current_files["pml_lambda"],
            "pml_mu": current_files["pml_mu"],
        }
        gradient_source_paths = current_files
        provenance_reference_path = current_contract["canonical_primal_summary"]
        extra_source_paths = {
            "iteration_engine_config": engine_path,
            "immutable_asset_manifest": asset_manifest_path,
        }
    elif args.iteration == 0 and reverse_result == HISTORICAL_REVERSE_PASS:
        _, _reference = load_certified_reference(
            repo, run, legacy_reference_path
        )
        reference_paths = common_paths(
            repo, run, reference_manifest=legacy_reference_path
        )
        operator_dir = Path(reference_paths["gll"]).parent
        topology_dir = Path(reference_paths["topology"])
        gradient_paths = {
            name: reverse_dir / f"gradient_{name}.npy"
            for name in ("solid_lam", "solid_mu", "pml_lam", "pml_mu")
        }
        gradient_source_paths = gradient_paths
        provenance_reference_path = legacy_reference_path
        extra_source_paths = {}
    else:
        raise RuntimeError(
            f"reverse source is not an allowed certified PASS: {reverse_result}"
        )
    bridge_result = (
        gradient_bridge_result(args.iteration)
        if current_contract is not None
        else PASS_RESULT
    )

    source_paths = {
        "config": config_path,
        "reference_manifest": provenance_reference_path,
        "reverse_summary": reverse_summary_path,
        **{
            f"gradient_{key}": value
            for key, value in gradient_source_paths.items()
        },
        **extra_source_paths,
        "solid_P": operator_dir / "P_sem_gll_from_h5_full.npz",
        "solid_row_xyz": operator_dir / "row_xyz.npy",
        "solid_row_sem_element": operator_dir / "row_sem_element.npy",
        "solid_connectivity": topology_dir / "solid_connectivity_compact.npy",
        "solid_xyz": topology_dir / "solid_compact_xyz.npy",
        "pml_connectivity": topology_dir / "pml_connectivity_compact.npy",
        "pml_xyz": topology_dir / "pml_compact_xyz.npy",
        "pml_region": topology_dir / "pml_element_region_code.npy",
        "mtilde": configured_path(config["mtilde_matrix_path"], repo),
        "mtilde_coords": configured_path(
            config["mtilde_matrix_coords_path"], repo
        ),
        "mtilde_active_indices": configured_path(
            config["mtilde_matrix_indices_path"], repo
        ),
    }
    missing = [str(path) for path in source_paths.values() if not path.is_file()]
    if missing:
        raise RuntimeError("missing optimizer-bridge inputs: " + ", ".join(missing))
    input_hashes = {name: sha256_file(path) for name, path in source_paths.items()}
    signature_payload = {
        "schema_version": 1,
        "iteration": int(args.iteration),
        "transition": runtime["transition"],
        "reverse_result": reverse_result,
        "input_sha256": input_hashes,
    }
    signature = hashlib.sha256(
        json.dumps(signature_payload, sort_keys=True).encode("utf-8")
    ).hexdigest()
    summary_path = out_dir / "summary.json"
    if summary_path.is_file():
        existing = json.loads(summary_path.read_text(encoding="utf-8"))
        if (
            existing.get("result") == bridge_result
            and existing.get("bridge_signature_sha256") == signature
        ):
            print(f"RESULT = {bridge_result}")
            print(f"OUTPUT = {out_dir}")
            print("IDEMPOTENT_REUSE = true")
            return
        raise RuntimeError(f"refusing non-matching existing bridge: {out_dir}")
    if out_dir.exists():
        raise RuntimeError(f"refusing to overwrite existing bridge directory: {out_dir}")

    shape, full_coords = material_grid_coordinates(config)
    full_count = int(np.prod(shape))
    domain = config["domain"]
    bounds = (
        float(domain["x_min_m"]),
        float(domain["x_max_m"]),
        float(domain["y_min_m"]),
        float(domain["y_max_m"]),
        float(domain["z_min_m"]),
        float(domain["z_max_m"]),
    )
    material_preflight = resolve_current_parent_material_contract(
        engine=engine,
        parent_workspace=runtime["parent_workspace"],
        expected_shape=shape,
    )
    h5_metadata = material_preflight["files"]

    gradients = {
        name: np.asarray(np.load(path), dtype=np.float64)
        for name, path in gradient_paths.items()
    }
    if not all(np.all(np.isfinite(value)) for value in gradients.values()):
        raise RuntimeError("non-finite raw external gradient")

    solid_conn = np.asarray(
        np.load(source_paths["solid_connectivity"]), dtype=np.int64
    )
    solid_xyz = np.asarray(np.load(source_paths["solid_xyz"]), dtype=np.float64)
    row_xyz = np.asarray(
        np.load(source_paths["solid_row_xyz"]), dtype=np.float64
    )
    row_sem_element = np.asarray(
        np.load(source_paths["solid_row_sem_element"]), dtype=np.int64
    )
    solid_row_map, solid_mapping = build_solid_row_map(
        solid_xyz,
        solid_conn,
        row_sem_element,
        row_xyz,
        args.batch_size,
    )
    if (
        gradients["solid_lam"].shape != solid_conn.shape
        or gradients["solid_mu"].shape != solid_conn.shape
    ):
        raise RuntimeError("solid gradient/topology shape mismatch")
    P_solid = load_npz(source_paths["solid_P"]).tocsr()
    if P_solid.shape != (solid_conn.size, full_count):
        raise RuntimeError("solid H5-to-GLL operator shape mismatch")
    solid_row_sum_error = float(
        np.max(np.abs(np.asarray(P_solid.sum(axis=1)).reshape(-1) - 1.0))
    )
    solid_gradient_lam_rows = gradients["solid_lam"].reshape(-1)[solid_row_map]
    solid_gradient_mu_rows = gradients["solid_mu"].reshape(-1)[solid_row_map]
    solid_lam = np.asarray(P_solid.T @ solid_gradient_lam_rows, dtype=np.float64)
    solid_mu = np.asarray(P_solid.T @ solid_gradient_mu_rows, dtype=np.float64)

    pml_conn = np.asarray(
        np.load(source_paths["pml_connectivity"]), dtype=np.int64
    )
    pml_xyz = np.asarray(np.load(source_paths["pml_xyz"]), dtype=np.float64)
    pml_region = np.asarray(
        np.load(source_paths["pml_region"]), dtype=np.uint8
    )
    if (
        gradients["pml_lam"].shape != pml_conn.shape
        or gradients["pml_mu"].shape != pml_conn.shape
    ):
        raise RuntimeError("PML gradient/topology shape mismatch")
    pml_sample_xyz = production_sample_xyz(
        pml_xyz[pml_conn], pml_region, config
    )
    pml_lam = trilinear_transpose(
        gradients["pml_lam"], pml_sample_xyz, shape, bounds, args.batch_size
    )
    pml_mu = trilinear_transpose(
        gradients["pml_mu"], pml_sample_xyz, shape, bounds, args.batch_size
    )
    total_lam = solid_lam + pml_lam
    total_mu = solid_mu + pml_mu
    dataset_kappa = total_lam.copy()
    dataset_mu = total_mu - (2.0 / 3.0) * total_lam

    mtilde_coords = np.asarray(
        np.load(source_paths["mtilde_coords"]), dtype=np.float64
    )[:, :3]
    mtilde_internal_indices = np.asarray(
        np.load(source_paths["mtilde_active_indices"]), dtype=np.int64
    )
    if mtilde_internal_indices.shape != (mtilde_coords.shape[0],):
        raise RuntimeError("Mtilde active-index shape mismatch")
    decimals = int(config.get("mtilde_artifact", {}).get("coordinate_decimals", 10))
    full_index = coordinate_index(full_coords, decimals)
    active_h5_indices = np.asarray(
        [full_index[tuple(np.round(row, decimals))] for row in mtilde_coords],
        dtype=np.int64,
    )
    active_coordinate_error = float(
        np.max(np.abs(full_coords[active_h5_indices] - mtilde_coords))
    )
    if np.unique(active_h5_indices).size != len(active_h5_indices):
        raise RuntimeError("active H5 mapping is not one-to-one")

    rng = np.random.default_rng(20260830)
    test_lam = rng.standard_normal(full_count)
    test_mu = rng.standard_normal(full_count)
    solid_left = float(
        np.dot(solid_gradient_lam_rows, P_solid @ test_lam)
        + np.dot(solid_gradient_mu_rows, P_solid @ test_mu)
    )
    solid_right = float(np.dot(solid_lam, test_lam) + np.dot(solid_mu, test_mu))
    pml_forward_lam = trilinear_sample(
        test_lam.reshape(shape), pml_sample_xyz.reshape(-1, 3), *bounds
    )
    pml_forward_mu = trilinear_sample(
        test_mu.reshape(shape), pml_sample_xyz.reshape(-1, 3), *bounds
    )
    pml_left = float(
        np.dot(gradients["pml_lam"].reshape(-1), pml_forward_lam)
        + np.dot(gradients["pml_mu"].reshape(-1), pml_forward_mu)
    )
    pml_right = float(np.dot(pml_lam, test_lam) + np.dot(pml_mu, test_mu))
    full_left = solid_left + pml_left
    full_right = float(np.dot(total_lam, test_lam) + np.dot(total_mu, test_mu))
    solid_transpose_error = relative_error(solid_left, solid_right)
    pml_transpose_error = relative_error(pml_left, pml_right)
    full_transpose_error = relative_error(full_left, full_right)

    test_kappa = rng.standard_normal(full_count)
    test_dataset_mu = rng.standard_normal(full_count)
    physical_left = float(
        np.dot(total_lam, test_kappa - (2.0 / 3.0) * test_dataset_mu)
        + np.dot(total_mu, test_dataset_mu)
    )
    dataset_right = float(
        np.dot(dataset_kappa, test_kappa) + np.dot(dataset_mu, test_dataset_mu)
    )
    variable_transform_error = relative_error(physical_left, dataset_right)

    rhs_lambda = total_lam[active_h5_indices]
    rhs_mu = total_mu[active_h5_indices]
    matrix = load_npz(source_paths["mtilde"]).tocsr()
    operator_index = coordinate_index(mtilde_coords, decimals)
    operator_order = np.asarray(
        [operator_index[tuple(np.round(row, decimals))] for row in mtilde_coords],
        dtype=np.int64,
    )
    mapped_coordinate_error = float(
        np.max(np.abs(mtilde_coords[operator_order] - mtilde_coords))
    )
    interior = matrix[operator_order, :][:, operator_order].tocsr()
    if interior.shape != (len(active_h5_indices), len(active_h5_indices)):
        raise RuntimeError("Mtilde shape does not match active H5 controls")
    gradient_lambda = np.asarray(spsolve(interior, rhs_lambda), dtype=np.float64)
    gradient_mu = np.asarray(spsolve(interior, rhs_mu), dtype=np.float64)
    residual_lambda = interior @ gradient_lambda - rhs_lambda
    residual_mu = interior @ gradient_mu - rhs_mu
    relative_lambda = float(
        np.linalg.norm(residual_lambda)
        / max(np.linalg.norm(rhs_lambda), np.finfo(np.float64).tiny)
    )
    relative_mu = float(
        np.linalg.norm(residual_mu)
        / max(np.linalg.norm(rhs_mu), np.finfo(np.float64).tiny)
    )
    gradient_sign = float(
        config.get("optimizer", {}).get("gradient_sign_from_mtilde", 1.0)
    )

    gates = {
        "reverse_source_certified": True,
        "reference_manifest_pass": True,
        "solid_gradient_shape": True,
        "pml_gradient_shape": True,
        "all_finite": bool(
            all(
                np.all(np.isfinite(value))
                for value in (
                    total_lam,
                    total_mu,
                    rhs_lambda,
                    rhs_mu,
                    gradient_lambda,
                    gradient_mu,
                )
            )
        ),
        "solid_row_coordinate_error_le_1e-12": float(
            solid_mapping["local_coordinate_error"]
        )
        <= 1.0e-12,
        "solid_row_mapping_bijective": bool(solid_mapping["bijective"]),
        "solid_row_sum_error_le_1e-14": solid_row_sum_error <= 1.0e-14,
        "solid_transpose_error_le_1e-12": solid_transpose_error <= 1.0e-12,
        "pml_transpose_error_le_1e-12": pml_transpose_error <= 1.0e-12,
        "full_chain_transpose_error_le_1e-12": full_transpose_error <= 1.0e-12,
        "variable_transform_error_le_1e-12": variable_transform_error <= 1.0e-12,
        "active_coordinate_error_le_1e-12": active_coordinate_error <= 1.0e-12,
        "mtilde_coordinate_error_le_1e-12": mapped_coordinate_error <= 1.0e-12,
        "mtilde_residual_lambda_le_1e-10": relative_lambda <= 1.0e-10,
        "mtilde_residual_mu_le_1e-10": relative_mu <= 1.0e-10,
        "gradient_sign_from_mtilde_is_plus_one": gradient_sign == 1.0,
        "optimizer_indices_selected_from_active_h5_mapping": True,
    }
    if not all(gates.values()):
        failed = [name for name, passed in gates.items() if not passed]
        raise RuntimeError("optimizer-bridge gates failed: " + ", ".join(failed))

    out_dir.mkdir(parents=True, exist_ok=False)
    rhs_dir = out_dir / "rhs"
    solve_dir = out_dir / "mtilde_solve"
    rhs_dir.mkdir()
    solve_dir.mkdir()
    arrays = {
        "full_control_coords.npy": full_coords,
        "solid_control_covector_lambda.npy": solid_lam,
        "solid_control_covector_mu.npy": solid_mu,
        "pml_control_covector_lambda.npy": pml_lam,
        "pml_control_covector_mu.npy": pml_mu,
        "full_control_covector_lambda.npy": total_lam,
        "full_control_covector_mu.npy": total_mu,
        "full_h5_dataset_covector_kappa.npy": dataset_kappa,
        "full_h5_dataset_covector_mu.npy": dataset_mu,
        "active_h5_indices.npy": active_h5_indices,
        "mtilde_active_full_indices.npy": mtilde_internal_indices,
        "mtilde_active_coords.npy": mtilde_coords,
        "solid_flat_index_for_P_row.npy": solid_row_map,
    }
    for filename, value in arrays.items():
        np.save(out_dir / filename, value)
    np.save(rhs_dir / "full_grid_trace_RHS_total_lambda.npy", rhs_lambda)
    np.save(rhs_dir / "full_grid_trace_RHS_total_mu.npy", rhs_mu)
    np.save(rhs_dir / "full_grid_trace_RHS_total_coords.npy", mtilde_coords)
    save_npz(solve_dir / "Mtilde_interior_sparse.npz", interior)
    np.save(solve_dir / "Mtilde_interior_indices.npy", active_h5_indices)
    np.save(solve_dir / "gradient_coords.npy", mtilde_coords)
    np.save(solve_dir / "g_lambda.npy", gradient_lambda)
    np.save(solve_dir / "g_mu.npy", gradient_mu)

    solve_summary = {
        "schema_version": 1,
        "result": bridge_result,
        "run_id": run,
        "parent_iteration": int(args.iteration),
        "child_iteration": int(args.iteration) + 1,
        "transition": runtime["transition"],
        "matrix": str(source_paths["mtilde"]),
        "rhs_count": int(len(rhs_lambda)),
        "interior_shape": list(interior.shape),
        "coordinate_error": mapped_coordinate_error,
        "active_h5_coordinate_error": active_coordinate_error,
        "relative_residual_lambda": relative_lambda,
        "relative_residual_mu": relative_mu,
        "gradient_sign_from_mtilde": gradient_sign,
        "Mtilde_interior_indices_space": "H5 full C-order indices",
    }
    atomic_json(solve_dir / "mtilde_gradient_summary.json", solve_summary)

    summary = {
        "schema_version": 1,
        "result": bridge_result,
        "run_id": run,
        "parent_iteration": int(args.iteration),
        "child_iteration": int(args.iteration) + 1,
        "bridge_signature_sha256": signature,
        "iteration": int(args.iteration),
        "transition": runtime["transition"],
        "reference_manifest": str(provenance_reference_path),
        "reverse_source": str(reverse_dir),
        "reverse_result": reverse_result,
        "sem3d_runs": 0,
        "parameterization": {
            "stored_h5_datasets": ["Kappa", "Mu"],
            "optimizer_variables": ["lambda", "mu"],
            "candidate_write_rule": "Kappa=lambda+(2/3)*mu; Mu=mu",
            "dataset_covector_rule": (
                "g_Kappa=g_lambda; g_Mu_dataset=g_mu-(2/3)*g_lambda"
            ),
            "gradient_sign_from_mtilde": gradient_sign,
        },
        "h5_metadata": h5_metadata,
        "counts": {
            "control_grid": full_count,
            "active_mtilde": int(len(active_h5_indices)),
            "solid_gll": int(solid_conn.size),
            "pml_gll": int(pml_conn.size),
        },
        "chain_rule": {
            "lambda": "P_solid^T*g_solid_lambda + P_pml^T*g_pml_lambda",
            "mu": "P_solid^T*g_solid_mu + P_pml^T*g_pml_mu",
            "active_rhs": (
                "restrict full lambda/mu covectors by H5 indices mapped from "
                "active Mtilde coordinates"
            ),
        },
        "index_contract": {
            "mtilde_internal_indices_space": "full Mtilde ordering",
            "optimizer_indices_space": "H5 full C-order",
            "active_coordinate_error": active_coordinate_error,
            "mtilde_internal_indices_equal_h5_indices": bool(
                np.array_equal(mtilde_internal_indices, active_h5_indices)
            ),
        },
        "transpose_tests": {
            "solid_relative_error": solid_transpose_error,
            "pml_relative_error": pml_transpose_error,
            "full_chain_relative_error": full_transpose_error,
            "kappa_mu_variable_transform_relative_error": variable_transform_error,
        },
        "covectors": {
            "solid_lambda": array_stats(solid_lam),
            "solid_mu": array_stats(solid_mu),
            "pml_lambda": array_stats(pml_lam),
            "pml_mu": array_stats(pml_mu),
            "full_lambda": array_stats(total_lam),
            "full_mu": array_stats(total_mu),
            "active_lambda_rhs": array_stats(rhs_lambda),
            "active_mu_rhs": array_stats(rhs_mu),
        },
        "mtilde": solve_summary,
        "gates": gates,
        "provenance": {
            "source_paths": {name: str(path) for name, path in source_paths.items()},
            "input_sha256": input_hashes,
            "current_contract": (
                None
                if current_contract is None
                else {
                    "material_covector_input": str(
                        current_contract["material_covector_input"]
                    ),
                    "optimizer_bridge_output": str(
                        current_contract["optimizer_bridge_output"]
                    ),
                    "registered_physical_gradient": str(
                        current_contract["registered_physical_gradient"]
                    ),
                    "canonical_primal_summary": str(
                        current_contract["canonical_primal_summary"]
                    ),
                    "operator_content_signature_sha256": current_contract[
                        "operator_content_signature_sha256"
                    ],
                    "topology_content_signature_sha256": current_contract[
                        "topology_content_signature_sha256"
                    ],
                }
            ),
        },
        "outputs": {
            "root": str(out_dir),
            "rhs": str(rhs_dir),
            "mtilde_solve": str(solve_dir),
        },
    }
    atomic_json(summary_path, summary)
    print(f"RESULT = {bridge_result}")
    print(f"MTILDE_RESIDUAL_LAMBDA = {relative_lambda:.17e}")
    print(f"MTILDE_RESIDUAL_MU = {relative_mu:.17e}")
    print(f"OUTPUT = {out_dir}")


if __name__ == "__main__":
    main()
