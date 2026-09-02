"""Map a certified external exact gradient into optimizer and Mtilde spaces."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

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
from scripts.fathi_benchmark.runtime_paths import (
    iteration_runtime_paths,
    resolve_path,
)


PASS_RESULT = "PASS_CERTIFIED_EXTERNAL_OPTIMIZER_BRIDGE"
GENERIC_REVERSE_PASS = "PASS_CERTIFIED_EXTERNAL_EXACT_REVERSE"
HISTORICAL_REVERSE_PASS = (
    "PASS_STAGE5O_EXTERNAL_PHYSICAL_EXACT_ADJOINT_CERTIFICATION"
)


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
    run = config_path.stem
    reference_path = (
        resolve_path(args.reference_manifest, base=repo)
        if args.reference_manifest
        else repo / "results" / run / "certified_external_reference.json"
    ).resolve()
    _, reference = load_certified_reference(repo, run, reference_path)
    reference_paths = common_paths(
        repo, run, reference_manifest=reference_path
    )
    reverse_dir = (
        resolve_path(args.reverse_dir, base=repo)
        if args.reverse_dir
        else Path(runtime["transition_root"])
        / "certified_iteration"
        / "exact_reverse"
    ).resolve()
    out_dir = (
        resolve_path(args.out_dir, base=repo)
        if args.out_dir
        else Path(runtime["transition_root"])
        / "certified_iteration"
        / "optimizer_bridge"
    ).resolve()

    reverse_summary_path = reverse_dir / "summary.json"
    reverse_summary = json.loads(
        reverse_summary_path.read_text(encoding="utf-8")
    )
    reverse_result = reverse_summary.get("result")
    allowed = {GENERIC_REVERSE_PASS}
    if args.iteration == 0:
        allowed.add(HISTORICAL_REVERSE_PASS)
    if reverse_result not in allowed:
        raise RuntimeError(
            f"reverse source is not an allowed certified PASS: {reverse_result}"
        )
    if reverse_result == GENERIC_REVERSE_PASS:
        if (
            int(reverse_summary["iteration"]) != int(args.iteration)
            or reverse_summary["transition"] != runtime["transition"]
        ):
            raise RuntimeError("generic reverse iteration provenance mismatch")
        if sha256_file(reference_path) != sha256_file(
            Path(reverse_summary["reference_manifest"])
        ):
            raise RuntimeError("generic reverse reference provenance mismatch")

    gradient_paths = {
        name: reverse_dir / f"gradient_{name}.npy"
        for name in ("solid_lam", "solid_mu", "pml_lam", "pml_mu")
    }
    operator_dir = Path(reference_paths["gll"]).parent
    topology_dir = Path(reference_paths["topology"])
    source_paths = {
        "config": config_path,
        "reference_manifest": reference_path,
        "reverse_summary": reverse_summary_path,
        **{f"gradient_{key}": value for key, value in gradient_paths.items()},
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
            existing.get("result") == PASS_RESULT
            and existing.get("bridge_signature_sha256") == signature
        ):
            print(f"RESULT = {PASS_RESULT}")
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
    material_spec = config["sem3d_mesh"]["material_spec"]
    parent_h5 = Path(runtime["parent_workspace"]) / "mat" / "h5"
    h5_metadata = {}
    for label, filename in (
        ("kappa", material_spec["kappa_file"].split("/")[-1]),
        ("mu", material_spec["mu_file"].split("/")[-1]),
    ):
        path = parent_h5 / filename
        with h5py.File(path, "r") as handle:
            dataset = config["material_grid"]["dataset"]
            if dataset not in handle or tuple(handle[dataset].shape) != shape:
                raise RuntimeError(f"invalid parent material field: {path}")
            h5_metadata[label] = {
                "path": str(path),
                "dataset": dataset,
                "shape": list(handle[dataset].shape),
                "dtype": str(handle[dataset].dtype),
            }

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
        "result": PASS_RESULT,
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
        "result": PASS_RESULT,
        "bridge_signature_sha256": signature,
        "iteration": int(args.iteration),
        "transition": runtime["transition"],
        "reference_manifest": str(reference_path),
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
        },
        "outputs": {
            "root": str(out_dir),
            "rhs": str(rhs_dir),
            "mtilde_solve": str(solve_dir),
        },
    }
    atomic_json(summary_path, summary)
    print(f"RESULT = {PASS_RESULT}")
    print(f"MTILDE_RESIDUAL_LAMBDA = {relative_lambda:.17e}")
    print(f"MTILDE_RESIDUAL_MU = {relative_mu:.17e}")
    print(f"OUTPUT = {out_dir}")


if __name__ == "__main__":
    main()
