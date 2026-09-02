"""Evaluate the certified Stage 5N external physical objective for a candidate.

The cheap ``objective-only`` mode consumes cached external receiver arrays.  The
``candidate-forward`` mode reuses the certified benchmark-side external forward
driver and is intentionally resumable.  Neither mode invokes SEM3D.
"""

import argparse
import hashlib
import json
import math
import os
from pathlib import Path

import h5py
import numpy as np

from scripts.fathi_benchmark.runtime_paths import (
    iteration_runtime_paths,
    resolve_path,
)


STAGE5N_DIR = "stage5n_external_physical_objective_certification"
STAGE5NR_DIR = "stage5nr_external_objective_fd_refinement"
STAGE5O_DIR = "stage5o_external_physical_exact_adjoint_certification"
CERTIFIED_BRIDGE_DIR = "stage5o_certified_optimizer_bridge"
OUTPUT_SUBDIR = "candidate_external_objectives"
REFERENCE_MANIFEST_NAME = "certified_external_reference.json"
PASS_REFERENCE = "PASS_CERTIFIED_EXTERNAL_REFERENCE_CONTRACT"
PASS_STAGE5NR = "PASS_EXTERNAL_OBJECTIVE_FD_STEP_REFINEMENT"
PASS_STAGE5O = "PASS_STAGE5O_EXTERNAL_PHYSICAL_EXACT_ADJOINT_CERTIFICATION"


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_arrays(*arrays):
    digest = hashlib.sha256()
    for value in arrays:
        array = np.ascontiguousarray(value)
        digest.update(str(array.shape).encode("ascii"))
        digest.update(array.dtype.str.encode("ascii"))
        digest.update(array.view(np.uint8))
    return digest.hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def read_json(path):
    path = Path(path)
    if not path.is_file():
        raise RuntimeError(f"missing certified artifact: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def explicit_or_default(value, default, repo):
    return resolve_path(value, base=repo) if value else Path(default).resolve()


def resolve_contract(args):
    repo = Path(args.repo).expanduser().resolve()

    config_path = resolve_path(
        args.config,
        base=repo,
    )

    config = read_json(config_path)
    run = config_path.stem

    runtime = iteration_runtime_paths(
        config,
        args.iter_k,
        repo_root=repo,
    )

    transition_root = Path(
        runtime["transition_root"]
    ).resolve()

    reference_manifest = explicit_or_default(
        args.reference_manifest,
        repo
        / "results"
        / run
        / REFERENCE_MANIFEST_NAME,
        repo,
    )

    reference = read_json(
        reference_manifest
    )

    if (
        reference.get("result")
        != PASS_REFERENCE
    ):
        raise RuntimeError(
            "certified external reference "
            "manifest is not PASS"
        )

    if reference.get("run") != run:
        raise RuntimeError(
            "certified external reference "
            "run mismatch"
        )

    def reference_path(value):
        path = Path(
            value
        ).expanduser()

        if not path.is_absolute():
            path = repo / path

        return path.resolve()

    operator = reference[
        "operator_assets"
    ]

    certification = reference[
        "certification_assets"
    ]

    reference_contract = reference[
        "contract"
    ]

    stage5n_summary_path = reference_path(
        certification[
            "stage5n_summary"
        ]
    )

    stage5nr_summary_path = reference_path(
        certification[
            "stage5nr_summary"
        ]
    )

    stage5o_summary_path = reference_path(
        certification[
            "stage5o_summary"
        ]
    )

    manifest_path = reference_path(
        certification[
            "fixed_dt_manifest"
        ]
    )

    stage5n_dir = (
        stage5n_summary_path.parent
    )

    if args.stage5n_dir:
        requested_stage5n = (
            explicit_or_default(
                args.stage5n_dir,
                stage5n_dir,
                repo,
            )
        )

        if (
            requested_stage5n
            != stage5n_dir
        ):
            raise RuntimeError(
                "--stage5n-dir differs from "
                "the frozen certified "
                "reference contract"
            )

    stage5n = read_json(
        stage5n_summary_path
    )

    stage5nr = read_json(
        stage5nr_summary_path
    )

    stage5o = read_json(
        stage5o_summary_path
    )

    manifest = read_json(
        manifest_path
    )

    if (
        stage5nr.get("result")
        != PASS_STAGE5NR
    ):
        raise RuntimeError(
            "Stage 5N-R certified "
            "objective refinement "
            "is not passing"
        )

    if (
        stage5o.get("result")
        != PASS_STAGE5O
    ):
        raise RuntimeError(
            "Stage 5O external "
            "exact-adjoint certification "
            "is not passing"
        )

    stage5n_contract = (
        stage5n.get(
            "contract",
            {},
        )
    )

    sample_count = int(
        reference_contract[
            "sample_count"
        ]
    )

    receiver_count = int(
        reference_contract[
            "receiver_count"
        ]
    )

    component_count = int(
        reference_contract[
            "component_count"
        ]
    )

    dt = float(
        reference_contract[
            "dt"
        ]
    )

    reference_parent_objective = float(
        stage5n[
            "objective"
        ][
            "J_external"
        ]
    )

    stage5o_objective = float(
        stage5o[
            "objective"
        ][
            "J_external"
        ]
    )

    checks = {
        "reference_result": (
            reference.get(
                "result"
            )
            == PASS_REFERENCE
        ),
        "manifest_sample_count": (
            int(
                manifest[
                    "baseline_sample_count"
                ]
            )
            == sample_count
        ),
        "manifest_dt": (
            math.isclose(
                float(
                    manifest[
                        "baseline_dt"
                    ]
                ),
                dt,
                rel_tol=0.0,
                abs_tol=1.0e-18,
            )
        ),
        "stage5o_sample_count": (
            int(
                stage5o[
                    "objective"
                ][
                    "sample_count"
                ]
            )
            == sample_count
        ),
        "stage5o_receiver_count": (
            int(
                stage5o[
                    "objective"
                ][
                    "receiver_count"
                ]
            )
            == receiver_count
        ),
        "stage5o_dt": (
            math.isclose(
                float(
                    stage5o[
                        "objective"
                    ][
                        "dt"
                    ]
                ),
                dt,
                rel_tol=0.0,
                abs_tol=1.0e-18,
            )
        ),
        "stage5o_objective": (
            stage5o_objective
            == reference_parent_objective
        ),
        "residual_sign": (
            stage5n_contract[
                "residual_sign"
            ]
            == reference_contract[
                "residual_sign"
            ]
        ),
        "time_weighting": (
            stage5n_contract[
                "time_weighting"
            ]
            == reference_contract[
                "time_weighting"
            ]
        ),
        "receiver_order": (
            stage5n_contract[
                "receiver_order"
            ]
            == reference_contract[
                "receiver_order"
            ]
        ),
    }

    if not all(
        checks.values()
    ):
        raise RuntimeError(
            "certified external "
            "reference mismatch: "
            f"{checks}"
        )

    true_default = reference_path(
        certification[
            "true_external"
        ]
    )

    true_external = explicit_or_default(
        args.true_external,
        true_default,
        repo,
    )

    if not true_default.is_file():
        raise RuntimeError(
            "missing certified true "
            f"external trace: "
            f"{true_default}"
        )

    if not true_external.is_file():
        raise RuntimeError(
            "missing requested true "
            f"external trace: "
            f"{true_external}"
        )

    true_hash = sha256_file(
        true_external
    )

    if (
        true_hash
        != sha256_file(
            true_default
        )
    ):
        raise RuntimeError(
            "true external override is "
            "not the exact certified "
            "reference artifact"
        )

    expected_true_hash = (
        reference.get(
            "hashes",
            {},
        ).get(
            "true_external_sha256"
        )
    )

    if (
        expected_true_hash
        and true_hash
        != expected_true_hash
    ):
        raise RuntimeError(
            "certified true external "
            "SHA256 mismatch"
        )

    iter_k = int(
        args.iter_k
    )

    reference_current = (
        stage5n_dir
        / "current_external_receiver.npy"
    )

    current_default = (
        reference_current
        if iter_k == 0
        else None
    )

    if args.current_external:
        current_external = (
            resolve_path(
                args.current_external,
                base=repo,
            )
        )
    else:
        current_external = (
            current_default
        )

    receiver_dir = reference_path(
        operator[
            "receiver"
        ]
    )

    output_dir = explicit_or_default(
        args.output_dir,
        transition_root
        / CERTIFIED_BRIDGE_DIR
        / OUTPUT_SUBDIR,
        repo,
    )

    candidate_root = explicit_or_default(
        args.candidate_root,
        transition_root
        / CERTIFIED_BRIDGE_DIR
        / "candidates",
        repo,
    )

    return {
        "repo": repo,
        "run": run,

        "config": config,
        "config_path": config_path,

        "iter_k": iter_k,

        "runtime": runtime,
        "transition_root": (
            transition_root
        ),

        "reference_manifest_path": (
            reference_manifest
        ),
        "reference_manifest": (
            reference
        ),

        "stage5n_dir": (
            stage5n_dir
        ),
        "stage5n_summary_path": (
            stage5n_summary_path
        ),
        "stage5nr_summary_path": (
            stage5nr_summary_path
        ),
        "stage5o_summary_path": (
            stage5o_summary_path
        ),

        "stage5n": stage5n,
        "stage5nr": stage5nr,
        "stage5o": stage5o,

        "manifest_path": (
            manifest_path
        ),

        "sample_count": sample_count,
        "receiver_count": receiver_count,
        "component_count": (
            component_count
        ),
        "dt": dt,

        "expected_parent_objective": (
            reference_parent_objective
            if iter_k == 0
            else None
        ),

        "true_default": (
            true_default
        ),
        "true_external": (
            true_external
        ),

        "current_default": (
            current_default
        ),
        "current_external": (
            current_external
        ),

        "receiver_dir": (
            receiver_dir
        ),

        "output_dir": (
            output_dir
        ),

        "candidate_root": (
            candidate_root
        ),

        "contract_checks": (
            checks
        ),
    }

def receiver_operator_audit(contract):
    nodes_path = contract["receiver_dir"] / "receiver_nodes.npy"
    weights_path = contract["receiver_dir"] / "receiver_weights.npy"
    nodes = np.asarray(np.load(nodes_path), dtype=np.int64)
    weights = np.asarray(np.load(weights_path), dtype=np.float64)
    if nodes.shape != weights.shape or nodes.ndim != 2:
        raise RuntimeError("physical receiver operator shape mismatch")
    if nodes.shape[0] != contract["receiver_count"]:
        raise RuntimeError("physical receiver count mismatch")
    row_error = float(np.max(np.abs(np.sum(weights, axis=1) - 1.0)))
    if row_error > 1.0e-14:
        raise RuntimeError("physical receiver interpolation rows do not sum to one")
    return {
        "ordering": contract["stage5n"]["contract"]["receiver_order"],
        "nodes_shape": list(nodes.shape),
        "weights_shape": list(weights.shape),
        "row_sum_max_abs_error": row_error,
        "nodes_sha256": sha256_file(nodes_path),
        "weights_sha256": sha256_file(weights_path),
        "operator_array_sha256": sha256_arrays(nodes, weights),
    }


def load_external(path, contract):
    path = Path(path)
    if not path.is_file():
        raise RuntimeError(f"missing external receiver array: {path}")
    value = np.load(path)
    if value.dtype != np.float64:
        raise RuntimeError(f"external receiver dtype is not float64: {path}")
    expected = (
        contract["sample_count"],
        contract["receiver_count"],
        contract["component_count"],
    )
    if value.shape != expected:
        raise RuntimeError(
            f"external receiver shape mismatch for {path}: {value.shape} != {expected}"
        )
    if not np.all(np.isfinite(value)):
        raise RuntimeError(f"nonfinite external receiver array: {path}")
    return np.asarray(value, dtype=np.float64)


def objective_record(contract, current_path, expected_objective=None, tolerance=1.0e-12):
    from scripts.exact_adjoint.certify_exact_adjoint_with_fixed_dt_fd import (
        trapezoid_weights,
    )

    current = load_external(current_path, contract)
    true = load_external(contract["true_external"], contract)
    residual = current - true
    time_grid = np.arange(contract["sample_count"], dtype=np.float64) * contract["dt"]
    weights = trapezoid_weights(time_grid)
    objective = 0.5 * float(
        np.sum(weights[:, None, None] * residual * residual)
    )
    regression = None
    if expected_objective is not None:
        expected_objective = float(expected_objective)
        relative_error = abs(objective - expected_objective) / max(
            abs(expected_objective), np.finfo(np.float64).tiny
        )
        regression = {
            "expected": expected_objective,
            "recomputed": objective,
            "relative_error": relative_error,
            "tolerance": float(tolerance),
            "pass": bool(relative_error <= float(tolerance)),
        }
    return {
        "objective": objective,
        "formula": "0.5 * sum_t,r,c(w_t * residual[t,r,c]^2)",
        "residual_sign": "current_external - true_external",
        "time_quadrature": "native fixed-dt trapezoidal quadrature",
        "dt": contract["dt"],
        "sample_count": contract["sample_count"],
        "receiver_count": contract["receiver_count"],
        "component_count": contract["component_count"],
        "time_start": float(time_grid[0]),
        "time_end": float(time_grid[-1]),
        "endpoint_weight_first": float(weights[0]),
        "interior_weight_first": float(weights[1]),
        "endpoint_weight_last": float(weights[-1]),
        "weight_sum": float(np.sum(weights)),
        "current_external": str(Path(current_path).resolve()),
        "current_external_sha256": sha256_file(current_path),
        "true_external": str(contract["true_external"]),
        "true_external_sha256": sha256_file(contract["true_external"]),
        "residual_l2": float(np.linalg.norm(residual.reshape(-1))),
        "residual_finite": bool(np.all(np.isfinite(residual))),
        "regression": regression,
    }


def safe_label(value):
    if not value or Path(value).name != value or value in {".", ".."}:
        raise RuntimeError(f"invalid output label: {value!r}")
    return value


def provenance(contract):
    return {
        "config": str(
            contract[
                "config_path"
            ]
        ),

        "iteration": int(
            contract[
                "iter_k"
            ]
        ),

        "transition": (
            contract[
                "runtime"
            ][
                "transition"
            ]
        ),

        "reference_manifest": str(
            contract[
                "reference_manifest_path"
            ]
        ),

        "reference_manifest_sha256": (
            sha256_file(
                contract[
                    "reference_manifest_path"
                ]
            )
        ),

        "stage5n_summary": str(
            contract[
                "stage5n_summary_path"
            ]
        ),

        "stage5nr_summary": str(
            contract[
                "stage5nr_summary_path"
            ]
        ),

        "stage5o_summary": str(
            contract[
                "stage5o_summary_path"
            ]
        ),

        "fixed_dt_manifest": str(
            contract[
                "manifest_path"
            ]
        ),

        "contract_checks": (
            contract[
                "contract_checks"
            ]
        ),
    }

def write_audit(path, summary):
    objective = summary.get("objective")
    lines = [
        "CERTIFIED EXTERNAL CANDIDATE OBJECTIVE",
        "=" * 72,
        f"RESULT = {summary['result']}",
        f"mode = {summary['mode']}",
        f"candidate = {summary.get('candidate')}",
        "SEM3D runs = 0",
        f"full external replays = {summary['full_external_replays']}",
    ]
    if objective:
        lines.extend(
            [
                f"J_external = {objective['objective']:.17e}",
                f"residual sign = {objective['residual_sign']}",
                f"dt = {objective['dt']:.17e}",
                f"sample/receiver/component count = "
                f"{objective['sample_count']}/{objective['receiver_count']}/"
                f"{objective['component_count']}",
            ]
        )
        if objective["regression"] is not None:
            regression = objective["regression"]
            lines.extend(
                [
                    f"expected = {regression['expected']:.17e}",
                    f"relative error = {regression['relative_error']:.17e}",
                    f"regression pass = {regression['pass']}",
                ]
            )
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_objective_only(args, contract):
    if contract["current_external"] is None:
        raise RuntimeError(
            "objective-only mode for iter_k > 0 "
            "requires --current-external; "
            "current-model external receiver arrays "
            "are iteration-specific"
        )

    if not Path(contract["current_external"]).is_file():
        raise RuntimeError(
            f"missing requested current external trace: {contract['current_external']}"
        )
    label = safe_label(args.candidate or contract["current_external"].stem)
    expected = args.expected_objective
    if expected is None and contract["current_external"] == contract["current_default"]:
        expected = contract["expected_parent_objective"]
    objective = objective_record(
        contract,
        contract["current_external"],
        expected_objective=expected,
        tolerance=args.relative_tolerance,
    )
    regression_pass = (
        objective["regression"] is None or objective["regression"]["pass"]
    )
    summary = {
        "result": (
            "PASS_CERTIFIED_EXTERNAL_OBJECTIVE_REGRESSION"
            if regression_pass
            else "FAIL_CERTIFIED_EXTERNAL_OBJECTIVE_REGRESSION"
        ),
        "mode": "objective-only",
        "candidate": label,
        "sem3d_runs": 0,
        "full_external_replays": 0,
        "provenance": provenance(contract),
        "receiver_operator": receiver_operator_audit(contract),
        "objective": objective,
    }
    target = contract["output_dir"] / label
    atomic_json(target / "summary.json", summary)
    write_audit(target / "audit.txt", summary)
    print(json.dumps(summary, indent=2))
    if not regression_pass:
        raise SystemExit(2)


def load_material_samples(path):
    with h5py.File(path, "r") as handle:
        if "samples" not in handle:
            raise RuntimeError(f"material H5 has no samples dataset: {path}")
        value = np.asarray(handle["samples"], dtype=np.float64)
    if not np.all(np.isfinite(value)) or np.min(value) <= 0.0:
        raise RuntimeError(f"candidate material is nonfinite or nonpositive: {path}")
    return value


def candidate_runtime(args, contract):
    from scripts.exact_adjoint.s43_external_forward import ExternalForwardDriver

    if args.material_dir:
        material_dir = resolve_path(args.material_dir, base=contract["repo"])
    else:
        if not args.candidate:
            raise RuntimeError("--candidate or --material-dir is required")
        material_dir = contract["candidate_root"] / args.candidate / "mat" / "h5"
    material_dir = material_dir.resolve()
    files = {
        name: material_dir / name
        for name in ("Mat_0_Kappa.h5", "Mat_0_Mu.h5", "Mat_0_Density.h5")
    }
    for path in files.values():
        if not path.is_file():
            raise RuntimeError(f"missing candidate material: {path}")
    fields = {name: load_material_samples(path) for name, path in files.items()}
    shapes = {tuple(value.shape) for value in fields.values()}
    if len(shapes) != 1:
        raise RuntimeError(f"candidate material shape mismatch: {shapes}")
    parent_density = (
        Path(contract["runtime"]["parent_workspace"])
        / "mat"
        / "h5"
        / "Mat_0_Density.h5"
    )
    if sha256_file(files["Mat_0_Density.h5"]) != sha256_file(parent_density):
        raise RuntimeError(
            "candidate Density differs from the fixed coupled-mass density contract"
        )

    driver = ExternalForwardDriver(
        contract["repo"],
        contract["run"],
        material_dir,
        args.batch_size,
        reference_manifest=(
            contract[
                "reference_manifest_path"
            ]
        ),
    )

    if (
        Path(
            driver.paths[
                "reference_manifest"
            ]
        ).resolve()
        != Path(
            contract[
                "reference_manifest_path"
            ]
        ).resolve()
    ):
        raise RuntimeError(
            "external driver did not use "
            "the requested certified "
            "reference manifest"
        )

    if (
        Path(
            driver.paths[
                "receiver"
            ]
        ).resolve()
        != Path(
            contract[
                "receiver_dir"
            ]
        ).resolve()
    ):
        raise RuntimeError(
            "external driver receiver "
            "operator differs from "
            "certified reference"
        )
    if not math.isclose(driver.dt, contract["dt"], rel_tol=0.0, abs_tol=1.0e-18):
        raise RuntimeError("candidate external driver fixed-dt mismatch")
    if driver.receiver_count != contract["receiver_count"]:
        raise RuntimeError("candidate external driver receiver-count mismatch")
    if driver.audit["material_kappa_sha256"] != sha256_file(
        files["Mat_0_Kappa.h5"]
    ) or driver.audit["material_mu_sha256"] != sha256_file(files["Mat_0_Mu.h5"]):
        raise RuntimeError("external driver did not load the requested candidate material")
    audit = {
        "material_dir": str(material_dir),
        "material_shape": list(next(iter(fields.values())).shape),
        "material_sha256": {name: sha256_file(path) for name, path in files.items()},
        "density_parent_sha256": sha256_file(parent_density),
        "density_matches_fixed_coupled_mass_contract": True,
        "density_dynamics": (
            "fixed precomputed coupled mass; candidate Density must match parent"
        ),
        "solid_kappa_mu_loaded_from_candidate": True,
        "pml_lambda_mu_sampled_from_candidate": True,
        "pml_rebuild_applied": bool(
            driver.pml_material_audit["rebuild_applied"]
        ),
        "driver": driver.audit,
    }
    return driver, audit


def run_candidate_preflight(args, contract):
    label = safe_label(args.candidate or Path(args.material_dir).resolve().parent.name)
    _, material = candidate_runtime(args, contract)
    summary = {
        "result": "PASS_CERTIFIED_EXTERNAL_CANDIDATE_PREFLIGHT",
        "mode": "candidate-preflight",
        "candidate": label,
        "sem3d_runs": 0,
        "full_external_replays": 0,
        "provenance": provenance(contract),
        "receiver_operator": receiver_operator_audit(contract),
        "material": material,
        "expensive_candidate_mode_ready": True,
    }
    target = contract["output_dir"] / label
    atomic_json(target / "preflight_summary.json", summary)
    write_audit(target / "preflight_audit.txt", summary)
    print(json.dumps(summary, indent=2))


def run_candidate_forward(args, contract):
    from scripts.exact_adjoint.s43_external_forward import run_external_forward

    label = safe_label(args.candidate or Path(args.material_dir).resolve().parent.name)
    driver, material = candidate_runtime(args, contract)
    target = contract["output_dir"] / label
    trace_path = target / "candidate_external_receiver.npy"
    checkpoint_path = target / "checkpoint" / "candidate_latest.npz"
    run = run_external_forward(
        driver,
        contract["sample_count"],
        {"primal": trace_path},
        checkpoint_path,
        checkpoint_interval=args.checkpoint_interval,
    )
    objective = objective_record(contract, trace_path)
    summary = {
        "result": "PASS_CERTIFIED_EXTERNAL_CANDIDATE_OBJECTIVE",
        "mode": "candidate-forward",
        "candidate": label,
        "sem3d_runs": 0,
        "full_external_replays": 1,
        "provenance": provenance(contract),
        "receiver_operator": receiver_operator_audit(contract),
        "material": material,
        "external_forward": run,
        "objective": objective,
    }
    atomic_json(target / "summary.json", summary)
    write_audit(target / "audit.txt", summary)
    print(json.dumps(summary, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=("objective-only", "candidate-preflight", "candidate-forward"),
        required=True,
    )
    parser.add_argument("--repo", default=".")
    parser.add_argument("--config", required=True)
    parser.add_argument("--iter-k", type=int, required=True)
    parser.add_argument("--candidate")
    parser.add_argument("--candidate-root")
    parser.add_argument("--material-dir")
    parser.add_argument("--current-external")
    parser.add_argument("--true-external")
    parser.add_argument("--stage5n-dir")
    parser.add_argument(
        "--reference-manifest",
        default=None,
        help=(
            "Frozen certified external "
            "operator/objective reference. "
            "Defaults to "
            "results/<run>/"
            "certified_external_reference.json."
        ),
    )
    parser.add_argument("--output-dir")
    parser.add_argument("--expected-objective", type=float)
    parser.add_argument("--relative-tolerance", type=float, default=1.0e-12)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--checkpoint-interval", type=int, default=100)
    args = parser.parse_args()
    if args.relative_tolerance < 0.0:
        parser.error("--relative-tolerance must be nonnegative")
    contract = resolve_contract(args)
    contract["output_dir"].mkdir(parents=True, exist_ok=True)
    if args.mode == "objective-only":
        run_objective_only(args, contract)
    elif args.mode == "candidate-preflight":
        run_candidate_preflight(args, contract)
    else:
        run_candidate_forward(args, contract)


if __name__ == "__main__":
    main()
