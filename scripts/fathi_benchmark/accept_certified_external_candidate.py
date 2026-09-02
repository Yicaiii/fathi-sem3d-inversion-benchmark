"""Non-mutating acceptance audit for a certified external candidate objective."""

import argparse
import hashlib
import json
import os
from pathlib import Path

import h5py
import numpy as np

from scripts.fathi_benchmark.runtime_paths import (
    iteration_runtime_paths,
    resolve_path,
)


BRIDGE_DIR = "stage5o_certified_optimizer_bridge"
PARENT_OBJECTIVE_DIR = "stage5n_external_physical_objective_certification"
CANDIDATE_OBJECTIVE_SUBDIR = "candidate_external_objectives"
ACCEPTANCE_SUBDIR = "certified_acceptance"
CANDIDATE_PASS = "PASS_CERTIFIED_EXTERNAL_CANDIDATE_OBJECTIVE"
AUDIT_PASS = "PASS_CERTIFIED_DESCENT_CANDIDATE"


def read_json(path):
    path = Path(path)
    if not path.is_file():
        raise RuntimeError(f"missing required summary: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


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


def safe_candidate(value):
    if not value or Path(value).name != value or value in {".", ".."}:
        raise RuntimeError(f"invalid candidate name: {value!r}")
    return value


def explicit_or_default(value, default, repo):
    return resolve_path(value, base=repo) if value else Path(default).resolve()


def read_samples(path):
    with h5py.File(path, "r") as handle:
        if "samples" not in handle:
            raise RuntimeError(f"material H5 has no samples dataset: {path}")
        return np.asarray(handle["samples"], dtype=np.float64)


def resolve_contract(args):
    repo = Path(args.repo).expanduser().resolve()
    config_path = resolve_path(args.config, base=repo)
    config = read_json(config_path)
    runtime = iteration_runtime_paths(config, args.iter_k, repo_root=repo)
    transition_root = Path(runtime["transition_root"]).resolve()
    bridge = transition_root / BRIDGE_DIR
    candidate = safe_candidate(args.candidate)
    candidate_root = explicit_or_default(
        args.candidate_root, bridge / "candidates", repo
    )
    objective_root = explicit_or_default(
        args.candidate_objective_root,
        bridge / CANDIDATE_OBJECTIVE_SUBDIR,
        repo,
    )
    output_root = explicit_or_default(
        args.output_dir, bridge / ACCEPTANCE_SUBDIR, repo
    )
    parent_summary_path = explicit_or_default(
        args.parent_summary,
        transition_root / PARENT_OBJECTIVE_DIR / "summary.json",
        repo,
    )
    candidate_summary_path = objective_root / candidate / "summary.json"
    candidate_dir = candidate_root / candidate
    state_files = sorted(candidate_dir.glob("*_state_candidate.npz"))
    if len(state_files) != 1:
        raise RuntimeError(
            f"expected one candidate state under {candidate_dir}, got {state_files}"
        )
    return {
        "repo": repo,
        "config": config,
        "config_path": config_path,
        "runtime": runtime,
        "transition_root": transition_root,
        "bridge": bridge,
        "candidate": candidate,
        "candidate_root": candidate_root,
        "candidate_dir": candidate_dir,
        "candidate_state": state_files[0],
        "parent_summary_path": parent_summary_path,
        "candidate_summary_path": candidate_summary_path,
        "output_root": output_root,
    }


def objective_contract_audit(contract):
    parent = read_json(contract["parent_summary_path"])
    candidate = read_json(contract["candidate_summary_path"])
    parent_contract = parent["contract"]
    candidate_objective = candidate["objective"]
    candidate_operator = candidate["receiver_operator"]
    candidate_provenance = candidate["provenance"]

    parent_j = float(parent["objective"]["J_external"])
    candidate_j = float(candidate_objective["objective"])
    true_external = Path(parent["files"]["true"]).expanduser().resolve()
    if not true_external.is_file():
        raise RuntimeError(f"missing parent true external artifact: {true_external}")
    receiver_dir = contract["transition_root"] / "real_s43_receiver_spatial_operator"
    receiver_nodes = np.asarray(
        np.load(receiver_dir / "receiver_nodes.npy"), dtype=np.int64
    )
    receiver_weights = np.asarray(
        np.load(receiver_dir / "receiver_weights.npy"), dtype=np.float64
    )
    operator_hash = sha256_arrays(receiver_nodes, receiver_weights)
    contract_checks = candidate_provenance.get("contract_checks", {})
    checks = {
        "candidate_result": candidate.get("result") == CANDIDATE_PASS,
        "parent_objective_finite": bool(np.isfinite(parent_j)),
        "candidate_objective_finite": bool(np.isfinite(candidate_j)),
        "dt": float(candidate_objective["dt"]) == float(parent_contract["dt"]),
        "sample_count": int(candidate_objective["sample_count"])
        == int(parent_contract["sample_count"]),
        "receiver_count": int(candidate_objective["receiver_count"])
        == int(parent_contract["receiver_count"]),
        "component_count": int(candidate_objective["component_count"])
        == int(contract["config"]["forward_operator"]["dimension"]),
        "residual_sign": candidate_objective["residual_sign"]
        == "current_external - true_external"
        and parent_contract["residual_sign"].startswith(
            "current_external - true_external"
        ),
        "time_quadrature": candidate_objective["time_quadrature"]
        == parent_contract["time_weighting"],
        "receiver_order": candidate_operator["ordering"]
        == parent_contract["receiver_order"],
        "receiver_operator_hash": candidate_operator["operator_array_sha256"]
        == operator_hash,
        "true_external_path": Path(candidate_objective["true_external"]).resolve()
        == true_external,
        "true_external_sha256": candidate_objective["true_external_sha256"]
        == sha256_file(true_external),
        "stage5n_summary_provenance": Path(
            candidate_provenance["stage5n_summary"]
        ).resolve()
        == contract["parent_summary_path"],
        "transition_provenance": candidate_provenance["transition"]
        == contract["runtime"]["transition"],
        "iteration_provenance": int(candidate_provenance["iteration"])
        == int(contract["runtime"]["transition"].split("_")[1]),
        "certified_contract_checks": bool(contract_checks)
        and all(bool(value) for value in contract_checks.values()),
        "physical_formula": candidate_objective["formula"]
        == "0.5 * sum_t,r,c(w_t * residual[t,r,c]^2)",
    }
    parity = bool(all(checks.values()))
    delta_j = candidate_j - parent_j
    relative_decrease = (parent_j - candidate_j) / max(
        abs(parent_j), np.finfo(np.float64).tiny
    )
    descent = bool(parity and candidate_j < parent_j)
    return {
        "parent": parent,
        "candidate_summary": candidate,
        "parent_objective": parent_j,
        "candidate_objective": candidate_j,
        "delta_J": delta_j,
        "relative_decrease": relative_decrease,
        "checks": checks,
        "parity": parity,
        "descent": descent,
        "true_external_sha256": sha256_file(true_external),
        "receiver_operator_sha256": operator_hash,
    }


def candidate_state_audit(contract, objective_audit):
    candidate_dir = contract["candidate_dir"]
    h5_dir = candidate_dir / "mat" / "h5"
    h5_paths = {
        "kappa": h5_dir / "Mat_0_Kappa.h5",
        "mu": h5_dir / "Mat_0_Mu.h5",
        "density": h5_dir / "Mat_0_Density.h5",
    }
    fields = {name: read_samples(path) for name, path in h5_paths.items()}
    lam = fields["kappa"] - (2.0 / 3.0) * fields["mu"]
    required_candidate_keys = {
        "lambda_field",
        "lambda_",
        "mu",
        "kappa",
        "density",
        "active_indices",
        "direction_coords",
        "p_lambda",
        "p_mu",
        "direction_scale",
        "iteration",
        "step_mpa",
        "step_pa",
        "direction",
        "normalization",
        "update_rule",
    }
    with np.load(contract["candidate_state"]) as state:
        keys = set(state.files)
        lambda_difference = state["lambda_field"] - lam
        lambda_max_abs_error = float(np.max(np.abs(lambda_difference)))
        lambda_relative_l2 = float(
            np.linalg.norm(lambda_difference.reshape(-1))
            / max(
                np.linalg.norm(state["lambda_field"].reshape(-1)),
                np.finfo(np.float64).tiny,
            )
        )
        lambda_one_ulp = float(
            np.spacing(np.max(np.abs(state["lambda_field"])))
        )
        candidate_checks = {
            "required_keys": required_candidate_keys.issubset(keys),
            "lambda_h5_roundtrip_within_one_ulp": bool(
                lambda_max_abs_error <= lambda_one_ulp
            ),
            "lambda_alias_matches": bool(
                np.array_equal(state["lambda_"], state["lambda_field"])
            ),
            "mu_matches_h5": bool(np.array_equal(state["mu"], fields["mu"])),
            "kappa_matches_h5": bool(
                np.array_equal(state["kappa"], fields["kappa"])
            ),
            "density_matches_h5": bool(
                np.array_equal(state["density"], fields["density"])
            ),
            "iteration": int(state["iteration"]) == int(
                contract["runtime"]["transition"].split("_")[1]
            ),
            "step_configured": float(state["step_mpa"])
            in [float(value) for value in contract["config"]["line_search"]["steps_mpa"]],
            "active_shapes": state["active_indices"].shape
            == state["p_lambda"].shape
            == state["p_mu"].shape
            and state["direction_coords"].shape
            == (state["active_indices"].size, 3),
            "finite": all(
                np.all(np.isfinite(state[name]))
                for name in (
                    "lambda_field",
                    "mu",
                    "kappa",
                    "density",
                    "direction_coords",
                    "p_lambda",
                    "p_mu",
                )
            ),
        }
        state_metadata = {
            "keys": sorted(keys),
            "material_shape": list(state["kappa"].shape),
            "active_count": int(state["active_indices"].size),
            "iteration": int(state["iteration"]),
            "step_mpa": float(state["step_mpa"]),
            "step_pa": float(state["step_pa"]),
            "direction": str(state["direction"].item()),
            "normalization": str(state["normalization"].item()),
            "update_rule": str(state["update_rule"].item()),
            "lambda_h5_roundtrip_bitwise": bool(
                np.array_equal(state["lambda_field"], lam)
            ),
            "lambda_h5_roundtrip_max_abs_error": lambda_max_abs_error,
            "lambda_h5_roundtrip_relative_l2": lambda_relative_l2,
            "lambda_h5_roundtrip_one_ulp": lambda_one_ulp,
        }
    objective_material_hashes = objective_audit["candidate_summary"]["material"][
        "material_sha256"
    ]
    candidate_checks["objective_material_hashes"] = all(
        objective_material_hashes[path.name] == sha256_file(path)
        for path in h5_paths.values()
    )
    raw_candidate_valid = bool(all(candidate_checks.values()))
    legacy_acceptance_keys = {
        "J",
        "parent_J",
        "delta_J",
        "iter_k",
        "iter",
        "accepted_from",
        "accepted_dir",
        "transition",
        "descent",
        "candidate_misfit_summary",
    }
    certified_acceptance_keys = {
        "certified_candidate_objective_summary",
        "true_external_sha256",
        "receiver_operator_sha256",
        "objective_contract",
    }
    missing_acceptance_keys = sorted(
        (legacy_acceptance_keys | certified_acceptance_keys)
        - set(state_metadata["keys"])
    )
    return {
        "path": str(contract["candidate_state"]),
        "metadata": state_metadata,
        "checks": candidate_checks,
        "raw_candidate_state_valid": raw_candidate_valid,
        "missing_accepted_state_keys": missing_acceptance_keys,
        "ready_for_promotion_as_accepted_state": bool(
            raw_candidate_valid and not missing_acceptance_keys
        ),
    }


def promotion_contract(contract):
    runtime = contract["runtime"]
    k = int(runtime["transition"].split("_")[1])
    kp1 = k + 1
    next_transition = f"iter_{kp1:03d}_to_iter_{kp1 + 1:03d}"
    return {
        "target_iteration": kp1,
        "accepted_model_target": str(runtime["accepted_dir"]),
        "accepted_state_target": str(runtime["state_out"]),
        "accepted_model_contract": [
            "full parent-compatible ordinary workspace",
            "candidate Mat_0_Kappa.h5, Mat_0_Mu.h5, Mat_0_Density.h5",
            "inherited mesh/source/receiver/STF/runtime inputs",
            "certified objective and material-hash provenance manifest",
            "accepted_summary",
            (
                "ordinary parent traces required by the current "
                "create_iteration_context_generic preflight"
            ),
        ],
        "accepted_state_contract": [
            "full lambda/lambda_field, mu, kappa, density arrays",
            "J, parent_J, delta_J, descent",
            "iter_k=0 and iter=1 for this transition",
            "accepted_from, accepted_dir, transition",
            "certified candidate summary path and objective-contract hashes",
        ],
        "lbfgs_history_input_contract": {
            "s0": "active lambda/mu from iter_001 accepted minus iter_000 accepted",
            "y0": "certified optimizer gradient g1 minus stored certified g0",
            "model_sources": [
                "iteration_runtime_paths(config, 0).parent_workspace",
                "iteration_runtime_paths(config, 1).parent_workspace",
            ],
            "gradient_sources": [
                "iter_000_to_iter_001/search_direction/grad_lambda.npy and grad_mu.npy",
                "new iter_001 certified optimizer gradient",
            ],
            "coordinate_gate": "direction/gradient coordinates must match within configured tolerance",
            "curvature_gate": "s0.T y0 must exceed configured relative curvature threshold",
        },
        "next_transition": next_transition,
        "promotion_blockers": [
            "raw candidate state lacks accepted-state objective/provenance keys",
            (
                "certified external acceptance has not defined how the current "
                "iter_001 parent ordinary-trace requirement is satisfied"
            ),
        ],
    }


def write_audit(path, summary):
    lines = [
        "CERTIFIED EXTERNAL CANDIDATE ACCEPTANCE AUDIT",
        "=" * 72,
        f"RESULT = {summary['result']}",
        f"candidate = {summary['candidate']}",
        f"parent J = {summary['objective']['parent_objective']:.17e}",
        f"candidate J = {summary['objective']['candidate_objective']:.17e}",
        f"delta J = {summary['objective']['delta_J']:.17e}",
        f"relative decrease = {summary['objective']['relative_decrease']:.17e}",
        f"contract parity = {summary['objective']['parity']}",
        f"descent = {summary['objective']['descent']}",
        (
            "candidate state ready for promotion = "
            f"{summary['candidate_state']['ready_for_promotion_as_accepted_state']}"
        ),
        "model promotion performed = False",
        "SEM3D runs = 0",
        "external forwards = 0",
    ]
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("audit",), required=True)
    parser.add_argument("--repo", default=".")
    parser.add_argument("--config", required=True)
    parser.add_argument("--iter-k", type=int, required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--candidate-root")
    parser.add_argument("--candidate-objective-root")
    parser.add_argument("--parent-summary")
    parser.add_argument("--output-dir")
    args = parser.parse_args()

    contract = resolve_contract(args)
    objective = objective_contract_audit(contract)
    candidate_state = candidate_state_audit(contract, objective)
    descent_pass = bool(objective["parity"] and objective["descent"])
    summary = {
        "result": AUDIT_PASS if descent_pass else "FAIL_CERTIFIED_DESCENT_CANDIDATE",
        "mode": "audit",
        "candidate": contract["candidate"],
        "transition": contract["runtime"]["transition"],
        "line_search_policy": (
            "independent caller-selected candidate; strict J_candidate < J_parent gate"
        ),
        "configured_steps_mpa": [
            float(value) for value in contract["config"]["line_search"]["steps_mpa"]
        ],
        "configured_step_order_has_acceptance_semantics": False,
        "more_candidate_objectives_required": False,
        "objective": {
            key: value
            for key, value in objective.items()
            if key not in {"parent", "candidate_summary"}
        },
        "candidate_state": candidate_state,
        "promotion": promotion_contract(contract),
        "model_promotion_performed": False,
        "sem3d_runs": 0,
        "external_forwards": 0,
    }
    target = contract["output_root"] / contract["candidate"]
    atomic_json(target / "summary.json", summary)
    write_audit(target / "audit.txt", summary)
    print(json.dumps(summary, indent=2))
    if not descent_pass:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
