#!/usr/bin/env python3

from pathlib import Path
import argparse
import hashlib
import json
import sys


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def load_json(path):
    if not path.is_file():
        raise RuntimeError(f"MISSING: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def require(cond, message):
    if not cond:
        raise RuntimeError(message)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", default=".")
    parser.add_argument("--run", default="fathi_s43_repro_p20_t052")
    parser.add_argument("--parent-iteration", type=int, required=True)
    args = parser.parse_args()

    repo = Path(args.repo).resolve()
    run = args.run
    k = int(args.parent_iteration)
    child = k + 1

    parent_name = f"iter_{k:03d}"
    child_name = f"iter_{child:03d}"
    transition = f"{parent_name}_to_{child_name}"

    result_root = repo / "results" / run / transition
    data_root = repo / "data" / "reproduction" / run

    reverse_p = (
        result_root
        / "exact_reverse"
        / "production_reverse"
        / "summary.json"
    )

    gradient_p = (
        result_root
        / "corrected_gradient"
        / "registered_gradient.json"
    )

    history_candidates = list(
        (repo / "results" / run / "optimizer_history").glob(
            f"{parent_name}_to_{child_name}*/**/*.json"
        )
    )

    direction_p = (
        result_root
        / "physical_optimizer"
        / f"{parent_name}_lbfgs_eq25_direction"
        / "direction_summary.json"
    )

    armijo_p = (
        result_root
        / "external_armijo"
        / "armijo_summary.json"
    )

    accepted_p = (
        data_root
        / "iterations"
        / child_name
        / "accepted"
        / "accepted_summary.json"
    )

    state_p = (
        repo
        / "results"
        / run
        / "states"
        / f"{child_name}_state.npz"
    )

    reverse = load_json(reverse_p)
    gradient = load_json(gradient_p)
    direction = load_json(direction_p)
    armijo = load_json(armijo_p)
    accepted = load_json(accepted_p)

    require(
        reverse.get("result")
        == f"PASS_ITER{k:03d}_EXACT_REVERSE_MATERIAL_COVECTOR",
        "exact reverse result contract mismatch",
    )

    require(
        gradient.get("result")
        == f"PASS_ITER{k:03d}_REGISTERED_PHYSICAL_GRADIENT",
        "registered gradient result contract mismatch",
    )

    require(
        armijo.get("accepted") is True,
        "Armijo did not accept a candidate",
    )

    require(
        int(armijo.get("parent_iteration", -1)) == k,
        "Armijo parent iteration mismatch",
    )

    require(
        int(armijo.get("child_iteration", -1)) == child,
        "Armijo child iteration mismatch",
    )

    require(
        int(accepted.get("iter", -1)) == child,
        "accepted child iteration mismatch",
    )

    require(
        accepted.get("result")
        == f"PASS_CURRENT_T052_ITER{child:03d}_ACCEPTED_MODEL",
        "accepted child result contract mismatch",
    )

    require(
        state_p.is_file(),
        "child state is missing",
    )

    objective = accepted.get("objective", {})
    parent_j = float(objective["parent"])
    accepted_j = float(objective["accepted"])

    require(
        accepted_j <= parent_j,
        "accepted objective exceeds parent objective",
    )

    alpha = float(accepted.get("accepted_alpha"))

    relative_reduction = (
        (parent_j - accepted_j) / parent_j
        if parent_j != 0.0
        else 0.0
    )

    mtilde_lambda = None
    mtilde_mu = None

    corrected_summary = result_root / "corrected_gradient" / "summary.json"

    if corrected_summary.is_file():
        d = load_json(corrected_summary)

        for key in (
            "mtilde_residual_lambda",
            "MTILDE_RESIDUAL_LAMBDA",
        ):
            if key in d:
                mtilde_lambda = d[key]

        for key in (
            "mtilde_residual_mu",
            "MTILDE_RESIDUAL_MU",
        ):
            if key in d:
                mtilde_mu = d[key]

    report = {
        "schema_version": 1,
        "result": f"PASS_CURRENT_ITERATION_{k:03d}_TO_{child:03d}_FINAL_CLOSURE_AUDIT",
        "run": run,
        "parent_iteration": k,
        "child_iteration": child,
        "transition": transition,
        "objective": {
            "parent": parent_j,
            "accepted": accepted_j,
            "relative_reduction": relative_reduction,
        },
        "accepted_alpha": alpha,
        "results": {
            "exact_reverse": reverse.get("result"),
            "registered_gradient": gradient.get("result"),
            "armijo": armijo.get("result"),
            "accepted_child": accepted.get("result"),
        },
        "hashes": {
            "accepted_summary_sha256": sha256_file(accepted_p),
            "child_state_sha256": sha256_file(state_p),
            "reverse_summary_sha256": sha256_file(reverse_p),
            "registered_gradient_sha256": sha256_file(gradient_p),
            "direction_summary_sha256": sha256_file(direction_p),
            "armijo_summary_sha256": sha256_file(armijo_p),
        },
        "paths": {
            "accepted_summary": str(accepted_p),
            "child_state": str(state_p),
        },
        "audit_only": True,
        "numerical_reruns": 0,
    }

    audit_dir = result_root / "closure_audit"
    audit_dir.mkdir(parents=True, exist_ok=True)

    audit_p = audit_dir / "final_closure_audit.json"
    audit_p.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    print("=" * 76)
    print(f" CURRENT ITERATION {k:03d} -> {child:03d} FINAL CLOSURE AUDIT")
    print("=" * 76)

    print(f"PARENT_J = {parent_j:.17e}")
    print(f"ACCEPTED_J = {accepted_j:.17e}")
    print(f"RELATIVE_REDUCTION = {relative_reduction:.17e}")
    print(f"ACCEPTED_ALPHA = {alpha:.17g}")

    if mtilde_lambda is not None:
        print(f"MTILDE_RESIDUAL_LAMBDA = {mtilde_lambda}")

    if mtilde_mu is not None:
        print(f"MTILDE_RESIDUAL_MU = {mtilde_mu}")

    print(
        "ACCEPTED_SUMMARY_SHA256 =",
        report["hashes"]["accepted_summary_sha256"],
    )

    print(
        "CHILD_STATE_SHA256 =",
        report["hashes"]["child_state_sha256"],
    )

    print(f"AUDIT = {audit_p}")
    print("NUMERICAL_RERUNS = 0")
    print(
        "RESULT =",
        report["result"],
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(
            f"RESULT = BLOCK_CURRENT_ITERATION_FINAL_CLOSURE_AUDIT: {exc}",
            file=sys.stderr,
        )
        raise
