from pathlib import Path
from datetime import datetime
import argparse
import json
import shutil
import sys

import h5py
import numpy as np

from scripts.fathi_benchmark.runtime_paths import repository_root, resolve_path


ROOT = repository_root()

parser = argparse.ArgumentParser()
parser.add_argument("--iter-k", type=int, required=True)
parser.add_argument("--candidate", default="line_search_neg_mtilde_1p00MPa")
parser.add_argument(
    "--force",
    action="store_true",
    help=(
        "Legacy compatibility alias enabling both "
        "--allow-non-descent and --overwrite-existing."
    ),
)
parser.add_argument(
    "--allow-non-descent",
    action="store_true",
)
parser.add_argument(
    "--overwrite-existing",
    action="store_true",
)
parser.add_argument("--config", default="benchmark_fathi_strict/config/benchmark_config.json")
args = parser.parse_args()

allow_non_descent = (
    args.allow_non_descent
    or args.force
)

overwrite_existing = (
    args.overwrite_existing
    or args.force
)

config_path = resolve_path(
    args.config,
    base=ROOT,
)

config = json.loads(
    config_path.read_text(encoding="utf-8")
)

k = args.iter_k
kp1 = k + 1
transition = f"iter_{k:03d}_to_iter_{kp1:03d}"
shape = tuple(config.get("material_shape", [41, 33, 33]))

run_result_root = (
    resolve_path(
        config["run_result_root"],
        base=ROOT,
    )
    / transition
)

run_data_root = (
    resolve_path(
        config["run_data_root"],
        base=ROOT,
    )
    / f"iter_{kp1:03d}"
)

state_dir = resolve_path(
    config["state_dir"],
    base=ROOT,
)
state_out = state_dir / f"iter_{kp1:03d}_state_v2_corrected.npz"

candidate_workspace = run_data_root / "candidate_forward_workspaces" / args.candidate
misfit_summary = run_result_root / "candidate_misfits" / f"{args.candidate}_misfit_summary_v2.json"
accepted_dir = run_data_root / "accepted"

report_dir = resolve_path(
    "benchmark_fathi_strict/reports/acceptance",
    base=ROOT,
)
report_dir.mkdir(parents=True, exist_ok=True)

def find_dataset_name(path: Path, expected_shape):
    matches = []
    with h5py.File(path, "r") as h:
        def visit(name, obj):
            if isinstance(obj, h5py.Dataset) and tuple(obj.shape) == tuple(expected_shape):
                matches.append(name)
        h.visititems(visit)
    if not matches:
        raise RuntimeError(f"No dataset with shape {expected_shape} in {path}")
    return matches[0]

def read_h5_field(path: Path):
    ds = find_dataset_name(path, shape)
    with h5py.File(path, "r") as h:
        arr = np.asarray(h[ds], dtype=np.float64)
    return arr, ds

def ignore_runtime(dirpath, names):
    ignored = set()
    runtime_dirs = {
        "traces",
        "prot",
        "res",
        "logs",
        "__pycache__",
    }
    for n in names:
        if n in runtime_dirs:
            ignored.add(n)
        if n.endswith((".log", ".out", ".err")):
            ignored.add(n)
    return ignored

created = datetime.now().isoformat()

required = [
    candidate_workspace,
    candidate_workspace / "mat/h5/Mat_0_Kappa.h5",
    candidate_workspace / "mat/h5/Mat_0_Mu.h5",
    candidate_workspace / "mat/h5/Mat_0_Density.h5",
    misfit_summary,
]
missing = [p for p in required if not p.exists()]

record = {
    "created": created,
    "transition": transition,
    "candidate": args.candidate,
    "force": args.force,
    "allow_non_descent": allow_non_descent,
    "overwrite_existing": overwrite_existing,
    "state_out": str(state_out),
    "accepted_dir": str(accepted_dir),
    "misfit_summary": str(misfit_summary),
    "missing": [str(p) for p in missing],
    "result": None,
}

lines = []
lines.append("Task 5C accept candidate if descent v2")
lines.append("======================================")
lines.append("")
lines.append(f"created = {created}")
lines.append(f"transition = {transition}")
lines.append(f"candidate = {args.candidate}")
lines.append(f"force = {args.force}")
lines.append(
    f"allow_non_descent = {allow_non_descent}"
)
lines.append(
    f"overwrite_existing = {overwrite_existing}"
)
lines.append("")

if missing:
    lines.append("Missing required inputs:")
    for p in missing:
        lines.append(f"  {p}")
    record["result"] = "FAIL_MISSING_INPUTS"

else:
    misfit = json.loads(
        misfit_summary.read_text(
            encoding="utf-8"
        )
    )

    misfit_result = misfit.get("result")
    record["misfit_result"] = misfit_result

    if misfit_result != "PASS":
        lines.append(
            "Candidate misfit result is not PASS."
        )
        lines.append(
            f"misfit_result = {misfit_result}"
        )
        lines.append(
            "Candidate will not be accepted."
        )

        record["result"] = (
            "FAIL_MISFIT_NOT_PASS"
        )

    else:
        candidate_raw = misfit.get(
            "total_J"
        )
        parent_raw = misfit.get(
            "parent_J"
        )

        if (
            candidate_raw is None
            or parent_raw is None
        ):
            lines.append(
                "Candidate or parent J is missing "
                "from the v2 misfit summary."
            )

            record["result"] = (
                "FAIL_MISSING_J"
            )

        else:
            try:
                candidate_J = float(
                    candidate_raw
                )
                parent_J = float(
                    parent_raw
                )

            except (
                TypeError,
                ValueError,
                OverflowError,
            ) as exc:
                record["result"] = (
                    "FAIL_INVALID_J"
                )
                record["J_error"] = repr(exc)

                lines.append(
                    "Candidate or parent J "
                    "could not be parsed."
                )
                lines.append(
                    f"J_error = {exc!r}"
                )

            else:
                if not (
                    np.isfinite(candidate_J)
                    and np.isfinite(parent_J)
                ):
                    record["result"] = (
                        "FAIL_NONFINITE_J"
                    )

                    lines.append(
                        "Candidate or parent J "
                        "is non-finite."
                    )

                else:
                    delta_J = (
                        candidate_J
                        - parent_J
                    )

                    descent = (
                        candidate_J
                        < parent_J
                    )

                    record.update({
                        "parent_J": parent_J,
                        "candidate_J": candidate_J,
                        "delta_J": delta_J,
                        "descent": descent,
                        "parent_J_source": misfit.get(
                            "parent_J_source"
                        ),
                    })

                    lines.append(
                        "Misfit comparison:"
                    )
                    lines.append(
                        f"  parent_J = "
                        f"{parent_J:.16e}"
                    )
                    lines.append(
                        f"  candidate_J = "
                        f"{candidate_J:.16e}"
                    )
                    lines.append(
                        f"  delta_J = "
                        f"{delta_J:.16e}"
                    )
                    lines.append(
                        f"  descent = {descent}"
                    )
                    lines.append("")

                    if (
                        not descent
                        and not allow_non_descent
                    ):
                        lines.append(
                            "Candidate is not descent."
                        )
                        lines.append(
                            "Not accepting this "
                            "candidate."
                        )
                        lines.append(
                            "Use a smaller line-search "
                            "candidate."
                        )

                        record["result"] = (
                            "CHECK_NO_DESCENT"
                        )

                    else:
                        candidate_h5_dir = (
                            candidate_workspace
                            / "mat"
                            / "h5"
                        )

                        try:
                            kappa, kappa_ds = (
                                read_h5_field(
                                    candidate_h5_dir
                                    / "Mat_0_Kappa.h5"
                                )
                            )

                            mu, mu_ds = (
                                read_h5_field(
                                    candidate_h5_dir
                                    / "Mat_0_Mu.h5"
                                )
                            )

                            density, density_ds = (
                                read_h5_field(
                                    candidate_h5_dir
                                    / "Mat_0_Density.h5"
                                )
                            )

                            lam = (
                                kappa
                                - (2.0 / 3.0) * mu
                            )

                        except Exception as exc:
                            record["result"] = (
                                "FAIL_BAD_CANDIDATE_FIELDS"
                            )
                            record["field_error"] = (
                                repr(exc)
                            )

                            lines.append(
                                "Candidate H5 fields "
                                "could not be read."
                            )
                            lines.append(
                                f"field_error = {exc!r}"
                            )

                        else:
                            finite_ok = (
                                np.all(
                                    np.isfinite(lam)
                                )
                                and np.all(
                                    np.isfinite(mu)
                                )
                                and np.all(
                                    np.isfinite(kappa)
                                )
                                and np.all(
                                    np.isfinite(
                                        density
                                    )
                                )
                            )

                            positive_ok = (
                                np.min(lam) > 0
                                and np.min(mu) > 0
                                and np.min(kappa) > 0
                                and np.min(density) > 0
                            )

                            record[
                                "finite_ok"
                            ] = bool(finite_ok)

                            record[
                                "positive_ok"
                            ] = bool(positive_ok)

                            if (
                                not finite_ok
                                or not positive_ok
                            ):
                                record["result"] = (
                                    "FAIL_BAD_CANDIDATE_FIELDS"
                                )

                                lines.append(
                                    "Candidate fields "
                                    "failed validation."
                                )
                                lines.append(
                                    f"finite_ok = "
                                    f"{finite_ok}"
                                )
                                lines.append(
                                    f"positive_ok = "
                                    f"{positive_ok}"
                                )

                            else:
                                existing_outputs = [
                                    output
                                    for output in [
                                        accepted_dir,
                                        state_out,
                                    ]
                                    if output.exists()
                                ]

                                record[
                                    "existing_outputs"
                                ] = [
                                    str(output)
                                    for output
                                    in existing_outputs
                                ]

                                if (
                                    existing_outputs
                                    and not overwrite_existing
                                ):
                                    lines.append(
                                        "Refusing to overwrite "
                                        "existing accepted/state "
                                        "outputs."
                                    )

                                    for output in (
                                        existing_outputs
                                    ):
                                        lines.append(
                                            f"  {output}"
                                        )

                                    lines.append(
                                        "Use "
                                        "--overwrite-existing "
                                        "only after review."
                                    )

                                    record["result"] = (
                                        "REFUSE_EXISTING_OUTPUTS"
                                    )

                                else:
                                    state_dir.mkdir(
                                        parents=True,
                                        exist_ok=True,
                                    )

                                    accepted_dir.parent.mkdir(
                                        parents=True,
                                        exist_ok=True,
                                    )

                                    staging_dir = (
                                        accepted_dir.with_name(
                                            "."
                                            + accepted_dir.name
                                            + "."
                                            + args.candidate
                                            + ".staging"
                                        )
                                    )

                                    backup_dir = (
                                        accepted_dir.with_name(
                                            "."
                                            + accepted_dir.name
                                            + "."
                                            + args.candidate
                                            + ".backup"
                                        )
                                    )

                                    temporary_state = (
                                        state_out.with_name(
                                            "."
                                            + state_out.stem
                                            + "."
                                            + args.candidate
                                            + ".staging.npz"
                                        )
                                    )

                                    for stale_dir in [
                                        staging_dir,
                                        backup_dir,
                                    ]:
                                        if stale_dir.exists():
                                            shutil.rmtree(
                                                stale_dir
                                            )

                                    if temporary_state.exists():
                                        temporary_state.unlink()

                                    shutil.copytree(
                                        candidate_workspace,
                                        staging_dir,
                                        ignore=ignore_runtime,
                                    )

                                    staged_h5_dir = (
                                        staging_dir
                                        / "mat"
                                        / "h5"
                                    )

                                    try:
                                        staged_kappa, _ = (
                                            read_h5_field(
                                                staged_h5_dir
                                                / "Mat_0_Kappa.h5"
                                            )
                                        )

                                        staged_mu, _ = (
                                            read_h5_field(
                                                staged_h5_dir
                                                / "Mat_0_Mu.h5"
                                            )
                                        )

                                        staged_density, _ = (
                                            read_h5_field(
                                                staged_h5_dir
                                                / "Mat_0_Density.h5"
                                            )
                                        )

                                        staged_lambda = (
                                            staged_kappa
                                            - (2.0 / 3.0)
                                            * staged_mu
                                        )

                                        staged_valid = (
                                            np.all(
                                                np.isfinite(
                                                    staged_lambda
                                                )
                                            )
                                            and np.all(
                                                np.isfinite(
                                                    staged_mu
                                                )
                                            )
                                            and np.all(
                                                np.isfinite(
                                                    staged_kappa
                                                )
                                            )
                                            and np.all(
                                                np.isfinite(
                                                    staged_density
                                                )
                                            )
                                            and np.min(
                                                staged_lambda
                                            ) > 0
                                            and np.min(
                                                staged_mu
                                            ) > 0
                                            and np.min(
                                                staged_kappa
                                            ) > 0
                                            and np.min(
                                                staged_density
                                            ) > 0
                                        )

                                    except Exception as exc:
                                        staged_valid = False
                                        record[
                                            "staging_error"
                                        ] = repr(exc)

                                    if not staged_valid:
                                        shutil.rmtree(
                                            staging_dir
                                        )

                                        record["result"] = (
                                            "FAIL_STAGED_FIELDS"
                                        )

                                        lines.append(
                                            "Staged accepted "
                                            "fields failed "
                                            "validation."
                                        )

                                    else:
                                        np.savez_compressed(
                                            temporary_state,
                                            **{
                                                "lambda": lam,
                                                "lambda_field": lam,
                                                "mu": mu,
                                                "kappa": kappa,
                                                "density": density,
                                                "J": np.array(
                                                    candidate_J
                                                ),
                                                "parent_J": np.array(
                                                    parent_J
                                                ),
                                                "delta_J": np.array(
                                                    delta_J
                                                ),
                                                "iter_k": np.array(k),
                                                "iter": np.array(
                                                    kp1
                                                ),
                                                "accepted_from":
                                                    args.candidate,
                                                "accepted_dir":
                                                    str(
                                                        accepted_dir
                                                    ),
                                                "transition":
                                                    transition,
                                                "descent": np.array(
                                                    descent
                                                ),
                                                (
                                                    "candidate_"
                                                    "misfit_summary"
                                                ):
                                                    str(
                                                        misfit_summary
                                                    ),
                                            }
                                        )

                                        moved_old_accepted = False
                                        commit_error = None

                                        try:
                                            if accepted_dir.exists():
                                                accepted_dir.rename(
                                                    backup_dir
                                                )

                                                moved_old_accepted = (
                                                    True
                                                )

                                            staging_dir.rename(
                                                accepted_dir
                                            )

                                            temporary_state.replace(
                                                state_out
                                            )

                                        except Exception as exc:
                                            commit_error = exc

                                            if accepted_dir.exists():
                                                shutil.rmtree(
                                                    accepted_dir
                                                )

                                            if (
                                                moved_old_accepted
                                                and backup_dir.exists()
                                            ):
                                                backup_dir.rename(
                                                    accepted_dir
                                                )

                                            if staging_dir.exists():
                                                shutil.rmtree(
                                                    staging_dir
                                                )

                                            if (
                                                temporary_state.exists()
                                            ):
                                                temporary_state.unlink()

                                        else:
                                            if backup_dir.exists():
                                                shutil.rmtree(
                                                    backup_dir
                                                )

                                        if commit_error is not None:
                                            record["result"] = (
                                                "FAIL_ATOMIC_COMMIT"
                                            )
                                            record[
                                                "commit_error"
                                            ] = repr(
                                                commit_error
                                            )

                                            lines.append(
                                                "Acceptance commit "
                                                "failed and old "
                                                "accepted output "
                                                "was restored."
                                            )
                                            lines.append(
                                                f"commit_error = "
                                                f"{commit_error!r}"
                                            )

                                        else:
                                            accepted_summary = (
                                                accepted_dir
                                                / "accepted_summary.txt"
                                            )

                                            accepted_summary.write_text(
                                                "\n".join([
                                                    (
                                                        "Accepted "
                                                        "candidate "
                                                        "summary"
                                                    ),
                                                    (
                                                        "==========="
                                                        "==========="
                                                    ),
                                                    "",
                                                    (
                                                        f"created = "
                                                        f"{created}"
                                                    ),
                                                    (
                                                        f"transition = "
                                                        f"{transition}"
                                                    ),
                                                    (
                                                        f"candidate = "
                                                        f"{args.candidate}"
                                                    ),
                                                    (
                                                        f"parent_J = "
                                                        f"{parent_J:.16e}"
                                                    ),
                                                    (
                                                        f"candidate_J = "
                                                        f"{candidate_J:.16e}"
                                                    ),
                                                    (
                                                        f"delta_J = "
                                                        f"{delta_J:.16e}"
                                                    ),
                                                    (
                                                        f"descent = "
                                                        f"{descent}"
                                                    ),
                                                    (
                                                        f"state_out = "
                                                        f"{state_out}"
                                                    ),
                                                    "",
                                                    (
                                                        "lambda min/max = "
                                                        f"{np.min(lam):.16e} "
                                                        f"{np.max(lam):.16e}"
                                                    ),
                                                    (
                                                        "mu min/max = "
                                                        f"{np.min(mu):.16e} "
                                                        f"{np.max(mu):.16e}"
                                                    ),
                                                    (
                                                        "kappa min/max = "
                                                        f"{np.min(kappa):.16e} "
                                                        f"{np.max(kappa):.16e}"
                                                    ),
                                                    (
                                                        "density min/max = "
                                                        f"{np.min(density):.16e} "
                                                        f"{np.max(density):.16e}"
                                                    ),
                                                    "",
                                                    "RESULT = PASS",
                                                ]) + "\n",
                                                encoding="utf-8",
                                            )

                                            record[
                                                "accepted_summary"
                                            ] = str(
                                                accepted_summary
                                            )

                                            record["result"] = (
                                                "PASS_ACCEPTED"
                                            )

                                            lines.append(
                                                "Accepted candidate."
                                            )
                                            lines.append(
                                                f"accepted_dir = "
                                                f"{accepted_dir}"
                                            )
                                            lines.append(
                                                f"state_out = "
                                                f"{state_out}"
                                            )
                                            lines.append("")
                                            lines.append(
                                                "lambda min/max = "
                                                f"{np.min(lam):.16e} "
                                                f"{np.max(lam):.16e}"
                                            )
                                            lines.append(
                                                "mu min/max = "
                                                f"{np.min(mu):.16e} "
                                                f"{np.max(mu):.16e}"
                                            )
                                            lines.append(
                                                "kappa min/max = "
                                                f"{np.min(kappa):.16e} "
                                                f"{np.max(kappa):.16e}"
                                            )
                                            lines.append(
                                                "density min/max = "
                                                f"{np.min(density):.16e} "
                                                f"{np.max(density):.16e}"
                                            )


json_path = report_dir / f"{transition}_{args.candidate}_acceptance_v2.json"
txt_path = report_dir / f"{transition}_{args.candidate}_acceptance_v2.txt"

lines.append("")
lines.append(f"json = {json_path}")
lines.append("")
lines.append(f"RESULT = {record['result']}")

json_path.write_text(json.dumps(record, indent=2), encoding="utf-8")
txt_path.write_text("\n".join(lines), encoding="utf-8")

print("\n".join(lines))

if record["result"] not in ["PASS_ACCEPTED", "CHECK_NO_DESCENT"]:
    sys.exit(1)
