from pathlib import Path
import os
from datetime import datetime
import argparse
import json
import shutil
import sys

import h5py
import numpy as np

ROOT = Path(os.environ.get("FATHI_BENCHMARK_ROOT", str(Path.home() / "sem3d_fathi_clean"))).expanduser().resolve()

parser = argparse.ArgumentParser()
parser.add_argument("--iter-k", type=int, required=True)
parser.add_argument("--candidate", default="line_search_neg_mtilde_1p00MPa")
parser.add_argument("--force", action="store_true")
parser.add_argument("--config", default="benchmark_fathi_strict/config/benchmark_config.json")
args = parser.parse_args()

config = json.loads((ROOT / args.config).read_text())

k = args.iter_k
kp1 = k + 1
transition = f"iter_{k:03d}_to_iter_{kp1:03d}"
shape = tuple(config.get("material_shape", [41, 33, 33]))

run_result_root = ROOT / config["run_result_root"] / transition
run_data_root = ROOT / config["run_data_root"] / f"iter_{kp1:03d}"
state_dir = ROOT / config["state_dir"]
state_out = state_dir / f"iter_{kp1:03d}_state_v2_corrected.npz"

candidate_workspace = run_data_root / "candidate_forward_workspaces" / args.candidate
misfit_summary = run_result_root / "candidate_misfits" / f"{args.candidate}_misfit_summary_v2.json"
accepted_dir = run_data_root / "accepted"

report_dir = ROOT / "benchmark_fathi_strict/reports/acceptance"
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
lines.append("")

if missing:
    lines.append("Missing required inputs:")
    for p in missing:
        lines.append(f"  {p}")
    record["result"] = "FAIL_MISSING_INPUTS"

else:
    misfit = json.loads(misfit_summary.read_text())

    candidate_J = float(misfit["total_J"])
    parent_J = misfit.get("parent_J", None)

    if parent_J is None:
        lines.append("parent_J is missing from v2 misfit summary.")
        record["result"] = "FAIL_NO_PARENT_J"
    else:
        parent_J = float(parent_J)
        delta_J = candidate_J - parent_J
        descent = candidate_J < parent_J

        record.update({
            "parent_J": parent_J,
            "candidate_J": candidate_J,
            "delta_J": delta_J,
            "descent": descent,
            "parent_J_source": misfit.get("parent_J_source"),
        })

        lines.append("Misfit comparison:")
        lines.append(f"  parent_J = {parent_J:.16e}")
        lines.append(f"  candidate_J = {candidate_J:.16e}")
        lines.append(f"  delta_J = {delta_J:.16e}")
        lines.append(f"  descent = {descent}")
        lines.append("")

        if not descent and not args.force:
            lines.append("Candidate is not descent.")
            lines.append("Not accepting this candidate.")
            lines.append("Next action: try smaller candidates, e.g. 0p50MPa or 0p25MPa.")
            record["result"] = "CHECK_NO_DESCENT"

        else:
            if accepted_dir.exists():
                shutil.rmtree(accepted_dir)

            shutil.copytree(candidate_workspace, accepted_dir, ignore=ignore_runtime)

            kappa, kappa_ds = read_h5_field(accepted_dir / "mat/h5/Mat_0_Kappa.h5")
            mu, mu_ds = read_h5_field(accepted_dir / "mat/h5/Mat_0_Mu.h5")
            density, density_ds = read_h5_field(accepted_dir / "mat/h5/Mat_0_Density.h5")
            lam = kappa - (2.0 / 3.0) * mu

            finite_ok = (
                np.all(np.isfinite(lam))
                and np.all(np.isfinite(mu))
                and np.all(np.isfinite(kappa))
                and np.all(np.isfinite(density))
            )
            positive_ok = (
                np.min(lam) > 0
                and np.min(mu) > 0
                and np.min(kappa) > 0
                and np.min(density) > 0
            )

            if not finite_ok or not positive_ok:
                record["result"] = "FAIL_BAD_ACCEPTED_FIELDS"
                lines.append(f"finite_ok = {finite_ok}")
                lines.append(f"positive_ok = {positive_ok}")
            else:
                state_dir.mkdir(parents=True, exist_ok=True)

                np.savez_compressed(
                    state_out,
                    **{
                        "lambda": lam,
                        "lambda_field": lam,
                        "mu": mu,
                        "kappa": kappa,
                        "density": density,
                        "J": np.array(candidate_J),
                        "parent_J": np.array(parent_J),
                        "delta_J": np.array(delta_J),
                        "iter_k": np.array(k),
                        "iter": np.array(kp1),
                        "accepted_from": args.candidate,
                        "accepted_dir": str(accepted_dir),
                        "transition": transition,
                        "descent": np.array(descent),
                        "candidate_misfit_summary": str(misfit_summary),
                    }
                )

                accepted_summary = accepted_dir / "accepted_summary.txt"
                accepted_summary.write_text(
                    "\n".join([
                        "Accepted candidate summary",
                        "==========================",
                        "",
                        f"created = {created}",
                        f"transition = {transition}",
                        f"candidate = {args.candidate}",
                        f"parent_J = {parent_J:.16e}",
                        f"candidate_J = {candidate_J:.16e}",
                        f"delta_J = {delta_J:.16e}",
                        f"descent = {descent}",
                        f"state_out = {state_out}",
                        "",
                        f"lambda min/max = {np.min(lam):.16e} {np.max(lam):.16e}",
                        f"mu min/max = {np.min(mu):.16e} {np.max(mu):.16e}",
                        f"kappa min/max = {np.min(kappa):.16e} {np.max(kappa):.16e}",
                        f"density min/max = {np.min(density):.16e} {np.max(density):.16e}",
                        "",
                        "RESULT = PASS",
                    ]) + "\n",
                    encoding="utf-8",
                )

                record["finite_ok"] = bool(finite_ok)
                record["positive_ok"] = bool(positive_ok)
                record["accepted_summary"] = str(accepted_summary)
                record["result"] = "PASS_ACCEPTED"

                lines.append("Accepted candidate.")
                lines.append(f"accepted_dir = {accepted_dir}")
                lines.append(f"state_out = {state_out}")
                lines.append("")
                lines.append(f"lambda min/max = {np.min(lam):.16e} {np.max(lam):.16e}")
                lines.append(f"mu min/max = {np.min(mu):.16e} {np.max(mu):.16e}")
                lines.append(f"kappa min/max = {np.min(kappa):.16e} {np.max(kappa):.16e}")
                lines.append(f"density min/max = {np.min(density):.16e} {np.max(density):.16e}")

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
