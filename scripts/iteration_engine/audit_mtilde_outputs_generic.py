from pathlib import Path
import os
from datetime import datetime
import argparse
import json
import numpy as np
import sys
from scipy.sparse import load_npz

ROOT = Path(os.environ.get("FATHI_BENCHMARK_ROOT", str(Path.home() / "sem3d_fathi_clean"))).expanduser().resolve()

parser = argparse.ArgumentParser()
parser.add_argument("--iter-k", type=int, required=True)
parser.add_argument("--config", default="benchmark_fathi_strict/config/benchmark_config.json")
args = parser.parse_args()

config_path = ROOT / args.config
if not config_path.exists():
    print(f"Missing config: {config_path}")
    sys.exit(1)

config = json.loads(config_path.read_text())

k = args.iter_k
kp1 = k + 1
transition = f"iter_{k:03d}_to_iter_{kp1:03d}"
expected_n = int(config.get("interior_gradient_size", 38440))

run_result_root = ROOT / config["run_result_root"] / transition
component_rhs_dir = run_result_root / "component_rhs"
mtilde_solve_dir = run_result_root / "mtilde_solve"

paths = {
    "rhs_lambda": component_rhs_dir / "full_grid_trace_RHS_total_lambda.npy",
    "rhs_mu": component_rhs_dir / "full_grid_trace_RHS_total_mu.npy",
    "rhs_coords": component_rhs_dir / "full_grid_trace_RHS_total_coords.npy",
    "M": mtilde_solve_dir / "Mtilde_q1_consistent_interior_38440_sparse.npz",
    "g_lambda": mtilde_solve_dir / "g_lambda_mtilde_q1_interior_solve_rhs_total.npy",
    "g_mu": mtilde_solve_dir / "g_mu_mtilde_q1_interior_solve_rhs_total.npy",
    "g_coords": mtilde_solve_dir / "g_mtilde_q1_interior_solve_rhs_total_coords.npy",
    "summary": mtilde_solve_dir / "mtilde_q1_interior_solve_rhs_total_summary.txt",
}

missing = [name for name, p in paths.items() if not p.exists()]

report_dir = ROOT / "benchmark_fathi_strict/reports/mtilde_run"
report_dir.mkdir(parents=True, exist_ok=True)

created = datetime.now().isoformat()

lines = []
payload = {
    "created": created,
    "transition": transition,
    "missing": missing,
}

lines.append("Task 3K Mtilde output audit")
lines.append("===========================")
lines.append("")
lines.append(f"created = {created}")
lines.append(f"transition = {transition}")
lines.append(f"expected_n = {expected_n}")
lines.append("")

if missing:
    lines.append("Missing files:")
    for name in missing:
        lines.append(f"  {name}: {paths[name]}")
    result = "FAIL_MISSING"
else:
    rhs_lam = np.load(paths["rhs_lambda"])
    rhs_mu = np.load(paths["rhs_mu"])
    rhs_coords = np.load(paths["rhs_coords"])
    M = load_npz(paths["M"])
    g_lam = np.load(paths["g_lambda"])
    g_mu = np.load(paths["g_mu"])
    g_coords = np.load(paths["g_coords"])

    checks = {}

    checks["rhs_lambda_shape_ok"] = rhs_lam.shape == (expected_n,)
    checks["rhs_mu_shape_ok"] = rhs_mu.shape == (expected_n,)
    checks["rhs_coords_shape_ok"] = rhs_coords.shape == (expected_n, 3)
    checks["M_shape_ok"] = M.shape == (expected_n, expected_n)
    checks["g_lambda_shape_ok"] = g_lam.shape == (expected_n,)
    checks["g_mu_shape_ok"] = g_mu.shape == (expected_n,)
    checks["g_coords_shape_ok"] = g_coords.shape == (expected_n, 3)

    checks["rhs_lambda_finite_ok"] = bool(np.all(np.isfinite(rhs_lam)))
    checks["rhs_mu_finite_ok"] = bool(np.all(np.isfinite(rhs_mu)))
    checks["rhs_coords_finite_ok"] = bool(np.all(np.isfinite(rhs_coords)))
    checks["g_lambda_finite_ok"] = bool(np.all(np.isfinite(g_lam)))
    checks["g_mu_finite_ok"] = bool(np.all(np.isfinite(g_mu)))
    checks["g_coords_finite_ok"] = bool(np.all(np.isfinite(g_coords)))

    coord_diff = float(np.max(np.abs(rhs_coords - g_coords)))
    checks["coords_match_ok"] = coord_diff < 1e-12

    res_lam = M @ g_lam - rhs_lam
    res_mu = M @ g_mu - rhs_mu

    rel_res_lam = float(np.linalg.norm(res_lam) / max(np.linalg.norm(rhs_lam), 1e-300))
    rel_res_mu = float(np.linalg.norm(res_mu) / max(np.linalg.norm(rhs_mu), 1e-300))

    checks["rel_res_lambda_ok"] = rel_res_lam < 1e-10
    checks["rel_res_mu_ok"] = rel_res_mu < 1e-10

    summary_txt = paths["summary"].read_text(errors="ignore")
    checks["summary_pass_ok"] = "RESULT = PASS" in summary_txt

    def arr_stats(a):
        return {
            "shape": list(a.shape),
            "finite": int(np.count_nonzero(np.isfinite(a))),
            "size": int(a.size),
            "min": float(np.nanmin(a)),
            "max": float(np.nanmax(a)),
            "maxabs": float(np.nanmax(np.abs(a))),
            "l2": float(np.sqrt(np.nansum(a * a))),
        }

    payload.update({
        "checks": checks,
        "coord_diff": coord_diff,
        "rel_res_lambda": rel_res_lam,
        "rel_res_mu": rel_res_mu,
        "stats": {
            "rhs_lambda": arr_stats(rhs_lam),
            "rhs_mu": arr_stats(rhs_mu),
            "g_lambda": arr_stats(g_lam),
            "g_mu": arr_stats(g_mu),
        },
    })

    lines.append("Shapes:")
    lines.append(f"  rhs_lambda = {rhs_lam.shape}")
    lines.append(f"  rhs_mu     = {rhs_mu.shape}")
    lines.append(f"  rhs_coords = {rhs_coords.shape}")
    lines.append(f"  M          = {M.shape}, nnz={M.nnz}")
    lines.append(f"  g_lambda   = {g_lam.shape}")
    lines.append(f"  g_mu       = {g_mu.shape}")
    lines.append(f"  g_coords   = {g_coords.shape}")
    lines.append("")
    lines.append("Residuals:")
    lines.append(f"  rel_res_lambda = {rel_res_lam:.16e}")
    lines.append(f"  rel_res_mu     = {rel_res_mu:.16e}")
    lines.append("")
    lines.append("Coordinate consistency:")
    lines.append(f"  max |rhs_coords - g_coords| = {coord_diff:.16e}")
    lines.append("")
    lines.append("Gradient stats:")
    lines.append(f"  g_lambda min    = {payload['stats']['g_lambda']['min']:.16e}")
    lines.append(f"  g_lambda max    = {payload['stats']['g_lambda']['max']:.16e}")
    lines.append(f"  g_lambda maxabs = {payload['stats']['g_lambda']['maxabs']:.16e}")
    lines.append(f"  g_lambda l2     = {payload['stats']['g_lambda']['l2']:.16e}")
    lines.append(f"  g_mu min        = {payload['stats']['g_mu']['min']:.16e}")
    lines.append(f"  g_mu max        = {payload['stats']['g_mu']['max']:.16e}")
    lines.append(f"  g_mu maxabs     = {payload['stats']['g_mu']['maxabs']:.16e}")
    lines.append(f"  g_mu l2         = {payload['stats']['g_mu']['l2']:.16e}")
    lines.append("")
    lines.append("Checks:")
    for k2, v in checks.items():
        lines.append(f"  {k2} = {v}")

    result = "PASS" if all(checks.values()) else "CHECK_NEEDED"

payload["result"] = result

out_json = report_dir / f"{transition}_mtilde_output_audit.json"
out_txt = report_dir / f"{transition}_mtilde_output_audit.txt"

lines.append("")
lines.append(f"json = {out_json}")
lines.append("")
lines.append(f"RESULT = {result}")

out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
out_txt.write_text("\n".join(lines), encoding="utf-8")

print("\n".join(lines))

if result != "PASS":
    sys.exit(2)
