from pathlib import Path
import os
import argparse
import csv
import h5py
import numpy as np
import re
from collections import defaultdict

parser = argparse.ArgumentParser()
parser.add_argument("--component", required=True, choices=["x", "y", "z"])
parser.add_argument("--forward-manifest", required=True)
parser.add_argument("--adjoint-manifest", required=True)
parser.add_argument("--out-dir", required=True)
parser.add_argument("--label", required=True)
args = parser.parse_args()

ROOT = Path(os.environ.get("FATHI_BENCHMARK_ROOT", str(Path.home() / "sem3d_fathi_clean"))).expanduser().resolve()

FWD_MANIFEST = Path(args.forward_manifest).expanduser()
ADJ_MANIFEST = Path(args.adjoint_manifest).expanduser()
OUT_DIR = Path(args.out_dir).expanduser()
if not FWD_MANIFEST.is_absolute():
    FWD_MANIFEST = ROOT / FWD_MANIFEST
if not ADJ_MANIFEST.is_absolute():
    ADJ_MANIFEST = ROOT / ADJ_MANIFEST
if not OUT_DIR.is_absolute():
    OUT_DIR = ROOT / OUT_DIR
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_LAM = OUT_DIR / f"{args.label}_RHS_{args.component}_lambda.npy"
OUT_MU = OUT_DIR / f"{args.label}_RHS_{args.component}_mu.npy"
OUT_COORDS = OUT_DIR / f"{args.label}_RHS_{args.component}_coords.npy"
OUT_ORDER = OUT_DIR / f"{args.label}_RHS_{args.component}_order.csv"
OUT_TXT = OUT_DIR / f"{args.label}_RHS_{args.component}_summary.txt"

def read_manifest(path):
    with path.open(newline="") as f:
        return list(csv.DictReader(f))

def norm_coord(v):
    x = round(float(v), 8)
    if abs(x) < 1e-10:
        x = 0.0
    return x

def coord_key(row):
    return (norm_coord(row["x"]), norm_coord(row["y"]), norm_coord(row["z"]))

def decode_vars(f):
    if "Variables" not in f:
        return []
    out = []
    for x in f["Variables"][()]:
        if isinstance(x, bytes):
            out.append(x.decode(errors="ignore").strip())
        else:
            out.append(str(x).strip())
    return out

def find_dudx_cols(variables):
    cols = {}
    for j, v in enumerate(variables):
        vv = " ".join(str(v).split())
        m = re.match(r"DUDX\s+([0-9]+)$", vv)
        if m:
            idx = int(m.group(1))
            if 1 <= idx <= 9:
                cols[idx - 1] = j
    if sorted(cols.keys()) != list(range(9)):
        raise RuntimeError(f"Could not find DUDX 1..9 in variables: {variables}")
    return [cols[i] for i in range(9)]

def interp_adjoint_to_reverse_time(f_time, a_time, a_dudx):
    T = min(float(f_time[-1]), float(a_time[-1]))

    mask = (f_time >= 0.0) & (f_time <= T)
    t = f_time[mask]
    tau = T - t

    out = np.empty((len(t), 9), dtype=np.float64)
    for j in range(9):
        out[:, j] = np.interp(tau, a_time, a_dudx[:, j])

    return t, out, mask, T

def compute_rhs_one_receiver(f_arr, a_arr, f_dudx_cols, a_dudx_cols):
    f_time = np.asarray(f_arr[:, 0], dtype=np.float64)
    a_time = np.asarray(a_arr[:, 0], dtype=np.float64)

    f_d = np.asarray(f_arr[:, f_dudx_cols], dtype=np.float64)
    a_d_raw = np.asarray(a_arr[:, a_dudx_cols], dtype=np.float64)

    t, a_d, mask, T = interp_adjoint_to_reverse_time(f_time, a_time, a_d_raw)
    f_d = f_d[mask, :]

    # DUDX order assumed:
    # 0 dUx/dx, 1 dUx/dy, 2 dUx/dz,
    # 3 dUy/dx, 4 dUy/dy, 5 dUy/dz,
    # 6 dUz/dx, 7 dUz/dy, 8 dUz/dz

    div_f = f_d[:, 0] + f_d[:, 4] + f_d[:, 8]
    div_a = a_d[:, 0] + a_d[:, 4] + a_d[:, 8]

    q_lambda = - div_f * div_a

    diag = 2.0 * (
        f_d[:, 0] * a_d[:, 0]
        + f_d[:, 4] * a_d[:, 4]
        + f_d[:, 8] * a_d[:, 8]
    )

    xy = (f_d[:, 1] + f_d[:, 3]) * (a_d[:, 1] + a_d[:, 3])
    xz = (f_d[:, 2] + f_d[:, 6]) * (a_d[:, 2] + a_d[:, 6])
    yz = (f_d[:, 5] + f_d[:, 7]) * (a_d[:, 5] + a_d[:, 7])

    q_mu = - (diag + xy + xz + yz)

    # NumPy 2.x uses np.trapezoid instead of the old np.trapz.
    integrate_trapezoid = getattr(np, "trapezoid", None)
    if integrate_trapezoid is None:
        integrate_trapezoid = getattr(np, "trapz")

    rhs_lam = float(integrate_trapezoid(q_lambda, t))
    rhs_mu = float(integrate_trapezoid(q_mu, t))

    return rhs_lam, rhs_mu, len(t), T

fwd_rows = read_manifest(FWD_MANIFEST)
adj_rows = read_manifest(ADJ_MANIFEST)

if len(fwd_rows) != 38440:
    raise RuntimeError(f"forward manifest rows != 38440: {len(fwd_rows)}")
if len(adj_rows) != 38440:
    raise RuntimeError(f"adjoint manifest rows != 38440: {len(adj_rows)}")

adj_by_coord = {}
for r in adj_rows:
    k = coord_key(r)
    if k in adj_by_coord:
        raise RuntimeError(f"duplicate adjoint coordinate: {k}")
    adj_by_coord[k] = r

pairs = []
missing = []
for idx, fr in enumerate(fwd_rows):
    k = coord_key(fr)
    ar = adj_by_coord.get(k)
    if ar is None:
        missing.append((idx, k))
        continue
    pairs.append((idx, fr, ar))

if missing:
    raise RuntimeError(f"missing adjoint coordinates: {missing[:10]} total={len(missing)}")

n = len(pairs)
rhs_lambda = np.full(n, np.nan, dtype=np.float64)
rhs_mu = np.full(n, np.nan, dtype=np.float64)
coords = np.full((n, 3), np.nan, dtype=np.float64)
used_steps = np.zeros(n, dtype=np.int64)
used_T = np.full(n, np.nan, dtype=np.float64)

# group by forward trace file
pairs_by_fwd_file = defaultdict(list)
for item in pairs:
    idx, fr, ar = item
    pairs_by_fwd_file[fr["trace_file"]].append(item)

total_done = 0
bad = []

for fwd_rel, group in sorted(pairs_by_fwd_file.items()):
    fwd_path = ROOT / fwd_rel
    print("")
    print(f"Forward file: {fwd_rel}")
    print(f"  receivers in this file: {len(group)}")

    # inside this forward file, group by adjoint file
    by_adj = defaultdict(list)
    for item in group:
        _, _, ar = item
        by_adj[ar["trace_file"]].append(item)

    with h5py.File(fwd_path, "r") as ff:
        f_vars = decode_vars(ff)
        f_dudx_cols = find_dudx_cols(f_vars)

        for adj_rel, sub in sorted(by_adj.items()):
            adj_path = ROOT / adj_rel
            print(f"  Adjoint file: {adj_rel} receivers={len(sub)}")

            with h5py.File(adj_path, "r") as af:
                a_vars = decode_vars(af)
                a_dudx_cols = find_dudx_cols(a_vars)

                for idx, fr, ar in sub:
                    try:
                        f_arr = np.asarray(ff[fr["receiver_key"]], dtype=np.float64)
                        a_arr = np.asarray(af[ar["receiver_key"]], dtype=np.float64)

                        rl, rm, ns, T = compute_rhs_one_receiver(f_arr, a_arr, f_dudx_cols, a_dudx_cols)

                        rhs_lambda[idx] = rl
                        rhs_mu[idx] = rm
                        coords[idx, :] = [float(fr["x"]), float(fr["y"]), float(fr["z"])]
                        used_steps[idx] = ns
                        used_T[idx] = T

                    except Exception as e:
                        bad.append((idx, fr["batch"], fr["receiver_key"], ar["batch"], ar["receiver_key"], str(e)))

                    total_done += 1
                    if total_done % 1000 == 0:
                        print(f"    done {total_done}/{n}")

finite_lam = np.isfinite(rhs_lambda)
finite_mu = np.isfinite(rhs_mu)
finite_coords = np.all(np.isfinite(coords), axis=1)

np.save(OUT_LAM, rhs_lambda)
np.save(OUT_MU, rhs_mu)
np.save(OUT_COORDS, coords)

with OUT_ORDER.open("w", newline="") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=[
            "index", "x", "y", "z",
            "forward_batch", "forward_trace_file", "forward_receiver_key",
            "adjoint_batch", "adjoint_trace_file", "adjoint_receiver_key",
            "used_steps", "used_T"
        ]
    )
    writer.writeheader()
    for idx, fr, ar in pairs:
        writer.writerow({
            "index": idx,
            "x": fr["x"], "y": fr["y"], "z": fr["z"],
            "forward_batch": fr["batch"],
            "forward_trace_file": fr["trace_file"],
            "forward_receiver_key": fr["receiver_key"],
            "adjoint_batch": ar["batch"],
            "adjoint_trace_file": ar["trace_file"],
            "adjoint_receiver_key": ar["receiver_key"],
            "used_steps": int(used_steps[idx]),
            "used_T": float(used_T[idx]) if np.isfinite(used_T[idx]) else np.nan,
        })

summary = []
summary.append(f"RHS component summary: {args.label} component={args.component}")
summary.append("====================================================")
summary.append("")
summary.append(f"FWD_MANIFEST = {FWD_MANIFEST}")
summary.append(f"ADJ_MANIFEST = {ADJ_MANIFEST}")
summary.append("")
summary.append(f"n_pairs = {n}")
summary.append(f"bad_count = {len(bad)}")
summary.append("")
summary.append(f"lambda finite = {int(np.count_nonzero(finite_lam))} / {n}")
summary.append(f"mu finite = {int(np.count_nonzero(finite_mu))} / {n}")
summary.append(f"coords finite = {int(np.count_nonzero(finite_coords))} / {n}")
summary.append("")
summary.append(f"used_steps unique = {sorted(set(int(x) for x in used_steps if x > 0))[:20]}")
summary.append(f"used_T min/max = {float(np.nanmin(used_T))} {float(np.nanmax(used_T))}")
summary.append("")
if np.any(finite_lam):
    summary.append(f"rhs_lambda min = {float(np.nanmin(rhs_lambda))}")
    summary.append(f"rhs_lambda max = {float(np.nanmax(rhs_lambda))}")
    summary.append(f"rhs_lambda maxabs = {float(np.nanmax(np.abs(rhs_lambda)))}")
    summary.append(f"rhs_lambda l2 = {float(np.sqrt(np.nansum(rhs_lambda * rhs_lambda)))}")
if np.any(finite_mu):
    summary.append(f"rhs_mu min = {float(np.nanmin(rhs_mu))}")
    summary.append(f"rhs_mu max = {float(np.nanmax(rhs_mu))}")
    summary.append(f"rhs_mu maxabs = {float(np.nanmax(np.abs(rhs_mu)))}")
    summary.append(f"rhs_mu l2 = {float(np.sqrt(np.nansum(rhs_mu * rhs_mu)))}")
summary.append("")
summary.append(f"OUT_LAM = {OUT_LAM}")
summary.append(f"OUT_MU = {OUT_MU}")
summary.append(f"OUT_COORDS = {OUT_COORDS}")
summary.append(f"OUT_ORDER = {OUT_ORDER}")
summary.append("")
if len(bad) == 0 and np.all(finite_lam) and np.all(finite_mu) and np.all(finite_coords):
    summary.append("RESULT = PASS")
    summary.append("This RHS component can be kept; corresponding adjoint traces may be deleted after backup/check.")
else:
    summary.append("RESULT = FAIL")
    summary.append("Do not delete adjoint traces before fixing this component.")
    summary.append("")
    summary.append("First bad entries:")
    for item in bad[:30]:
        summary.append(f"  {item}")

OUT_TXT.write_text("\n".join(summary) + "\n")
print("")
print("\n".join(summary))
