from pathlib import Path
import numpy as np
from scipy.sparse import load_npz, save_npz
from scipy.sparse.linalg import spsolve

ROOT = Path.home() / "sem3d_fathi_clean"

M_PATH = ROOT / "results/audit_teacher_feedback/iter005_mass_matrix/Mtilde_q1_consistent_sparse.npz"
RHS_DIR = ROOT / "results/longterm_capteurs_material_grid/component_rhs"
OUT = ROOT / "results/longterm_capteurs_material_grid/mtilde_solve_full_grid"
OUT.mkdir(parents=True, exist_ok=True)

M = load_npz(M_PATH).tocsr()

rhs_lam = np.load(RHS_DIR / "full_grid_trace_RHS_total_lambda.npy")
rhs_mu = np.load(RHS_DIR / "full_grid_trace_RHS_total_mu.npy")
rhs_coords = np.load(RHS_DIR / "full_grid_trace_RHS_total_coords.npy")

if M.shape != (44649, 44649):
    raise RuntimeError(f"Unexpected Mtilde shape: {M.shape}")

if rhs_lam.shape != (38440,) or rhs_mu.shape != (38440,) or rhs_coords.shape != (38440, 3):
    raise RuntimeError(
        f"Unexpected RHS shapes: lambda={rhs_lam.shape}, mu={rhs_mu.shape}, coords={rhs_coords.shape}"
    )

# IMPORTANT:
# This follows scripts/audit/302_build_q1_consistent_mtilde.py exactly:
#   shape = nz, ny, nx
#   x = linspace(-20, 20, nx)
#   y = linspace(-20, 20, ny)
#   z = linspace(0, -50, nz)
#   M3 = kron(Mz, kron(My, Mx))
#   flattening convention: field[iz, iy, ix], x fastest in C-order
nx, ny, nz = 33, 33, 41
x = np.linspace(-20.0, 20.0, nx)
y = np.linspace(-20.0, 20.0, ny)
z = np.linspace(0.0, -50.0, nz)

def key(p):
    return tuple(round(float(v), 8) for v in p)

full_coords = []
for iz in range(nz):
    for iy in range(ny):
        for ix in range(nx):
            full_coords.append((x[ix], y[iy], z[iz]))
full_coords = np.asarray(full_coords, dtype=np.float64)

if full_coords.shape != (44649, 3):
    raise RuntimeError(f"full_coords shape mismatch: {full_coords.shape}")

full_map = {key(p): i for i, p in enumerate(full_coords)}

idx = []
missing = []
for p in rhs_coords:
    k = key(p)
    j = full_map.get(k)
    if j is None:
        missing.append(k)
    else:
        idx.append(j)

if missing:
    print("First missing coordinates:")
    for m in missing[:20]:
        print("  ", m)
    raise RuntimeError(f"Missing RHS coordinates in full Mtilde grid: {len(missing)}")

idx = np.asarray(idx, dtype=np.int64)

if len(np.unique(idx)) != len(idx):
    raise RuntimeError("Duplicate Mtilde indices found in RHS mapping.")

mapped = full_coords[idx]
max_coord_diff = float(np.max(np.abs(mapped - rhs_coords)))

if max_coord_diff > 1e-8:
    raise RuntimeError(f"Mapped coords do not match RHS coords. max diff = {max_coord_diff}")

print("Mapping PASS")
print("  row order = z_y_x from field[iz, iy, ix], x fastest")
print("  idx size =", idx.size)
print("  max_coord_diff =", max_coord_diff)

M_interior = M[idx, :][:, idx].tocsr()

save_npz(OUT / "Mtilde_q1_consistent_interior_38440_sparse.npz", M_interior)
np.save(OUT / "Mtilde_q1_consistent_interior_38440_indices.npy", idx)
np.save(OUT / "Mtilde_q1_consistent_interior_38440_coords.npy", rhs_coords)

print("Solving M_interior g_lambda = RHS_total_lambda ...")
g_lam = spsolve(M_interior, rhs_lam)

print("Solving M_interior g_mu = RHS_total_mu ...")
g_mu = spsolve(M_interior, rhs_mu)

np.save(OUT / "g_lambda_mtilde_q1_interior_solve_rhs_total.npy", g_lam)
np.save(OUT / "g_mu_mtilde_q1_interior_solve_rhs_total.npy", g_mu)
np.save(OUT / "g_mtilde_q1_interior_solve_rhs_total_coords.npy", rhs_coords)

res_lam = M_interior @ g_lam - rhs_lam
res_mu = M_interior @ g_mu - rhs_mu

rel_res_lam = np.linalg.norm(res_lam) / max(np.linalg.norm(rhs_lam), 1e-300)
rel_res_mu = np.linalg.norm(res_mu) / max(np.linalg.norm(rhs_mu), 1e-300)

summary = []
summary.append("Mtilde interior solve RHS_total summary")
summary.append("======================================")
summary.append("")
summary.append(f"M_PATH = {M_PATH}")
summary.append(f"full M shape = {M.shape}")
summary.append(f"full M nnz = {M.nnz}")
summary.append("")
summary.append("Mtilde grid source:")
summary.append("  scripts/audit/302_build_q1_consistent_mtilde.py")
summary.append("  shape convention = field[iz, iy, ix]")
summary.append("  row order = z_y_x")
summary.append("  x fastest in C-order")
summary.append("  z = linspace(0.0, -50.0, 41)")
summary.append("")
summary.append(f"full grid nx ny nz = {nx} {ny} {nz}")
summary.append(f"full grid total nodes = {full_coords.shape[0]}")
summary.append("")
summary.append(f"interior idx size = {idx.size}")
summary.append(f"interior idx unique = {len(np.unique(idx))}")
summary.append(f"max mapped coord diff = {max_coord_diff:.16e}")
summary.append("")
summary.append(f"M_interior shape = {M_interior.shape}")
summary.append(f"M_interior nnz = {M_interior.nnz}")
summary.append("")
summary.append(f"rhs_lambda finite = {np.count_nonzero(np.isfinite(rhs_lam))} / {rhs_lam.size}")
summary.append(f"rhs_mu finite = {np.count_nonzero(np.isfinite(rhs_mu))} / {rhs_mu.size}")
summary.append("")
summary.append(f"g_lambda finite = {np.count_nonzero(np.isfinite(g_lam))} / {g_lam.size}")
summary.append(f"g_mu finite = {np.count_nonzero(np.isfinite(g_mu))} / {g_mu.size}")
summary.append("")
summary.append(f"g_lambda min = {np.min(g_lam):.16e}")
summary.append(f"g_lambda max = {np.max(g_lam):.16e}")
summary.append(f"g_lambda maxabs = {np.max(np.abs(g_lam)):.16e}")
summary.append(f"g_lambda l2 = {np.sqrt(np.sum(g_lam * g_lam)):.16e}")
summary.append("")
summary.append(f"g_mu min = {np.min(g_mu):.16e}")
summary.append(f"g_mu max = {np.max(g_mu):.16e}")
summary.append(f"g_mu maxabs = {np.max(np.abs(g_mu)):.16e}")
summary.append(f"g_mu l2 = {np.sqrt(np.sum(g_mu * g_mu)):.16e}")
summary.append("")
summary.append(f"relative residual lambda = {rel_res_lam:.16e}")
summary.append(f"relative residual mu = {rel_res_mu:.16e}")
summary.append("")
summary.append("Outputs:")
summary.append(f"  {OUT / 'Mtilde_q1_consistent_interior_38440_sparse.npz'}")
summary.append(f"  {OUT / 'Mtilde_q1_consistent_interior_38440_indices.npy'}")
summary.append(f"  {OUT / 'Mtilde_q1_consistent_interior_38440_coords.npy'}")
summary.append(f"  {OUT / 'g_lambda_mtilde_q1_interior_solve_rhs_total.npy'}")
summary.append(f"  {OUT / 'g_mu_mtilde_q1_interior_solve_rhs_total.npy'}")
summary.append(f"  {OUT / 'g_mtilde_q1_interior_solve_rhs_total_coords.npy'}")
summary.append("")
summary.append("NEXT_STEP_NOTE:")
summary.append("  g_lambda and g_mu are defined on the 38440 interior material-grid nodes.")
summary.append("  Before line-search, decide whether to update only these interior nodes or embed them back into the 44649-node full grid.")
summary.append("")
if (
    np.all(np.isfinite(g_lam))
    and np.all(np.isfinite(g_mu))
    and rel_res_lam < 1e-8
    and rel_res_mu < 1e-8
):
    summary.append("RESULT = PASS")
    summary.append("Meaning: interior Q1 Mtilde solve completed.")
else:
    summary.append("RESULT = CHECK")
    summary.append("Meaning: solve finished, but inspect residual/finite stats before using direction.")

txt = "\n".join(summary) + "\n"
(OUT / "mtilde_q1_interior_solve_rhs_total_summary.txt").write_text(txt)
print("")
print(txt)
