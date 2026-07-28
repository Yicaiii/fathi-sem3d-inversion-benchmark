from pathlib import Path
import numpy as np
import scipy.sparse as sp

ROOT = Path.home() / "sem3d_fathi_clean"
OUT = ROOT / "results/audit_teacher_feedback/iter005_mass_matrix"
OUT.mkdir(parents=True, exist_ok=True)

STATE = ROOT / "results/inversion_linear/states/iter_005_state.npz"

X_MIN, X_MAX = -20.0, 20.0
Y_MIN, Y_MAX = -20.0, 20.0
Z_TOP, Z_BOTTOM = 0.0, -50.0

def load_shape():
    st = np.load(STATE, allow_pickle=True)
    if "lambda_field" in st.files:
        return st["lambda_field"].shape
    raise RuntimeError(f"lambda_field not found in {STATE}. keys={st.files}")

def one_dimensional_p1_mass(coords):
    """
    Consistent 1D P1 mass matrix on a possibly nonuniform grid.
    For each segment [i,i+1] of length h:
        local mass = h/6 [[2,1],[1,2]]
    """
    coords = np.asarray(coords, dtype=float)
    n = len(coords)

    diag = np.zeros(n, dtype=float)
    off = np.zeros(n - 1, dtype=float)

    for i in range(n - 1):
        h = abs(coords[i + 1] - coords[i])
        diag[i] += h / 3.0
        diag[i + 1] += h / 3.0
        off[i] += h / 6.0

    return sp.diags(
        diagonals=[off, diag, off],
        offsets=[-1, 0, 1],
        shape=(n, n),
        format="csr",
    )

shape = load_shape()
nz, ny, nx = shape

x = np.linspace(X_MIN, X_MAX, nx)
y = np.linspace(Y_MIN, Y_MAX, ny)
z = np.linspace(Z_TOP, Z_BOTTOM, nz)

Mx = one_dimensional_p1_mass(x)
My = one_dimensional_p1_mass(y)
Mz = one_dimensional_p1_mass(z)

# Tensor-product Q1 mass matrix for structured grid.
# Flattening convention: field[iz, iy, ix], x fastest in C-order.
M3 = sp.kron(Mz, sp.kron(My, Mx, format="csr"), format="csr")

lumped = np.asarray(M3.sum(axis=1)).ravel().reshape(shape)

sp.save_npz(OUT / "Mtilde_q1_consistent_sparse.npz", M3)
np.save(OUT / "Mtilde_q1_rowsum_lumped_from_sparse.npy", lumped)

summary = []
summary.append("Q1/P1-like consistent material-grid mass matrix audit")
summary.append("====================================================")
summary.append("")
summary.append(f"state = {STATE}")
summary.append(f"shape = {shape}")
summary.append(f"n_dofs = {M3.shape[0]}")
summary.append(f"matrix_shape = {M3.shape}")
summary.append(f"nnz = {M3.nnz}")
summary.append(f"density = {M3.nnz / (M3.shape[0] * M3.shape[1]):.16e}")
summary.append("")
summary.append("Coordinate fallback:")
summary.append(f"  x = [{x[0]}, {x[-1]}], nx={nx}, dx={abs(x[1]-x[0]) if nx > 1 else np.nan}")
summary.append(f"  y = [{y[0]}, {y[-1]}], ny={ny}, dy={abs(y[1]-y[0]) if ny > 1 else np.nan}")
summary.append(f"  z = [{z[0]}, {z[-1]}], nz={nz}, dz={abs(z[1]-z[0]) if nz > 1 else np.nan}")
summary.append("")
summary.append("Lumped row-sum stats:")
summary.append(f"  sum = {float(np.sum(lumped)):.16e}")
summary.append(f"  min positive = {float(np.min(lumped[lumped > 0])):.16e}")
summary.append(f"  max = {float(np.max(lumped)):.16e}")
summary.append(f"  n_positive = {int(np.count_nonzero(lumped > 0))}")
summary.append("")
summary.append("Outputs:")
summary.append(f"  {OUT / 'Mtilde_q1_consistent_sparse.npz'}")
summary.append(f"  {OUT / 'Mtilde_q1_rowsum_lumped_from_sparse.npy'}")
summary.append("")
summary.append("Interpretation:")
summary.append("  This is a structured-grid tensor-product Q1 mass matrix prototype.")
summary.append("  It is a concrete candidate for the supervisor-requested material-grid Mtilde.")
summary.append("  A strict implementation should solve Mtilde g = RHS.")
summary.append("  Row-sum lumping is only a documented diagonal approximation, not the exact solve.")
summary.append("")

summary_path = OUT / "Mtilde_q1_consistent_summary.txt"
summary_path.write_text("\n".join(summary) + "\n")
print("\n".join(summary))
