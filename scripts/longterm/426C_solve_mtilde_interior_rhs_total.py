import argparse

import numpy as np
from scipy.sparse import load_npz, save_npz
from scipy.sparse.linalg import spsolve

from scripts.fathi_benchmark.runtime_paths import (
    repository_root,
    resolve_path,
)


parser = argparse.ArgumentParser()
parser.add_argument(
    "--matrix",
    required=True,
)
parser.add_argument(
    "--matrix-indices",
)
parser.add_argument(
    "--matrix-coords",
)
parser.add_argument(
    "--rhs-dir",
    required=True,
)
parser.add_argument(
    "--out-dir",
    required=True,
)
args = parser.parse_args()

ROOT = repository_root()

matrix_path = resolve_path(
    args.matrix,
    base=ROOT,
)

matrix_indices_path = (
    resolve_path(
        args.matrix_indices,
        base=ROOT,
    )
    if args.matrix_indices
    else None
)

matrix_coords_path = (
    resolve_path(
        args.matrix_coords,
        base=ROOT,
    )
    if args.matrix_coords
    else None
)

rhs_dir = resolve_path(
    args.rhs_dir,
    base=ROOT,
)

out_dir = resolve_path(
    args.out_dir,
    base=ROOT,
)

out_dir.mkdir(
    parents=True,
    exist_ok=True,
)

matrix = load_npz(
    matrix_path
).tocsr()

rhs_lambda = np.load(
    rhs_dir
    / "full_grid_trace_RHS_total_lambda.npy"
)

rhs_mu = np.load(
    rhs_dir
    / "full_grid_trace_RHS_total_mu.npy"
)

rhs_coords = np.load(
    rhs_dir
    / "full_grid_trace_RHS_total_coords.npy"
)

if rhs_lambda.ndim != 1:
    raise RuntimeError(
        "Lambda RHS must be one-dimensional: "
        f"{rhs_lambda.shape}"
    )

expected_n = int(rhs_lambda.size)

if rhs_mu.shape != (expected_n,):
    raise RuntimeError(
        "Unexpected mu RHS shape: "
        f"{rhs_mu.shape}"
    )

if rhs_coords.shape != (expected_n, 3):
    raise RuntimeError(
        "Unexpected RHS coordinate shape: "
        f"{rhs_coords.shape}"
    )

if not np.all(np.isfinite(rhs_lambda)):
    raise RuntimeError(
        "Lambda RHS contains non-finite values"
    )

if not np.all(np.isfinite(rhs_mu)):
    raise RuntimeError(
        "Mu RHS contains non-finite values"
    )

if not np.all(np.isfinite(rhs_coords)):
    raise RuntimeError(
        "RHS coordinates contain non-finite values"
    )

nx, ny, nz = 33, 33, 41

x_values = np.linspace(
    -20.0,
    20.0,
    nx,
)

y_values = np.linspace(
    -20.0,
    20.0,
    ny,
)

z_values = np.linspace(
    0.0,
    -50.0,
    nz,
)

full_coords = np.asarray(
    [
        (x, y, z)
        for z in z_values
        for y in y_values
        for x in x_values
    ],
    dtype=np.float64,
)


def key(point):
    return tuple(
        round(float(value), 8)
        for value in point
    )


full_map = {
    key(point): index
    for index, point in enumerate(full_coords)
}

missing = [
    key(point)
    for point in rhs_coords
    if key(point) not in full_map
]

if missing:
    raise RuntimeError(
        "RHS coordinates are missing from the "
        "full Mtilde grid. First missing: "
        f"{missing[:20]}"
    )

rhs_indices = np.asarray(
    [
        full_map[key(point)]
        for point in rhs_coords
    ],
    dtype=np.int64,
)

if np.unique(rhs_indices).size != expected_n:
    raise RuntimeError(
        "Duplicate full-grid indices found in "
        "the RHS coordinate mapping"
    )

mapped = full_coords[rhs_indices]

max_coord_diff = float(
    np.max(np.abs(mapped - rhs_coords))
)

if max_coord_diff > 1e-8:
    raise RuntimeError(
        "Mapped coordinates do not match RHS "
        f"coordinates: {max_coord_diff}"
    )

full_shape = (
    full_coords.shape[0],
    full_coords.shape[0],
)

matrix_source_mode = None
matrix_permutation_applied = False

if matrix.shape == full_shape:
    active_matrix = matrix[
        rhs_indices, :
    ][
        :, rhs_indices
    ].tocsr()

    matrix_source_mode = "full_matrix_extract"

elif matrix.shape == (
    expected_n,
    expected_n,
):
    if matrix_indices_path is None:
        raise RuntimeError(
            "A reduced matrix requires "
            "--matrix-indices so it can be "
            "reordered to the RHS coordinate order"
        )

    artifact_indices = np.load(
        matrix_indices_path
    ).astype(np.int64)

    if artifact_indices.shape != (
        expected_n,
    ):
        raise RuntimeError(
            "Unexpected reduced matrix index shape: "
            f"{artifact_indices.shape}"
        )

    if np.unique(artifact_indices).size != expected_n:
        raise RuntimeError(
            "Reduced matrix indices are not unique"
        )

    artifact_position = {
        int(full_index): position
        for position, full_index
        in enumerate(artifact_indices)
    }

    missing_indices = [
        int(full_index)
        for full_index in rhs_indices
        if int(full_index)
        not in artifact_position
    ]

    if missing_indices:
        raise RuntimeError(
            "RHS controls are not covered by the "
            "reduced Mtilde artifact. First missing: "
            f"{missing_indices[:20]}"
        )

    permutation = np.asarray(
        [
            artifact_position[int(full_index)]
            for full_index in rhs_indices
        ],
        dtype=np.int64,
    )

    if np.unique(permutation).size != expected_n:
        raise RuntimeError(
            "Reduced matrix reorder permutation "
            "is not unique"
        )

    if matrix_coords_path is not None:
        artifact_coords = np.load(
            matrix_coords_path
        ).astype(np.float64)

        if artifact_coords.shape != (
            expected_n,
            3,
        ):
            raise RuntimeError(
                "Unexpected reduced matrix "
                "coordinate shape: "
                f"{artifact_coords.shape}"
            )

        artifact_coord_diff = float(
            np.max(
                np.abs(
                    full_coords[artifact_indices]
                    - artifact_coords
                )
            )
        )

        if artifact_coord_diff > 1e-8:
            raise RuntimeError(
                "Reduced matrix coordinate artifact "
                "does not match its indices: "
                f"{artifact_coord_diff}"
            )

    active_matrix = matrix[
        permutation, :
    ][
        :, permutation
    ].tocsr()

    matrix_source_mode = "reduced_matrix_reorder"
    matrix_permutation_applied = not np.array_equal(
        permutation,
        np.arange(expected_n),
    )

else:
    raise RuntimeError(
        "Mtilde matrix must be either the full "
        f"{full_shape} matrix or an "
        f"{(expected_n, expected_n)} reduced matrix. "
        f"Received {matrix.shape}"
    )

matrix_name = (
    "Mtilde_q1_consistent_interior_"
    f"{expected_n}"
)

matrix_out = (
    out_dir
    / f"{matrix_name}_sparse.npz"
)

indices_out = (
    out_dir
    / f"{matrix_name}_indices.npy"
)

coords_out = (
    out_dir
    / f"{matrix_name}_coords.npy"
)

save_npz(
    matrix_out,
    active_matrix,
)

np.save(
    indices_out,
    rhs_indices,
)

np.save(
    coords_out,
    rhs_coords,
)

print(
    "Solving active Mtilde g_lambda "
    "= RHS_total_lambda ..."
)

gradient_lambda = spsolve(
    active_matrix,
    rhs_lambda,
)

print(
    "Solving active Mtilde g_mu "
    "= RHS_total_mu ..."
)

gradient_mu = spsolve(
    active_matrix,
    rhs_mu,
)

gradient_lambda_path = (
    out_dir
    / "g_lambda_mtilde_q1_"
    "interior_solve_rhs_total.npy"
)

gradient_mu_path = (
    out_dir
    / "g_mu_mtilde_q1_"
    "interior_solve_rhs_total.npy"
)

gradient_coords_path = (
    out_dir
    / "g_mtilde_q1_interior_"
    "solve_rhs_total_coords.npy"
)

np.save(
    gradient_lambda_path,
    gradient_lambda,
)

np.save(
    gradient_mu_path,
    gradient_mu,
)

np.save(
    gradient_coords_path,
    rhs_coords,
)

residual_lambda = (
    active_matrix @ gradient_lambda
    - rhs_lambda
)

residual_mu = (
    active_matrix @ gradient_mu
    - rhs_mu
)

relative_residual_lambda = (
    np.linalg.norm(residual_lambda)
    / max(
        np.linalg.norm(rhs_lambda),
        1e-300,
    )
)

relative_residual_mu = (
    np.linalg.norm(residual_mu)
    / max(
        np.linalg.norm(rhs_mu),
        1e-300,
    )
)

summary = [
    "Mtilde active solve RHS_total summary",
    "======================================",
    "",
    f"matrix_path = {matrix_path}",
    f"input matrix shape = {matrix.shape}",
    f"input matrix nnz = {matrix.nnz}",
    f"matrix_source_mode = {matrix_source_mode}",
    (
        "matrix_permutation_applied = "
        f"{matrix_permutation_applied}"
    ),
    "",
    f"active_count = {expected_n}",
    (
        "active full-grid indices unique = "
        f"{np.unique(rhs_indices).size}"
    ),
    (
        "max mapped coordinate difference = "
        f"{max_coord_diff:.16e}"
    ),
    "",
    f"active matrix shape = {active_matrix.shape}",
    f"active matrix nnz = {active_matrix.nnz}",
    "",
    (
        "relative residual lambda = "
        f"{relative_residual_lambda:.16e}"
    ),
    (
        "relative residual mu = "
        f"{relative_residual_mu:.16e}"
    ),
    "",
    "Outputs:",
    f"  {matrix_out}",
    f"  {indices_out}",
    f"  {coords_out}",
    f"  {gradient_lambda_path}",
    f"  {gradient_mu_path}",
    f"  {gradient_coords_path}",
    "",
]

passed = (
    np.all(np.isfinite(gradient_lambda))
    and np.all(np.isfinite(gradient_mu))
    and relative_residual_lambda < 1e-8
    and relative_residual_mu < 1e-8
)

if passed:
    summary.extend(
        [
            "RESULT = PASS",
            (
                "Meaning: active Q1 Mtilde solve "
                "completed."
            ),
        ]
    )
else:
    summary.extend(
        [
            "RESULT = CHECK",
            (
                "Meaning: solve finished, but "
                "inspect residual and finite stats."
            ),
        ]
    )

summary_text = "\n".join(summary) + "\n"

summary_path = (
    out_dir
    / "mtilde_q1_interior_solve_"
    "rhs_total_summary.txt"
)

summary_path.write_text(
    summary_text,
    encoding="utf-8",
)

print()
print(summary_text)

if not passed:
    raise SystemExit(2)
