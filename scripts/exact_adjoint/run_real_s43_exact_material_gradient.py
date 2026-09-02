import argparse
import json
import re
import resource
import time
from pathlib import Path

import h5py
import numpy as np
import scipy.sparse as sp

from scripts.exact_adjoint.real_s43_global_operator import (
    adjoint_step,
    force_interface_adjoint,
    load_global_data,
    pml_corrector_adjoint,
    pml_main_adjoint,
    pml_pred_adjoint,
    solid_corrector_adjoint,
    solid_internal_force,
    state_norm,
    velocity_interface_adjoint,
)


def rss_gib():
    return (
        resource.getrusage(
            resource.RUSAGE_SELF
        ).ru_maxrss
        /
        1024.0
        /
        1024.0
    )


def zero_state(data):
    solid = data["solid"]
    pml = data["pml"]

    return (
        np.zeros(
            (solid["nn"], 3),
            dtype=np.float64,
        ),
        np.zeros(
            (solid["nn"], 3),
            dtype=np.float64,
        ),
        np.zeros(
            (pml["nn"], 3, 3),
            dtype=np.float64,
        ),
        np.zeros(
            (pml["ne"], 27, 3, 6),
            dtype=np.float64,
        ),
    )


def load_displ(path):
    with h5py.File(
        path,
        "r",
    ) as h5:

        if "displ" in h5:
            arr = h5["displ"][...]

        else:
            found = []

            def visit(name, obj):
                if (
                    isinstance(
                        obj,
                        h5py.Dataset,
                    )
                    and
                    name.split("/")[-1]
                    ==
                    "displ"
                ):
                    found.append(
                        name
                    )

            h5.visititems(
                visit
            )

            if len(found) != 1:
                raise RuntimeError(
                    f"cannot uniquely find displ "
                    f"in {path}: {found}"
                )

            arr = h5[
                found[0]
            ][...]

    return np.asarray(
        arr,
        dtype=np.float64,
    )


def find_snapshots(root):
    rows = []

    for path in root.rglob(
        "sem_field.0000.h5"
    ):
        match = re.fullmatch(
            r"Rsem(\d+)",
            path.parent.name,
        )

        if match is None:
            continue

        rows.append(
            (
                int(
                    match.group(1)
                ),
                path,
            )
        )

    rows.sort(
        key=lambda x:
            x[0]
    )

    ids = [
        x[0]
        for x in rows
    ]

    paths = [
        x[1]
        for x in rows
    ]

    return ids, paths


def first_dataset_shape(path):
    with h5py.File(
        path,
        "r",
    ) as h5:

        if "samples" in h5:
            return tuple(
                int(x)
                for x in h5[
                    "samples"
                ].shape
            )

        found = []

        def visit(name, obj):
            if isinstance(
                obj,
                h5py.Dataset,
            ):
                found.append(
                    tuple(
                        int(x)
                        for x in obj.shape
                    )
                )

        h5.visititems(
            visit
        )

        if len(found) != 1:
            raise RuntimeError(
                f"cannot identify material dataset "
                f"in {path}: {found}"
            )

        return found[0]


def find_material_operator(
    directory,
    target_shape,
):
    candidates = []

    for path in Path(
        directory
    ).rglob(
        "*.npz"
    ):
        try:
            matrix = sp.load_npz(
                path
            )
        except Exception:
            continue

        if matrix.shape == target_shape:
            candidates.append(
                (
                    path,
                    matrix.tocsr(),
                )
            )

    if len(candidates) != 1:
        raise RuntimeError(
            "expected exactly one sparse material "
            f"operator with shape {target_shape}; "
            f"found "
            f"{[(str(p), m.shape) for p, m in candidates]}"
        )

    return candidates[0]


def derivatives(
    data,
    field,
    ids,
):
    hp = data["hp"]
    scale = data["scale"]
    ngll = data["ngll"]

    f = field[
        ids
    ].reshape(
        -1,
        ngll,
        ngll,
        ngll,
        3,
    )

    dx = (
        np.einsum(
            "eljkc,li->eijkc",
            f,
            hp,
            optimize=True,
        )
        *
        scale
    )

    dy = (
        np.einsum(
            "eilkc,lj->eijkc",
            f,
            hp,
            optimize=True,
        )
        *
        scale
    )

    dz = (
        np.einsum(
            "eijlc,lk->eijkc",
            f,
            hp,
            optimize=True,
        )
        *
        scale
    )

    return dx, dy, dz


def material_gradient_local(
    data,
    U,
    force_seed,
):
    conn = data["conn"]

    ne = data["ne"]
    ngll = data["ngll"]
    batch = data["batch_size"]

    weights = data["weights"]
    jac = data["jac"]

    w3 = (
        np.einsum(
            "i,j,k->ijk",
            weights,
            weights,
            weights,
            optimize=True,
        )
        *
        jac
    )

    g_lam = np.zeros(
        (
            ne,
            ngll ** 3,
        ),
        dtype=np.float64,
    )

    g_mu = np.zeros(
        (
            ne,
            ngll ** 3,
        ),
        dtype=np.float64,
    )

    for e0 in range(
        0,
        ne,
        batch,
    ):
        e1 = min(
            e0
            +
            batch,
            ne,
        )

        ids = conn[
            e0:e1
        ]

        ux, uy, uz = derivatives(
            data,
            U,
            ids,
        )

        vx, vy, vz = derivatives(
            data,
            force_seed,
            ids,
        )

        div_u = (
            ux[..., 0]
            +
            uy[..., 1]
            +
            uz[..., 2]
        )

        div_v = (
            vx[..., 0]
            +
            vy[..., 1]
            +
            vz[..., 2]
        )

        shear_xy_u = (
            uy[..., 0]
            +
            ux[..., 1]
        )

        shear_xy_v = (
            vy[..., 0]
            +
            vx[..., 1]
        )

        shear_xz_u = (
            uz[..., 0]
            +
            ux[..., 2]
        )

        shear_xz_v = (
            vz[..., 0]
            +
            vx[..., 2]
        )

        shear_yz_u = (
            uz[..., 1]
            +
            uy[..., 2]
        )

        shear_yz_v = (
            vz[..., 1]
            +
            vy[..., 2]
        )

        local_lam = (
            -w3[
                None,
                :,
                :,
                :,
            ]
            *
            div_u
            *
            div_v
        )

        local_mu = (
            -w3[
                None,
                :,
                :,
                :,
            ]
            *
            (
                2.0
                *
                ux[..., 0]
                *
                vx[..., 0]

                +
                2.0
                *
                uy[..., 1]
                *
                vy[..., 1]

                +
                2.0
                *
                uz[..., 2]
                *
                vz[..., 2]

                +
                shear_xy_u
                *
                shear_xy_v

                +
                shear_xz_u
                *
                shear_xz_v

                +
                shear_yz_u
                *
                shear_yz_v
            )
        )

        g_lam[
            e0:e1
        ] = local_lam.reshape(
            e1 - e0,
            -1,
        )

        g_mu[
            e0:e1
        ] = local_mu.reshape(
            e1 - e0,
            -1,
        )

    return g_lam, g_mu


def adjoint_step_with_force_seed(
    data,
    bar_state_out,
):
    solid = data["solid"]
    pml = data["pml"]
    interface = data["interface"]

    (
        barUs_new,
        barVs_new,
        barVp_new,
        barSp_output,
    ) = bar_state_out

    (
        barUs,
        barVs_after,
        barFs_total,
    ) = solid_corrector_adjoint(
        solid,
        barUs_new,
        barVs_new,
    )

    (
        barVp_pred_direct,
        barFp_after,
    ) = pml_corrector_adjoint(
        pml,
        barVp_new,
    )

    (
        barFs_internal,
        barFp_internal,
    ) = force_interface_adjoint(
        interface,
        barFs_total,
        barFp_after,
    )

    barSp_from_force = (
        pml_main_adjoint(
            pml,
            barFp_internal,
        )
    )

    barSp_new = (
        np.asarray(
            barSp_output,
            dtype=np.float64,
        )
        +
        barSp_from_force
    )

    (
        barVp_pred_internal,
        barSp_old,
    ) = pml_pred_adjoint(
        pml,
        barSp_new,
    )

    barVp_pred = (
        barVp_pred_direct
        +
        barVp_pred_internal
    )

    barUs += solid_internal_force(
        solid,
        barFs_internal,
    )

    (
        barVs_old,
        barVp_old,
    ) = velocity_interface_adjoint(
        interface,
        barVs_after,
        barVp_pred,
    )

    return (
        (
            barUs,
            barVs_old,
            barVp_old,
            barSp_old,
        ),
        barFs_internal,
    )


def relative_error(
    a,
    b,
):
    den = max(
        abs(a),
        abs(b),
        np.finfo(
            np.float64
        ).tiny,
    )

    return abs(
        a - b
    ) / den


def main():
    p = argparse.ArgumentParser()

    p.add_argument(
        "--config",
        required=True,
    )

    p.add_argument(
        "--topology",
        required=True,
    )

    p.add_argument(
        "--coefficients",
        required=True,
    )

    p.add_argument(
        "--coupled-mass",
        required=True,
    )

    p.add_argument(
        "--gll",
        required=True,
    )

    p.add_argument(
        "--weights",
        required=True,
    )

    p.add_argument(
        "--kappa",
        required=True,
    )

    p.add_argument(
        "--mu",
        required=True,
    )

    p.add_argument(
        "--operator-dir",
        required=True,
    )

    p.add_argument(
        "--forward",
        required=True,
    )

    p.add_argument(
        "--receiver-nodes",
        required=True,
    )

    p.add_argument(
        "--native-gradient",
        required=True,
    )

    p.add_argument(
        "--native-time",
        required=True,
    )

    p.add_argument(
        "--solid-to-snapshot",
        required=True,
    )

    p.add_argument(
        "--output",
        required=True,
    )

    p.add_argument(
        "--batch-size",
        type=int,
        default=2048,
    )

    args = p.parse_args()

    output = Path(
        args.output
    )

    output.mkdir(
        parents=True,
        exist_ok=True,
    )

    cfg = json.loads(
        Path(
            args.config
        ).read_text(
            encoding="utf-8"
        )
    )

    data = load_global_data(
        args.config,
        args.topology,
        args.coefficients,
        args.coupled_mass,
        args.gll,
        args.weights,
        args.kappa,
        args.mu,
        batch_size=args.batch_size,
    )

    solid = data["solid"]

    receiver_nodes = np.asarray(
        np.load(
            args.receiver_nodes
        ),
        dtype=np.int64,
    )

    native = np.asarray(
        np.load(
            args.native_gradient
        ),
        dtype=np.float64,
    )

    native_time = np.asarray(
        np.load(
            args.native_time
        ),
        dtype=np.float64,
    )

    solid_to_snapshot = np.asarray(
        np.load(
            args.solid_to_snapshot
        ),
        dtype=np.int64,
    )

    print(
        "============================================================"
    )

    print(
        "STAGE 4C PREFLIGHT"
    )

    print(
        "============================================================"
    )

    print(
        "native gradient shape =",
        native.shape,
    )

    print(
        "native time shape =",
        native_time.shape,
    )

    print(
        "native time first/last =",
        [
            float(
                native_time[0]
            ),
            float(
                native_time[-1]
            ),
        ],
    )

    print(
        "dt =",
        f'{data["dt"]:.17e}',
    )

    if native.ndim != 3:
        raise RuntimeError(
            "native objective gradient must be 3D"
        )

    if native.shape[1:] != (
        225,
        3,
    ):
        raise RuntimeError(
            f"unexpected native shape {native.shape}"
        )

    if len(
        native_time
    ) != native.shape[0]:
        raise RuntimeError(
            "native time/count mismatch"
        )

    n_states = len(
        native_time
    )

    n_transitions = (
        n_states
        -
        1
    )

    derived_transitions = int(
        round(
            (
                native_time[-1]
                -
                native_time[0]
            )
            /
            data["dt"]
        )
    )

    print(
        "state/time samples =",
        n_states,
    )

    print(
        "correct transition count =",
        n_transitions,
    )

    print(
        "transition count from T/dt =",
        derived_transitions,
    )

    time_reconstruction = (
        native_time[0]
        +
        np.arange(
            n_states,
            dtype=np.float64,
        )
        *
        data["dt"]
    )

    time_error = float(
        np.max(
            np.abs(
                native_time
                -
                time_reconstruction
            )
        )
    )

    print(
        "native time-grid max error =",
        f"{time_error:.17e}",
    )

    if (
        n_transitions
        !=
        derived_transitions
    ):
        raise RuntimeError(
            "state/transition time contract mismatch"
        )

    snapshot_ids, snapshots = (
        find_snapshots(
            Path(
                args.forward
            )
            /
            "res"
        )
    )

    print(
        "snapshot count =",
        len(
            snapshots
        ),
    )

    print(
        "snapshot ID first/last =",
        [
            snapshot_ids[0],
            snapshot_ids[-1],
        ],
    )

    if len(
        snapshots
    ) != n_states:
        raise RuntimeError(
            f"snapshot count {len(snapshots)} "
            f"!= state count {n_states}"
        )

    if len(
        solid_to_snapshot
    ) != solid["nn"]:
        raise RuntimeError(
            "solid snapshot map size mismatch"
        )

    first_full = load_displ(
        snapshots[0]
    )

    second_full = load_displ(
        snapshots[1]
    )

    last_full = load_displ(
        snapshots[-1]
    )

    U0 = first_full[
        solid_to_snapshot
    ]

    U1 = second_full[
        solid_to_snapshot
    ]

    Ulast = last_full[
        solid_to_snapshot
    ]

    first_norm = float(
        np.linalg.norm(
            U0
        )
    )

    second_norm = float(
        np.linalg.norm(
            U1
        )
    )

    last_norm = float(
        np.linalg.norm(
            Ulast
        )
    )

    print(
        "snapshot U[0] norm =",
        f"{first_norm:.17e}",
    )

    print(
        "snapshot U[1] norm =",
        f"{second_norm:.17e}",
    )

    print(
        "snapshot U[last] norm =",
        f"{last_norm:.17e}",
    )

    # ----------------------------------------------------------
    # Discover validated material interpolation P
    # ----------------------------------------------------------

    material_shape = first_dataset_shape(
        args.kappa
    )

    n_controls = int(
        np.prod(
            material_shape
        )
    )

    target_shape = (
        solid["ne"]
        *
        (
            solid["ngll"]
            **
            3
        ),
        n_controls,
    )

    P_path, P = find_material_operator(
        args.operator_dir,
        target_shape,
    )

    print(
        "material control shape =",
        material_shape,
    )

    print(
        "material control count =",
        n_controls,
    )

    print(
        "P file =",
        P_path,
    )

    print(
        "P shape =",
        P.shape,
    )

    print(
        "P nnz =",
        P.nnz,
    )

    # ----------------------------------------------------------
    # Verify P row ordering against THIS solid element/GLL order
    # ----------------------------------------------------------

    domain = cfg["domain"]

    nz, ny, nx = material_shape

    x_axis = np.linspace(
        float(
            domain["x_min_m"]
        ),
        float(
            domain["x_max_m"]
        ),
        nx,
    )

    y_axis = np.linspace(
        float(
            domain["y_min_m"]
        ),
        float(
            domain["y_max_m"]
        ),
        ny,
    )

    z_axis = np.linspace(
        float(
            domain["z_min_m"]
        ),
        float(
            domain["z_max_m"]
        ),
        nz,
    )

    zz, yy, xx = np.meshgrid(
        z_axis,
        y_axis,
        x_axis,
        indexing="ij",
    )

    control_x = xx.reshape(-1)
    control_y = yy.reshape(-1)
    control_z = zz.reshape(-1)

    row_xyz = solid["xyz"][
        solid["conn"]
    ].reshape(
        -1,
        3,
    )

    px = np.asarray(
        P
        @
        control_x
    ).reshape(-1)

    py = np.asarray(
        P
        @
        control_y
    ).reshape(-1)

    pz = np.asarray(
        P
        @
        control_z
    ).reshape(-1)

    p_xyz_error = float(
        np.max(
            np.abs(
                np.column_stack(
                    (
                        px,
                        py,
                        pz,
                    )
                )
                -
                row_xyz
            )
        )
    )

    row_sum_error = float(
        np.max(
            np.abs(
                np.asarray(
                    P.sum(
                        axis=1
                    )
                ).reshape(-1)
                -
                1.0
            )
        )
    )

    print(
        "P row-sum max error =",
        f"{row_sum_error:.17e}",
    )

    print(
        "P row-order XYZ max error =",
        f"{p_xyz_error:.17e}",
    )

    # ----------------------------------------------------------
    # Material derivative directional test
    # ----------------------------------------------------------

    mid = (
        n_states
        //
        2
    )

    Umid_full = load_displ(
        snapshots[
            mid
        ]
    )

    Umid = Umid_full[
        solid_to_snapshot
    ]

    rng = np.random.default_rng(
        20260828
    )

    seed = rng.standard_normal(
        (
            solid["nn"],
            3,
        )
    )

    seed /= np.linalg.norm(
        seed
    )

    g_test_lam, g_test_mu = (
        material_gradient_local(
            solid,
            Umid,
            seed,
        )
    )

    dm_lam = rng.standard_normal(
        g_test_lam.shape
    )

    dm_mu = rng.standard_normal(
        g_test_mu.shape
    )

    norm_dm = np.sqrt(
        np.vdot(
            dm_lam,
            dm_lam,
        )
        +
        np.vdot(
            dm_mu,
            dm_mu,
        )
    )

    dm_lam /= norm_dm
    dm_mu /= norm_dm

    direction_data = dict(
        solid
    )

    direction_data["lam"] = (
        dm_lam
    )

    direction_data["mu"] = (
        dm_mu
    )

    F_direction = solid_internal_force(
        direction_data,
        Umid,
    )

    directional_direct = float(
        np.vdot(
            seed,
            F_direction,
        )
    )

    directional_gradient = float(
        np.vdot(
            g_test_lam,
            dm_lam,
        )
        +
        np.vdot(
            g_test_mu,
            dm_mu,
        )
    )

    material_dot_error = relative_error(
        directional_direct,
        directional_gradient,
    )

    print(
        "material directional direct =",
        f"{directional_direct:.17e}",
    )

    print(
        "material directional gradient =",
        f"{directional_gradient:.17e}",
    )

    print(
        "material derivative dot error =",
        f"{material_dot_error:.17e}",
    )

    preflight_checks = {
        "state_transition_count":
            n_transitions
            ==
            derived_transitions,

        "native_time_grid":
            time_error
            <
            1.0e-12,

        "snapshot_count":
            len(
                snapshots
            )
            ==
            n_states,

        "initial_snapshot_zero":
            first_norm
            <
            1.0e-20,

        "P_row_sum":
            row_sum_error
            <
            1.0e-12,

        "P_row_order":
            p_xyz_error
            <
            1.0e-12,

        "material_derivative":
            material_dot_error
            <
            1.0e-12,
    }

    print(
        "preflight checks =",
        preflight_checks,
    )

    if not all(
        preflight_checks.values()
    ):
        print(
            "RESULT = FAIL_STAGE4C_PREFLIGHT"
        )

        return

    print(
        "RESULT = PASS_STAGE4C_PREFLIGHT"
    )

    # ==========================================================
    # CORRECT REVERSE:
    #
    # states q_0 ... q_N, N=1440
    #
    # bar_N += q_N
    #
    # for n=N-1..0:
    #   seed,bar_n = T_n^T(bar_{n+1})
    #   gradient += dT_n/dm ^T bar_{n+1}
    #   bar_n += q_n
    #
    # exactly 1440 transitions
    # ==========================================================

    print()
    print(
        "============================================================"
    )

    print(
        "START CORRECT 1440-TRANSITION MATERIAL REVERSE"
    )

    print(
        "============================================================"
    )

    bar = zero_state(
        data
    )

    barU = bar[0]

    np.add.at(
        barU,
        receiver_nodes,
        native[-1],
    )

    bar = (
        barU,
        bar[1],
        bar[2],
        bar[3],
    )

    g_lam_local = np.zeros(
        (
            solid["ne"],
            solid["ngll"]
            **
            3,
        ),
        dtype=np.float64,
    )

    g_mu_local = np.zeros_like(
        g_lam_local
    )

    progress_path = (
        output
        /
        "progress.txt"
    )

    progress_path.write_text(
        "",
        encoding="utf-8",
    )

    checkpoints = {
        n_transitions - 1,
        1200,
        1000,
        800,
        600,
        400,
        200,
        100,
        0,
    }

    t0 = time.perf_counter()

    for n in range(
        n_transitions - 1,
        -1,
        -1,
    ):
        full = load_displ(
            snapshots[
                n
            ]
        )

        U_forward = full[
            solid_to_snapshot
        ]

        (
            bar,
            force_seed,
        ) = adjoint_step_with_force_seed(
            data,
            bar,
        )

        gl,
        gm = material_gradient_local(
            solid,
            U_forward,
            force_seed,
        )

        g_lam_local += gl
        g_mu_local += gm

        barU = bar[0]

        np.add.at(
            barU,
            receiver_nodes,
            native[
                n
            ],
        )

        bar = (
            barU,
            bar[1],
            bar[2],
            bar[3],
        )

        if not all(
            np.isfinite(
                x
            ).all()
            for x in bar
        ):
            raise RuntimeError(
                f"non-finite adjoint state "
                f"after transition {n}"
            )

        if (
            not np.isfinite(
                g_lam_local
            ).all()
            or
            not np.isfinite(
                g_mu_local
            ).all()
        ):
            raise RuntimeError(
                f"non-finite material gradient "
                f"after transition {n}"
            )

        if n in checkpoints:
            elapsed = (
                time.perf_counter()
                -
                t0
            )

            completed = (
                n_transitions
                -
                n
            )

            sec_per = (
                elapsed
                /
                completed
            )

            eta = (
                sec_per
                *
                n
            )

            line = (
                f"transition n={n:04d} "
                f"completed={completed}/{n_transitions} "
                f"adj_norm={state_norm(bar):.17e} "
                f"g_lambda_l2={np.linalg.norm(g_lam_local):.17e} "
                f"g_mu_l2={np.linalg.norm(g_mu_local):.17e} "
                f"elapsed_s={elapsed:.3f} "
                f"eta_s={eta:.3f} "
                f"rss_gib={rss_gib():.3f}"
            )

            print(
                line,
                flush=True,
            )

            with progress_path.open(
                "a",
                encoding="utf-8",
            ) as f:
                f.write(
                    line
                    +
                    "\n"
                )

    elapsed = (
        time.perf_counter()
        -
        t0
    )

    # ==========================================================
    # Map exact element-GLL gradients back to material H5 grid
    # ==========================================================

    g_lambda_control = np.asarray(
        P.T
        @
        g_lam_local.reshape(
            -1
        )
    ).reshape(-1)

    g_mu_physical_control = np.asarray(
        P.T
        @
        g_mu_local.reshape(
            -1
        )
    ).reshape(-1)

    # SEM material variables:
    #
    # lambda = Kappa - 2/3 Mu
    #
    # therefore
    #
    # dJ/dKappa = dJ/dlambda
    #
    # dJ/dMu_H5 =
    #     dJ/dmu_physical - 2/3 dJ/dlambda
    #

    g_kappa_control = np.array(
        g_lambda_control,
        copy=True,
    )

    g_mu_h5_control = (
        g_mu_physical_control
        -
        (
            2.0
            /
            3.0
        )
        *
        g_lambda_control
    )

    np.save(
        output
        /
        "gradient_lambda_control.npy",
        g_lambda_control,
    )

    np.save(
        output
        /
        "gradient_mu_physical_control.npy",
        g_mu_physical_control,
    )

    np.save(
        output
        /
        "gradient_kappa_h5_control.npy",
        g_kappa_control,
    )

    np.save(
        output
        /
        "gradient_mu_h5_control.npy",
        g_mu_h5_control,
    )

    np.save(
        output
        /
        "gradient_lambda_control_grid.npy",
        g_lambda_control.reshape(
            material_shape
        ),
    )

    np.save(
        output
        /
        "gradient_mu_physical_control_grid.npy",
        g_mu_physical_control.reshape(
            material_shape
        ),
    )

    final_norm = state_norm(
        bar
    )

    print()
    print(
        "============================================================"
    )

    print(
        "CORRECT MATERIAL REVERSE COMPLETE"
    )

    print(
        "============================================================"
    )

    print(
        "state samples =",
        n_states,
    )

    print(
        "transitions executed =",
        n_transitions,
    )

    print(
        "elapsed seconds =",
        f"{elapsed:.6f}",
    )

    print(
        "seconds per transition =",
        f"{elapsed/n_transitions:.6f}",
    )

    print(
        "final adjoint state norm =",
        f"{final_norm:.17e}",
    )

    print(
        "lambda local gradient L2 =",
        f"{np.linalg.norm(g_lam_local):.17e}",
    )

    print(
        "lambda control gradient L2 =",
        f"{np.linalg.norm(g_lambda_control):.17e}",
    )

    print(
        "lambda control gradient sum =",
        f"{np.sum(g_lambda_control):.17e}",
    )

    print(
        "mu physical control gradient L2 =",
        f"{np.linalg.norm(g_mu_physical_control):.17e}",
    )

    print(
        "mu physical control gradient sum =",
        f"{np.sum(g_mu_physical_control):.17e}",
    )

    print(
        "Kappa-H5 gradient sum =",
        f"{np.sum(g_kappa_control):.17e}",
    )

    print(
        "Mu-H5 gradient sum =",
        f"{np.sum(g_mu_h5_control):.17e}",
    )

    print(
        "peak RSS GiB =",
        f"{rss_gib():.3f}",
    )

    final_checks = {
        "transition_count":
            n_transitions
            ==
            derived_transitions,

        "finite_adjoint":
            all(
                np.isfinite(
                    x
                ).all()
                for x in bar
            ),

        "finite_lambda_gradient":
            bool(
                np.isfinite(
                    g_lambda_control
                ).all()
            ),

        "finite_mu_gradient":
            bool(
                np.isfinite(
                    g_mu_physical_control
                ).all()
            ),

        "nonzero_lambda_gradient":
            bool(
                np.linalg.norm(
                    g_lambda_control
                )
                >
                0.0
            ),

        "nonzero_mu_gradient":
            bool(
                np.linalg.norm(
                    g_mu_physical_control
                )
                >
                0.0
            ),
    }

    print(
        "final checks =",
        final_checks,
    )

    summary = {
        "state_samples":
            int(
                n_states
            ),

        "transitions":
            int(
                n_transitions
            ),

        "dt":
            float(
                data["dt"]
            ),

        "time_grid_max_error":
            time_error,

        "snapshot_count":
            int(
                len(
                    snapshots
                )
            ),

        "snapshot_first_norm":
            first_norm,

        "snapshot_second_norm":
            second_norm,

        "snapshot_last_norm":
            last_norm,

        "P_path":
            str(
                P_path
            ),

        "P_shape":
            list(
                P.shape
            ),

        "P_nnz":
            int(
                P.nnz
            ),

        "P_row_sum_error":
            row_sum_error,

        "P_row_order_xyz_error":
            p_xyz_error,

        "material_derivative_dot_error":
            material_dot_error,

        "elapsed_seconds":
            float(
                elapsed
            ),

        "seconds_per_transition":
            float(
                elapsed
                /
                n_transitions
            ),

        "final_adjoint_norm":
            float(
                final_norm
            ),

        "lambda_control_l2":
            float(
                np.linalg.norm(
                    g_lambda_control
                )
            ),

        "lambda_control_sum":
            float(
                np.sum(
                    g_lambda_control
                )
            ),

        "mu_physical_control_l2":
            float(
                np.linalg.norm(
                    g_mu_physical_control
                )
            ),

        "mu_physical_control_sum":
            float(
                np.sum(
                    g_mu_physical_control
                )
            ),

        "kappa_h5_sum":
            float(
                np.sum(
                    g_kappa_control
                )
            ),

        "mu_h5_sum":
            float(
                np.sum(
                    g_mu_h5_control
                )
            ),

        "preflight_checks":
            preflight_checks,

        "final_checks":
            final_checks,
    }

    (
        output
        /
        "summary.json"
    ).write_text(
        json.dumps(
            summary,
            indent=2,
        ),
        encoding="utf-8",
    )

    if all(
        final_checks.values()
    ):
        print(
            "RESULT = PASS_REAL_S43_EXACT_MATERIAL_GRADIENT"
        )
    else:
        print(
            "RESULT = FAIL_REAL_S43_EXACT_MATERIAL_GRADIENT"
        )


if __name__ == "__main__":
    main()
