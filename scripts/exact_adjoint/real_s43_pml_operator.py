import json
from pathlib import Path

import numpy as np


def derivative_matrix(x):
    x = np.asarray(
        x,
        dtype=np.float64,
    )

    n = len(x)

    bw = np.ones(
        n,
        dtype=np.float64,
    )

    for i in range(n):
        for j in range(n):
            if i != j:
                bw[i] /= (
                    x[i]
                    -
                    x[j]
                )

    D = np.zeros(
        (
            n,
            n,
        ),
        dtype=np.float64,
    )

    for i in range(n):
        for j in range(n):
            if i != j:
                D[i, j] = (
                    bw[j]
                    /
                    (
                        bw[i]
                        *
                        (
                            x[i]
                            -
                            x[j]
                        )
                    )
                )

        D[i, i] = (
            -np.sum(
                D[i]
            )
        )

    return D


def load_operator_data(
    topology_dir,
    coefficient_dir,
    gll_path,
    weights_path,
    batch_size=2048,
):
    topology_dir = Path(
        topology_dir
    )

    coefficient_dir = Path(
        coefficient_dir
    )

    gll = np.asarray(
        np.load(
            gll_path
        ),
        dtype=np.float64,
    )

    weights = np.asarray(
        np.load(
            weights_path
        ),
        dtype=np.float64,
    )

    if len(gll) != 3:
        raise RuntimeError(
            f"expected ngll=3, got {len(gll)}"
        )

    hp = derivative_matrix(
        gll
    ).T

    summary = json.loads(
        (
            coefficient_dir
            /
            "summary.json"
        ).read_text(
            encoding="utf-8"
        )
    )

    conn = np.asarray(
        np.load(
            topology_dir
            /
            "pml_connectivity_compact.npy"
        ),
        dtype=np.int64,
    )

    dirichlet = np.asarray(
        np.load(
            topology_dir
            /
            "pml_dirichlet_compact.npy"
        ),
        dtype=np.int64,
    )

    lam = np.asarray(
        np.load(
            coefficient_dir
            /
            "pml_lambda_gll.npy"
        ),
        dtype=np.float64,
    )

    mu = np.asarray(
        np.load(
            coefficient_dir
            /
            "pml_mu_gll.npy"
        ),
        dtype=np.float64,
    )

    rho = np.asarray(
        np.load(
            coefficient_dir
            /
            "pml_density_gll.npy"
        ),
        dtype=np.float64,
    )

    alpha = np.asarray(
        np.load(
            coefficient_dir
            /
            "pml_alpha_xyz.npy"
        ),
        dtype=np.float64,
    )

    local_mass = np.asarray(
        np.load(
            coefficient_dir
            /
            "pml_local_mass.npy"
        ),
        dtype=np.float64,
    )

    dS0 = np.asarray(
        np.load(
            coefficient_dir
            /
            "pml_dumpS0_xyz.npy"
        ),
        dtype=np.float64,
    )

    dS1 = np.asarray(
        np.load(
            coefficient_dir
            /
            "pml_dumpS1_xyz.npy"
        ),
        dtype=np.float64,
    )

    dV0 = np.asarray(
        np.load(
            coefficient_dir
            /
            "pml_dumpV0.npy"
        ),
        dtype=np.float64,
    )

    dV1 = np.asarray(
        np.load(
            coefficient_dir
            /
            "pml_dumpV1.npy"
        ),
        dtype=np.float64,
    )

    ne = conn.shape[0]
    nn = dV0.shape[0]

    if conn.shape != (
        ne,
        27,
    ):
        raise RuntimeError(
            f"unexpected connectivity {conn.shape}"
        )

    if lam.shape != (
        ne,
        27,
    ):
        raise RuntimeError(
            f"unexpected lambda shape {lam.shape}"
        )

    if mu.shape != (
        ne,
        27,
    ):
        raise RuntimeError(
            f"unexpected mu shape {mu.shape}"
        )

    if rho.shape != (
        ne,
        27,
    ):
        raise RuntimeError(
            f"unexpected density shape {rho.shape}"
        )

    if alpha.shape != (
        ne,
        27,
        3,
    ):
        raise RuntimeError(
            f"unexpected alpha shape {alpha.shape}"
        )

    if local_mass.shape != (
        ne,
        27,
    ):
        raise RuntimeError(
            f"unexpected local-mass shape {local_mass.shape}"
        )

    if dS0.shape != (
        ne,
        27,
        3,
    ):
        raise RuntimeError(
            f"unexpected DumpS0 shape {dS0.shape}"
        )

    if dS1.shape != (
        ne,
        27,
        3,
    ):
        raise RuntimeError(
            f"unexpected DumpS1 shape {dS1.shape}"
        )

    if dV0.shape != (
        nn,
        3,
    ):
        raise RuntimeError(
            f"unexpected DumpV0 shape {dV0.shape}"
        )

    if dV1.shape != (
        nn,
        3,
    ):
        raise RuntimeError(
            f"unexpected DumpV1 shape {dV1.shape}"
        )

    h = float(
        summary[
            "element_spacing_m"
        ]
    )

    return {
        "conn": conn,
        "dirichlet": dirichlet,
        "lam": lam,
        "mu": mu,
        "rho": rho,
        "alpha": alpha,
        "local_mass": local_mass,
        "dS0": dS0,
        "dS1": dS1,
        "dV0": dV0,
        "dV1": dV1,
        "gll": gll,
        "weights": weights,
        "hp": hp,
        "dt": float(
            summary[
                "dt"
            ]
        ),
        "jac": float(
            summary[
                "jacobian"
            ]
        ),
        "invgrad_scale": (
            2.0
            /
            h
        ),
        "ne": ne,
        "nn": nn,
        "batch_size": int(
            batch_size
        ),
    }


def zero_state(data):
    V = np.zeros(
        (
            data["nn"],
            3,
            3,
        ),
        dtype=np.float64,
    )

    S = np.zeros(
        (
            data["ne"],
            27,
            3,
            6,
        ),
        dtype=np.float64,
    )

    return V, S


def state_dot(a, b):
    Va, Sa = a
    Vb, Sb = b

    return float(
        np.vdot(
            Va,
            Vb,
        )
        +
        np.vdot(
            Sa,
            Sb,
        )
    )


def state_norm(a):
    return np.sqrt(
        max(
            state_dot(
                a,
                a,
            ),
            0.0,
        )
    )


def normalize_state(a):
    V, S = a

    n = state_norm(
        a
    )

    if not np.isfinite(n):
        raise RuntimeError(
            "non-finite state norm"
        )

    if n == 0.0:
        raise RuntimeError(
            "zero state norm"
        )

    V /= n
    S /= n

    return V, S


def random_state(
    data,
    rng,
):
    V = rng.standard_normal(
        (
            data["nn"],
            3,
            3,
        )
    )

    S = rng.standard_normal(
        (
            data["ne"],
            27,
            3,
            6,
        )
    )

    return normalize_state(
        (
            V,
            S,
        )
    )


def _derivatives(
    vloc,
    hp,
    scale,
):
    dx = np.einsum(
        "eljkc,li->eijkc",
        vloc,
        hp,
        optimize=True,
    )

    dy = np.einsum(
        "eilkc,lj->eijkc",
        vloc,
        hp,
        optimize=True,
    )

    dz = np.einsum(
        "eijlc,lk->eijkc",
        vloc,
        hp,
        optimize=True,
    )

    dx *= scale
    dy *= scale
    dz *= scale

    return dx, dy, dz


def _material_direction(
    data,
    dlam,
    dmu,
):
    shape = (
        data["ne"],
        27,
    )

    dlam = np.asarray(
        dlam,
        dtype=np.float64,
    )

    dmu = np.asarray(
        dmu,
        dtype=np.float64,
    )

    if dlam.shape != shape:
        raise RuntimeError(
            f"unexpected delta-lambda shape {dlam.shape}"
        )

    if dmu.shape != shape:
        raise RuntimeError(
            f"unexpected delta-mu shape {dmu.shape}"
        )

    return dlam, dmu


def _material_strain_basis(
    dx,
    dy,
    dz,
):
    shape = (
        *dx.shape[:-1],
        3,
        6,
    )

    lam_basis = np.zeros(
        shape,
        dtype=np.float64,
    )

    mu_basis = np.zeros_like(
        lam_basis
    )

    # Split x: xx, yy, zz receive lambda*dVx/dx;
    # xx additionally receives 2*mu*dVx/dx.
    lam_basis[..., 0, 0] = dx[..., 0]
    lam_basis[..., 0, 1] = dx[..., 0]
    lam_basis[..., 0, 2] = dx[..., 0]
    mu_basis[..., 0, 0] = 2.0 * dx[..., 0]
    mu_basis[..., 0, 3] = dx[..., 1]
    mu_basis[..., 0, 4] = dx[..., 2]

    # Split y.
    lam_basis[..., 1, 0] = dy[..., 1]
    lam_basis[..., 1, 1] = dy[..., 1]
    lam_basis[..., 1, 2] = dy[..., 1]
    mu_basis[..., 1, 1] = 2.0 * dy[..., 1]
    mu_basis[..., 1, 3] = dy[..., 0]
    mu_basis[..., 1, 5] = dy[..., 2]

    # Split z.
    lam_basis[..., 2, 0] = dz[..., 2]
    lam_basis[..., 2, 1] = dz[..., 2]
    lam_basis[..., 2, 2] = dz[..., 2]
    mu_basis[..., 2, 2] = 2.0 * dz[..., 2]
    mu_basis[..., 2, 4] = dz[..., 0]
    mu_basis[..., 2, 5] = dz[..., 1]

    return lam_basis, mu_basis


def _velocity_mass_from_dump_coefficients(data):
    dV0 = data["dV0"]
    dV1 = data["dV1"]

    if np.any(dV1 <= 0.0):
        raise RuntimeError(
            "non-positive PML DumpV1 coefficient"
        )

    # DumpV1 = 1/(M + D), DumpV0 = (M - D)/(M + D).
    # Recover M from the active coefficients so this remains correct
    # after the global operator installs the coupled solid/PML mass.
    return (
        0.5
        *
        (
            1.0
            +
            dV0
        )
        /
        dV1
    )


def material_coefficient_tangent(
    data,
    dlam,
    dmu,
):
    """Differentiate all material-dependent SolidPML coefficients.

    Density and geometry are fixed.  The active chain is
    (lambda, mu) -> Vp -> alpha -> DumpS/DumpMass -> DumpV.
    """
    dlam, dmu = _material_direction(
        data,
        dlam,
        dmu,
    )

    l2m = (
        data["lam"]
        +
        2.0
        *
        data["mu"]
    )

    if np.any(l2m <= 0.0):
        raise RuntimeError(
            "non-positive lambda+2mu in SolidPML"
        )

    relative_vp = (
        (
            dlam
            +
            2.0
            *
            dmu
        )
        /
        (
            2.0
            *
            l2m
        )
    )

    dalpha = (
        data["alpha"]
        *
        relative_vp[..., None]
    )

    dS1 = (
        -0.5
        *
        data["dt"]
        *
        data["dS1"] ** 2
        *
        dalpha
    )

    dS0 = (
        -data["dt"]
        *
        data["dS1"] ** 2
        *
        dalpha
    )

    d_dumpmass = np.zeros_like(
        data["dV0"]
    )

    local_scale = (
        0.5
        *
        data["local_mass"]
        *
        data["dt"]
    )

    ids = data["conn"].reshape(-1)

    for direction in range(3):
        np.add.at(
            d_dumpmass[:, direction],
            ids,
            (
                local_scale
                *
                dalpha[..., direction]
            ).reshape(-1),
        )

    velocity_mass = (
        _velocity_mass_from_dump_coefficients(
            data
        )
    )

    inverse_denominator = data["dV1"]

    dV1 = (
        -d_dumpmass
        *
        inverse_denominator ** 2
    )

    dV0 = (
        -2.0
        *
        velocity_mass
        *
        d_dumpmass
        *
        inverse_denominator ** 2
    )

    return {
        "alpha": dalpha,
        "dS0": dS0,
        "dS1": dS1,
        "dumpmass": d_dumpmass,
        "dV0": dV0,
        "dV1": dV1,
    }


def _alpha_material_vjp(
    data,
    bar_alpha,
):
    bar_alpha = np.asarray(
        bar_alpha,
        dtype=np.float64,
    )

    if bar_alpha.shape != data["alpha"].shape:
        raise RuntimeError(
            f"unexpected alpha-adjoint shape {bar_alpha.shape}"
        )

    l2m = (
        data["lam"]
        +
        2.0
        *
        data["mu"]
    )

    common = np.sum(
        bar_alpha
        *
        data["alpha"],
        axis=2,
    )

    g_lam = (
        common
        /
        (
            2.0
            *
            l2m
        )
    )

    g_mu = (
        common
        /
        l2m
    )

    return g_lam, g_mu


def pred_material_tangent(
    data,
    V,
    Sold,
    dlam,
    dmu,
    coefficient_tangent=None,
):
    dlam, dmu = _material_direction(
        data,
        dlam,
        dmu,
    )

    if coefficient_tangent is None:
        coefficient_tangent = (
            material_coefficient_tangent(
                data,
                dlam,
                dmu,
            )
        )

    Vtot = np.sum(
        V,
        axis=2,
    )

    out = np.empty_like(
        Sold
    )

    for e0 in range(
        0,
        data["ne"],
        data["batch_size"],
    ):
        e1 = min(
            e0
            +
            data["batch_size"],
            data["ne"],
        )

        ids = data["conn"][e0:e1]

        vloc = Vtot[ids].reshape(
            -1,
            3,
            3,
            3,
            3,
        )

        dx, dy, dz = _derivatives(
            vloc,
            data["hp"],
            data["invgrad_scale"],
        )

        lam_basis, mu_basis = (
            _material_strain_basis(
                dx,
                dy,
                dz,
            )
        )

        shape4 = (
            -1,
            3,
            3,
            3,
        )

        la = data["lam"][e0:e1].reshape(
            shape4
        )
        muv = data["mu"][e0:e1].reshape(
            shape4
        )
        dla = dlam[e0:e1].reshape(
            shape4
        )
        dmu_v = dmu[e0:e1].reshape(
            shape4
        )

        constitutive = (
            la[..., None, None]
            *
            lam_basis
            +
            muv[..., None, None]
            *
            mu_basis
        )

        dconstitutive = (
            dla[..., None, None]
            *
            lam_basis
            +
            dmu_v[..., None, None]
            *
            mu_basis
        )

        old = Sold[e0:e1].reshape(
            -1,
            3,
            3,
            3,
            3,
            6,
        )

        dS0 = coefficient_tangent["dS0"][e0:e1].reshape(
            -1,
            3,
            3,
            3,
            3,
        )

        dS1 = coefficient_tangent["dS1"][e0:e1].reshape(
            -1,
            3,
            3,
            3,
            3,
        )

        S1 = data["dS1"][e0:e1].reshape(
            -1,
            3,
            3,
            3,
            3,
        )

        local = (
            dS0[..., None]
            *
            old
            +
            data["dt"]
            *
            (
                dS1[..., None]
                *
                constitutive
                +
                S1[..., None]
                *
                dconstitutive
            )
        )

        out[e0:e1] = local.reshape(
            e1 - e0,
            27,
            3,
            6,
        )

    return out


def pred_material_vjp(
    data,
    V,
    Sold,
    barSnew,
):
    g_lam = np.zeros_like(
        data["lam"]
    )
    g_mu = np.zeros_like(
        data["mu"]
    )
    bar_alpha = np.zeros_like(
        data["alpha"]
    )

    Vtot = np.sum(
        V,
        axis=2,
    )

    for e0 in range(
        0,
        data["ne"],
        data["batch_size"],
    ):
        e1 = min(
            e0
            +
            data["batch_size"],
            data["ne"],
        )

        ids = data["conn"][e0:e1]

        vloc = Vtot[ids].reshape(
            -1,
            3,
            3,
            3,
            3,
        )

        dx, dy, dz = _derivatives(
            vloc,
            data["hp"],
            data["invgrad_scale"],
        )

        lam_basis, mu_basis = (
            _material_strain_basis(
                dx,
                dy,
                dz,
            )
        )

        shape4 = (
            -1,
            3,
            3,
            3,
        )

        la = data["lam"][e0:e1].reshape(
            shape4
        )
        muv = data["mu"][e0:e1].reshape(
            shape4
        )

        constitutive = (
            la[..., None, None]
            *
            lam_basis
            +
            muv[..., None, None]
            *
            mu_basis
        )

        bs = np.asarray(
            barSnew[e0:e1],
            dtype=np.float64,
        ).reshape(
            -1,
            3,
            3,
            3,
            3,
            6,
        )

        old = Sold[e0:e1].reshape(
            -1,
            3,
            3,
            3,
            3,
            6,
        )

        S1 = data["dS1"][e0:e1].reshape(
            -1,
            3,
            3,
            3,
            3,
        )

        common = (
            data["dt"]
            *
            S1[..., None]
            *
            bs
        )

        g_lam[e0:e1] = np.sum(
            common
            *
            lam_basis,
            axis=(4, 5),
        ).reshape(
            e1 - e0,
            27,
        )

        g_mu[e0:e1] = np.sum(
            common
            *
            mu_basis,
            axis=(4, 5),
        ).reshape(
            e1 - e0,
            27,
        )

        bar_dS0 = np.sum(
            bs
            *
            old,
            axis=5,
        )

        bar_dS1 = (
            data["dt"]
            *
            np.sum(
                bs
                *
                constitutive,
                axis=5,
            )
        )

        local_S1 = S1

        local_bar_alpha = (
            -data["dt"]
            *
            local_S1 ** 2
            *
            bar_dS0
            -
            0.5
            *
            data["dt"]
            *
            local_S1 ** 2
            *
            bar_dS1
        )

        bar_alpha[e0:e1] = local_bar_alpha.reshape(
            e1 - e0,
            27,
            3,
        )

    alpha_lam, alpha_mu = (
        _alpha_material_vjp(
            data,
            bar_alpha,
        )
    )

    g_lam += alpha_lam
    g_mu += alpha_mu

    return g_lam, g_mu


def corrector_material_tangent(
    data,
    V,
    F,
    dF,
    coefficient_tangent,
):
    out = (
        coefficient_tangent["dV0"][:, None, :]
        *
        V
        +
        data["dt"]
        *
        (
            coefficient_tangent["dV1"][:, None, :]
            *
            F
            +
            data["dV1"][:, None, :]
            *
            dF
        )
    )

    out[
        data["dirichlet"],
        :,
        :,
    ] = 0.0

    return out


def corrector_material_vjp(
    data,
    V,
    F,
    barVnew,
):
    b = np.asarray(
        barVnew,
        dtype=np.float64,
    ).copy()

    b[
        data["dirichlet"],
        :,
        :,
    ] = 0.0

    bar_dV0 = np.sum(
        b
        *
        V,
        axis=1,
    )

    bar_dV1 = (
        data["dt"]
        *
        np.sum(
            b
            *
            F,
            axis=1,
        )
    )

    velocity_mass = (
        _velocity_mass_from_dump_coefficients(
            data
        )
    )

    bar_dumpmass = (
        -data["dV1"] ** 2
        *
        (
            2.0
            *
            velocity_mass
            *
            bar_dV0
            +
            bar_dV1
        )
    )

    bar_alpha = (
        0.5
        *
        data["local_mass"][..., None]
        *
        data["dt"]
        *
        bar_dumpmass[
            data["conn"]
        ]
    )

    return _alpha_material_vjp(
        data,
        bar_alpha,
    )


def pred_forward(
    data,
    V,
    Sold,
):
    conn = data["conn"]
    hp = data["hp"]
    scale = data[
        "invgrad_scale"
    ]

    lam = data["lam"]
    mu = data["mu"]

    dS0 = data["dS0"]
    dS1 = data["dS1"]

    dt = data["dt"]
    batch = data[
        "batch_size"
    ]

    Vtot = np.sum(
        V,
        axis=2,
    )

    Snew = np.empty_like(
        Sold
    )

    ne = data["ne"]

    for e0 in range(
        0,
        ne,
        batch,
    ):
        e1 = min(
            e0 + batch,
            ne,
        )

        ids = conn[
            e0:e1
        ]

        vl = Vtot[
            ids
        ].reshape(
            -1,
            3,
            3,
            3,
            3,
        )

        dx, dy, dz = (
            _derivatives(
                vl,
                hp,
                scale,
            )
        )

        old = Sold[
            e0:e1
        ].reshape(
            -1,
            3,
            3,
            3,
            3,
            6,
        )

        new = np.empty_like(
            old
        )

        la = lam[
            e0:e1
        ].reshape(
            -1,
            3,
            3,
            3,
        )

        muv = mu[
            e0:e1
        ].reshape(
            -1,
            3,
            3,
            3,
        )

        l2m = (
            la
            +
            2.0
            *
            muv
        )

        a = dS0[
            e0:e1
        ].reshape(
            -1,
            3,
            3,
            3,
            3,
        )

        b = (
            dS1[
                e0:e1
            ].reshape(
                -1,
                3,
                3,
                3,
                3,
            )
            *
            dt
        )

        ax = a[..., 0]
        bx = b[..., 0]

        new[..., 0, 0] = (
            ax
            *
            old[..., 0, 0]
            +
            bx
            *
            l2m
            *
            dx[..., 0]
        )

        new[..., 0, 1] = (
            ax
            *
            old[..., 0, 1]
            +
            bx
            *
            la
            *
            dx[..., 0]
        )

        new[..., 0, 2] = (
            ax
            *
            old[..., 0, 2]
            +
            bx
            *
            la
            *
            dx[..., 0]
        )

        new[..., 0, 3] = (
            ax
            *
            old[..., 0, 3]
            +
            bx
            *
            muv
            *
            dx[..., 1]
        )

        new[..., 0, 4] = (
            ax
            *
            old[..., 0, 4]
            +
            bx
            *
            muv
            *
            dx[..., 2]
        )

        new[..., 0, 5] = (
            ax
            *
            old[..., 0, 5]
        )

        ay = a[..., 1]
        by = b[..., 1]

        new[..., 1, 0] = (
            ay
            *
            old[..., 1, 0]
            +
            by
            *
            la
            *
            dy[..., 1]
        )

        new[..., 1, 1] = (
            ay
            *
            old[..., 1, 1]
            +
            by
            *
            l2m
            *
            dy[..., 1]
        )

        new[..., 1, 2] = (
            ay
            *
            old[..., 1, 2]
            +
            by
            *
            la
            *
            dy[..., 1]
        )

        new[..., 1, 3] = (
            ay
            *
            old[..., 1, 3]
            +
            by
            *
            muv
            *
            dy[..., 0]
        )

        new[..., 1, 4] = (
            ay
            *
            old[..., 1, 4]
        )

        new[..., 1, 5] = (
            ay
            *
            old[..., 1, 5]
            +
            by
            *
            muv
            *
            dy[..., 2]
        )

        az = a[..., 2]
        bz = b[..., 2]

        new[..., 2, 0] = (
            az
            *
            old[..., 2, 0]
            +
            bz
            *
            la
            *
            dz[..., 2]
        )

        new[..., 2, 1] = (
            az
            *
            old[..., 2, 1]
            +
            bz
            *
            la
            *
            dz[..., 2]
        )

        new[..., 2, 2] = (
            az
            *
            old[..., 2, 2]
            +
            bz
            *
            l2m
            *
            dz[..., 2]
        )

        new[..., 2, 3] = (
            az
            *
            old[..., 2, 3]
        )

        new[..., 2, 4] = (
            az
            *
            old[..., 2, 4]
            +
            bz
            *
            muv
            *
            dz[..., 0]
        )

        new[..., 2, 5] = (
            az
            *
            old[..., 2, 5]
            +
            bz
            *
            muv
            *
            dz[..., 1]
        )

        Snew[
            e0:e1
        ] = new.reshape(
            e1 - e0,
            27,
            3,
            6,
        )

    return Snew


def main_forward(
    data,
    Snew,
):
    conn = data["conn"]
    hp = data["hp"]
    w = data["weights"]
    jac = data["jac"]

    scale = data[
        "invgrad_scale"
    ]

    batch = data[
        "batch_size"
    ]

    F = np.zeros(
        (
            data["nn"],
            3,
            3,
        ),
        dtype=np.float64,
    )

    ne = data["ne"]

    for e0 in range(
        0,
        ne,
        batch,
    ):
        e1 = min(
            e0 + batch,
            ne,
        )

        s = Snew[
            e0:e1
        ].reshape(
            -1,
            3,
            3,
            3,
            3,
            6,
        )

        st = np.sum(
            s,
            axis=4,
        )

        sxx = st[..., 0]
        syy = st[..., 1]
        szz = st[..., 2]
        sxy = st[..., 3]
        sxz = st[..., 4]
        syz = st[..., 5]

        tx = scale * np.stack(
            (
                sxx,
                sxy,
                sxz,
            ),
            axis=-1,
        )

        ty = scale * np.stack(
            (
                sxy,
                syy,
                syz,
            ),
            axis=-1,
        )

        tz = scale * np.stack(
            (
                sxz,
                syz,
                szz,
            ),
            axis=-1,
        )

        fx = (
            -jac
            *
            np.einsum(
                "li,eijkc,i,j,k->eljkc",
                hp,
                tx,
                w,
                w,
                w,
                optimize=True,
            )
        )

        fy = (
            -jac
            *
            np.einsum(
                "lj,eijkc,i,j,k->eilkc",
                hp,
                ty,
                w,
                w,
                w,
                optimize=True,
            )
        )

        fz = (
            -jac
            *
            np.einsum(
                "lk,eijkc,i,j,k->eijlc",
                hp,
                tz,
                w,
                w,
                w,
                optimize=True,
            )
        )

        ids = conn[
            e0:e1
        ].reshape(-1)

        np.add.at(
            F[:, :, 0],
            ids,
            fx.reshape(
                -1,
                3,
            ),
        )

        np.add.at(
            F[:, :, 1],
            ids,
            fy.reshape(
                -1,
                3,
            ),
        )

        np.add.at(
            F[:, :, 2],
            ids,
            fz.reshape(
                -1,
                3,
            ),
        )

    return F


def corrector_forward(
    data,
    V,
    F,
):
    Vnew = (
        data[
            "dV0"
        ][
            :,
            None,
            :,
        ]
        *
        V
        +
        data[
            "dt"
        ]
        *
        data[
            "dV1"
        ][
            :,
            None,
            :,
        ]
        *
        F
    )

    Vnew[
        data[
            "dirichlet"
        ],
        :,
        :,
    ] = 0.0

    return Vnew


def forward_step(
    data,
    state,
):
    V, Sold = state

    Snew = pred_forward(
        data,
        V,
        Sold,
    )

    F = main_forward(
        data,
        Snew,
    )

    Vnew = corrector_forward(
        data,
        V,
        F,
    )

    return Vnew, Snew


def material_tangent(
    data,
    state,
    dlam,
    dmu,
):
    """Apply the complete one-step SolidPML material derivative."""
    V, Sold = state

    coefficient_tangent = (
        material_coefficient_tangent(
            data,
            dlam,
            dmu,
        )
    )

    Snew = pred_forward(
        data,
        V,
        Sold,
    )

    dSnew = pred_material_tangent(
        data,
        V,
        Sold,
        dlam,
        dmu,
        coefficient_tangent=(
            coefficient_tangent
        ),
    )

    F = main_forward(
        data,
        Snew,
    )

    dF = main_forward(
        data,
        dSnew,
    )

    dVnew = corrector_material_tangent(
        data,
        V,
        F,
        dF,
        coefficient_tangent,
    )

    return dVnew, dSnew


def material_vjp(
    data,
    state,
    bar_state_out,
):
    """Transpose of :func:`material_tangent` at a fixed state."""
    V, Sold = state
    barVnew, barSout = bar_state_out

    Snew = pred_forward(
        data,
        V,
        Sold,
    )

    F = main_forward(
        data,
        Snew,
    )

    corrector_lam, corrector_mu = (
        corrector_material_vjp(
            data,
            V,
            F,
            barVnew,
        )
    )

    _, barF = corrector_adjoint(
        data,
        barVnew,
    )

    barSnew = np.asarray(
        barSout,
        dtype=np.float64,
    ).copy()

    barSnew += main_adjoint(
        data,
        barF,
    )

    pred_lam, pred_mu = (
        pred_material_vjp(
            data,
            V,
            Sold,
            barSnew,
        )
    )

    return (
        corrector_lam
        +
        pred_lam,
        corrector_mu
        +
        pred_mu,
    )


def corrector_adjoint(
    data,
    barVnew,
):
    b = np.array(
        barVnew,
        copy=True,
        dtype=np.float64,
    )

    b[
        data[
            "dirichlet"
        ],
        :,
        :,
    ] = 0.0

    barVold = (
        data[
            "dV0"
        ][
            :,
            None,
            :,
        ]
        *
        b
    )

    barF = (
        data[
            "dt"
        ]
        *
        data[
            "dV1"
        ][
            :,
            None,
            :,
        ]
        *
        b
    )

    return barVold, barF


def main_adjoint(
    data,
    barF,
):
    conn = data["conn"]
    hp = data["hp"]
    w = data["weights"]
    jac = data["jac"]

    scale = data[
        "invgrad_scale"
    ]

    batch = data[
        "batch_size"
    ]

    barS = np.empty(
        (
            data["ne"],
            27,
            3,
            6,
        ),
        dtype=np.float64,
    )

    ne = data["ne"]

    for e0 in range(
        0,
        ne,
        batch,
    ):
        e1 = min(
            e0 + batch,
            ne,
        )

        ids = conn[
            e0:e1
        ]

        gb = barF[
            ids
        ]

        bx = gb[
            ...,
            0,
        ].reshape(
            -1,
            3,
            3,
            3,
            3,
        )

        by = gb[
            ...,
            1,
        ].reshape(
            -1,
            3,
            3,
            3,
            3,
        )

        bz = gb[
            ...,
            2,
        ].reshape(
            -1,
            3,
            3,
            3,
            3,
        )

        btx = (
            -jac
            *
            np.einsum(
                "li,eljkc,i,j,k->eijkc",
                hp,
                bx,
                w,
                w,
                w,
                optimize=True,
            )
        )

        bty = (
            -jac
            *
            np.einsum(
                "lj,eilkc,i,j,k->eijkc",
                hp,
                by,
                w,
                w,
                w,
                optimize=True,
            )
        )

        btz = (
            -jac
            *
            np.einsum(
                "lk,eijlc,i,j,k->eijkc",
                hp,
                bz,
                w,
                w,
                w,
                optimize=True,
            )
        )

        bs = np.zeros(
            (
                e1 - e0,
                3,
                3,
                3,
                6,
            ),
            dtype=np.float64,
        )

        bs[..., 0] += (
            scale
            *
            btx[..., 0]
        )

        bs[..., 1] += (
            scale
            *
            bty[..., 1]
        )

        bs[..., 2] += (
            scale
            *
            btz[..., 2]
        )

        bs[..., 3] += (
            scale
            *
            (
                btx[..., 1]
                +
                bty[..., 0]
            )
        )

        bs[..., 4] += (
            scale
            *
            (
                btx[..., 2]
                +
                btz[..., 0]
            )
        )

        bs[..., 5] += (
            scale
            *
            (
                bty[..., 2]
                +
                btz[..., 1]
            )
        )

        out = np.repeat(
            bs[
                ...,
                None,
                :,
            ],
            3,
            axis=4,
        )

        barS[
            e0:e1
        ] = out.reshape(
            e1 - e0,
            27,
            3,
            6,
        )

    return barS


def pred_adjoint(
    data,
    barSnew,
):
    conn = data["conn"]
    hp = data["hp"]

    scale = data[
        "invgrad_scale"
    ]

    lam = data["lam"]
    mu = data["mu"]

    dS0 = data["dS0"]
    dS1 = data["dS1"]

    dt = data["dt"]

    batch = data[
        "batch_size"
    ]

    barVtot = np.zeros(
        (
            data["nn"],
            3,
        ),
        dtype=np.float64,
    )

    barSold = np.empty_like(
        barSnew
    )

    ne = data["ne"]

    for e0 in range(
        0,
        ne,
        batch,
    ):
        e1 = min(
            e0 + batch,
            ne,
        )

        bs = barSnew[
            e0:e1
        ].reshape(
            -1,
            3,
            3,
            3,
            3,
            6,
        )

        la = lam[
            e0:e1
        ].reshape(
            -1,
            3,
            3,
            3,
        )

        muv = mu[
            e0:e1
        ].reshape(
            -1,
            3,
            3,
            3,
        )

        l2m = (
            la
            +
            2.0
            *
            muv
        )

        a = dS0[
            e0:e1
        ].reshape(
            -1,
            3,
            3,
            3,
            3,
        )

        d = dS1[
            e0:e1
        ].reshape(
            -1,
            3,
            3,
            3,
            3,
        )

        oldbar = np.empty_like(
            bs
        )

        oldbar[..., 0, :] = (
            a[
                ...,
                0,
                None,
            ]
            *
            bs[..., 0, :]
        )

        oldbar[..., 1, :] = (
            a[
                ...,
                1,
                None,
            ]
            *
            bs[..., 1, :]
        )

        oldbar[..., 2, :] = (
            a[
                ...,
                2,
                None,
            ]
            *
            bs[..., 2, :]
        )

        barSold[
            e0:e1
        ] = oldbar.reshape(
            e1 - e0,
            27,
            3,
            6,
        )

        bx = bs[..., 0, :]
        by = bs[..., 1, :]
        bz = bs[..., 2, :]

        dxbar = np.zeros(
            (
                e1 - e0,
                3,
                3,
                3,
                3,
            ),
            dtype=np.float64,
        )

        dybar = np.zeros_like(
            dxbar
        )

        dzbar = np.zeros_like(
            dxbar
        )

        cx = (
            dt
            *
            d[..., 0]
        )

        cy = (
            dt
            *
            d[..., 1]
        )

        cz = (
            dt
            *
            d[..., 2]
        )

        dxbar[..., 0] += (
            cx
            *
            (
                l2m
                *
                bx[..., 0]
                +
                la
                *
                bx[..., 1]
                +
                la
                *
                bx[..., 2]
            )
        )

        dxbar[..., 1] += (
            cx
            *
            muv
            *
            bx[..., 3]
        )

        dxbar[..., 2] += (
            cx
            *
            muv
            *
            bx[..., 4]
        )

        dybar[..., 1] += (
            cy
            *
            (
                la
                *
                by[..., 0]
                +
                l2m
                *
                by[..., 1]
                +
                la
                *
                by[..., 2]
            )
        )

        dybar[..., 0] += (
            cy
            *
            muv
            *
            by[..., 3]
        )

        dybar[..., 2] += (
            cy
            *
            muv
            *
            by[..., 5]
        )

        dzbar[..., 2] += (
            cz
            *
            (
                la
                *
                bz[..., 0]
                +
                la
                *
                bz[..., 1]
                +
                l2m
                *
                bz[..., 2]
            )
        )

        dzbar[..., 0] += (
            cz
            *
            muv
            *
            bz[..., 4]
        )

        dzbar[..., 1] += (
            cz
            *
            muv
            *
            bz[..., 5]
        )

        dxbar *= scale
        dybar *= scale
        dzbar *= scale

        bvx = np.einsum(
            "li,eijkc->eljkc",
            hp,
            dxbar,
            optimize=True,
        )

        bvy = np.einsum(
            "lj,eijkc->eilkc",
            hp,
            dybar,
            optimize=True,
        )

        bvz = np.einsum(
            "lk,eijkc->eijlc",
            hp,
            dzbar,
            optimize=True,
        )

        bvl = (
            bvx
            +
            bvy
            +
            bvz
        )

        ids = conn[
            e0:e1
        ].reshape(-1)

        np.add.at(
            barVtot,
            ids,
            bvl.reshape(
                -1,
                3,
            ),
        )

    barV = np.repeat(
        barVtot[
            :,
            :,
            None,
        ],
        3,
        axis=2,
    )

    return barV, barSold


def adjoint_step(
    data,
    bar_state_out,
):
    barVnew, barSout = (
        bar_state_out
    )

    barVold, barF = (
        corrector_adjoint(
            data,
            barVnew,
        )
    )

    barSnew = (
        np.asarray(
            barSout,
            dtype=np.float64,
        ).copy()
    )

    barSnew += main_adjoint(
        data,
        barF,
    )

    barVpred, barSold = (
        pred_adjoint(
            data,
            barSnew,
        )
    )

    barVold += barVpred

    return barVold, barSold
