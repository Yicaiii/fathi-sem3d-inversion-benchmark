import json
from pathlib import Path

import numpy as np

from scripts.exact_adjoint.build_real_s43_coupled_mass import (
    read_samples,
    trilinear_sample,
)

from scripts.exact_adjoint.real_s43_pml_operator import (
    derivative_matrix,
)


def load_solid_data(
    config_path,
    topology_dir,
    coefficient_dir,
    coupled_mass_dir,
    gll_path,
    weights_path,
    kappa_path,
    mu_path,
    batch_size=2048,
):
    config_path = Path(
        config_path
    )

    topology_dir = Path(
        topology_dir
    )

    coefficient_dir = Path(
        coefficient_dir
    )

    coupled_mass_dir = Path(
        coupled_mass_dir
    )

    cfg = json.loads(
        config_path.read_text(
            encoding="utf-8"
        )
    )

    coef_summary = json.loads(
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
            "solid_connectivity_compact.npy"
        ),
        dtype=np.int64,
    )

    xyz = np.asarray(
        np.load(
            topology_dir
            /
            "solid_compact_xyz.npy"
        ),
        dtype=np.float64,
    )

    invmass = np.asarray(
        np.load(
            coupled_mass_dir
            /
            "solid_inverse_mass_coupled.npy"
        ),
        dtype=np.float64,
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

    hp = derivative_matrix(
        gll
    ).T

    if conn.ndim != 2:
        raise RuntimeError(
            f"invalid connectivity {conn.shape}"
        )

    ngll = len(
        gll
    )

    ngll3 = (
        ngll
        **
        3
    )

    if conn.shape[1] != ngll3:
        raise RuntimeError(
            "connectivity/GLL mismatch"
        )

    if len(
        invmass
    ) != len(
        xyz
    ):
        raise RuntimeError(
            "mass/node mismatch"
        )

    domain = cfg[
        "domain"
    ]

    kappa_grid = read_samples(
        kappa_path
    )

    mu_grid = read_samples(
        mu_path
    )

    kappa_nodes = trilinear_sample(
        kappa_grid,
        xyz,
        float(
            domain[
                "x_min_m"
            ]
        ),
        float(
            domain[
                "x_max_m"
            ]
        ),
        float(
            domain[
                "y_min_m"
            ]
        ),
        float(
            domain[
                "y_max_m"
            ]
        ),
        float(
            domain[
                "z_min_m"
            ]
        ),
        float(
            domain[
                "z_max_m"
            ]
        ),
    )

    mu_nodes = trilinear_sample(
        mu_grid,
        xyz,
        float(
            domain[
                "x_min_m"
            ]
        ),
        float(
            domain[
                "x_max_m"
            ]
        ),
        float(
            domain[
                "y_min_m"
            ]
        ),
        float(
            domain[
                "y_max_m"
            ]
        ),
        float(
            domain[
                "z_min_m"
            ]
        ),
        float(
            domain[
                "z_max_m"
            ]
        ),
    )

    lambda_nodes = (
        kappa_nodes
        -
        (
            2.0
            /
            3.0
        )
        *
        mu_nodes
    )

    lam = lambda_nodes[
        conn
    ]

    mu = mu_nodes[
        conn
    ]

    h = float(
        coef_summary[
            "element_spacing_m"
        ]
    )

    return {
        "conn":
            conn,

        "xyz":
            xyz,

        "invmass":
            invmass,

        "lam":
            lam,

        "mu":
            mu,

        "gll":
            gll,

        "weights":
            weights,

        "hp":
            hp,

        "jac":
            float(
                coef_summary[
                    "jacobian"
                ]
            ),

        "scale":
            (
                2.0
                /
                h
            ),

        "dt":
            float(
                coef_summary[
                    "dt"
                ]
            ),

        "ne":
            int(
                conn.shape[0]
            ),

        "nn":
            int(
                len(
                    xyz
                )
            ),

        "ngll":
            int(
                ngll
            ),

        "batch_size":
            int(
                batch_size
            ),
    }


def pair_dot(
    a,
    b,
):
    return float(
        np.vdot(
            a[0],
            b[0],
        )
        +
        np.vdot(
            a[1],
            b[1],
        )
    )


def pair_norm(
    a,
):
    return float(
        np.sqrt(
            max(
                pair_dot(
                    a,
                    a,
                ),
                0.0,
            )
        )
    )


def normalize_pair(
    a,
):
    n = pair_norm(
        a
    )

    if (
        n
        ==
        0.0
        or
        not
        np.isfinite(
            n
        )
    ):
        raise RuntimeError(
            "invalid state norm"
        )

    return (
        a[0]
        /
        n,
        a[1]
        /
        n,
    )


def random_state(
    data,
    rng,
):
    U = rng.standard_normal(
        (
            data[
                "nn"
            ],
            3,
        )
    )

    V = rng.standard_normal(
        (
            data[
                "nn"
            ],
            3,
        )
    )

    return normalize_pair(
        (
            U,
            V,
        )
    )


def internal_force(
    data,
    U,
):
    U = np.asarray(
        U,
        dtype=np.float64,
    )

    conn = data[
        "conn"
    ]

    hp = data[
        "hp"
    ]

    weights = data[
        "weights"
    ]

    jac = data[
        "jac"
    ]

    scale = data[
        "scale"
    ]

    lam = data[
        "lam"
    ]

    mu = data[
        "mu"
    ]

    batch = data[
        "batch_size"
    ]

    ngll = data[
        "ngll"
    ]

    F = np.zeros(
        (
            data[
                "nn"
            ],
            3,
        ),
        dtype=np.float64,
    )

    for e0 in range(
        0,
        data[
            "ne"
        ],
        batch,
    ):
        e1 = min(
            e0
            +
            batch,
            data[
                "ne"
            ],
        )

        ids = conn[
            e0:e1
        ]

        u = U[
            ids
        ].reshape(
            -1,
            ngll,
            ngll,
            ngll,
            3,
        )

        du_dx = (
            np.einsum(
                "eljkc,li->eijkc",
                u,
                hp,
                optimize=True,
            )
            *
            scale
        )

        du_dy = (
            np.einsum(
                "eilkc,lj->eijkc",
                u,
                hp,
                optimize=True,
            )
            *
            scale
        )

        du_dz = (
            np.einsum(
                "eijlc,lk->eijkc",
                u,
                hp,
                optimize=True,
            )
            *
            scale
        )

        la = lam[
            e0:e1
        ].reshape(
            -1,
            ngll,
            ngll,
            ngll,
        )

        muv = mu[
            e0:e1
        ].reshape(
            -1,
            ngll,
            ngll,
            ngll,
        )

        divu = (
            du_dx[
                ...,
                0
            ]
            +
            du_dy[
                ...,
                1
            ]
            +
            du_dz[
                ...,
                2
            ]
        )

        sxx = (
            la
            *
            divu
            +
            2.0
            *
            muv
            *
            du_dx[
                ...,
                0
            ]
        )

        syy = (
            la
            *
            divu
            +
            2.0
            *
            muv
            *
            du_dy[
                ...,
                1
            ]
        )

        szz = (
            la
            *
            divu
            +
            2.0
            *
            muv
            *
            du_dz[
                ...,
                2
            ]
        )

        sxy = (
            muv
            *
            (
                du_dy[
                    ...,
                    0
                ]
                +
                du_dx[
                    ...,
                    1
                ]
            )
        )

        sxz = (
            muv
            *
            (
                du_dz[
                    ...,
                    0
                ]
                +
                du_dx[
                    ...,
                    2
                ]
            )
        )

        syz = (
            muv
            *
            (
                du_dz[
                    ...,
                    1
                ]
                +
                du_dy[
                    ...,
                    2
                ]
            )
        )

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
                weights,
                weights,
                weights,
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
                weights,
                weights,
                weights,
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
                weights,
                weights,
                weights,
                optimize=True,
            )
        )

        flocal = (
            fx
            +
            fy
            +
            fz
        )

        np.add.at(
            F,
            ids.reshape(
                -1
            ),
            flocal.reshape(
                -1,
                3,
            ),
        )

    return F


def material_tangent(
    data,
    U,
    dlam,
    dmu,
):
    """Apply the regular-solid force derivative with respect to material."""
    direction_data = dict(
        data
    )

    direction_data["lam"] = np.asarray(
        dlam,
        dtype=np.float64,
    )

    direction_data["mu"] = np.asarray(
        dmu,
        dtype=np.float64,
    )

    if direction_data["lam"].shape != data["lam"].shape:
        raise RuntimeError(
            f"unexpected solid delta-lambda shape "
            f"{direction_data['lam'].shape}"
        )

    if direction_data["mu"].shape != data["mu"].shape:
        raise RuntimeError(
            f"unexpected solid delta-mu shape "
            f"{direction_data['mu'].shape}"
        )

    return internal_force(
        direction_data,
        U,
    )


def material_vjp(
    data,
    U,
    force_seed,
):
    """Transpose of :func:`material_tangent` at a fixed displacement."""
    conn = data["conn"]
    ngll = data["ngll"]
    hp = data["hp"]
    scale = data["scale"]

    w3 = (
        np.einsum(
            "i,j,k->ijk",
            data["weights"],
            data["weights"],
            data["weights"],
            optimize=True,
        )
        *
        data["jac"]
    )

    g_lam = np.zeros_like(
        data["lam"]
    )
    g_mu = np.zeros_like(
        data["mu"]
    )

    def derivatives(field, ids):
        local = field[ids].reshape(
            -1,
            ngll,
            ngll,
            ngll,
            3,
        )

        dx = (
            np.einsum(
                "eljkc,li->eijkc",
                local,
                hp,
                optimize=True,
            )
            *
            scale
        )

        dy = (
            np.einsum(
                "eilkc,lj->eijkc",
                local,
                hp,
                optimize=True,
            )
            *
            scale
        )

        dz = (
            np.einsum(
                "eijlc,lk->eijkc",
                local,
                hp,
                optimize=True,
            )
            *
            scale
        )

        return dx, dy, dz

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

        ids = conn[e0:e1]

        ux, uy, uz = derivatives(
            U,
            ids,
        )

        vx, vy, vz = derivatives(
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
            -w3[None, ...]
            *
            div_u
            *
            div_v
        )

        local_mu = (
            -w3[None, ...]
            *
            (
                2.0 * ux[..., 0] * vx[..., 0]
                +
                2.0 * uy[..., 1] * vy[..., 1]
                +
                2.0 * uz[..., 2] * vz[..., 2]
                +
                shear_xy_u * shear_xy_v
                +
                shear_xz_u * shear_xz_v
                +
                shear_yz_u * shear_yz_v
            )
        )

        g_lam[e0:e1] = local_lam.reshape(
            e1 - e0,
            -1,
        )

        g_mu[e0:e1] = local_mu.reshape(
            e1 - e0,
            -1,
        )

    return g_lam, g_mu


def forward_step(
    data,
    state,
):
    U, V = state

    F = internal_force(
        data,
        U,
    )

    A = (
        data[
            "invmass"
        ][
            :,
            None,
        ]
        *
        F
    )

    Vnew = (
        V
        +
        data[
            "dt"
        ]
        *
        A
    )

    Unew = (
        U
        +
        data[
            "dt"
        ]
        *
        Vnew
    )

    return (
        Unew,
        Vnew,
    )


def adjoint_step(
    data,
    bar_state_out,
):
    barUnew, barVnew = (
        bar_state_out
    )

    barU = np.array(
        barUnew,
        copy=True,
        dtype=np.float64,
    )

    barV_after = (
        np.asarray(
            barVnew,
            dtype=np.float64,
        )
        +
        data[
            "dt"
        ]
        *
        np.asarray(
            barUnew,
            dtype=np.float64,
        )
    )

    barV = np.array(
        barV_after,
        copy=True,
    )

    barF = (
        data[
            "dt"
        ]
        *
        data[
            "invmass"
        ][
            :,
            None,
        ]
        *
        barV_after
    )

    barU += internal_force(
        data,
        barF,
    )

    return (
        barU,
        barV,
    )
