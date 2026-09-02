from pathlib import Path

import numpy as np

from scripts.exact_adjoint.real_s43_pml_operator import (
    adjoint_step as unused_pml_full_adjoint,
    corrector_adjoint as pml_corrector_adjoint,
    corrector_forward as pml_corrector_forward,
    corrector_material_tangent as pml_corrector_material_tangent,
    corrector_material_vjp as pml_corrector_material_vjp,
    load_operator_data as load_pml_data,
    main_adjoint as pml_main_adjoint,
    main_forward as pml_main_forward,
    material_coefficient_tangent as pml_material_coefficient_tangent,
    pred_adjoint as pml_pred_adjoint,
    pred_forward as pml_pred_forward,
    pred_material_tangent as pml_pred_material_tangent,
    pred_material_vjp as pml_pred_material_vjp,
)

from scripts.exact_adjoint.real_s43_solid_operator import (
    internal_force as solid_internal_force,
    load_solid_data,
    material_tangent as solid_material_tangent,
    material_vjp as solid_material_vjp,
)

from scripts.exact_adjoint.real_s43_solid_pml_interface import (
    force_interface_adjoint,
    force_interface_forward,
    load_interface,
    velocity_interface_adjoint,
    velocity_interface_forward,
)


def load_global_data(
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
    topology_dir = Path(
        topology_dir
    )

    coefficient_dir = Path(
        coefficient_dir
    )

    coupled_mass_dir = Path(
        coupled_mass_dir
    )

    solid = load_solid_data(
        config_path,
        topology_dir,
        coefficient_dir,
        coupled_mass_dir,
        gll_path,
        weights_path,
        kappa_path,
        mu_path,
        batch_size=batch_size,
    )

    pml = load_pml_data(
        topology_dir,
        coefficient_dir,
        gll_path,
        weights_path,
        batch_size=batch_size,
    )

    pml["dV0"] = np.asarray(
        np.load(
            coupled_mass_dir
            /
            "pml_dumpV0_coupled.npy"
        ),
        dtype=np.float64,
    )

    pml["dV1"] = np.asarray(
        np.load(
            coupled_mass_dir
            /
            "pml_dumpV1_coupled.npy"
        ),
        dtype=np.float64,
    )

    interface = load_interface(
        topology_dir
    )

    if solid["nn"] != len(
        interface[
            "solid_xyz"
        ]
    ):
        raise RuntimeError(
            "solid/interface node mismatch"
        )

    if pml["nn"] != len(
        interface[
            "pml_xyz"
        ]
    ):
        raise RuntimeError(
            "PML/interface node mismatch"
        )

    if solid["dt"] != pml["dt"]:
        raise RuntimeError(
            "solid/PML dt mismatch"
        )

    if len(
        interface[
            "solid_idx"
        ]
    ) != 22657:
        raise RuntimeError(
            "unexpected interface size"
        )

    return {
        "solid":
            solid,

        "pml":
            pml,

        "interface":
            interface,

        "dt":
            solid[
                "dt"
            ],
    }


def state_dot(
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
        +
        np.vdot(
            a[2],
            b[2],
        )
        +
        np.vdot(
            a[3],
            b[3],
        )
    )


def state_norm(
    a,
):
    return float(
        np.sqrt(
            max(
                state_dot(
                    a,
                    a,
                ),
                0.0,
            )
        )
    )


def normalize_state(
    state,
):
    n = state_norm(
        state
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
            "invalid global state norm"
        )

    return tuple(
        x / n
        for x in state
    )


def random_state(
    data,
    rng,
):
    solid = data[
        "solid"
    ]

    pml = data[
        "pml"
    ]

    Us = rng.standard_normal(
        (
            solid[
                "nn"
            ],
            3,
        )
    )

    Vs = rng.standard_normal(
        (
            solid[
                "nn"
            ],
            3,
        )
    )

    Vp = rng.standard_normal(
        (
            pml[
                "nn"
            ],
            3,
            3,
        )
    )

    Sp = rng.standard_normal(
        (
            pml[
                "ne"
            ],
            27,
            3,
            6,
        )
    )

    return normalize_state(
        (
            Us,
            Vs,
            Vp,
            Sp,
        )
    )


def solid_corrector_forward(
    solid,
    U,
    V,
    F,
):
    acceleration = (
        solid[
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
        solid[
            "dt"
        ]
        *
        acceleration
    )

    Unew = (
        U
        +
        solid[
            "dt"
        ]
        *
        Vnew
    )

    return (
        Unew,
        Vnew,
    )


def solid_corrector_adjoint(
    solid,
    barUnew,
    barVnew,
):
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
        solid[
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
        solid[
            "dt"
        ]
        *
        solid[
            "invmass"
        ][
            :,
            None,
        ]
        *
        barV_after
    )

    return (
        barU,
        barV,
        barF,
    )


def forward_step(
    data,
    state,
):
    solid = data[
        "solid"
    ]

    pml = data[
        "pml"
    ]

    interface = data[
        "interface"
    ]

    Us, Vs, Vp_old, Sp_old = state

    # ----------------------------------------------------------
    # SEM3D Newmark predictor:
    #
    # regular solid:
    #   force workspace is zeroed
    #
    # SolidPML:
    #   interface PML velocity split-0 <- solid velocity
    #   interface split-1, split-2 <- 0
    #   predictor copy creates the PML velocity used internally.
    #
    # In our compact graph, Vs_after is the identity solid branch
    # and Vp_pred is the overwritten/copy-equivalent PML branch.
    # ----------------------------------------------------------

    Vs_after, Vp_pred = (
        velocity_interface_forward(
            interface,
            Vs,
            Vp_old,
        )
    )

    # ----------------------------------------------------------
    # Internal regular-solid force
    # ----------------------------------------------------------

    Fs_internal = solid_internal_force(
        solid,
        Us,
    )

    # ----------------------------------------------------------
    # Standard SolidPML internal recurrence
    # ----------------------------------------------------------

    Sp_new = pml_pred_forward(
        pml,
        Vp_pred,
        Sp_old,
    )

    Fp_internal = pml_main_forward(
        pml,
        Sp_new,
    )

    # ----------------------------------------------------------
    # SEM3D couplage_pml_solid:
    #
    # Fs <- Fs + Fp(split0)+Fp(split1)+Fp(split2)
    #
    # Fp is also retained for the PML corrector.
    # ----------------------------------------------------------

    Fs_total, Fp_after = (
        force_interface_forward(
            interface,
            Fs_internal,
            Fp_internal,
        )
    )

    # ----------------------------------------------------------
    # Correctors
    # ----------------------------------------------------------

    Vp_new = pml_corrector_forward(
        pml,
        Vp_pred,
        Fp_after,
    )

    Us_new, Vs_new = (
        solid_corrector_forward(
            solid,
            Us,
            Vs_after,
            Fs_total,
        )
    )

    return (
        Us_new,
        Vs_new,
        Vp_new,
        Sp_new,
    )


def material_tangent(
    data,
    state,
    solid_dlam,
    solid_dmu,
    pml_dlam,
    pml_dmu,
):
    """Apply the complete global one-step material derivative.

    The four direction arrays are element-GLL fields in their respective
    regular-solid and SolidPML orderings.
    """
    solid = data["solid"]
    pml = data["pml"]
    interface = data["interface"]

    Us, Vs, Vp_old, Sp_old = state

    # The frozen S43 candidate pairs have an exactly zero SolidPML
    # material direction.  In that case the PML material derivative is
    # identically zero, so evaluating the baseline PML predictor/main/
    # corrector graph is unnecessary.  This is an exact specialization of
    # the same operator (not a second tangent implementation), and it also
    # makes the expected independence from Vs/Vp/Sp explicit.
    if (
        not np.any(pml_dlam)
        and
        not np.any(pml_dmu)
    ):
        dFs_internal = solid_material_tangent(
            solid,
            Us,
            solid_dlam,
            solid_dmu,
        )

        dFs_total, unused_dFp_after = force_interface_forward(
            interface,
            dFs_internal,
            np.zeros_like(Vp_old),
        )

        dVs_new = (
            solid["dt"]
            *
            solid["invmass"][:, None]
            *
            dFs_total
        )

        dUs_new = solid["dt"] * dVs_new

        return (
            dUs_new,
            dVs_new,
            np.zeros_like(Vp_old),
            np.zeros_like(Sp_old),
        )

    Vs_after, Vp_pred = velocity_interface_forward(
        interface,
        Vs,
        Vp_old,
    )

    Fs_internal = solid_internal_force(
        solid,
        Us,
    )

    dFs_internal = solid_material_tangent(
        solid,
        Us,
        solid_dlam,
        solid_dmu,
    )

    coefficient_tangent = (
        pml_material_coefficient_tangent(
            pml,
            pml_dlam,
            pml_dmu,
        )
    )

    Sp_new = pml_pred_forward(
        pml,
        Vp_pred,
        Sp_old,
    )

    dSp_new = pml_pred_material_tangent(
        pml,
        Vp_pred,
        Sp_old,
        pml_dlam,
        pml_dmu,
        coefficient_tangent=(
            coefficient_tangent
        ),
    )

    Fp_internal = pml_main_forward(
        pml,
        Sp_new,
    )

    dFp_internal = pml_main_forward(
        pml,
        dSp_new,
    )

    Fs_total, Fp_after = force_interface_forward(
        interface,
        Fs_internal,
        Fp_internal,
    )

    dFs_total, dFp_after = force_interface_forward(
        interface,
        dFs_internal,
        dFp_internal,
    )

    dVp_new = pml_corrector_material_tangent(
        pml,
        Vp_pred,
        Fp_after,
        dFp_after,
        coefficient_tangent,
    )

    dVs_new = (
        solid["dt"]
        *
        solid["invmass"][:, None]
        *
        dFs_total
    )

    dUs_new = (
        solid["dt"]
        *
        dVs_new
    )

    return (
        dUs_new,
        dVs_new,
        dVp_new,
        dSp_new,
    )


def material_vjp(
    data,
    state,
    bar_state_out,
):
    """Transpose of :func:`material_tangent` at a fixed global state."""
    solid = data["solid"]
    pml = data["pml"]
    interface = data["interface"]

    Us, Vs, Vp_old, Sp_old = state
    barUs_new, barVs_new, barVp_new, barSp_output = (
        bar_state_out
    )

    Vs_after, Vp_pred = velocity_interface_forward(
        interface,
        Vs,
        Vp_old,
    )

    Fs_internal = solid_internal_force(
        solid,
        Us,
    )

    Sp_new = pml_pred_forward(
        pml,
        Vp_pred,
        Sp_old,
    )

    Fp_internal = pml_main_forward(
        pml,
        Sp_new,
    )

    Fs_total, Fp_after = force_interface_forward(
        interface,
        Fs_internal,
        Fp_internal,
    )

    _, _, barFs_total = solid_corrector_adjoint(
        solid,
        barUs_new,
        barVs_new,
    )

    corrector_lam, corrector_mu = (
        pml_corrector_material_vjp(
            pml,
            Vp_pred,
            Fp_after,
            barVp_new,
        )
    )

    _, barFp_after = pml_corrector_adjoint(
        pml,
        barVp_new,
    )

    barFs_internal, barFp_internal = (
        force_interface_adjoint(
            interface,
            barFs_total,
            barFp_after,
        )
    )

    barSp_new = (
        np.asarray(
            barSp_output,
            dtype=np.float64,
        )
        +
        pml_main_adjoint(
            pml,
            barFp_internal,
        )
    )

    pred_lam, pred_mu = pml_pred_material_vjp(
        pml,
        Vp_pred,
        Sp_old,
        barSp_new,
    )

    solid_lam, solid_mu = solid_material_vjp(
        solid,
        Us,
        barFs_internal,
    )

    return (
        solid_lam,
        solid_mu,
        corrector_lam + pred_lam,
        corrector_mu + pred_mu,
    )


def adjoint_step(
    data,
    bar_state_out,
):
    solid = data[
        "solid"
    ]

    pml = data[
        "pml"
    ]

    interface = data[
        "interface"
    ]

    (
        barUs_new,
        barVs_new,
        barVp_new,
        barSp_output,
    ) = bar_state_out

    # ==========================================================
    # REVERSE SEM3D TIMESTEP
    # ==========================================================

    # ----------------------------------------------------------
    # 1. Reverse regular-solid corrector
    # ----------------------------------------------------------

    (
        barUs,
        barVs_after,
        barFs_total,
    ) = solid_corrector_adjoint(
        solid,
        barUs_new,
        barVs_new,
    )

    # ----------------------------------------------------------
    # 2. Reverse PML corrector
    # ----------------------------------------------------------

    (
        barVp_pred_direct,
        barFp_after,
    ) = pml_corrector_adjoint(
        pml,
        barVp_new,
    )

    # ----------------------------------------------------------
    # 3. Reverse PML-force -> solid-force coupling
    #
    # barFs_internal gets solid force adjoint.
    #
    # barFp_internal gets:
    #   - identity contribution from PML corrector
    #   - same solid-force adjoint added to all three splits
    # ----------------------------------------------------------

    (
        barFs_internal,
        barFp_internal,
    ) = force_interface_adjoint(
        interface,
        barFs_total,
        barFp_after,
    )

    # ----------------------------------------------------------
    # 4. Reverse PML weak divergence
    # ----------------------------------------------------------

    barSp_from_force = pml_main_adjoint(
        pml,
        barFp_internal,
    )

    barSp_new = (
        np.asarray(
            barSp_output,
            dtype=np.float64,
        )
        +
        barSp_from_force
    )

    # ----------------------------------------------------------
    # 5. Reverse PML stress/history predictor
    # ----------------------------------------------------------

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

    # ----------------------------------------------------------
    # 6. Reverse regular-solid internal force.
    #
    # For this real isotropic linear S43 operator we validated:
    #
    #     K^T = K
    #
    # to ~1.8e-14 in Stage 3H.
    # ----------------------------------------------------------

    barUs += solid_internal_force(
        solid,
        barFs_internal,
    )

    # ----------------------------------------------------------
    # 7. Reverse predictor/interface overwrite.
    #
    # This also includes the identity solid-velocity path used
    # later by the regular-solid corrector.
    # ----------------------------------------------------------

    (
        barVs_old,
        barVp_old,
    ) = velocity_interface_adjoint(
        interface,
        barVs_after,
        barVp_pred,
    )

    return (
        barUs,
        barVs_old,
        barVp_old,
        barSp_old,
    )
