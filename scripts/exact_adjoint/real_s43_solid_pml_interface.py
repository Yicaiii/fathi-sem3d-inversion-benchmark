from pathlib import Path

import numpy as np


def load_interface(topology_dir):
    root = Path(topology_dir)

    solid_xyz = np.asarray(
        np.load(
            root
            /
            "solid_compact_xyz.npy"
        ),
        dtype=np.float64,
    )

    pml_xyz = np.asarray(
        np.load(
            root
            /
            "pml_compact_xyz.npy"
        ),
        dtype=np.float64,
    )

    solid_idx = np.asarray(
        np.load(
            root
            /
            "interface_solid_compact.npy"
        ),
        dtype=np.int64,
    )

    pml_idx = np.asarray(
        np.load(
            root
            /
            "interface_pml_compact.npy"
        ),
        dtype=np.int64,
    )

    interface_xyz = np.asarray(
        np.load(
            root
            /
            "interface_xyz.npy"
        ),
        dtype=np.float64,
    )

    if not (
        len(solid_idx)
        ==
        len(pml_idx)
        ==
        len(interface_xyz)
    ):
        raise RuntimeError(
            "interface array lengths differ"
        )

    if len(
        np.unique(
            solid_idx
        )
    ) != len(
        solid_idx
    ):
        raise RuntimeError(
            "solid interface IDs are not unique"
        )

    if len(
        np.unique(
            pml_idx
        )
    ) != len(
        pml_idx
    ):
        raise RuntimeError(
            "PML interface IDs are not unique"
        )

    solid_error = float(
        np.max(
            np.abs(
                solid_xyz[
                    solid_idx
                ]
                -
                interface_xyz
            )
        )
    )

    pml_error = float(
        np.max(
            np.abs(
                pml_xyz[
                    pml_idx
                ]
                -
                interface_xyz
            )
        )
    )

    if (
        solid_error
        >
        1.0e-12
        or
        pml_error
        >
        1.0e-12
    ):
        raise RuntimeError(
            "interface coordinate mismatch"
        )

    return {
        "solid_xyz":
            solid_xyz,

        "pml_xyz":
            pml_xyz,

        "solid_idx":
            solid_idx,

        "pml_idx":
            pml_idx,

        "interface_xyz":
            interface_xyz,

        "solid_coordinate_error":
            solid_error,

        "pml_coordinate_error":
            pml_error,
    }


def velocity_interface_forward(
    interface,
    solid_velocity,
    pml_velocity,
):
    solid_velocity = np.asarray(
        solid_velocity,
        dtype=np.float64,
    )

    pml_velocity = np.asarray(
        pml_velocity,
        dtype=np.float64,
    )

    if solid_velocity.shape != (
        len(
            interface[
                "solid_xyz"
            ]
        ),
        3,
    ):
        raise ValueError(
            "invalid solid velocity shape"
        )

    if pml_velocity.shape != (
        len(
            interface[
                "pml_xyz"
            ]
        ),
        3,
        3,
    ):
        raise ValueError(
            "invalid PML velocity shape"
        )

    solid_out = np.array(
        solid_velocity,
        copy=True,
    )

    pml_out = np.array(
        pml_velocity,
        copy=True,
    )

    si = interface[
        "solid_idx"
    ]

    pi = interface[
        "pml_idx"
    ]

    pml_out[
        pi,
        :,
        :,
    ] = 0.0

    pml_out[
        pi,
        :,
        0,
    ] = solid_velocity[
        si,
        :,
    ]

    return (
        solid_out,
        pml_out,
    )


def velocity_interface_adjoint(
    interface,
    bar_solid_out,
    bar_pml_out,
):
    bar_solid_out = np.asarray(
        bar_solid_out,
        dtype=np.float64,
    )

    bar_pml_out = np.asarray(
        bar_pml_out,
        dtype=np.float64,
    )

    bar_solid_in = np.array(
        bar_solid_out,
        copy=True,
    )

    bar_pml_in = np.array(
        bar_pml_out,
        copy=True,
    )

    si = interface[
        "solid_idx"
    ]

    pi = interface[
        "pml_idx"
    ]

    np.add.at(
        bar_solid_in,
        si,
        bar_pml_out[
            pi,
            :,
            0,
        ],
    )

    bar_pml_in[
        pi,
        :,
        :,
    ] = 0.0

    return (
        bar_solid_in,
        bar_pml_in,
    )


def force_interface_forward(
    interface,
    solid_force,
    pml_force,
):
    solid_force = np.asarray(
        solid_force,
        dtype=np.float64,
    )

    pml_force = np.asarray(
        pml_force,
        dtype=np.float64,
    )

    if solid_force.shape != (
        len(
            interface[
                "solid_xyz"
            ]
        ),
        3,
    ):
        raise ValueError(
            "invalid solid force shape"
        )

    if pml_force.shape != (
        len(
            interface[
                "pml_xyz"
            ]
        ),
        3,
        3,
    ):
        raise ValueError(
            "invalid PML force shape"
        )

    solid_out = np.array(
        solid_force,
        copy=True,
    )

    pml_out = np.array(
        pml_force,
        copy=True,
    )

    si = interface[
        "solid_idx"
    ]

    pi = interface[
        "pml_idx"
    ]

    contribution = np.sum(
        pml_force[
            pi,
            :,
            :,
        ],
        axis=2,
    )

    np.add.at(
        solid_out,
        si,
        contribution,
    )

    return (
        solid_out,
        pml_out,
    )


def force_interface_adjoint(
    interface,
    bar_solid_out,
    bar_pml_out,
):
    bar_solid_out = np.asarray(
        bar_solid_out,
        dtype=np.float64,
    )

    bar_pml_out = np.asarray(
        bar_pml_out,
        dtype=np.float64,
    )

    bar_solid_in = np.array(
        bar_solid_out,
        copy=True,
    )

    bar_pml_in = np.array(
        bar_pml_out,
        copy=True,
    )

    si = interface[
        "solid_idx"
    ]

    pi = interface[
        "pml_idx"
    ]

    bs = bar_solid_out[
        si,
        :,
    ]

    for split in range(3):
        np.add.at(
            bar_pml_in[
                :,
                :,
                split,
            ],
            pi,
            bs,
        )

    return (
        bar_solid_in,
        bar_pml_in,
    )


def pair_dot(a, b):
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


def pair_norm(a):
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


def normalize_pair(a):
    n = pair_norm(
        a
    )

    if (
        n == 0.0
        or
        not np.isfinite(
            n
        )
    ):
        raise RuntimeError(
            "invalid pair norm"
        )

    return (
        a[0] / n,
        a[1] / n,
    )
