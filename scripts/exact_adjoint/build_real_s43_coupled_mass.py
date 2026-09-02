import argparse
import json
from pathlib import Path

import h5py
import numpy as np


def read_samples(path):
    with h5py.File(path, "r") as f:
        if "samples" in f:
            return np.asarray(
                f["samples"][...],
                dtype=np.float64,
            )

        names = []

        def visit(name, obj):
            if isinstance(obj, h5py.Dataset):
                names.append(name)

        f.visititems(visit)

        if len(names) != 1:
            raise RuntimeError(
                f"cannot identify dataset in {path}: {names}"
            )

        return np.asarray(
            f[names[0]][...],
            dtype=np.float64,
        )


def trilinear_sample(
    field,
    xyz,
    xmin,
    xmax,
    ymin,
    ymax,
    zmin,
    zmax,
):
    field = np.asarray(
        field,
        dtype=np.float64,
    )

    xyz = np.asarray(
        xyz,
        dtype=np.float64,
    )

    nz, ny, nx = field.shape

    x = np.clip(
        xyz[:, 0],
        xmin,
        xmax,
    )

    y = np.clip(
        xyz[:, 1],
        ymin,
        ymax,
    )

    z = np.clip(
        xyz[:, 2],
        zmin,
        zmax,
    )

    fx = (
        (x - xmin)
        /
        (xmax - xmin)
        *
        (nx - 1)
    )

    fy = (
        (y - ymin)
        /
        (ymax - ymin)
        *
        (ny - 1)
    )

    fz = (
        (z - zmin)
        /
        (zmax - zmin)
        *
        (nz - 1)
    )

    ix0 = np.floor(fx).astype(np.int64)
    iy0 = np.floor(fy).astype(np.int64)
    iz0 = np.floor(fz).astype(np.int64)

    ix1 = np.minimum(
        ix0 + 1,
        nx - 1,
    )

    iy1 = np.minimum(
        iy0 + 1,
        ny - 1,
    )

    iz1 = np.minimum(
        iz0 + 1,
        nz - 1,
    )

    tx = fx - ix0
    ty = fy - iy0
    tz = fz - iz0

    c000 = field[iz0, iy0, ix0]
    c001 = field[iz0, iy0, ix1]
    c010 = field[iz0, iy1, ix0]
    c011 = field[iz0, iy1, ix1]

    c100 = field[iz1, iy0, ix0]
    c101 = field[iz1, iy0, ix1]
    c110 = field[iz1, iy1, ix0]
    c111 = field[iz1, iy1, ix1]

    c00 = (
        c000
        *
        (1.0 - tx)
        +
        c001
        *
        tx
    )

    c01 = (
        c010
        *
        (1.0 - tx)
        +
        c011
        *
        tx
    )

    c10 = (
        c100
        *
        (1.0 - tx)
        +
        c101
        *
        tx
    )

    c11 = (
        c110
        *
        (1.0 - tx)
        +
        c111
        *
        tx
    )

    c0 = (
        c00
        *
        (1.0 - ty)
        +
        c01
        *
        ty
    )

    c1 = (
        c10
        *
        (1.0 - ty)
        +
        c11
        *
        ty
    )

    return (
        c0
        *
        (1.0 - tz)
        +
        c1
        *
        tz
    )


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
        "--weights",
        required=True,
    )

    p.add_argument(
        "--density",
        required=True,
    )

    p.add_argument(
        "--output",
        required=True,
    )

    args = p.parse_args()

    topo = Path(
        args.topology
    )

    coef = Path(
        args.coefficients
    )

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

    coef_summary = json.loads(
        (
            coef
            /
            "summary.json"
        ).read_text(
            encoding="utf-8"
        )
    )

    weights = np.asarray(
        np.load(
            args.weights
        ),
        dtype=np.float64,
    )

    if len(weights) != 3:
        raise RuntimeError(
            "expected ngll=3 weights"
        )

    solid_conn = np.asarray(
        np.load(
            topo
            /
            "solid_connectivity_compact.npy"
        ),
        dtype=np.int64,
    )

    solid_xyz = np.asarray(
        np.load(
            topo
            /
            "solid_compact_xyz.npy"
        ),
        dtype=np.float64,
    )

    solid_interface = np.asarray(
        np.load(
            topo
            /
            "interface_solid_compact.npy"
        ),
        dtype=np.int64,
    )

    pml_interface = np.asarray(
        np.load(
            topo
            /
            "interface_pml_compact.npy"
        ),
        dtype=np.int64,
    )

    pml_mass_base = np.asarray(
        np.load(
            coef
            /
            "pml_mass_global.npy"
        ),
        dtype=np.float64,
    )

    pml_dumpmass = np.asarray(
        np.load(
            coef
            /
            "pml_dumpmass_global.npy"
        ),
        dtype=np.float64,
    )

    old_dumpV0 = np.asarray(
        np.load(
            coef
            /
            "pml_dumpV0.npy"
        ),
        dtype=np.float64,
    )

    old_dumpV1 = np.asarray(
        np.load(
            coef
            /
            "pml_dumpV1.npy"
        ),
        dtype=np.float64,
    )

    if solid_conn.shape != (
        36864,
        27,
    ):
        raise RuntimeError(
            f"unexpected solid connectivity {solid_conn.shape}"
        )

    if len(
        solid_interface
    ) != len(
        pml_interface
    ):
        raise RuntimeError(
            "interface map length mismatch"
        )

    if len(
        np.unique(
            solid_interface
        )
    ) != len(
        solid_interface
    ):
        raise RuntimeError(
            "solid interface map not unique"
        )

    if len(
        np.unique(
            pml_interface
        )
    ) != len(
        pml_interface
    ):
        raise RuntimeError(
            "PML interface map not unique"
        )

    jac = float(
        coef_summary[
            "jacobian"
        ]
    )

    density_grid = read_samples(
        args.density
    )

    domain = cfg[
        "domain"
    ]

    rho_solid_nodes = trilinear_sample(
        density_grid,
        solid_xyz,
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

    if not np.all(
        rho_solid_nodes
        >
        0.0
    ):
        raise RuntimeError(
            "non-positive solid density"
        )

    local_weight = []

    for i in range(3):
        for j in range(3):
            for k in range(3):
                local_weight.append(
                    weights[i]
                    *
                    weights[j]
                    *
                    weights[k]
                    *
                    jac
                )

    local_weight = np.asarray(
        local_weight,
        dtype=np.float64,
    )

    local_rho = rho_solid_nodes[
        solid_conn
    ]

    local_mass = (
        local_rho
        *
        local_weight[
            None,
            :,
        ]
    )

    solid_mass_base = np.zeros(
        len(
            solid_xyz
        ),
        dtype=np.float64,
    )

    np.add.at(
        solid_mass_base,
        solid_conn.reshape(-1),
        local_mass.reshape(-1),
    )

    if not np.all(
        solid_mass_base
        >
        0.0
    ):
        raise RuntimeError(
            "non-positive solid assembled mass"
        )

    solid_mass_coupled = np.array(
        solid_mass_base,
        copy=True,
    )

    pml_mass_coupled = np.array(
        pml_mass_base,
        copy=True,
    )

    interface_mass = (
        solid_mass_base[
            solid_interface
        ]
        +
        pml_mass_base[
            pml_interface
        ]
    )

    solid_mass_coupled[
        solid_interface
    ] = interface_mass

    pml_mass_coupled[
        pml_interface
    ] = interface_mass

    solid_inverse_mass = (
        1.0
        /
        solid_mass_coupled
    )

    denominator = (
        pml_mass_coupled[
            :,
            None,
        ]
        +
        pml_dumpmass
    )

    if not np.all(
        denominator
        >
        0.0
    ):
        raise RuntimeError(
            "non-positive coupled PML denominator"
        )

    pml_dumpV1_coupled = (
        1.0
        /
        denominator
    )

    pml_dumpV0_coupled = (
        pml_mass_coupled[
            :,
            None,
        ]
        -
        pml_dumpmass
    ) * pml_dumpV1_coupled

    solid_noninterface = np.ones(
        len(
            solid_mass_base
        ),
        dtype=bool,
    )

    solid_noninterface[
        solid_interface
    ] = False

    pml_noninterface = np.ones(
        len(
            pml_mass_base
        ),
        dtype=bool,
    )

    pml_noninterface[
        pml_interface
    ] = False

    solid_outside_error = float(
        np.max(
            np.abs(
                solid_mass_coupled[
                    solid_noninterface
                ]
                -
                solid_mass_base[
                    solid_noninterface
                ]
            )
        )
    )

    pml_outside_error = float(
        np.max(
            np.abs(
                pml_mass_coupled[
                    pml_noninterface
                ]
                -
                pml_mass_base[
                    pml_noninterface
                ]
            )
        )
    )

    interface_solid_error = float(
        np.max(
            np.abs(
                solid_mass_coupled[
                    solid_interface
                ]
                -
                interface_mass
            )
        )
    )

    interface_pml_error = float(
        np.max(
            np.abs(
                pml_mass_coupled[
                    pml_interface
                ]
                -
                interface_mass
            )
        )
    )

    dumpV0_change = np.abs(
        pml_dumpV0_coupled
        -
        old_dumpV0
    )

    dumpV1_change = np.abs(
        pml_dumpV1_coupled
        -
        old_dumpV1
    )

    changed0 = np.flatnonzero(
        np.any(
            dumpV0_change
            >
            0.0,
            axis=1,
        )
    )

    changed1 = np.flatnonzero(
        np.any(
            dumpV1_change
            >
            0.0,
            axis=1,
        )
    )

    print(
        "============================================================"
    )

    print(
        "REAL S43 COUPLED MASS"
    )

    print(
        "============================================================"
    )

    print(
        "solid nodes =",
        len(
            solid_mass_base
        ),
    )

    print(
        "PML nodes =",
        len(
            pml_mass_base
        ),
    )

    print(
        "interface pairs =",
        len(
            solid_interface
        ),
    )

    print()

    print(
        "solid base mass min/max =",
        [
            float(
                solid_mass_base.min()
            ),
            float(
                solid_mass_base.max()
            ),
        ],
    )

    print(
        "PML base mass min/max =",
        [
            float(
                pml_mass_base.min()
            ),
            float(
                pml_mass_base.max()
            ),
        ],
    )

    print(
        "interface combined mass min/max =",
        [
            float(
                interface_mass.min()
            ),
            float(
                interface_mass.max()
            ),
        ],
    )

    print(
        "solid coupled mass min/max =",
        [
            float(
                solid_mass_coupled.min()
            ),
            float(
                solid_mass_coupled.max()
            ),
        ],
    )

    print(
        "PML coupled mass min/max =",
        [
            float(
                pml_mass_coupled.min()
            ),
            float(
                pml_mass_coupled.max()
            ),
        ],
    )

    print()

    print(
        "solid outside-interface error =",
        f"{solid_outside_error:.17e}",
    )

    print(
        "PML outside-interface error =",
        f"{pml_outside_error:.17e}",
    )

    print(
        "interface solid mass error =",
        f"{interface_solid_error:.17e}",
    )

    print(
        "interface PML mass error =",
        f"{interface_pml_error:.17e}",
    )

    print()

    print(
        "PML DumpV0 changed nodes =",
        len(
            changed0
        ),
    )

    print(
        "PML DumpV1 changed nodes =",
        len(
            changed1
        ),
    )

    print(
        "DumpV0 max absolute correction =",
        f"{float(dumpV0_change.max()):.17e}",
    )

    print(
        "DumpV1 max absolute correction =",
        f"{float(dumpV1_change.max()):.17e}",
    )

    np.save(
        output
        /
        "solid_mass_base.npy",
        solid_mass_base,
    )

    np.save(
        output
        /
        "solid_mass_coupled.npy",
        solid_mass_coupled,
    )

    np.save(
        output
        /
        "solid_inverse_mass_coupled.npy",
        solid_inverse_mass,
    )

    np.save(
        output
        /
        "pml_mass_base.npy",
        pml_mass_base,
    )

    np.save(
        output
        /
        "pml_mass_coupled.npy",
        pml_mass_coupled,
    )

    np.save(
        output
        /
        "pml_dumpV0_coupled.npy",
        pml_dumpV0_coupled,
    )

    np.save(
        output
        /
        "pml_dumpV1_coupled.npy",
        pml_dumpV1_coupled,
    )

    np.save(
        output
        /
        "interface_combined_mass.npy",
        interface_mass,
    )

    checks = {
        "interface_count":
            len(
                solid_interface
            )
            ==
            22657,

        "solid_outside_unchanged":
            solid_outside_error
            ==
            0.0,

        "pml_outside_unchanged":
            pml_outside_error
            ==
            0.0,

        "solid_interface_exact":
            interface_solid_error
            ==
            0.0,

        "pml_interface_exact":
            interface_pml_error
            ==
            0.0,

        "solid_inverse_mass_finite":
            bool(
                np.isfinite(
                    solid_inverse_mass
                ).all()
            ),

        "pml_dumpV_finite":
            bool(
                np.isfinite(
                    pml_dumpV0_coupled
                ).all()
                and
                np.isfinite(
                    pml_dumpV1_coupled
                ).all()
            ),

        "interface_dumpmass_zero":
            bool(
                np.all(
                    pml_dumpmass[
                        pml_interface
                    ]
                    ==
                    0.0
                )
            ),

        "dumpV0_interface_invariant":
            len(
                changed0
            )
            ==
            0,

        "dumpV1_changed_exact_interface":
            np.array_equal(
                np.sort(
                    changed1
                ),
                np.sort(
                    pml_interface
                ),
            ),
    }

    print()

    print(
        "checks =",
        checks,
    )

    summary = {
        "solid_nodes":
            int(
                len(
                    solid_mass_base
                )
            ),

        "pml_nodes":
            int(
                len(
                    pml_mass_base
                )
            ),

        "interface_pairs":
            int(
                len(
                    solid_interface
                )
            ),

        "solid_base_mass_minmax":
            [
                float(
                    solid_mass_base.min()
                ),
                float(
                    solid_mass_base.max()
                ),
            ],

        "pml_base_mass_minmax":
            [
                float(
                    pml_mass_base.min()
                ),
                float(
                    pml_mass_base.max()
                ),
            ],

        "interface_mass_minmax":
            [
                float(
                    interface_mass.min()
                ),
                float(
                    interface_mass.max()
                ),
            ],

        "dumpV0_changed_nodes":
            int(
                len(
                    changed0
                )
            ),

        "dumpV1_changed_nodes":
            int(
                len(
                    changed1
                )
            ),

        "checks":
            checks,
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
        checks.values()
    ):
        print(
            "RESULT = PASS_REAL_S43_COUPLED_MASS"
        )
    else:
        print(
            "RESULT = FAIL_REAL_S43_COUPLED_MASS"
        )


if __name__ == "__main__":
    main()
