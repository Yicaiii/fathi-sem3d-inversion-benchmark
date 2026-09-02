import argparse
import json
import re
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


def parse_native_dt(path):
    text = Path(path).read_text(
        encoding="utf-8",
        errors="replace",
    )

    m = re.search(
        r"Time step size:\s*([0-9Ee+\-.]+)",
        text,
    )

    if not m:
        raise RuntimeError(
            f"cannot recover native dt from {path}"
        )

    dt = float(m.group(1))

    n = re.search(
        r"Number of time steps:\s*(\d+)",
        text,
    )

    nt = int(
        n.group(1)
    ) if n else None

    return dt, nt


def trilinear_sample(
    field,
    xyz,
    x_min,
    x_max,
    y_min,
    y_max,
    z_min,
    z_max,
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
        x_min,
        x_max,
    )

    y = np.clip(
        xyz[:, 1],
        y_min,
        y_max,
    )

    z = np.clip(
        xyz[:, 2],
        z_min,
        z_max,
    )

    fx = (
        (x - x_min)
        /
        (x_max - x_min)
        *
        (nx - 1)
    )

    fy = (
        (y - y_min)
        /
        (y_max - y_min)
        *
        (ny - 1)
    )

    fz = (
        (z - z_min)
        /
        (z_max - z_min)
        *
        (nz - 1)
    )

    ix0 = np.floor(fx).astype(np.int64)
    iy0 = np.floor(fy).astype(np.int64)
    iz0 = np.floor(fz).astype(np.int64)

    ix1 = np.minimum(ix0 + 1, nx - 1)
    iy1 = np.minimum(iy0 + 1, ny - 1)
    iz1 = np.minimum(iz0 + 1, nz - 1)

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

    c00 = c000 * (1.0 - tx) + c001 * tx
    c01 = c010 * (1.0 - tx) + c011 * tx
    c10 = c100 * (1.0 - tx) + c101 * tx
    c11 = c110 * (1.0 - tx) + c111 * tx

    c0 = c00 * (1.0 - ty) + c01 * ty
    c1 = c10 * (1.0 - ty) + c11 * ty

    return c0 * (1.0 - tz) + c1 * tz


def dump_stress(alpha, dt):
    d1 = 1.0 / (
        1.0
        +
        0.5
        *
        dt
        *
        alpha
    )

    d0 = (
        1.0
        -
        0.5
        *
        dt
        *
        alpha
    ) * d1

    return d0, d1


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--config", required=True)
    parser.add_argument("--mesh", required=True)
    parser.add_argument("--strict-log", required=True)

    parser.add_argument("--kappa", required=True)
    parser.add_argument("--mu", required=True)
    parser.add_argument("--density", required=True)

    parser.add_argument("--gll", required=True)
    parser.add_argument("--weights", required=True)
    parser.add_argument("--row-jacobian", required=True)

    parser.add_argument("--pml-element-ids", required=True)
    parser.add_argument("--pml-connectivity", required=True)
    parser.add_argument("--pml-region-code", required=True)
    parser.add_argument("--pml-xyz", required=True)

    parser.add_argument("--output", required=True)

    args = parser.parse_args()

    cfg = json.loads(
        Path(args.config).read_text(
            encoding="utf-8"
        )
    )

    output = Path(args.output)

    output.mkdir(
        parents=True,
        exist_ok=True,
    )

    dt, nt = parse_native_dt(
        args.strict_log
    )

    gll = np.asarray(
        np.load(args.gll),
        dtype=np.float64,
    )

    weights = np.asarray(
        np.load(args.weights),
        dtype=np.float64,
    )

    if len(gll) != 3:
        raise RuntimeError(
            f"expected ngll=3, got {len(gll)}"
        )

    pml_element_ids = np.asarray(
        np.load(args.pml_element_ids),
        dtype=np.int64,
    )

    pml_conn = np.asarray(
        np.load(args.pml_connectivity),
        dtype=np.int64,
    )

    region_code = np.asarray(
        np.load(args.pml_region_code),
        dtype=np.uint8,
    )

    pml_xyz = np.asarray(
        np.load(args.pml_xyz),
        dtype=np.float64,
    )

    if len(pml_element_ids) != len(region_code):
        raise RuntimeError(
            "PML element/region length mismatch"
        )

    if pml_conn.shape != (
        len(pml_element_ids),
        27,
    ):
        raise RuntimeError(
            f"unexpected connectivity shape {pml_conn.shape}"
        )

    with h5py.File(
        args.mesh,
        "r",
    ) as f:

        mesh_nodes = np.asarray(
            f["local_nodes"][...],
            dtype=np.float64,
        )

        mesh_elements = np.rint(
            f["elements"][...]
        ).astype(np.int64)

        mesh_material = np.rint(
            f["material"][...]
        ).astype(np.int64)

    corner_xyz = mesh_nodes[
        mesh_elements[
            pml_element_ids
        ]
    ]

    elem_min = corner_xyz.min(
        axis=1
    )

    elem_max = corner_xyz.max(
        axis=1
    )

    hvec = elem_max - elem_min

    if not np.allclose(
        hvec,
        hvec[0],
        atol=1e-12,
        rtol=0.0,
    ):
        raise RuntimeError(
            "nonuniform PML element geometry"
        )

    h = float(
        hvec[0, 0]
    )

    if not np.allclose(
        hvec,
        h,
        atol=1e-12,
        rtol=0.0,
    ):
        raise RuntimeError(
            "non-cubic PML elements"
        )

    jac = (
        h
        /
        2.0
    ) ** 3

    row_jac = np.asarray(
        np.load(args.row_jacobian),
        dtype=np.float64,
    )

    physical_jac_unique = np.unique(
        row_jac
    )

    jac_match = bool(
        np.allclose(
            physical_jac_unique,
            jac,
            atol=1e-15,
            rtol=0.0,
        )
    )

    offsets = []

    for i in range(3):
        for j in range(3):
            for k in range(3):
                offsets.append(
                    (
                        i,
                        j,
                        k,
                    )
                )

    offsets = np.asarray(
        offsets,
        dtype=np.int64,
    )

    q = gll[
        offsets
    ]

    local_xyz = (
        elem_min[:, None, :]
        +
        0.5
        *
        (
            q[None, :, :]
            +
            1.0
        )
        *
        (
            elem_max
            -
            elem_min
        )[:, None, :]
    )

    flat_xyz = local_xyz.reshape(
        -1,
        3,
    )

    kappa_grid = read_samples(
        args.kappa
    )

    mu_grid = read_samples(
        args.mu
    )

    rho_grid = read_samples(
        args.density
    )

    material_cfg = cfg[
        "material_grid"
    ]

    x_min = float(
        cfg["domain"]["x_min_m"]
    )

    x_max = float(
        cfg["domain"]["x_max_m"]
    )

    y_min = float(
        cfg["domain"]["y_min_m"]
    )

    y_max = float(
        cfg["domain"]["y_max_m"]
    )

    z_min = float(
        cfg["domain"]["z_min_m"]
    )

    z_max = float(
        cfg["domain"]["z_max_m"]
    )

    kappa = trilinear_sample(
        kappa_grid,
        flat_xyz,
        x_min,
        x_max,
        y_min,
        y_max,
        z_min,
        z_max,
    ).reshape(
        len(pml_element_ids),
        27,
    )

    mu = trilinear_sample(
        mu_grid,
        flat_xyz,
        x_min,
        x_max,
        y_min,
        y_max,
        z_min,
        z_max,
    ).reshape(
        len(pml_element_ids),
        27,
    )

    rho = trilinear_sample(
        rho_grid,
        flat_xyz,
        x_min,
        x_max,
        y_min,
        y_max,
        z_min,
        z_max,
    ).reshape(
        len(pml_element_ids),
        27,
    )

    lam = (
        kappa
        -
        (
            2.0
            /
            3.0
        )
        *
        mu
    )

    vp = np.sqrt(
        (
            lam
            +
            2.0
            *
            mu
        )
        /
        rho
    )

    pml_cfg = cfg[
        "sem3d_mesh"
    ][
        "pml"
    ]

    Apow = float(
        pml_cfg[
            "amplitude"
        ][
            "value"
        ]
    )

    npow = int(
        pml_cfg[
            "polynomial_order"
        ]
    )

    x_positions = pml_cfg[
        "x"
    ][
        "positions_m"
    ]

    x_widths = pml_cfg[
        "x"
    ][
        "widths_m"
    ]

    y_positions = pml_cfg[
        "y"
    ][
        "positions_m"
    ]

    y_widths = pml_cfg[
        "y"
    ][
        "widths_m"
    ]

    z_position = float(
        pml_cfg[
            "bottom"
        ][
            "position_m"
        ]
    )

    z_width = float(
        pml_cfg[
            "bottom"
        ][
            "width_m"
        ]
    )

    alpha = np.zeros(
        (
            len(pml_element_ids),
            27,
            3,
        ),
        dtype=np.float64,
    )

    def apply_direction(
        mask,
        direction,
        position,
        width,
    ):
        if not np.any(mask):
            return

        ri = (
            (
                local_xyz[
                    mask,
                    :,
                    direction,
                ]
                -
                position
            )
            /
            width
        )

        if np.min(ri) < -1e-12:
            raise RuntimeError(
                "negative PML normalized coordinate"
            )

        if np.max(ri) > 1.0 + 1e-12:
            raise RuntimeError(
                "PML normalized coordinate exceeds 1"
            )

        alpha[
            mask,
            :,
            direction,
        ] = (
            Apow
            *
            vp[
                mask,
                :,
            ]
            *
            (
                1.0
                /
                abs(width)
            )
            *
            ri ** npow
        )

    left_x = (
        region_code & 1
    ) != 0

    right_x = (
        region_code & 2
    ) != 0

    left_y = (
        region_code & 4
    ) != 0

    right_y = (
        region_code & 8
    ) != 0

    bottom_z = (
        region_code & 16
    ) != 0

    if np.any(
        left_x
        &
        right_x
    ):
        raise RuntimeError(
            "PML element active on both x sides"
        )

    if np.any(
        left_y
        &
        right_y
    ):
        raise RuntimeError(
            "PML element active on both y sides"
        )

    apply_direction(
        left_x,
        0,
        float(x_positions[0]),
        float(x_widths[0]),
    )

    apply_direction(
        right_x,
        0,
        float(x_positions[2]),
        float(x_widths[2]),
    )

    apply_direction(
        left_y,
        1,
        float(y_positions[0]),
        float(y_widths[0]),
    )

    apply_direction(
        right_y,
        1,
        float(y_positions[2]),
        float(y_widths[2]),
    )

    apply_direction(
        bottom_z,
        2,
        z_position,
        z_width,
    )

    if np.any(
        alpha < -1e-14
    ):
        raise RuntimeError(
            "negative alpha detected"
        )

    alpha = np.maximum(
        alpha,
        0.0,
    )

    dumpS0 = np.empty_like(
        alpha
    )

    dumpS1 = np.empty_like(
        alpha
    )

    for direction in range(3):
        d0, d1 = dump_stress(
            alpha[:, :, direction],
            dt,
        )

        dumpS0[:, :, direction] = d0
        dumpS1[:, :, direction] = d1

    local_weight = np.empty(
        27,
        dtype=np.float64,
    )

    n = 0

    for i in range(3):
        for j in range(3):
            for k in range(3):

                local_weight[n] = (
                    weights[i]
                    *
                    weights[j]
                    *
                    weights[k]
                    *
                    jac
                )

                n += 1

    local_mass = (
        rho
        *
        local_weight[
            None,
            :,
        ]
    )

    node_count = int(
        pml_conn.max()
        +
        1
    )

    mass_global = np.zeros(
        node_count,
        dtype=np.float64,
    )

    np.add.at(
        mass_global,
        pml_conn.reshape(-1),
        local_mass.reshape(-1),
    )

    dumpmass_global = np.zeros(
        (
            node_count,
            3,
        ),
        dtype=np.float64,
    )

    for direction in range(3):
        local_dumpmass = (
            0.5
            *
            local_mass
            *
            alpha[
                :,
                :,
                direction,
            ]
            *
            dt
        )

        np.add.at(
            dumpmass_global[
                :,
                direction,
            ],
            pml_conn.reshape(-1),
            local_dumpmass.reshape(-1),
        )

    denom = (
        mass_global[:, None]
        +
        dumpmass_global
    )

    if np.any(
        denom <= 0.0
    ):
        raise RuntimeError(
            "non-positive PML velocity denominator"
        )

    dumpV1 = (
        1.0
        /
        denom
    )

    dumpV0 = (
        mass_global[:, None]
        -
        dumpmass_global
    ) * dumpV1

    material_pml = mesh_material[
        pml_element_ids
    ]

    material_region_map = {}

    material_region_one_to_one = True

    for material_id in np.unique(
        material_pml
    ):
        values = np.unique(
            region_code[
                material_pml
                ==
                material_id
            ]
        )

        material_region_map[
            str(
                int(
                    material_id
                )
            )
        ] = [
            int(x)
            for x in values
        ]

        if len(values) != 1:
            material_region_one_to_one = False

    summary = {
        "dt": float(dt),
        "nt": nt,
        "Apow": Apow,
        "npow": npow,
        "element_spacing_m": h,
        "jacobian": jac,
        "physical_jacobian_match": jac_match,
        "pml_elements": int(
            len(pml_element_ids)
        ),
        "pml_nodes": node_count,
        "material_region_map": material_region_map,
        "material_region_one_to_one": material_region_one_to_one,
        "lambda_minmax": [
            float(lam.min()),
            float(lam.max()),
        ],
        "mu_minmax": [
            float(mu.min()),
            float(mu.max()),
        ],
        "density_minmax": [
            float(rho.min()),
            float(rho.max()),
        ],
        "vp_minmax": [
            float(vp.min()),
            float(vp.max()),
        ],
        "alpha_minmax": {
            "x": [
                float(
                    alpha[:, :, 0].min()
                ),
                float(
                    alpha[:, :, 0].max()
                ),
            ],
            "y": [
                float(
                    alpha[:, :, 1].min()
                ),
                float(
                    alpha[:, :, 1].max()
                ),
            ],
            "z": [
                float(
                    alpha[:, :, 2].min()
                ),
                float(
                    alpha[:, :, 2].max()
                ),
            ],
        },
        "dumpS0_minmax": [
            float(dumpS0.min()),
            float(dumpS0.max()),
        ],
        "dumpS1_minmax": [
            float(dumpS1.min()),
            float(dumpS1.max()),
        ],
        "mass_global_minmax": [
            float(mass_global.min()),
            float(mass_global.max()),
        ],
        "dumpmass_global_minmax": [
            float(dumpmass_global.min()),
            float(dumpmass_global.max()),
        ],
        "dumpV0_minmax": [
            float(dumpV0.min()),
            float(dumpV0.max()),
        ],
        "dumpV1_minmax": [
            float(dumpV1.min()),
            float(dumpV1.max()),
        ],
    }

    np.save(
        output
        /
        "pml_lambda_gll.npy",
        lam,
    )

    np.save(
        output
        /
        "pml_mu_gll.npy",
        mu,
    )

    np.save(
        output
        /
        "pml_density_gll.npy",
        rho,
    )

    np.save(
        output
        /
        "pml_vp_gll.npy",
        vp,
    )

    np.save(
        output
        /
        "pml_alpha_xyz.npy",
        alpha,
    )

    np.save(
        output
        /
        "pml_dumpS0_xyz.npy",
        dumpS0,
    )

    np.save(
        output
        /
        "pml_dumpS1_xyz.npy",
        dumpS1,
    )

    np.save(
        output
        /
        "pml_local_mass.npy",
        local_mass,
    )

    np.save(
        output
        /
        "pml_mass_global.npy",
        mass_global,
    )

    np.save(
        output
        /
        "pml_dumpmass_global.npy",
        dumpmass_global,
    )

    np.save(
        output
        /
        "pml_dumpV0.npy",
        dumpV0,
    )

    np.save(
        output
        /
        "pml_dumpV1.npy",
        dumpV1,
    )

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

    checks = {
        "dt_positive":
            dt > 0.0,

        "nt_positive":
            nt is not None
            and
            nt > 0,

        "jacobian_match":
            jac_match,

        "material_region_one_to_one":
            material_region_one_to_one,

        "lambda_positive":
            bool(
                np.all(
                    lam > 0.0
                )
            ),

        "mu_positive":
            bool(
                np.all(
                    mu > 0.0
                )
            ),

        "density_positive":
            bool(
                np.all(
                    rho > 0.0
                )
            ),

        "alpha_nonnegative":
            bool(
                np.all(
                    alpha >= 0.0
                )
            ),

        "mass_positive":
            bool(
                np.all(
                    mass_global > 0.0
                )
            ),

        "dumpS_finite":
            bool(
                np.isfinite(
                    dumpS0
                ).all()
                and
                np.isfinite(
                    dumpS1
                ).all()
            ),

        "dumpV_finite":
            bool(
                np.isfinite(
                    dumpV0
                ).all()
                and
                np.isfinite(
                    dumpV1
                ).all()
            ),

        "pml_node_count_matches_topology":
            node_count
            ==
            len(
                pml_xyz
            ),
    }

    print(
        "============================================================"
    )
    print(
        "REAL S43 PML COEFFICIENTS"
    )
    print(
        "============================================================"
    )

    print(
        "dt =",
        f"{dt:.17e}",
    )

    print(
        "nt =",
        nt,
    )

    print(
        "Apow =",
        Apow,
    )

    print(
        "npow =",
        npow,
    )

    print(
        "element spacing =",
        h,
    )

    print(
        "Jacobian =",
        f"{jac:.17e}",
    )

    print(
        "physical Jacobian match =",
        jac_match,
    )

    print()
    print(
        "PML elements =",
        len(
            pml_element_ids
        ),
    )

    print(
        "PML nodes =",
        node_count,
    )

    print()
    print(
        "material -> region code =",
        material_region_map,
    )

    print(
        "material/region one-to-one =",
        material_region_one_to_one,
    )

    print()
    print(
        "lambda min/max =",
        summary[
            "lambda_minmax"
        ],
    )

    print(
        "mu min/max =",
        summary[
            "mu_minmax"
        ],
    )

    print(
        "density min/max =",
        summary[
            "density_minmax"
        ],
    )

    print(
        "Vp min/max =",
        summary[
            "vp_minmax"
        ],
    )

    print()
    print(
        "alpha x min/max =",
        summary[
            "alpha_minmax"
        ][
            "x"
        ],
    )

    print(
        "alpha y min/max =",
        summary[
            "alpha_minmax"
        ][
            "y"
        ],
    )

    print(
        "alpha z min/max =",
        summary[
            "alpha_minmax"
        ][
            "z"
        ],
    )

    print()
    print(
        "DumpS0 min/max =",
        summary[
            "dumpS0_minmax"
        ],
    )

    print(
        "DumpS1 min/max =",
        summary[
            "dumpS1_minmax"
        ],
    )

    print(
        "Mass global min/max =",
        summary[
            "mass_global_minmax"
        ],
    )

    print(
        "DumpMass global min/max =",
        summary[
            "dumpmass_global_minmax"
        ],
    )

    print(
        "DumpV0 min/max =",
        summary[
            "dumpV0_minmax"
        ],
    )

    print(
        "DumpV1 min/max =",
        summary[
            "dumpV1_minmax"
        ],
    )

    print()
    print(
        "checks =",
        checks,
    )

    if all(
        checks.values()
    ):
        print(
            "RESULT = PASS_REAL_S43_PML_COEFFICIENT_RECONSTRUCTION"
        )
    else:
        print(
            "RESULT = FAIL_REAL_S43_PML_COEFFICIENT_RECONSTRUCTION"
        )


if __name__ == "__main__":
    main()
