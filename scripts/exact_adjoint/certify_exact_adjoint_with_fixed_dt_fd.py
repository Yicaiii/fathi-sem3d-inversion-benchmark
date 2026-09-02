import argparse
import json
import math
import re
from pathlib import Path

import h5py
import numpy as np


def decode(x):
    if isinstance(x, bytes):
        return x.decode(
            "utf-8",
            errors="replace",
        )

    if isinstance(x, np.bytes_):
        return bytes(x).decode(
            "utf-8",
            errors="replace",
        )

    return str(x)


def trapezoid_weights(t):
    t = np.asarray(
        t,
        dtype=np.float64,
    )

    if (
        t.ndim != 1
        or len(t) < 2
    ):
        raise RuntimeError(
            "invalid objective time grid"
        )

    if not np.all(
        np.diff(t) > 0.0
    ):
        raise RuntimeError(
            "objective time grid "
            "not increasing"
        )

    w = np.empty_like(t)

    w[0] = (
        t[1] - t[0]
    ) / 2.0

    w[-1] = (
        t[-1] - t[-2]
    ) / 2.0

    if len(t) > 2:
        w[1:-1] = (
            t[2:]
            -
            t[:-2]
        ) / 2.0

    return w


def receiver_groups(h5):
    pat = re.compile(
        r"^station_[0-9]+$"
    )

    rows = [
        k
        for k in sorted(
            h5.keys()
        )
        if (
            pat.fullmatch(k)
            and
            isinstance(
                h5[k],
                h5py.Group,
            )
        )
    ]

    if not rows:
        raise RuntimeError(
            "no station groups"
        )

    return rows


def read_trace(h5, key):
    if key not in h5:
        raise RuntimeError(
            f"trace dataset missing: {key}"
        )

    a = np.asarray(
        h5[key][...],
        dtype=np.float64,
    )

    if (
        a.ndim != 2
        or
        a.shape[1] < 4
    ):
        raise RuntimeError(
            f"{key}: unexpected shape "
            f"{a.shape}"
        )

    t = a[:, 0]

    u = a[:, 1:4]

    if (
        len(t) < 2
        or
        not np.all(
            np.diff(t) > 0.0
        )
    ):
        raise RuntimeError(
            f"{key}: invalid time grid"
        )

    return t, u


def interp_xyz(
    query,
    native_time,
    native_values,
):
    tol = max(
        1.0e-13,
        1.0e-9
        *
        float(
            np.median(
                np.diff(
                    native_time
                )
            )
        ),
    )

    if (
        query[0]
        <
        native_time[0]
        -
        tol
        or
        query[-1]
        >
        native_time[-1]
        +
        tol
    ):
        raise RuntimeError(
            "objective time grid outside "
            "fixed-dt native trace"
        )

    out = np.empty(
        (
            len(query),
            3,
        ),
        dtype=np.float64,
    )

    for c in range(3):
        out[:, c] = np.interp(
            query,
            native_time,
            native_values[:, c],
        )

    return out


def relerr(value, reference):
    return (
        abs(
            value - reference
        )
        /
        max(
            abs(reference),
            np.finfo(
                np.float64
            ).tiny,
        )
    )


def main():
    p = argparse.ArgumentParser()

    p.add_argument(
        "--residual",
        required=True,
    )

    p.add_argument(
        "--final",
        required=True,
    )

    p.add_argument(
        "--manifest",
        required=True,
    )

    p.add_argument(
        "--output",
        required=True,
    )

    args = p.parse_args()

    residual_path = Path(
        args.residual
    )

    final_dir = Path(
        args.final
    )

    manifest_path = Path(
        args.manifest
    )

    output = Path(
        args.output
    )

    output.mkdir(
        parents=True,
        exist_ok=True,
    )

    final = json.loads(
        (
            final_dir
            /
            "summary.json"
        ).read_text(
            encoding="utf-8"
        )
    )

    manifest = json.loads(
        manifest_path.read_text(
            encoding="utf-8"
        )
    )

    cases = {}

    for row in manifest[
        "cases"
    ]:
        key = (
            row[
                "component"
            ],
            row[
                "sign"
            ],
        )

        cases[key] = Path(
            row[
                "workspace"
            ]
        )

    required = {
        ("lambda", "plus"),
        ("lambda", "minus"),
        ("mu", "plus"),
        ("mu", "minus"),
    }

    if set(
        cases.keys()
    ) != required:
        raise RuntimeError(
            f"unexpected fixed-dt cases: "
            f"{cases.keys()}"
        )

    for key, ws in cases.items():
        marker = (
            ws
            /
            "FIXED_DT_RUN_PASS"
        )

        if not marker.is_file():
            raise RuntimeError(
                f"{key}: missing "
                f"FIXED_DT_RUN_PASS"
            )

    with h5py.File(
        residual_path,
        "r",
    ) as rh5:

        groups = receiver_groups(
            rh5
        )

        if len(groups) != 225:
            raise RuntimeError(
                f"station count = "
                f"{len(groups)}"
            )

        g0 = rh5[
            groups[0]
        ]

        baseline_trace = Path(
            decode(
                g0.attrs[
                    "sim_file"
                ]
            )
        )

    print(
        "============================================================"
    )

    print(
        "FIXED-DT CERTIFICATION CONTRACT"
    )

    print(
        "============================================================"
    )

    print(
        "stations =",
        len(groups),
    )

    print(
        "baseline trace =",
        baseline_trace,
    )

    print(
        "fixed-dt manifest baseline dt =",
        f'{float(manifest["baseline_dt"]):.17e}',
    )

    print(
        "fixed-dt manifest baseline count =",
        manifest[
            "baseline_sample_count"
        ],
    )

    results = {}

    for component in [
        "lambda",
        "mu",
    ]:
        info = final[
            "components"
        ][
            component
        ]

        step = float(
            info[
                "step_pa"
            ]
        )

        adj = float(
            info[
                "exact_adjoint_per_pa"
            ]
        )

        old_adaptive_fd = float(
            info[
                "frozen_fd_per_pa"
            ]
        )

        plus_ws = cases[
            (
                component,
                "plus",
            )
        ]

        minus_ws = cases[
            (
                component,
                "minus",
            )
        ]

        plus_trace = (
            plus_ws
            /
            "traces"
            /
            "capteurs.0000.h5"
        )

        minus_trace = (
            minus_ws
            /
            "traces"
            /
            "capteurs.0000.h5"
        )

        print()
        print(
            "============================================================"
        )

        print(
            component.upper()
        )

        print(
            "============================================================"
        )

        print(
            "step Pa =",
            f"{step:.17e}",
        )

        print(
            "plus trace =",
            plus_trace,
        )

        print(
            "minus trace =",
            minus_trace,
        )

        J0 = 0.0
        Jp = 0.0
        Jm = 0.0

        D_tangent = 0.0
        D_center = 0.0

        D_tangent_xyz = np.zeros(
            3,
            dtype=np.float64,
        )

        D_center_xyz = np.zeros(
            3,
            dtype=np.float64,
        )

        center_sq = 0.0
        anti_sq = 0.0

        max_time_grid_error = 0.0

        endpoint_hold_count = 0

        with (
            h5py.File(
                residual_path,
                "r",
            ) as rh5,
            h5py.File(
                baseline_trace,
                "r",
            ) as bh5,
            h5py.File(
                plus_trace,
                "r",
            ) as ph5,
            h5py.File(
                minus_trace,
                "r",
            ) as mh5,
        ):

            for name in groups:
                g = rh5[
                    name
                ]

                key = decode(
                    g.attrs[
                        "sim_dataset"
                    ]
                )

                tq = np.asarray(
                    g[
                        "time_true_grid"
                    ][...],
                    dtype=np.float64,
                )

                r0 = np.asarray(
                    g[
                        "residual_forward_time_xyz"
                    ][...],
                    dtype=np.float64,
                )

                tb, ub_native = read_trace(
                    bh5,
                    key,
                )

                tp, up_native = read_trace(
                    ph5,
                    key,
                )

                tm, um_native = read_trace(
                    mh5,
                    key,
                )

                if not (
                    len(tb)
                    ==
                    len(tp)
                    ==
                    len(tm)
                ):
                    raise RuntimeError(
                        f"{key}: fixed-dt "
                        "sample-count mismatch"
                    )

                local_time_error = max(
                    float(
                        np.max(
                            np.abs(
                                tb - tp
                            )
                        )
                    ),
                    float(
                        np.max(
                            np.abs(
                                tb - tm
                            )
                        )
                    ),
                    float(
                        np.max(
                            np.abs(
                                tp - tm
                            )
                        )
                    ),
                )

                max_time_grid_error = max(
                    max_time_grid_error,
                    local_time_error,
                )

                endpoint_hold_count += int(
                    np.count_nonzero(
                        tq > tb[-1]
                    )
                )

                endpoint_hold_count += int(
                    np.count_nonzero(
                        tq > tp[-1]
                    )
                )

                endpoint_hold_count += int(
                    np.count_nonzero(
                        tq > tm[-1]
                    )
                )

                ub = interp_xyz(
                    tq,
                    tb,
                    ub_native,
                )

                up = interp_xyz(
                    tq,
                    tp,
                    up_native,
                )

                um = interp_xyz(
                    tq,
                    tm,
                    um_native,
                )

                delta_plus = (
                    up - ub
                )

                delta_minus = (
                    um - ub
                )

                rp = (
                    r0
                    +
                    delta_plus
                )

                rm = (
                    r0
                    +
                    delta_minus
                )

                tangent = (
                    up
                    -
                    um
                ) / (
                    2.0
                    *
                    step
                )

                center = (
                    (
                        up
                        +
                        um
                    )
                    /
                    2.0
                    -
                    ub
                )

                anti = (
                    up
                    -
                    um
                ) / 2.0

                w = trapezoid_weights(
                    tq
                )

                W = w[
                    :,
                    None
                ]

                J0 += (
                    0.5
                    *
                    float(
                        np.sum(
                            W
                            *
                            r0
                            *
                            r0
                        )
                    )
                )

                Jp += (
                    0.5
                    *
                    float(
                        np.sum(
                            W
                            *
                            rp
                            *
                            rp
                        )
                    )
                )

                Jm += (
                    0.5
                    *
                    float(
                        np.sum(
                            W
                            *
                            rm
                            *
                            rm
                        )
                    )
                )

                tangent_xyz = np.sum(
                    W
                    *
                    r0
                    *
                    tangent,
                    axis=0,
                )

                center_xyz = np.sum(
                    W
                    *
                    center
                    *
                    tangent,
                    axis=0,
                )

                D_tangent_xyz += (
                    tangent_xyz
                )

                D_center_xyz += (
                    center_xyz
                )

                D_tangent += float(
                    np.sum(
                        tangent_xyz
                    )
                )

                D_center += float(
                    np.sum(
                        center_xyz
                    )
                )

                center_sq += float(
                    np.sum(
                        W
                        *
                        center
                        *
                        center
                    )
                )

                anti_sq += float(
                    np.sum(
                        W
                        *
                        anti
                        *
                        anti
                    )
                )

        D_identity = (
            D_tangent
            +
            D_center
        )

        D_fd = (
            Jp
            -
            Jm
        ) / (
            2.0
            *
            step
        )

        plus_one_sided = (
            Jp
            -
            J0
        ) / step

        minus_one_sided = (
            J0
            -
            Jm
        ) / step

        adj_fd_ratio = (
            adj
            /
            D_fd
            if D_fd != 0.0
            else math.nan
        )

        tangent_adj_ratio = (
            D_tangent
            /
            adj
            if adj != 0.0
            else math.nan
        )

        fd_vs_adj = relerr(
            D_fd,
            adj,
        )

        tangent_vs_adj = relerr(
            D_tangent,
            adj,
        )

        identity_error = relerr(
            D_identity,
            D_fd,
        )

        center_norm = float(
            np.sqrt(
                max(
                    center_sq,
                    0.0,
                )
            )
        )

        anti_norm = float(
            np.sqrt(
                max(
                    anti_sq,
                    0.0,
                )
            )
        )

        center_over_anti = (
            center_norm
            /
            max(
                anti_norm,
                np.finfo(
                    np.float64
                ).tiny,
            )
        )

        same_sign = bool(
            (
                adj > 0.0
                and
                D_fd > 0.0
            )
            or
            (
                adj < 0.0
                and
                D_fd < 0.0
            )
        )

        agreement_5pct = bool(
            same_sign
            and
            fd_vs_adj
            <=
            5.0e-2
        )

        print()
        print(
            "===== TIME GRID ====="
        )

        print(
            "max baseline/plus/minus "
            "native-time mismatch =",
            f"{max_time_grid_error:.17e}",
        )

        print(
            "objective endpoint-hold "
            "sample count =",
            endpoint_hold_count,
        )

        print()
        print(
            "===== OBJECTIVES ====="
        )

        print(
            "J0 =",
            f"{J0:.17e}",
        )

        print(
            "J_plus fixed-dt =",
            f"{Jp:.17e}",
        )

        print(
            "J_minus fixed-dt =",
            f"{Jm:.17e}",
        )

        print(
            "J_plus - J0 =",
            f"{Jp-J0:.17e}",
        )

        print(
            "J_minus - J0 =",
            f"{Jm-J0:.17e}",
        )

        print()
        print(
            "===== CENTRAL FD ====="
        )

        print(
            "FIXED-DT FD / Pa =",
            f"{D_fd:.17e}",
        )

        print(
            "old adaptive-CFL FD / Pa =",
            f"{old_adaptive_fd:.17e}",
        )

        print(
            "exact adjoint / Pa =",
            f"{adj:.17e}",
        )

        print(
            "ADJ / fixed-DT FD ratio =",
            f"{adj_fd_ratio:.17e}",
        )

        print(
            "fixed-DT FD vs ADJ "
            "relative error =",
            f"{fd_vs_adj:.17e}",
        )

        print(
            "same sign =",
            same_sign,
        )

        print(
            "<=5% agreement =",
            agreement_5pct,
        )

        print()
        print(
            "===== TRACE-LEVEL IDENTITY ====="
        )

        print(
            "baseline residual dot "
            "fixed-dt central tangent =",
            f"{D_tangent:.17e}",
        )

        print(
            "center correction =",
            f"{D_center:.17e}",
        )

        print(
            "identity sum =",
            f"{D_identity:.17e}",
        )

        print(
            "identity relative error =",
            f"{identity_error:.17e}",
        )

        print(
            "fixed-dt trace tangent / "
            "exact adjoint =",
            f"{tangent_adj_ratio:.17e}",
        )

        print(
            "fixed-dt trace tangent vs "
            "ADJ relative error =",
            f"{tangent_vs_adj:.17e}",
        )

        print(
            "center / antisymmetric "
            "trace norm =",
            f"{center_over_anti:.17e}",
        )

        print()
        print(
            "===== ONE-SIDED CHECK ====="
        )

        print(
            "plus one-sided derivative =",
            f"{plus_one_sided:.17e}",
        )

        print(
            "minus one-sided derivative =",
            f"{minus_one_sided:.17e}",
        )

        print()
        print(
            "===== XYZ CONTRIBUTIONS ====="
        )

        print(
            "tangent xyz =",
            [
                float(x)
                for x in D_tangent_xyz
            ],
        )

        print(
            "center xyz =",
            [
                float(x)
                for x in D_center_xyz
            ],
        )

        results[
            component
        ] = {
            "step_pa":
                step,

            "J0":
                J0,

            "J_plus_fixed_dt":
                Jp,

            "J_minus_fixed_dt":
                Jm,

            "fixed_dt_fd_per_pa":
                D_fd,

            "old_adaptive_fd_per_pa":
                old_adaptive_fd,

            "exact_adjoint_per_pa":
                adj,

            "adj_over_fixed_fd":
                adj_fd_ratio,

            "fixed_fd_vs_adjoint_relative_error":
                fd_vs_adj,

            "same_sign":
                same_sign,

            "agreement_within_5pct":
                agreement_5pct,

            "trace_tangent":
                D_tangent,

            "center_correction":
                D_center,

            "identity_sum":
                D_identity,

            "identity_relative_error":
                identity_error,

            "trace_tangent_vs_adjoint_relative_error":
                tangent_vs_adj,

            "center_over_antisymmetric_trace_norm":
                center_over_anti,

            "plus_one_sided":
                plus_one_sided,

            "minus_one_sided":
                minus_one_sided,

            "max_native_time_grid_mismatch":
                max_time_grid_error,

            "endpoint_hold_count":
                endpoint_hold_count,

            "tangent_xyz":
                D_tangent_xyz.tolist(),

            "center_xyz":
                D_center_xyz.tolist(),
        }

    print()
    print(
        "============================================================"
    )

    print(
        "FINAL FIXED-DT SCIENTIFIC VERDICT"
    )

    print(
        "============================================================"
    )

    contract_pass = all(
        r[
            "max_native_time_grid_mismatch"
        ]
        <=
        1.0e-15
        and
        r[
            "endpoint_hold_count"
        ]
        ==
        0
        and
        r[
            "identity_relative_error"
        ]
        <=
        1.0e-8
        for r in results.values()
    )

    numerical_pass = all(
        r[
            "agreement_within_5pct"
        ]
        for r in results.values()
    )

    print(
        "fixed-dt trace/objective "
        "contract PASS =",
        contract_pass,
    )

    print(
        "lambda exact adjoint =",
        f'{results["lambda"]["exact_adjoint_per_pa"]:.17e}',
    )

    print(
        "lambda fixed-dt FD =",
        f'{results["lambda"]["fixed_dt_fd_per_pa"]:.17e}',
    )

    print(
        "lambda relative error =",
        f'{results["lambda"]["fixed_fd_vs_adjoint_relative_error"]:.17e}',
    )

    print(
        "mu exact adjoint =",
        f'{results["mu"]["exact_adjoint_per_pa"]:.17e}',
    )

    print(
        "mu fixed-dt FD =",
        f'{results["mu"]["fixed_dt_fd_per_pa"]:.17e}',
    )

    print(
        "mu relative error =",
        f'{results["mu"]["fixed_fd_vs_adjoint_relative_error"]:.17e}',
    )

    print(
        "both same sign and <=5% =",
        numerical_pass,
    )

    payload = {
        "contract_pass":
            contract_pass,

        "numerical_pass":
            numerical_pass,

        "components":
            results,
    }

    (
        output
        /
        "summary.json"
    ).write_text(
        json.dumps(
            payload,
            indent=2,
        )
        +
        "\n",
        encoding="utf-8",
    )

    if (
        contract_pass
        and
        numerical_pass
    ):
        print(
            "RESULT = "
            "PASS_FINAL_FIXED_DT_EXACT_ADJOINT_CERTIFICATION"
        )

    elif contract_pass:
        print(
            "RESULT = "
            "FAIL_FINAL_FIXED_DT_EXACT_ADJOINT_NUMERICAL"
        )

    else:
        print(
            "RESULT = "
            "FAIL_FINAL_FIXED_DT_EXACT_ADJOINT_CONTRACT"
        )


if __name__ == "__main__":
    main()
