import argparse
import gc
import json
from pathlib import Path

import numpy as np

from scripts.exact_adjoint.build_real_s43_pml_coefficients import (
    dump_stress,
    read_samples,
    trilinear_sample,
)
from scripts.exact_adjoint.real_s43_global_operator import (
    load_global_data,
    material_tangent as global_material_tangent,
    material_vjp as global_material_vjp,
    random_state as random_global_state,
    state_dot as global_state_dot,
    state_norm as global_state_norm,
)
from scripts.exact_adjoint.real_s43_pml_operator import (
    forward_step as pml_forward_step,
    load_operator_data,
    material_coefficient_tangent,
    material_tangent as pml_material_tangent,
    material_vjp as pml_material_vjp,
    random_state as random_pml_state,
    state_dot as pml_state_dot,
    state_norm as pml_state_norm,
)


def relative_error(a, b):
    return float(
        abs(a - b)
        /
        max(
            abs(a),
            abs(b),
            np.finfo(np.float64).tiny,
        )
    )


def array_stats(value):
    value = np.asarray(value, dtype=np.float64)
    return {
        "support": int(np.count_nonzero(value)),
        "max_abs": float(np.max(np.abs(value))),
        "l2": float(np.linalg.norm(value.reshape(-1))),
    }


def node_support(delta, connectivity, node_count):
    support = np.zeros(node_count, dtype=bool)
    np.logical_or.at(
        support,
        connectivity.reshape(-1),
        np.asarray(delta).reshape(-1) != 0.0,
    )
    return int(np.count_nonzero(support))


def production_sample_xyz(local_xyz, region_code, cfg):
    """Reproduce build_prop_files.F90 PML boundary clamping."""
    out = np.array(local_xyz, copy=True, dtype=np.float64)
    pml = cfg["sem3d_mesh"]["pml"]

    def clamp(mask, direction, position, width):
        if not np.any(mask):
            return
        values = out[mask, :, direction]
        if width > 0.0:
            values[values > position] = position
        elif width < 0.0:
            values[values < position] = position
        out[mask, :, direction] = values

    clamp(
        (region_code & 1) != 0,
        0,
        float(pml["x"]["positions_m"][0]),
        float(pml["x"]["widths_m"][0]),
    )
    clamp(
        (region_code & 2) != 0,
        0,
        float(pml["x"]["positions_m"][2]),
        float(pml["x"]["widths_m"][2]),
    )
    clamp(
        (region_code & 4) != 0,
        1,
        float(pml["y"]["positions_m"][0]),
        float(pml["y"]["widths_m"][0]),
    )
    clamp(
        (region_code & 8) != 0,
        1,
        float(pml["y"]["positions_m"][2]),
        float(pml["y"]["widths_m"][2]),
    )
    clamp(
        (region_code & 16) != 0,
        2,
        float(pml["bottom"]["position_m"]),
        float(pml["bottom"]["width_m"]),
    )
    return out


def sample_lambda_mu(material_dir, sample_xyz, cfg, shape):
    material_dir = Path(material_dir)
    domain = cfg["domain"]
    bounds = (
        float(domain["x_min_m"]),
        float(domain["x_max_m"]),
        float(domain["y_min_m"]),
        float(domain["y_max_m"]),
        float(domain["z_min_m"]),
        float(domain["z_max_m"]),
    )
    flat_xyz = sample_xyz.reshape(-1, 3)
    kappa = trilinear_sample(
        read_samples(material_dir / "Mat_0_Kappa.h5"),
        flat_xyz,
        *bounds,
    ).reshape(shape)
    mu = trilinear_sample(
        read_samples(material_dir / "Mat_0_Mu.h5"),
        flat_xyz,
        *bounds,
    ).reshape(shape)
    return kappa - (2.0 / 3.0) * mu, mu


def candidate_sensitivity(
    workspace,
    sample_xyz,
    cfg,
    conn,
    pml,
    coupled_mass_dir,
):
    lam, mu = sample_lambda_mu(
        Path(workspace) / "mat" / "h5",
        sample_xyz,
        cfg,
        pml["lam"].shape,
    )
    dlam = lam - pml["lam"]
    dmu = mu - pml["mu"]

    l2m = lam + 2.0 * mu
    base_l2m = pml["lam"] + 2.0 * pml["mu"]
    vp = np.sqrt(l2m / pml["rho"])
    base_vp = np.sqrt(base_l2m / pml["rho"])
    geometry_factor = np.divide(
        pml["alpha"],
        base_vp[..., None],
        out=np.zeros_like(pml["alpha"]),
        where=base_vp[..., None] != 0.0,
    )
    d_alpha = geometry_factor * (vp - base_vp)[..., None]
    alpha = pml["alpha"] + d_alpha
    dS0_candidate, dS1_candidate = dump_stress(alpha, pml["dt"])

    base_dV0 = np.asarray(
        np.load(Path(coupled_mass_dir) / "pml_dumpV0_coupled.npy"),
        dtype=np.float64,
    )
    base_dV1 = np.asarray(
        np.load(Path(coupled_mass_dir) / "pml_dumpV1_coupled.npy"),
        dtype=np.float64,
    )
    velocity_mass = 0.5 * (1.0 + base_dV0) / base_dV1
    base_dumpmass = 0.5 * (1.0 - base_dV0) / base_dV1
    delta_dumpmass = np.zeros_like(base_dumpmass)
    local_scale = 0.5 * pml["local_mass"] * pml["dt"]
    for direction in range(3):
        np.add.at(
            delta_dumpmass[:, direction],
            conn.reshape(-1),
            (local_scale * d_alpha[..., direction]).reshape(-1),
        )
    dumpmass = base_dumpmass + delta_dumpmass
    candidate_dV1 = 1.0 / (velocity_mass + dumpmass)
    candidate_dV0 = (velocity_mass - dumpmass) * candidate_dV1

    return {
        "delta_lambda": {
            **array_stats(dlam),
            "unique_node_support": node_support(
                dlam, conn, pml["nn"]
            ),
        },
        "delta_mu": {
            **array_stats(dmu),
            "unique_node_support": node_support(
                dmu, conn, pml["nn"]
            ),
        },
        "delta_vp": array_stats(vp - base_vp),
        "delta_alpha": array_stats(d_alpha),
        "delta_dumpS0": array_stats(
            dS0_candidate - pml["dS0"]
        ),
        "delta_dumpS1": array_stats(
            dS1_candidate - pml["dS1"]
        ),
        "delta_dumpV0": array_stats(
            candidate_dV0 - base_dV0
        ),
        "delta_dumpV1": array_stats(
            candidate_dV1 - base_dV1
        ),
    }


def dot_test_record(lhs, rhs, b_norm, bt_norm):
    return {
        "lhs": float(lhs),
        "rhs": float(rhs),
        "relative_error": relative_error(lhs, rhs),
        "B_direction_norm": float(b_norm),
        "BT_seed_norm": float(bt_norm),
        "nondegenerate": bool(b_norm > 0.0 and bt_norm > 0.0),
    }


def pml_data_at_material(data, dlam, dmu, scale):
    """Rebuild the primal coefficients at a finite material perturbation."""
    out = dict(data)
    out["lam"] = data["lam"] + scale * dlam
    out["mu"] = data["mu"] + scale * dmu
    base_vp = np.sqrt(
        (data["lam"] + 2.0 * data["mu"]) / data["rho"]
    )
    vp = np.sqrt(
        (out["lam"] + 2.0 * out["mu"]) / data["rho"]
    )
    geometry_factor = np.divide(
        data["alpha"],
        base_vp[..., None],
        out=np.zeros_like(data["alpha"]),
        where=base_vp[..., None] != 0.0,
    )
    out["alpha"] = geometry_factor * vp[..., None]
    out["dS0"], out["dS1"] = dump_stress(
        out["alpha"], data["dt"]
    )
    velocity_mass = 0.5 * (1.0 + data["dV0"]) / data["dV1"]
    dumpmass = np.zeros_like(data["dV0"])
    local_scale = 0.5 * data["local_mass"] * data["dt"]
    ids = data["conn"].reshape(-1)
    for direction in range(3):
        np.add.at(
            dumpmass[:, direction],
            ids,
            (
                local_scale * out["alpha"][..., direction]
            ).reshape(-1),
        )
    out["dV1"] = 1.0 / (velocity_mass + dumpmass)
    out["dV0"] = (velocity_mass - dumpmass) * out["dV1"]
    return out


def finite_difference_record(data, state, dlam, dmu, tangent, step):
    plus = pml_forward_step(
        pml_data_at_material(data, dlam, dmu, step),
        state,
    )
    minus = pml_forward_step(
        pml_data_at_material(data, dlam, dmu, -step),
        state,
    )
    finite_difference = tuple(
        (positive - negative) / (2.0 * step)
        for positive, negative in zip(plus, minus)
    )
    residual = tuple(
        fd_value - tangent_value
        for fd_value, tangent_value in zip(finite_difference, tangent)
    )
    return {
        "central_step_pa": float(step),
        "finite_difference_norm": float(pml_state_norm(finite_difference)),
        "analytic_tangent_norm": float(pml_state_norm(tangent)),
        "relative_state_error": float(
            pml_state_norm(residual)
            /
            max(
                pml_state_norm(finite_difference),
                pml_state_norm(tangent),
                np.finfo(np.float64).tiny,
            )
        ),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--topology", required=True)
    parser.add_argument("--coefficients", required=True)
    parser.add_argument("--coupled-mass", required=True)
    parser.add_argument("--gll", required=True)
    parser.add_argument("--weights", required=True)
    parser.add_argument("--baseline-material", required=True)
    parser.add_argument("--workspace-manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--seed", type=int, default=20260829)
    args = parser.parse_args()

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    cfg = json.loads(Path(args.config).read_text(encoding="utf-8"))
    manifest = json.loads(
        Path(args.workspace_manifest).read_text(encoding="utf-8")
    )
    topology = Path(args.topology)
    conn = np.asarray(
        np.load(topology / "pml_connectivity_compact.npy"),
        dtype=np.int64,
    )
    xyz = np.asarray(
        np.load(topology / "pml_compact_xyz.npy"),
        dtype=np.float64,
    )
    region = np.asarray(
        np.load(topology / "pml_element_region_code.npy"),
        dtype=np.uint8,
    )
    local_xyz = xyz[conn]
    sample_xyz = production_sample_xyz(local_xyz, region, cfg)

    pml = load_operator_data(
        args.topology,
        args.coefficients,
        args.gll,
        args.weights,
        batch_size=args.batch_size,
    )
    pml["dV0"] = np.asarray(
        np.load(Path(args.coupled_mass) / "pml_dumpV0_coupled.npy"),
        dtype=np.float64,
    )
    pml["dV1"] = np.asarray(
        np.load(Path(args.coupled_mass) / "pml_dumpV1_coupled.npy"),
        dtype=np.float64,
    )

    baseline_lam, baseline_mu = sample_lambda_mu(
        args.baseline_material,
        sample_xyz,
        cfg,
        pml["lam"].shape,
    )
    baseline_reconstruction = {
        "lambda_max_abs_error": float(
            np.max(np.abs(baseline_lam - pml["lam"]))
        ),
        "mu_max_abs_error": float(
            np.max(np.abs(baseline_mu - pml["mu"]))
        ),
    }

    candidates = {}
    for case in manifest["cases"]:
        key = f"{case['component']}_{case['sign']}"
        candidates[key] = candidate_sensitivity(
            case["workspace"],
            sample_xyz,
            cfg,
            conn,
            pml,
            args.coupled_mass,
        )

    rng = np.random.default_rng(args.seed)
    pml_state = random_pml_state(pml, rng)
    pml_seed = random_pml_state(pml, rng)
    pml_grad_lam, pml_grad_mu = pml_material_vjp(
        pml, pml_state, pml_seed
    )
    pml_tests = {}
    pml_finite_difference = {}
    for component in ("lambda", "mu"):
        direction = rng.standard_normal(pml["lam"].shape)
        direction /= np.linalg.norm(direction)
        dlam = np.zeros_like(pml["lam"])
        dmu = np.zeros_like(pml["mu"])
        if component == "lambda":
            dlam = direction
            gradient = pml_grad_lam
        else:
            dmu = direction
            gradient = pml_grad_mu
        tangent = pml_material_tangent(
            pml, pml_state, dlam, dmu
        )
        lhs = pml_state_dot(tangent, pml_seed)
        rhs = float(np.vdot(direction, gradient))
        coefficients = material_coefficient_tangent(
            pml, dlam, dmu
        )
        record = dot_test_record(
            lhs,
            rhs,
            pml_state_norm(tangent),
            np.linalg.norm(gradient),
        )
        record["coefficient_tangent_l2"] = {
            name: float(np.linalg.norm(value.reshape(-1)))
            for name, value in coefficients.items()
        }
        pml_tests[component] = record
        pml_finite_difference[component] = finite_difference_record(
            pml,
            pml_state,
            dlam,
            dmu,
            tangent,
            1.0e6,
        )

    del pml_state, pml_seed, pml_grad_lam, pml_grad_mu
    gc.collect()

    global_data = load_global_data(
        args.config,
        args.topology,
        args.coefficients,
        args.coupled_mass,
        args.gll,
        args.weights,
        Path(args.baseline_material) / "Mat_0_Kappa.h5",
        Path(args.baseline_material) / "Mat_0_Mu.h5",
        batch_size=args.batch_size,
    )
    global_state = random_global_state(global_data, rng)
    global_seed = random_global_state(global_data, rng)
    material_direction = [
        rng.standard_normal(global_data["solid"]["lam"].shape),
        rng.standard_normal(global_data["solid"]["mu"].shape),
        rng.standard_normal(global_data["pml"]["lam"].shape),
        rng.standard_normal(global_data["pml"]["mu"].shape),
    ]
    direction_norm = np.sqrt(
        sum(np.vdot(x, x).real for x in material_direction)
    )
    material_direction = [x / direction_norm for x in material_direction]
    global_tangent = global_material_tangent(
        global_data, global_state, *material_direction
    )
    global_gradient = global_material_vjp(
        global_data, global_state, global_seed
    )
    global_lhs = global_state_dot(global_tangent, global_seed)
    global_rhs = float(
        sum(
            np.vdot(direction, gradient)
            for direction, gradient in zip(
                material_direction, global_gradient
            )
        )
    )
    global_test = dot_test_record(
        global_lhs,
        global_rhs,
        global_state_norm(global_tangent),
        np.sqrt(sum(np.vdot(x, x).real for x in global_gradient)),
    )

    summary = {
        "pml_elements": int(conn.shape[0]),
        "pml_element_gll_entries": int(conn.size),
        "pml_unique_nodes": int(xyz.shape[0]),
        "baseline_reconstruction": baseline_reconstruction,
        "candidate_sensitivity": candidates,
        "dot_tests": {
            "pml_lambda_only": pml_tests["lambda"],
            "pml_mu_only": pml_tests["mu"],
            "combined_global_material": global_test,
        },
        "finite_difference_tests": {
            "pml_lambda_only": pml_finite_difference["lambda"],
            "pml_mu_only": pml_finite_difference["mu"],
        },
    }
    checks = {
        "baseline_reconstruction_exact": all(
            value == 0.0 for value in baseline_reconstruction.values()
        ),
        "all_frozen_candidates_zero_in_pml": all(
            row["delta_lambda"]["support"] == 0
            and row["delta_mu"]["support"] == 0
            for row in candidates.values()
        ),
        "dot_tests_nondegenerate": all(
            row["nondegenerate"]
            for row in summary["dot_tests"].values()
        ),
        "dot_tests_below_1e-12": all(
            row["relative_error"] < 1.0e-12
            for row in summary["dot_tests"].values()
        ),
        "finite_difference_tests_below_1e-9": all(
            row["relative_state_error"] < 1.0e-9
            for row in summary["finite_difference_tests"].values()
        ),
    }
    summary["checks"] = checks
    summary["result"] = (
        "PASS_PML_MATERIAL_FORENSIC_AUDIT"
        if all(checks.values())
        else "FAIL_PML_MATERIAL_FORENSIC_AUDIT"
    )
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    (output / "audit.txt").write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
