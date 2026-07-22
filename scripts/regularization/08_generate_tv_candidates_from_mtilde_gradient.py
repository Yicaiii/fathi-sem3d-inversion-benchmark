from __future__ import annotations

from pathlib import Path
from datetime import datetime
import argparse
import json
import shutil

import h5py
import numpy as np


ROOT = Path.home() / "sem3d_fathi_clean"


def resolve(path: str) -> Path:
    p = Path(path)
    return p if p.is_absolute() else ROOT / p


def load_config(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Missing config: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def find_dataset_name(path: Path, expected_shape):
    matches = []
    with h5py.File(path, "r") as h:
        def visit(name, obj):
            if isinstance(obj, h5py.Dataset):
                if tuple(obj.shape) == tuple(expected_shape):
                    matches.append(name)
        h.visititems(visit)

    if not matches:
        raise RuntimeError(
            f"No dataset with shape {expected_shape} in {path}"
        )

    return matches[0]


def read_h5_field(path: Path, expected_shape):
    name = find_dataset_name(path, expected_shape)
    with h5py.File(path, "r") as h:
        arr = np.asarray(h[name], dtype=np.float64)
    return arr, name


def overwrite_h5_field(path: Path, data: np.ndarray, dataset_name: str):
    with h5py.File(path, "r+") as h:
        if dataset_name not in h:
            raise RuntimeError(
                f"Dataset {dataset_name} not found in {path}"
            )

        if tuple(h[dataset_name].shape) != tuple(data.shape):
            raise RuntimeError(
                f"Shape mismatch for {path}:{dataset_name}: "
                f"{h[dataset_name].shape} vs {data.shape}"
            )

        h[dataset_name][...] = data


def stats(a: np.ndarray) -> dict:
    return {
        "shape": list(a.shape),
        "finite": int(np.count_nonzero(np.isfinite(a))),
        "size": int(a.size),
        "min": float(np.nanmin(a)),
        "max": float(np.nanmax(a)),
        "mean": float(np.nanmean(a)),
        "maxabs": float(np.nanmax(np.abs(a))),
        "l2": float(np.sqrt(np.nansum(a * a))),
    }


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config",
        default="benchmark_fathi_tv/config/tv_config_iter008_to_iter009.json",
    )

    parser.add_argument(
        "--label",
        default="parent_iter008_gradientratio5pct",
        help="TV branch label under mtilde_solve/",
    )

    parser.add_argument(
        "--parent-h5-dir",
        required=True,
        help="Parent accepted material HDF5 directory containing Mat_0_Kappa/Mu/Density.h5",
    )

    parser.add_argument(
        "--steps-mpa",
        nargs="+",
        type=float,
        default=[0.10, 0.25, 0.50, 1.00],
    )

    args = parser.parse_args()

    config_path = resolve(args.config)
    config = load_config(config_path)

    transition_dir = resolve(config["tv_transition_dir"])
    parent_h5_dir = resolve(args.parent_h5_dir)

    nx = int(config["mesh"]["nx"])
    ny = int(config["mesh"]["ny"])
    nz = int(config["mesh"]["nz"])
    shape = (nz, ny, nx)
    n_total = nx * ny * nz

    active_indices_path = resolve(config["active_indices"])

    tv_mtilde_dir = transition_dir / "mtilde_solve" / args.label
    gL_path = tv_mtilde_dir / "g_lambda_total.npy"
    gM_path = tv_mtilde_dir / "g_mu_total.npy"

    candidate_root = transition_dir / "candidates" / args.label
    candidate_root.mkdir(parents=True, exist_ok=True)

    required = [
        active_indices_path,
        gL_path,
        gM_path,
        parent_h5_dir / "Mat_0_Kappa.h5",
        parent_h5_dir / "Mat_0_Mu.h5",
        parent_h5_dir / "Mat_0_Density.h5",
    ]

    missing = [p for p in required if not p.exists()]
    if missing:
        print("Missing inputs:")
        for p in missing:
            print(" ", p)
        raise SystemExit(1)

    idx = np.load(active_indices_path).astype(np.int64).reshape(-1)
    gL = np.load(gL_path).astype(np.float64).reshape(-1)
    gM = np.load(gM_path).astype(np.float64).reshape(-1)

    if idx.shape != (38440,):
        raise RuntimeError(f"Unexpected active index shape: {idx.shape}")

    if gL.shape != (38440,) or gM.shape != (38440,):
        raise RuntimeError(
            f"Unexpected TV gradient shapes: gL={gL.shape}, gM={gM.shape}"
        )

    if len(np.unique(idx)) != idx.size:
        raise RuntimeError("Duplicate active indices detected")

    if int(idx.min()) < 0 or int(idx.max()) >= n_total:
        raise RuntimeError(
            f"indices out of range for shape {shape}: "
            f"min={idx.min()} max={idx.max()}"
        )

    if not np.all(np.isfinite(gL)) or not np.all(np.isfinite(gM)):
        raise RuntimeError("Non-finite TV Mtilde gradient values")

    kappa_parent, kappa_dataset = read_h5_field(
        parent_h5_dir / "Mat_0_Kappa.h5",
        shape,
    )

    mu_parent, mu_dataset = read_h5_field(
        parent_h5_dir / "Mat_0_Mu.h5",
        shape,
    )

    density_parent, density_dataset = read_h5_field(
        parent_h5_dir / "Mat_0_Density.h5",
        shape,
    )

    lambda_parent = kappa_parent - (2.0 / 3.0) * mu_parent

    flat_lambda_parent = lambda_parent.reshape(-1, order="C")
    flat_mu_parent = mu_parent.reshape(-1, order="C")

    scaleL = float(np.max(np.abs(gL)))
    scaleM = float(np.max(np.abs(gM)))

    if scaleL <= 0.0 or scaleM <= 0.0:
        raise RuntimeError(f"Bad gradient scales: {scaleL}, {scaleM}")

    records = []
    created = datetime.now().isoformat()

    for step_mpa in args.steps_mpa:
        step_pa = float(step_mpa) * 1e6
        label_step = f"{step_mpa:.2f}".replace(".", "p") + "MPa"
        cand_name = f"tv_{args.label}_neg_mtilde_{label_step}"

        cand_dir = candidate_root / cand_name
        cand_h5_dir = cand_dir / "mat/h5"
        cand_dir.mkdir(parents=True, exist_ok=True)

        if cand_h5_dir.exists():
            shutil.rmtree(cand_h5_dir)

        shutil.copytree(parent_h5_dir, cand_h5_dir)

        flat_lambda = flat_lambda_parent.copy()
        flat_mu = flat_mu_parent.copy()

        flat_lambda[idx] = flat_lambda[idx] - step_pa * (gL / scaleL)
        flat_mu[idx] = flat_mu[idx] - step_pa * (gM / scaleM)

        lambda_new = flat_lambda.reshape(shape, order="C")
        mu_new = flat_mu.reshape(shape, order="C")
        kappa_new = lambda_new + (2.0 / 3.0) * mu_new
        density_new = density_parent.copy()

        finite_ok = (
            np.all(np.isfinite(lambda_new))
            and np.all(np.isfinite(mu_new))
            and np.all(np.isfinite(kappa_new))
            and np.all(np.isfinite(density_new))
        )

        positive_ok = (
            np.min(lambda_new) > 0.0
            and np.min(mu_new) > 0.0
            and np.min(kappa_new) > 0.0
            and np.min(density_new) > 0.0
        )

        overwrite_h5_field(
            cand_h5_dir / "Mat_0_Kappa.h5",
            kappa_new,
            kappa_dataset,
        )

        overwrite_h5_field(
            cand_h5_dir / "Mat_0_Mu.h5",
            mu_new,
            mu_dataset,
        )

        overwrite_h5_field(
            cand_h5_dir / "Mat_0_Density.h5",
            density_new,
            density_dataset,
        )

        state_npz = cand_dir / f"{cand_name}_state_candidate.npz"

        np.savez_compressed(
            state_npz,
            lambda_field=lambda_new,
            mu=mu_new,
            kappa=kappa_new,
            density=density_new,
            parent_h5_dir=str(parent_h5_dir),
            tv_config=str(config_path),
            tv_label=args.label,
            step_mpa=step_mpa,
            step_pa=step_pa,
            gradient_lambda=str(gL_path),
            gradient_mu=str(gM_path),
            gradient_indices=str(active_indices_path),
            direction="negative_tv_data_mtilde_gradient_maxabs_normalized",
        )

        rec = {
            "candidate": cand_name,
            "step_mpa": step_mpa,
            "step_pa": step_pa,
            "candidate_dir": str(cand_dir),
            "mat_h5_dir": str(cand_h5_dir),
            "state_npz": str(state_npz),
            "finite_ok": bool(finite_ok),
            "positive_ok": bool(positive_ok),
            "lambda": stats(lambda_new),
            "mu": stats(mu_new),
            "kappa": stats(kappa_new),
            "density": stats(density_new),
            "max_abs_delta_lambda": float(
                np.max(np.abs(lambda_new - lambda_parent))
            ),
            "max_abs_delta_mu": float(
                np.max(np.abs(mu_new - mu_parent))
            ),
        }

        records.append(rec)

    payload = {
        "created": created,
        "config": str(config_path),
        "tv_label": args.label,
        "shape": list(shape),
        "active_indices": str(active_indices_path),
        "active_count": int(idx.size),
        "parent_h5_dir": str(parent_h5_dir),
        "g_lambda": str(gL_path),
        "g_mu": str(gM_path),
        "gradient_scale_lambda": scaleL,
        "gradient_scale_mu": scaleM,
        "candidate_root": str(candidate_root),
        "records": records,
        "result": (
            "PASS_TV_CANDIDATE_GENERATION"
            if all(r["finite_ok"] and r["positive_ok"] for r in records)
            else "CHECK_NEEDED"
        ),
    }

    json_out = candidate_root / "tv_candidate_generation_summary.json"
    txt_out = candidate_root / "tv_candidate_generation_summary.txt"

    json_out.write_text(
        json.dumps(payload, indent=2),
        encoding="utf-8",
    )

    lines = []
    lines.append("TV CANDIDATE GENERATION SUMMARY")
    lines.append("================================")
    lines.append("")
    lines.append(f"tv label = {args.label}")
    lines.append(f"parent_h5_dir = {parent_h5_dir}")
    lines.append(f"shape = {shape}")
    lines.append(f"active count = {idx.size}")
    lines.append(f"g_lambda = {gL_path}")
    lines.append(f"g_mu = {gM_path}")
    lines.append(f"gradient scale lambda = {scaleL:.16e}")
    lines.append(f"gradient scale mu = {scaleM:.16e}")
    lines.append("")

    for r in records:
        lines.append(f"Candidate: {r['candidate']}")
        lines.append(f"  step_mpa = {r['step_mpa']}")
        lines.append(f"  finite_ok = {r['finite_ok']}")
        lines.append(f"  positive_ok = {r['positive_ok']}")
        lines.append(
            f"  max_abs_delta_lambda = {r['max_abs_delta_lambda']:.16e}"
        )
        lines.append(
            f"  max_abs_delta_mu = {r['max_abs_delta_mu']:.16e}"
        )
        lines.append(f"  mat_h5_dir = {r['mat_h5_dir']}")
        lines.append("")

    lines.append(f"RESULT = {payload['result']}")

    txt_out.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print("\n".join(lines))


if __name__ == "__main__":
    main()
