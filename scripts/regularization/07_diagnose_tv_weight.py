from __future__ import annotations

from pathlib import Path
import argparse
import json
import re

import numpy as np


ROOT = Path(__file__).resolve().parents[2]


def resolve(path: str) -> Path:
    value = Path(path).expanduser()
    return value.resolve() if value.is_absolute() else (ROOT / value).resolve()


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    denominator = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denominator == 0.0:
        return float("nan")
    return float(np.dot(a, b) / denominator)


def relative_norm(a: np.ndarray, reference: np.ndarray) -> float:
    return float(
        np.linalg.norm(a)
        / max(float(np.linalg.norm(reference)), 1e-300)
    )


def objective_from_file(path: Path) -> float:
    if not path.exists():
        raise FileNotFoundError(f"Missing objective file: {path}")

    if path.suffix.lower() == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        for key in [
            "parent_J",
            "total_J",
            "J_parent",
            "parent_objective",
            "objective",
            "J",
        ]:
            if key in payload:
                return float(payload[key])

    text = path.read_text(encoding="utf-8", errors="replace")
    pattern = re.compile(
        r"(?:parent_J|total_J|J_parent|parent_objective|objective|J)"
        r"\s*[:=]\s*"
        r"([-+]?\d+(?:\.\d*)?(?:[eE][-+]?\d+)?)"
    )
    match = pattern.search(text)
    if not match:
        raise RuntimeError(f"Cannot find parent objective in {path}")
    return float(match.group(1))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--label",
        default=None,
        help="Output branch label. Default: transition from config.",
    )
    parser.add_argument("--j-data", type=float, default=None)
    parser.add_argument("--j-data-file", default=None)
    args = parser.parse_args()

    config_path = resolve(args.config)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    label = args.label or config.get("transition", "tv_parent")
    transition_dir = resolve(config["tv_transition_dir"])

    tv_values_path = transition_dir / "tv_full_grid" / label / "tv_values_and_stats.json"
    combined_dir = transition_dir / "combined_rhs" / label
    solve_dir = transition_dir / "mtilde_solve" / label

    alpha_lambda = float(config["tv"]["alpha_lambda"])
    alpha_mu = float(config["tv"]["alpha_mu"])

    solve_metadata_path = solve_dir / "mtilde_data_tv_solve_metadata.json"
    solve_metadata = json.loads(solve_metadata_path.read_text(encoding="utf-8"))

    for name, configured, solved in [
        ("lambda", alpha_lambda, float(solve_metadata["alpha_lambda"])),
        ("mu", alpha_mu, float(solve_metadata["alpha_mu"])),
    ]:
        if not np.isclose(configured, solved, rtol=1e-12, atol=0.0):
            raise RuntimeError(
                f"Stale {name} gradient: config={configured:.16e}, solve={solved:.16e}"
            )

    objective_file_value = args.j_data_file or config.get("parent_objective_path")
    if args.j_data is not None:
        j_data = float(args.j_data)
        objective_source = "command line"
    elif objective_file_value:
        objective_path = resolve(objective_file_value)
        j_data = objective_from_file(objective_path)
        objective_source = str(objective_path)
    else:
        raise SystemExit(
            "Provide --j-data or --j-data-file, or configure parent_objective_path."
        )

    if not np.isfinite(j_data) or j_data <= 0.0:
        raise RuntimeError(f"J_data must be positive and finite: {j_data}")

    tv_values = json.loads(tv_values_path.read_text(encoding="utf-8"))
    r_data_lambda = np.load(combined_dir / "rhs_data_lambda.npy")
    r_data_mu = np.load(combined_dir / "rhs_data_mu.npy")
    r_tv_lambda = np.load(combined_dir / "rhs_tv_lambda_physical.npy")
    r_tv_mu = np.load(combined_dir / "rhs_tv_mu_physical.npy")
    g_total_lambda = np.load(solve_dir / "g_lambda_total.npy")
    g_total_mu = np.load(solve_dir / "g_mu_total.npy")
    g_data_lambda = np.load(resolve(config["baseline_gradient_lambda"]))
    g_data_mu = np.load(resolve(config["baseline_gradient_mu"]))

    g_tv_weighted_lambda = g_total_lambda - g_data_lambda
    g_tv_weighted_mu = g_total_mu - g_data_mu

    tv_lambda = float(tv_values["tv_lambda_hat"])
    tv_mu = float(tv_values["tv_mu_hat"])
    j_reg_lambda = alpha_lambda * tv_lambda
    j_reg_mu = alpha_mu * tv_mu
    j_reg_total = j_reg_lambda + j_reg_mu
    j_total_parent = j_data + j_reg_total

    diagnostic = {
        "config": str(config_path),
        "label": label,
        "j_data": j_data,
        "j_data_source": objective_source,
        "tv_lambda_hat": tv_lambda,
        "tv_mu_hat": tv_mu,
        "alpha_lambda": alpha_lambda,
        "alpha_mu": alpha_mu,
        "j_reg_lambda": j_reg_lambda,
        "j_reg_mu": j_reg_mu,
        "j_reg_total": j_reg_total,
        "j_total_parent": j_total_parent,
        "j_reg_over_j_data": j_reg_total / j_data,
        "rhs_cosine_lambda": cosine(r_data_lambda, r_tv_lambda),
        "rhs_cosine_mu": cosine(r_data_mu, r_tv_mu),
        "weighted_tv_gradient_over_data_lambda": relative_norm(
            g_tv_weighted_lambda,
            g_data_lambda,
        ),
        "weighted_tv_gradient_over_data_mu": relative_norm(
            g_tv_weighted_mu,
            g_data_mu,
        ),
        "total_gradient_cosine_data_lambda": cosine(
            g_total_lambda,
            g_data_lambda,
        ),
        "total_gradient_cosine_data_mu": cosine(
            g_total_mu,
            g_data_mu,
        ),
    }

    output_json = solve_dir / "tv_weight_diagnostic.json"
    output_txt = solve_dir / "tv_weight_diagnostic.txt"
    output_json.write_text(json.dumps(diagnostic, indent=2), encoding="utf-8")
    lines = ["TV WEIGHT DIAGNOSTIC", "====================", ""]
    for key, value in diagnostic.items():
        lines.append(f"{key} = {value}")
    lines.extend(["", f"json = {output_json}", "RESULT = PASS_TV_WEIGHT_DIAGNOSTIC"])
    output_txt.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
