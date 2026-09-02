"""Pure helpers for the certified physical-step line-search contract."""

from __future__ import annotations

import math
from decimal import Decimal, InvalidOperation


ACCEPTANCE_POLICIES = ("strict_descent", "armijo")


def positive_decimal(value, name: str) -> Decimal:
    try:
        result = Decimal(str(value))
    except (InvalidOperation, ValueError) as error:
        raise ValueError(f"{name} is not a decimal value: {value}") from error
    if not result.is_finite() or result <= 0:
        raise ValueError(f"{name} must be finite and positive: {value}")
    return result


def decimal_text(value) -> str:
    result = positive_decimal(value, "step_mpa")
    text = format(result.normalize(), "f")
    if "." in text:
        text = text.rstrip("0").rstrip(".")
    return text


def candidate_name_from_step_mpa(value) -> str:
    return "line_search_direction_" + decimal_text(value).replace(".", "p") + "MPa"


def backtracking_steps(
    initial_step_mpa,
    rho,
    max_attempts: int,
    min_step_mpa,
) -> list[Decimal]:
    initial = positive_decimal(initial_step_mpa, "initial_step_mpa")
    reduction = positive_decimal(rho, "rho")
    minimum = positive_decimal(min_step_mpa, "min_step_mpa")
    if reduction >= 1:
        raise ValueError("rho must satisfy 0 < rho < 1")
    if int(max_attempts) != max_attempts or int(max_attempts) < 1:
        raise ValueError("max_attempts must be a positive integer")
    if minimum > initial:
        raise ValueError("min_step_mpa cannot exceed initial_step_mpa")

    result = []
    step = initial
    for _ in range(int(max_attempts)):
        if step < minimum:
            break
        result.append(step)
        step *= reduction
    if not result:
        raise ValueError("line search has no admissible step")
    return result


def acceptance_metrics(
    *,
    policy: str,
    parent_objective: float,
    candidate_objective: float,
    step_mpa,
    direction_scale: float,
    g_dot_p: float,
    armijo_c1: float,
) -> dict[str, float | bool | str | None]:
    if policy not in ACCEPTANCE_POLICIES:
        raise ValueError(f"unsupported acceptance policy: {policy}")
    values = (
        parent_objective,
        candidate_objective,
        direction_scale,
        g_dot_p,
        armijo_c1,
    )
    if not all(math.isfinite(float(value)) for value in values):
        raise ValueError("line-search metric is non-finite")
    if direction_scale <= 0:
        raise ValueError("direction_scale must be positive")
    if g_dot_p >= 0:
        raise ValueError("g_dot_p must be negative for a descent direction")
    if not 0 < armijo_c1 < 1:
        raise ValueError("armijo_c1 must satisfy 0 < c1 < 1")

    step_pa = float(positive_decimal(step_mpa, "step_mpa") * Decimal("1e6"))
    perturbation_multiplier = step_pa / float(direction_scale)
    directional_linear_prediction = perturbation_multiplier * float(g_dot_p)
    armijo_rhs = float(parent_objective) + float(armijo_c1) * (
        directional_linear_prediction
    )
    threshold = float(parent_objective) if policy == "strict_descent" else armijo_rhs
    delta = float(candidate_objective) - float(parent_objective)
    relative_change = delta / max(
        abs(float(parent_objective)),
        float.fromhex("0x0.0000000000001p-1022"),
    )
    return {
        "policy": policy,
        "accepted": bool(float(candidate_objective) < threshold),
        "acceptance_rhs": threshold,
        "delta_objective": delta,
        "relative_change": relative_change,
        "step_pa": step_pa,
        "direction_scale": float(direction_scale),
        "actual_perturbation_multiplier": perturbation_multiplier,
        "g_dot_p": float(g_dot_p),
        "directional_linear_prediction": directional_linear_prediction,
        "armijo_c1": float(armijo_c1) if policy == "armijo" else None,
        "armijo_rhs": armijo_rhs,
    }
