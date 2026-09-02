"""Physical Pa-space L-BFGS primitives for the Fathi reproduction.

The certified gradient is a physical Riesz gradient: ``Mtilde g = RHS``.
Consequently every optimizer pairing is evaluated in the Mtilde metric.
Only the paper's Eq. 25 lambda-bias normalization uses the Euclidean norm,
matching the norm convention defined by the paper on the same page.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np


VectorPair = tuple[np.ndarray, np.ndarray]


@dataclass(frozen=True)
class CurvatureAudit:
    """Acceptance record for one physical-space L-BFGS history pair."""

    accepted: bool
    s_m_y: float
    s_norm_m: float
    y_norm_m: float
    threshold: float
    reason: str


def _vector(value: np.ndarray, *, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional, got {array.shape}")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} contains non-finite values")
    return array


def _pair(value: Sequence[np.ndarray], *, name: str) -> VectorPair:
    if len(value) != 2:
        raise ValueError(f"{name} must contain lambda and mu vectors")
    lam = _vector(value[0], name=f"{name}.lambda")
    mu = _vector(value[1], name=f"{name}.mu")
    if lam.shape != mu.shape:
        raise ValueError(
            f"{name} lambda/mu shapes differ: {lam.shape} vs {mu.shape}"
        )
    return lam, mu


def physical_curvature_pair(
    parent_model: Sequence[np.ndarray],
    child_model: Sequence[np.ndarray],
    parent_gradient: Sequence[np.ndarray],
    child_gradient: Sequence[np.ndarray],
) -> tuple[VectorPair, VectorPair]:
    """Construct ``s`` and ``y`` from accepted physical/control history.

    The inputs and outputs are lambda/mu vectors in physical Pa/control
    coordinates.  Curvature acceptance remains the responsibility of
    :func:`audit_curvature_pair`, which evaluates ``s.T @ Mtilde @ y``.
    """

    model_k = _pair(parent_model, name="parent_model")
    model_kp1 = _pair(child_model, name="child_model")
    gradient_k = _pair(parent_gradient, name="parent_gradient")
    gradient_kp1 = _pair(child_gradient, name="child_gradient")
    shapes = {
        model_k[0].shape,
        model_kp1[0].shape,
        gradient_k[0].shape,
        gradient_kp1[0].shape,
    }
    if len(shapes) != 1:
        raise ValueError(f"model/gradient history shapes differ: {shapes}")
    s_pair = (
        np.asarray(model_kp1[0] - model_k[0], dtype=np.float64),
        np.asarray(model_kp1[1] - model_k[1], dtype=np.float64),
    )
    y_pair = (
        np.asarray(gradient_kp1[0] - gradient_k[0], dtype=np.float64),
        np.asarray(gradient_kp1[1] - gradient_k[1], dtype=np.float64),
    )
    return s_pair, y_pair


def mtilde_inner(
    left: np.ndarray,
    right: np.ndarray,
    mtilde,
) -> float:
    """Return ``left.T @ Mtilde @ right`` with strict shape checks."""

    a = _vector(left, name="left")
    b = _vector(right, name="right")
    if a.shape != b.shape:
        raise ValueError(f"inner-product shape mismatch: {a.shape} vs {b.shape}")
    if tuple(mtilde.shape) != (a.size, a.size):
        raise ValueError(
            f"Mtilde shape {mtilde.shape} does not match vector size {a.size}"
        )
    value = float(a @ (mtilde @ b))
    if not np.isfinite(value):
        raise ValueError("Mtilde inner product is non-finite")
    return value


def joint_mtilde_inner(
    left: Sequence[np.ndarray],
    right: Sequence[np.ndarray],
    mtilde,
) -> float:
    """Return the block-diagonal joint lambda/mu Mtilde pairing."""

    left_lam, left_mu = _pair(left, name="left")
    right_lam, right_mu = _pair(right, name="right")
    if left_lam.shape != right_lam.shape:
        raise ValueError(
            "joint inner-product shape mismatch: "
            f"{left_lam.shape} vs {right_lam.shape}"
        )
    return mtilde_inner(left_lam, right_lam, mtilde) + mtilde_inner(
        left_mu, right_mu, mtilde
    )


def mtilde_norm(vector: np.ndarray, mtilde) -> float:
    """Return the Mtilde norm, allowing only roundoff-sized negativity."""

    array = _vector(vector, name="vector")
    squared = mtilde_inner(array, array, mtilde)
    if squared < 0.0:
        roundoff_floor = 64.0 * np.finfo(np.float64).eps * max(
            1.0, float(np.max(np.abs(array))) ** 2
        )
        if squared < -roundoff_floor:
            raise ValueError(f"negative Mtilde norm squared: {squared}")
        squared = 0.0
    return float(np.sqrt(squared))


def joint_mtilde_norm(vector: Sequence[np.ndarray], mtilde) -> float:
    pair = _pair(vector, name="vector")
    squared = joint_mtilde_inner(pair, pair, mtilde)
    if squared < 0.0:
        raise ValueError(f"negative joint Mtilde norm squared: {squared}")
    return float(np.sqrt(squared))


def audit_curvature_pair(
    s: Sequence[np.ndarray],
    y: Sequence[np.ndarray],
    mtilde,
    *,
    relative_tolerance: float,
) -> CurvatureAudit:
    """Apply a positive, relative Mtilde-metric curvature safeguard."""

    if not np.isfinite(relative_tolerance) or relative_tolerance < 0.0:
        raise ValueError("relative_tolerance must be finite and non-negative")
    s_pair = _pair(s, name="s")
    y_pair = _pair(y, name="y")
    s_m_y = joint_mtilde_inner(s_pair, y_pair, mtilde)
    s_norm_m = joint_mtilde_norm(s_pair, mtilde)
    y_norm_m = joint_mtilde_norm(y_pair, mtilde)
    threshold = float(relative_tolerance * s_norm_m * y_norm_m)

    if s_norm_m == 0.0:
        return CurvatureAudit(
            False, s_m_y, s_norm_m, y_norm_m, threshold, "zero_s_mnorm"
        )
    if y_norm_m == 0.0:
        return CurvatureAudit(
            False, s_m_y, s_norm_m, y_norm_m, threshold, "zero_y_mnorm"
        )
    if s_m_y <= 0.0:
        return CurvatureAudit(
            False, s_m_y, s_norm_m, y_norm_m, threshold, "nonpositive_sMy"
        )
    if s_m_y <= threshold:
        return CurvatureAudit(
            False,
            s_m_y,
            s_norm_m,
            y_norm_m,
            threshold,
            "numerically_degenerate_sMy",
        )
    return CurvatureAudit(
        True, s_m_y, s_norm_m, y_norm_m, threshold, "accepted"
    )


def physical_lbfgs_direction(
    gradient: Sequence[np.ndarray],
    history: Iterable[tuple[Sequence[np.ndarray], Sequence[np.ndarray]]],
    mtilde,
    *,
    gamma0: float,
    memory: int = 15,
    curvature_relative_tolerance: float = 1.0e-12,
) -> tuple[VectorPair, list[CurvatureAudit], float]:
    """Compute an Mtilde-metric L-BFGS direction in physical Pa space.

    Invalid history pairs are skipped. With no admissible pair, the frozen
    reproduction assumption ``H0_phys = gamma0 I`` is used. With history,
    the usual scaling is adapted to ``gamma = sMy / yMy``.
    """

    grad = _pair(gradient, name="gradient")
    if not np.isfinite(gamma0) or gamma0 <= 0.0:
        raise ValueError("gamma0 must be finite and positive")
    if memory <= 0:
        raise ValueError("memory must be positive")

    raw_history = list(history)[-memory:]
    accepted = []
    audits: list[CurvatureAudit] = []
    for s_value, y_value in raw_history:
        s_pair = _pair(s_value, name="s")
        y_pair = _pair(y_value, name="y")
        audit = audit_curvature_pair(
            s_pair,
            y_pair,
            mtilde,
            relative_tolerance=curvature_relative_tolerance,
        )
        audits.append(audit)
        if audit.accepted:
            accepted.append((s_pair, y_pair, 1.0 / audit.s_m_y))

    if not accepted:
        direction = (-gamma0 * grad[0], -gamma0 * grad[1])
        return direction, audits, float(gamma0)

    q = (np.array(grad[0], copy=True), np.array(grad[1], copy=True))
    alphas = []
    for s_pair, y_pair, rho in reversed(accepted):
        alpha = rho * joint_mtilde_inner(s_pair, q, mtilde)
        q = (q[0] - alpha * y_pair[0], q[1] - alpha * y_pair[1])
        alphas.append(alpha)

    last_s, last_y, _ = accepted[-1]
    s_m_y = joint_mtilde_inner(last_s, last_y, mtilde)
    y_m_y = joint_mtilde_inner(last_y, last_y, mtilde)
    if y_m_y <= 0.0:
        raise ValueError(f"invalid yMy for L-BFGS scaling: {y_m_y}")
    gamma = float(s_m_y / y_m_y)
    if not np.isfinite(gamma) or gamma <= 0.0:
        raise ValueError(f"invalid L-BFGS history scaling: {gamma}")

    r = (gamma * q[0], gamma * q[1])
    for (s_pair, y_pair, rho), alpha in zip(accepted, reversed(alphas)):
        beta = rho * joint_mtilde_inner(y_pair, r, mtilde)
        r = (
            r[0] + s_pair[0] * (alpha - beta),
            r[1] + s_pair[1] * (alpha - beta),
        )

    return (-r[0], -r[1]), audits, gamma


def lambda_bias_weight(iteration: int) -> float:
    """Paper weight schedule ``W(k) = max(1 - k/50, 0)``."""

    if int(iteration) != iteration or iteration < 0:
        raise ValueError("iteration must be a non-negative integer")
    return float(max(1.0 - float(iteration) / 50.0, 0.0))


def apply_lambda_bias_euclidean(
    direction: Sequence[np.ndarray],
    *,
    weight: float,
) -> VectorPair:
    """Apply paper Eq. 25 using its Euclidean norm convention."""

    lam, mu = _pair(direction, name="direction")
    if not np.isfinite(weight) or not 0.0 <= weight <= 1.0:
        raise ValueError("weight must be finite and in [0, 1]")
    lam_norm = float(np.linalg.norm(lam))
    mu_norm = float(np.linalg.norm(mu))
    if lam_norm == 0.0:
        if weight == 0.0:
            return np.array(lam, copy=True), np.array(mu, copy=True)
        raise ValueError("Eq. 25 lambda direction has zero Euclidean norm")
    if weight > 0.0 and mu_norm == 0.0:
        raise ValueError("Eq. 25 mu direction has zero Euclidean norm")

    normalized_mu = np.zeros_like(mu) if weight == 0.0 else mu / mu_norm
    biased_lambda = lam_norm * (
        weight * normalized_mu + (1.0 - weight) * lam / lam_norm
    )
    if not np.all(np.isfinite(biased_lambda)):
        raise ValueError("Eq. 25 produced a non-finite lambda direction")
    return biased_lambda, np.array(mu, copy=True)
