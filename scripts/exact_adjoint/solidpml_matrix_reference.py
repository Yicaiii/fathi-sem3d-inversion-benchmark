from pathlib import Path

import numpy as np


def load_operator(path):
    T = np.asarray(
        np.load(Path(path)),
        dtype=np.float64,
    )

    if T.ndim != 2:
        raise ValueError(
            f"operator must be 2D, got {T.ndim}"
        )

    if T.shape[0] != T.shape[1]:
        raise ValueError(
            f"operator must be square, got {T.shape}"
        )

    if not np.isfinite(T).all():
        raise ValueError(
            "operator contains non-finite values"
        )

    return T


def forward_step(T, state):
    state = np.asarray(
        state,
        dtype=np.float64,
    )

    if state.shape != (T.shape[1],):
        raise ValueError(
            f"state shape {state.shape} incompatible with {T.shape}"
        )

    return T @ state


def adjoint_step(T, adjoint_state):
    adjoint_state = np.asarray(
        adjoint_state,
        dtype=np.float64,
    )

    if adjoint_state.shape != (T.shape[0],):
        raise ValueError(
            f"adjoint shape {adjoint_state.shape} incompatible with {T.shape}"
        )

    return T.T @ adjoint_state


def relative_dot_error(lhs, rhs):
    scale = max(
        abs(lhs),
        abs(rhs),
        np.finfo(np.float64).tiny,
    )

    return abs(lhs - rhs) / scale
