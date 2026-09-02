import unittest

import numpy as np
from scipy.linalg import block_diag
from scipy.sparse import csr_matrix

from scripts.fathi_benchmark.physical_space_optimizer import (
    apply_lambda_bias_euclidean,
    audit_curvature_pair,
    joint_mtilde_inner,
    lambda_bias_weight,
    mtilde_inner,
    physical_lbfgs_direction,
)


class PhysicalSpaceOptimizerTest(unittest.TestCase):
    def setUp(self):
        self.m_dense = np.array(
            [
                [3.0, 0.5, 0.0],
                [0.5, 2.0, 0.25],
                [0.0, 0.25, 1.5],
            ],
            dtype=np.float64,
        )
        self.mtilde = csr_matrix(self.m_dense)

    def test_mtilde_inner_matches_dense_definition(self):
        left = np.array([1.5, -2.0, 0.25])
        right = np.array([-0.5, 4.0, 2.0])
        expected = float(left @ self.m_dense @ right)
        self.assertEqual(mtilde_inner(left, right, self.mtilde), expected)

    def test_joint_inner_matches_block_diagonal_metric(self):
        left = (np.array([1.0, 2.0, 3.0]), np.array([-1.0, 0.5, 4.0]))
        right = (np.array([0.5, -2.0, 1.0]), np.array([3.0, 2.0, -0.5]))
        block = block_diag(self.m_dense, self.m_dense)
        expected = float(np.concatenate(left) @ block @ np.concatenate(right))
        self.assertEqual(
            joint_mtilde_inner(left, right, self.mtilde), expected
        )

    def test_iteration_zero_uses_frozen_physical_gamma(self):
        gradient = (
            np.array([1.0, -2.0, 0.5]),
            np.array([-1.5, 0.25, 3.0]),
        )
        direction, audits, gamma = physical_lbfgs_direction(
            gradient, [], self.mtilde, gamma0=7.5
        )
        self.assertEqual(audits, [])
        self.assertEqual(gamma, 7.5)
        np.testing.assert_array_equal(direction[0], -7.5 * gradient[0])
        np.testing.assert_array_equal(direction[1], -7.5 * gradient[1])
        self.assertLess(
            joint_mtilde_inner(gradient, direction, self.mtilde), 0.0
        )

    def test_two_loop_uses_mtilde_products(self):
        gradient = (
            np.array([0.6, -1.2, 0.4]),
            np.array([-0.8, 0.3, 1.1]),
        )
        history = [
            (
                (
                    np.array([0.2, -0.1, 0.4]),
                    np.array([-0.3, 0.2, 0.1]),
                ),
                (
                    np.array([0.4, -0.2, 0.3]),
                    np.array([-0.1, 0.5, 0.2]),
                ),
            ),
            (
                (
                    np.array([-0.15, 0.3, 0.2]),
                    np.array([0.25, -0.1, 0.35]),
                ),
                (
                    np.array([-0.2, 0.45, 0.25]),
                    np.array([0.3, -0.05, 0.5]),
                ),
            ),
        ]
        direction, audits, gamma = physical_lbfgs_direction(
            gradient,
            history,
            self.mtilde,
            gamma0=11.0,
            curvature_relative_tolerance=1.0e-14,
        )
        self.assertTrue(all(audit.accepted for audit in audits))

        block = block_diag(self.m_dense, self.m_dense)
        g = np.concatenate(gradient)
        dense_history = [
            (np.concatenate(s_pair), np.concatenate(y_pair))
            for s_pair, y_pair in history
        ]
        rhos = [1.0 / float(s @ block @ y) for s, y in dense_history]
        q = g.copy()
        alphas = []
        for (s, y), rho in reversed(list(zip(dense_history, rhos))):
            alpha = rho * float(s @ block @ q)
            q -= alpha * y
            alphas.append(alpha)
        last_s, last_y = dense_history[-1]
        expected_gamma = float(
            (last_s @ block @ last_y) / (last_y @ block @ last_y)
        )
        r = expected_gamma * q
        for ((s, y), rho), alpha in zip(
            zip(dense_history, rhos), reversed(alphas)
        ):
            beta = rho * float(y @ block @ r)
            r += s * (alpha - beta)

        self.assertEqual(gamma, expected_gamma)
        np.testing.assert_allclose(
            np.concatenate(direction), -r, rtol=2.0e-15, atol=0.0
        )

    def test_curvature_safeguard_rejects_nonpositive_pair(self):
        s = (np.array([1.0, 0.0, 0.0]), np.zeros(3))
        y = (np.array([-1.0, 0.0, 0.0]), np.zeros(3))
        audit = audit_curvature_pair(
            s, y, self.mtilde, relative_tolerance=1.0e-12
        )
        self.assertFalse(audit.accepted)
        self.assertEqual(audit.reason, "nonpositive_sMy")

    def test_eq25_bias_uses_euclidean_norm_and_weight_schedule(self):
        direction = (
            np.array([3.0, 4.0, 0.0]),
            np.array([0.0, 0.0, 2.0]),
        )
        self.assertEqual(lambda_bias_weight(0), 1.0)
        self.assertEqual(lambda_bias_weight(25), 0.5)
        self.assertEqual(lambda_bias_weight(50), 0.0)
        biased = apply_lambda_bias_euclidean(direction, weight=1.0)
        np.testing.assert_array_equal(biased[0], np.array([0.0, 0.0, 5.0]))
        np.testing.assert_array_equal(biased[1], direction[1])


if __name__ == "__main__":
    unittest.main()
