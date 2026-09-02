import unittest

from scripts.fathi_benchmark.line_search_contract import (
    acceptance_metrics,
    backtracking_steps,
    candidate_name_from_step_mpa,
    decimal_text,
)


class CertifiedLineSearchContractTest(unittest.TestCase):
    def test_backtracking_sequence(self):
        steps = backtracking_steps("0.1", "0.5", 4, "0.0125")
        self.assertEqual(
            [decimal_text(value) for value in steps],
            ["0.1", "0.05", "0.025", "0.0125"],
        )

    def test_candidate_names_are_stable(self):
        self.assertEqual(
            candidate_name_from_step_mpa("0.05"),
            "line_search_direction_0p05MPa",
        )
        self.assertEqual(
            candidate_name_from_step_mpa("0.025"),
            "line_search_direction_0p025MPa",
        )
        self.assertEqual(
            candidate_name_from_step_mpa(1.0),
            "line_search_direction_1MPa",
        )

    def test_actual_iter1_normalization_and_rejection(self):
        metrics = acceptance_metrics(
            policy="strict_descent",
            parent_objective=4.43978280400354824e-20,
            candidate_objective=4.440410211345994e-20,
            step_mpa="0.1",
            direction_scale=5.3490158278608622e4,
            g_dot_p=-1.7870991495370614e-23,
            armijo_c1=1.0e-4,
        )
        self.assertFalse(metrics["accepted"])
        self.assertAlmostEqual(
            metrics["actual_perturbation_multiplier"],
            100000.0 / 53490.158278608622,
            places=15,
        )
        self.assertAlmostEqual(
            metrics["directional_linear_prediction"],
            (100000.0 / 53490.158278608622)
            * -1.7870991495370614e-23,
            places=38,
        )

    def test_armijo_uses_normalized_directional_term(self):
        metrics = acceptance_metrics(
            policy="armijo",
            parent_objective=4.43978280400354824e-20,
            candidate_objective=4.440410211345994e-20,
            step_mpa="0.1",
            direction_scale=5.3490158278608622e4,
            g_dot_p=-1.7870991495370614e-23,
            armijo_c1=1.0e-4,
        )
        self.assertFalse(metrics["accepted"])
        self.assertAlmostEqual(
            metrics["armijo_rhs"],
            4.43978246990486434e-20,
            places=34,
        )


if __name__ == "__main__":
    unittest.main()
