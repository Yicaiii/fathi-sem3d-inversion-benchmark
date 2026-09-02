import ast
import inspect
from pathlib import Path
import textwrap
import unittest

from scripts.exact_adjoint.real_s43_global_operator import (
    adjoint_step,
    material_vjp,
)
from scripts.exact_adjoint.s43_external_reverse_core import (
    cleanup_replay_cache,
    ensure_replay_cache,
    load_replay_state,
)
from scripts.fathi_benchmark import run_certified_external_exact_reverse as core
from scripts.fathi_benchmark import run_exact_reverse_gradient_generic as generic

FROZEN_CERTIFIED_WEIGHTED_RESIDUAL_AST = "BinOp(left=Subscript(value=Subscript(value=Name(id='runtime', ctx=Load()), slice=Constant(value='objective_weights'), ctx=Load()), slice=Tuple(elts=[Slice(), Constant(value=None), Constant(value=None)], ctx=Load()), ctx=Load()), op=Mult(), right=Subscript(value=Name(id='runtime', ctx=Load()), slice=Constant(value='residual'), ctx=Load()))"
FROZEN_CERTIFIED_SEED_AST = "Subscript(value=Name(id='weighted_residual', ctx=Load()), slice=Name(id='transition', ctx=Load()), ctx=Load())"
FROZEN_CERTIFIED_REVERSE_RANGE_AST = "Call(func=Name(id='range', ctx=Load()), args=[BinOp(left=Name(id='active_end', ctx=Load()), op=Sub(), right=Constant(value=1)), BinOp(left=Name(id='sub_start', ctx=Load()), op=Sub(), right=Constant(value=1)), UnaryOp(op=USub(), operand=Constant(value=1))], keywords=[])"


def _function_tree(function):
    return ast.parse(textwrap.dedent(inspect.getsource(function))).body[0]


def _assignment(function, name):
    for node in ast.walk(_function_tree(function)):
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == name
            for target in node.targets
        ):
            return ast.dump(node.value, include_attributes=False)
    raise AssertionError(f"assignment not found: {name}")


def _reverse_range(function):
    for node in ast.walk(_function_tree(function)):
        if (
            isinstance(node, ast.For)
            and isinstance(node.target, ast.Name)
            and node.target.id == "transition"
            and isinstance(node.iter, ast.Call)
            and isinstance(node.iter.func, ast.Name)
            and node.iter.func.id == "range"
            and len(node.iter.args) == 3
        ):
            return ast.dump(node.iter, include_attributes=False)
    raise AssertionError("reverse transition range not found")


class ExactReverseGradientGenericTest(unittest.TestCase):
    def test_wrapper_delegates_without_a_second_reverse_implementation(self):
        source = inspect.getsource(generic.run_reverse)
        self.assertIn("certified_reverse.run_reverse", source)
        self.assertNotIn("material_vjp", source)
        self.assertNotIn("adjoint_step", source)

    def test_delegate_uses_exact_certified_api_objects(self):
        self.assertIs(core.material_vjp, material_vjp)
        self.assertIs(core.adjoint_step, adjoint_step)
        self.assertIs(core.ensure_replay_cache, ensure_replay_cache)
        self.assertIs(core.load_replay_state, load_replay_state)
        self.assertIs(core.cleanup_replay_cache, cleanup_replay_cache)

    def test_seed_formula_and_reverse_order_match_certification_driver(self):
        self.assertEqual(
            _assignment(core.run_reverse, "weighted_residual"),
            FROZEN_CERTIFIED_WEIGHTED_RESIDUAL_AST,
        )
        self.assertEqual(
            _assignment(core.run_reverse, "seed"),
            FROZEN_CERTIFIED_SEED_AST,
        )
        self.assertEqual(
            _reverse_range(core.run_reverse),
            FROZEN_CERTIFIED_REVERSE_RANGE_AST,
        )
        source = inspect.getsource(core.run_reverse)
        for call in (
            "ensure_replay_cache(",
            "load_replay_state(",
            "material_vjp(",
            "adjoint_step(",
            "cleanup_replay_cache(",
            "np.add.at(",
        ):
            self.assertIn(call, source)

    def test_production_contract_is_generic_and_material_covector_only(self):
        source_path = Path(generic.__file__)
        source = source_path.read_text(encoding="utf-8")
        self.assertNotIn("/home/crellamaybe", source)
        self.assertNotIn("iter_000/", source)
        self.assertNotIn("iter_001/", source)
        self.assertNotIn("run_external_forward", source)
        self.assertNotIn("SEM3D", source)
        self.assertEqual(
            generic.GRADIENT_NAMES,
            ("solid_lambda", "solid_mu", "pml_lambda", "pml_mu"),
        )
        for option in (
            "--repo",
            "--config",
            "--iter-k",
            "--current-trace",
            "--true-trace",
            "--retained-primal-dir",
            "--output-dir",
            "--batch-size",
            "--replay-stride",
            "--reverse-checkpoint-interval",
            "--action",
        ):
            self.assertIn(option, source)

    def test_progress_contract_contains_required_resume_fields(self):
        source = inspect.getsource(core.run_reverse)
        for field in (
            '"reverse_steps"',
            '"next_transition"',
            '"elapsed_seconds"',
            '"finite"',
            '"coarse_interval"',
        ):
            self.assertIn(field, source)


if __name__ == "__main__":
    unittest.main()
