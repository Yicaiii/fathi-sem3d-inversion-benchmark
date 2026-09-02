import ast
import hashlib
from pathlib import Path
import unittest


class CertifiedGradientBridgeUtilsEquivalenceTests(unittest.TestCase):
    def test_helper_ast_fingerprints_are_frozen(self):
        root = Path(__file__).resolve().parents[1]
        path = root / 'scripts/fathi_benchmark/certified_gradient_bridge_utils.py'
        tree = ast.parse(path.read_text(encoding='utf-8'))
        funcs = {
            node.name: node
            for node in tree.body
            if isinstance(node, ast.FunctionDef)
        }
        expected = {
            'array_stats': '269469dd48f5e9ad16b645544ace6b9b3f22a3682fff69d54048c8291f5652c8',
            'build_solid_row_map': '270604f9da18ab93cdb34e4c316b2838ab847e1b74dca8222a32a1602bdf0143',
            'configured_path': '2df7c342ca474d81a12dac918d867ded049c99f395fb99e3def006fcfce20998',
            'material_grid_coordinates': 'ceb62df9d9702c257f09d3f60700192820615a3d202b1d0fcffde3762f3e1783',
            'relative_error': 'ff92b72003155b5eb572bb5ba17545f00f5a13b054d14ae8aaa0cb3345c373a8',
            'trilinear_transpose': '68b9365b07c73fb52f3ce85dae58b9aa0adde2e222726997ea139a2b41d5748e',
        }
        self.assertEqual(set(funcs), set(expected))
        for name, digest in expected.items():
            actual = hashlib.sha256(
                ast.dump(funcs[name], include_attributes=False).encode('utf-8')
            ).hexdigest()
            self.assertEqual(actual, digest, name)

    def test_current_bridge_does_not_import_stage5o(self):
        root = Path(__file__).resolve().parents[1]
        path = root / 'scripts/fathi_benchmark/bridge_certified_external_gradient.py'
        text = path.read_text(encoding='utf-8')
        self.assertNotIn('bridge_stage5o_certified_gradient', text)
        self.assertIn('certified_gradient_bridge_utils', text)


if __name__ == '__main__':
    unittest.main()
