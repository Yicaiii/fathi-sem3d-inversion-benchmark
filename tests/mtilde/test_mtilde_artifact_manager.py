from scripts.mtilde.ensure_mtilde import (
    generate_full_matrix,
    map_active_indices,
    structured_grid,
)


def test_structured_q1_and_active_mapping():
    spec = {
        "full_grid": {
            "shape_zyx": [3, 3, 3],
            "x_min_m": 0.0,
            "x_max_m": 2.0,
            "y_min_m": 0.0,
            "y_max_m": 2.0,
            "z_top_m": 0.0,
            "z_bottom_m": -2.0,
        }
    }

    x, y, z, coords = structured_grid(spec)
    matrix = generate_full_matrix(x, y, z)

    assert matrix.shape == (27, 27)
    assert coords.shape == (27, 3)
    assert matrix.diagonal().min() > 0.0

    active = coords[[0, 2, 18, 20]]
    indices = map_active_indices(coords, active, 10)

    assert indices.tolist() == [0, 2, 18, 20]
