from pathlib import Path
import os
import h5py
import numpy as np

ROOT = Path(os.environ.get("FATHI_BENCHMARK_ROOT", str(Path.home() / "sem3d_fathi_clean"))).expanduser().resolve()

files = [
    ROOT / "data/inversion_linear/iter_008/forward_dudx_mgcap_full_batches/strict_full_forward_000/traces/capteurs.0000.h5",
    ROOT / "data/inversion_linear/iter_008/adjoint_full_grid_batches/x/batch_000/traces/capteurs.0000.h5",
]

def walk(name, obj):
    if isinstance(obj, h5py.Dataset):
        print(f"DATASET {name} shape={obj.shape} dtype={obj.dtype}")
    elif isinstance(obj, h5py.Group):
        print(f"GROUP   {name}")

for p in files:
    print("=" * 100)
    print("FILE =", p)
    if not p.exists():
        print("MISSING")
        continue

    with h5py.File(p, "r") as h:
        print("attrs =", dict(h.attrs))
        print("top keys =", list(h.keys())[:30])
        print("")
        h.visititems(walk)
