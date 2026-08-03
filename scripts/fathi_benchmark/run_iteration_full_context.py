from pathlib import Path
from runpy import run_module
import sys

ROOT = Path(__file__).resolve().parents[2]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

if __name__ == "__main__":
    run_module(
        "scripts.fathi_benchmark.iteration_pipeline",
        run_name="__main__",
    )
