from __future__ import annotations

from runpy import run_module


if __name__ == '__main__':
    run_module(
        'scripts.fathi_benchmark.iteration_pipeline',
        run_name='__main__',
    )
