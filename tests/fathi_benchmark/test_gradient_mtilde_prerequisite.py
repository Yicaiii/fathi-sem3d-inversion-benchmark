from pathlib import Path
import json

from scripts.fathi_benchmark.run_task3_gradient import (
    remove_option,
    resolve_context_path,
)


def test_remove_context_option():
    argv = [
        "--iter-k",
        "0",
        "--context",
        "/tmp/context.json",
        "--stage",
        "all",
    ]

    assert remove_option(
        argv,
        "--context",
    ) == [
        "--iter-k",
        "0",
        "--stage",
        "all",
    ]


def test_resolve_context_from_config(tmp_path):
    run_root = tmp_path / "results"
    transition = "iter_000_to_iter_001"
    context = (
        run_root
        / transition
        / (
            transition
            + "_iteration_context.json"
        )
    )
    context.parent.mkdir(
        parents=True,
    )
    context.write_text(
        "{}\n",
        encoding="utf-8",
    )

    config = tmp_path / "config.json"
    config.write_text(
        json.dumps(
            {
                "run_result_root": str(
                    run_root
                )
            }
        )
        + "\n",
        encoding="utf-8",
    )

    resolved = resolve_context_path(
        tmp_path,
        config,
        0,
        None,
    )

    assert resolved == context.resolve()
