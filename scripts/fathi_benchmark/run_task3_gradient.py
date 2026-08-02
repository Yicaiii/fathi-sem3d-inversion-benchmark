from __future__ import annotations

from argparse import ArgumentParser
from pathlib import Path
from typing import Sequence
import json
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[2]
CORE = Path(__file__).resolve().with_name(
    "run_task3_gradient_core.py"
)


def resolve_path(
    value: str | Path,
    base: Path,
) -> Path:
    path = Path(value).expanduser()

    if not path.is_absolute():
        path = base / path

    return path.resolve()


def remove_option(
    argv: Sequence[str],
    option: str,
) -> list[str]:
    result: list[str] = []
    index = 0

    while index < len(argv):
        token = argv[index]

        if token == option:
            index += 2
            continue

        if token.startswith(option + "="):
            index += 1
            continue

        result.append(token)
        index += 1

    return result


def resolve_context_path(
    root: Path,
    config_path: Path,
    iter_k: int,
    explicit_context: str | None,
) -> Path:
    if explicit_context:
        context_path = resolve_path(
            explicit_context,
            root,
        )
    else:
        config = json.loads(
            config_path.read_text(
                encoding="utf-8",
            )
        )

        run_result_root = resolve_path(
            config["run_result_root"],
            root,
        )

        transition = (
            f"iter_{iter_k:03d}_to_"
            f"iter_{iter_k + 1:03d}"
        )

        context_path = (
            run_result_root
            / transition
            / (
                f"{transition}_"
                "iteration_context.json"
            )
        )

    if not context_path.is_file():
        raise SystemExit(
            "Gradient requires an existing iteration "
            f"context: {context_path}"
        )

    return context_path


def main() -> int:
    parser = ArgumentParser(
        add_help=False,
    )
    parser.add_argument(
        "--iter-k",
        type=int,
        required=True,
    )
    parser.add_argument(
        "--config",
        required=True,
    )
    parser.add_argument(
        "--context",
    )

    known, _ = parser.parse_known_args()

    config_path = resolve_path(
        known.config,
        ROOT,
    )

    if not config_path.is_file():
        raise SystemExit(
            f"Missing benchmark config: {config_path}"
        )

    context_path = resolve_context_path(
        ROOT,
        config_path,
        known.iter_k,
        known.context,
    )

    ensure_command = [
        sys.executable,
        "-m",
        "scripts.mtilde.ensure_mtilde",
        "--config",
        str(config_path),
        "--context",
        str(context_path),
        "--execute",
    ]

    print(
        "Gradient prerequisite: ensuring Mtilde artifact"
    )
    print(
        "  " + " ".join(ensure_command)
    )

    ensure_process = subprocess.run(
        ensure_command,
        cwd=ROOT,
    )

    if ensure_process.returncode != 0:
        print(
            "RESULT = FAIL_MTILDE_PREREQUISITE"
        )
        return ensure_process.returncode

    if not CORE.is_file():
        print(
            f"Missing gradient core: {CORE}"
        )
        print(
            "RESULT = FAIL_MISSING_GRADIENT_CORE"
        )
        return 1

    forwarded = remove_option(
        sys.argv[1:],
        "--context",
    )

    core_command = [
        sys.executable,
        str(CORE),
        *forwarded,
    ]

    print(
        "Mtilde ready. Running gradient core"
    )
    print(
        "  " + " ".join(core_command)
    )

    core_process = subprocess.run(
        core_command,
        cwd=ROOT,
    )

    return core_process.returncode


if __name__ == "__main__":
    raise SystemExit(main())
