from __future__ import annotations

from argparse import ArgumentParser
from pathlib import Path
from runpy import run_module
import json
import subprocess
import sys


def repository_root() -> Path:
    return Path(
        subprocess.check_output(
            ['git', 'rev-parse', '--show-toplevel'],
            text=True,
        ).strip()
    ).resolve()


def resolve_path(value: str, root: Path) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = root / path
    return path.resolve()


def create_or_resolve_context(
    context_value: str | None,
    config_value: str | None,
    iter_k: int | None,
    force_context: bool,
) -> Path:
    root = repository_root()

    if context_value:
        context_path = resolve_path(context_value, root)
        if not context_path.is_file():
            raise SystemExit(f'Missing context: {context_path}')
        return context_path

    if config_value is None or iter_k is None:
        raise SystemExit('Provide either --context, or both --config and --iter-k.')

    config_path = resolve_path(config_value, root)
    if not config_path.is_file():
        raise SystemExit(f'Missing config: {config_path}')

    config = json.loads(config_path.read_text(encoding='utf-8'))
    run_result_root = resolve_path(config['run_result_root'], root)
    transition = f'iter_{iter_k:03d}_to_iter_{iter_k + 1:03d}'
    context_path = (
        run_result_root
        / transition
        / f'{transition}_iteration_context.json'
    )

    if context_path.is_file() and not force_context:
        return context_path

    command = [
        sys.executable,
        'scripts/fathi_benchmark/create_iteration_context_generic.py',
        '--iter-k',
        str(iter_k),
        '--config',
        str(config_path),
        '--write',
    ]
    if force_context:
        command.append('--overwrite')

    process = subprocess.run(command, cwd=root)
    if process.returncode != 0:
        raise SystemExit(process.returncode)
    if not context_path.is_file():
        raise SystemExit(f'Expected context was not created: {context_path}')
    return context_path


def main() -> None:
    parser = ArgumentParser(add_help=False)
    parser.add_argument('--context')
    parser.add_argument('--config')
    parser.add_argument('--iter-k', type=int)
    parser.add_argument('--force-context', action='store_true')
    known, remaining = parser.parse_known_args()
    context_path = create_or_resolve_context(
        known.context,
        known.config,
        known.iter_k,
        known.force_context,
    )
    sys.argv = [
        'scripts.fathi_benchmark.iteration_pipeline',
        '--context',
        str(context_path),
        *remaining,
    ]
    run_module(
        'scripts.fathi_benchmark.iteration_pipeline',
        run_name='__main__',
    )


if __name__ == '__main__':
    main()
