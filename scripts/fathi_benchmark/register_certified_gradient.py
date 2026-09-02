"""Register immutable certified optimizer gradients for L-BFGS history."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path


PASS_RESULT = "PASS_CERTIFIED_GRADIENT_REGISTRY"
REQUIRED_FILES = (
    "grad_lambda.npy",
    "grad_mu.npy",
    "direction_coords.npy",
    "search_direction_summary.json",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def relative_path(path: Path, repo: Path) -> str:
    try:
        return str(path.resolve().relative_to(repo.resolve()))
    except ValueError:
        return str(path.resolve())


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", default=".")
    parser.add_argument("--registry", required=True)
    parser.add_argument("--iteration", type=int, required=True)
    parser.add_argument("--gradient-dir", required=True)
    parser.add_argument("--source", required=True)
    args = parser.parse_args()
    if args.iteration < 0:
        parser.error("--iteration must be nonnegative")

    repo = Path(args.repo).expanduser().resolve()
    registry_path = Path(args.registry).expanduser()
    if not registry_path.is_absolute():
        registry_path = repo / registry_path
    registry_path = registry_path.resolve()
    gradient_dir = Path(args.gradient_dir).expanduser()
    if not gradient_dir.is_absolute():
        gradient_dir = repo / gradient_dir
    gradient_dir = gradient_dir.resolve()
    paths = {name: gradient_dir / name for name in REQUIRED_FILES}
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise RuntimeError("missing certified gradient files: " + ", ".join(missing))

    direction_summary = json.loads(
        paths["search_direction_summary.json"].read_text(encoding="utf-8")
    )
    if direction_summary.get("result") not in {
        "PASS",
        "PASS_CERTIFIED_SEARCH_DIRECTION",
    }:
        raise RuntimeError("search direction summary is not PASS")
    if int(direction_summary["iteration"]) != int(args.iteration):
        raise RuntimeError("search direction iteration mismatch")

    key = str(int(args.iteration))
    entry = {
        "iteration": int(args.iteration),
        "gradient_dir": relative_path(gradient_dir, repo),
        "gradient_contract": (
            "certified optimizer-space physical lambda/mu gradient"
        ),
        "source": str(args.source),
        "sha256": {name: sha256_file(path) for name, path in paths.items()},
    }

    if registry_path.is_file():
        registry = json.loads(registry_path.read_text(encoding="utf-8"))
        if registry.get("result") != PASS_RESULT:
            raise RuntimeError("existing certified gradient registry is not PASS")
        existing = registry.get("iterations", {}).get(key)
        if existing is not None:
            if existing == entry:
                print(f"RESULT = {PASS_RESULT}")
                print(f"ITERATION = {args.iteration}")
                print("IDEMPOTENT_REUSE = true")
                return
            raise RuntimeError(
                f"refusing conflicting certified gradient iteration {key}"
            )
    else:
        run = registry_path.parent.name
        reference = registry_path.parent / "certified_external_reference.json"
        if not reference.is_file():
            raise RuntimeError(
                "cannot initialize registry without certified external reference"
            )
        registry = {
            "schema_version": 1,
            "result": PASS_RESULT,
            "run": run,
            "path_base": "repository_root",
            "reference_manifest": relative_path(reference, repo),
            "iterations": {},
        }

    registry.setdefault("iterations", {})[key] = entry
    atomic_json(registry_path, registry)
    print(f"RESULT = {PASS_RESULT}")
    print(f"ITERATION = {args.iteration}")
    print(f"REGISTRY = {registry_path}")


if __name__ == "__main__":
    main()
