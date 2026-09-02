"""Register the canonical CURRENT corrected physical gradient.

CURRENT registration consumes the files emitted by the certified bridge
directly. It does not copy or rename gradient arrays and performs no Mtilde
solve. The explicitly selected legacy mode preserves the historical registry
format for reproduction audits.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping

from scripts.fathi_benchmark.current_pipeline_contracts import (
    SCHEMA_VERSION,
    accepted_model_result,
    artifact_record,
    atomic_json,
    canonical_sha256,
    exact_reverse_result,
    gradient_bridge_result,
    registered_gradient_result,
    require_identity,
    require_result,
    sha256_file,
)
from scripts.fathi_benchmark.iteration_context import (
    IterationPaths,
    build_iteration_paths,
)
from scripts.fathi_benchmark.path_consistency import (
    validate_path_config_consistency,
)
from scripts.fathi_benchmark.runtime_paths import iteration_runtime_paths


CURRENT_MANIFEST_NAME = "registered_gradient.json"
LEGACY_PASS_RESULT = "PASS_CERTIFIED_GRADIENT_REGISTRY"
LEGACY_REQUIRED_FILES = (
    "grad_lambda.npy",
    "grad_mu.npy",
    "direction_coords.npy",
    "search_direction_summary.json",
)


def _read_json(path: str | Path) -> dict[str, Any]:
    source = Path(path).expanduser().resolve()
    value = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON manifest must be an object: {source}")
    return value


def _paths(
    *,
    repo: Path,
    runtime_config: Mapping[str, Any],
    engine_config: Mapping[str, Any],
    iteration: int,
) -> IterationPaths:
    validate_path_config_consistency(
        runtime_config, engine_config, repository_root=repo
    )
    runtime = iteration_runtime_paths(
        runtime_config, int(iteration), repo_root=repo
    )
    paths = build_iteration_paths(
        engine_config,
        int(iteration),
        child_iteration=int(iteration) + 1,
        repository_root=repo,
        runtime_root=runtime["runtime_root"],
    )
    if paths.transition_root != Path(runtime["transition_root"]):
        raise ValueError("runtime/iteration-engine transition path mismatch")
    return paths


def build_current_registered_gradient_manifest(
    *,
    repo: str | Path,
    paths: IterationPaths,
) -> dict[str, Any]:
    """Validate bridge bytes and construct one loadable gradient manifest."""

    root = Path(repo).expanduser().resolve()
    iteration = paths.identity.parent_iteration
    bridge_summary_path = paths.gradient_root / "summary.json"
    solve_summary_path = paths.mtilde_solve / "mtilde_gradient_summary.json"
    reverse_summary_path = paths.exact_reverse / "production_reverse" / "summary.json"
    accepted_summary_path = paths.parent_accepted / "accepted_summary.json"

    bridge = _read_json(bridge_summary_path)
    solve = _read_json(solve_summary_path)
    reverse = _read_json(reverse_summary_path)
    accepted = _read_json(accepted_summary_path)
    expected_bridge = gradient_bridge_result(iteration)
    require_result(bridge, expected_bridge, label="CURRENT bridge")
    require_identity(bridge, paths, label="CURRENT bridge")
    require_result(solve, expected_bridge, label="CURRENT Mtilde solve")
    require_identity(solve, paths, label="CURRENT Mtilde solve")
    require_result(
        reverse, exact_reverse_result(iteration), label="CURRENT exact reverse"
    )
    if (
        int(reverse.get("iteration", -1)) != iteration
        or reverse.get("transition") != paths.identity.transition_id
    ):
        raise ValueError("CURRENT exact reverse iteration/transition mismatch")
    if (
        Path(str(bridge.get("reverse_source", ""))).expanduser().resolve()
        != reverse_summary_path.parent.resolve()
        or bridge.get("reverse_result") != exact_reverse_result(iteration)
        or bridge.get("provenance", {})
        .get("input_sha256", {})
        .get("reverse_summary")
        != sha256_file(reverse_summary_path)
    ):
        raise ValueError("bridge-to-reverse provenance mismatch")
    if (
        accepted.get("result") != accepted_model_result(iteration)
        or accepted.get("run") != paths.identity.run_id
        or int(accepted.get("iter", -1)) != iteration
    ):
        raise ValueError("accepted parent result/identity mismatch")

    files = {
        "lambda": paths.mtilde_solve / "g_lambda.npy",
        "mu": paths.mtilde_solve / "g_mu.npy",
        "coordinates": paths.mtilde_solve / "gradient_coords.npy",
        "active_indices": paths.gradient_root / "mtilde_active_full_indices.npy",
        "active_h5_indices": paths.gradient_root / "active_h5_indices.npy",
        "mtilde": paths.mtilde_solve / "Mtilde_interior_sparse.npz",
    }
    records = {
        name: artifact_record(path, repo=root) for name, path in files.items()
    }
    canonical_coords = paths.gradient_root / "mtilde_active_coords.npy"
    if sha256_file(canonical_coords) != records["coordinates"]["sha256"]:
        raise ValueError("gradient coordinate SHA differs from canonical coordinates")
    h5_solve_indices = paths.mtilde_solve / "Mtilde_interior_indices.npy"
    if sha256_file(h5_solve_indices) != records["active_h5_indices"]["sha256"]:
        raise ValueError("Mtilde solve H5 index SHA differs from bridge mapping")

    source_records = {
        "source_reverse_summary": artifact_record(reverse_summary_path, repo=root),
        "source_bridge_summary": artifact_record(bridge_summary_path, repo=root),
        "source_mtilde_summary": artifact_record(solve_summary_path, repo=root),
    }
    parent_material = {
        str(key): str(value)
        for key, value in accepted.get("material_sha256", {}).items()
    }
    if not parent_material:
        raise ValueError("accepted parent lacks material SHA provenance")
    parent = {
        "summary": artifact_record(accepted_summary_path, repo=root),
        "result": accepted_model_result(iteration),
        "material_sha256": parent_material,
    }
    signature_payload = {
        **records,
        **source_records,
        "parent_accepted_model": parent,
        "run_id": paths.identity.run_id,
        "iteration": iteration,
        "transition": paths.identity.transition_id,
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "result": registered_gradient_result(iteration),
        "run_id": paths.identity.run_id,
        "iteration": iteration,
        "parent_iteration": iteration,
        "child_iteration": paths.identity.child_iteration,
        "transition": paths.identity.transition_id,
        **records,
        **source_records,
        "parent_accepted_model": parent,
        "units": "physical Pa-space Riesz gradient",
        "ordering": "canonical active-control order",
        "registration_signature_sha256": canonical_sha256(signature_payload),
    }


def register_current_gradient(
    *,
    repo: str | Path,
    paths: IterationPaths,
    output_path: str | Path | None = None,
) -> Path:
    """Write or idempotently validate the canonical registered manifest."""

    destination = (
        Path(output_path).expanduser().resolve()
        if output_path is not None
        else (paths.gradient_root / CURRENT_MANIFEST_NAME).resolve()
    )
    if destination != (paths.gradient_root / CURRENT_MANIFEST_NAME).resolve():
        raise ValueError("CURRENT registered-gradient path is not canonical")
    payload = build_current_registered_gradient_manifest(repo=repo, paths=paths)
    if destination.is_file():
        existing = _read_json(destination)
        if existing != payload:
            raise ValueError("existing registered-gradient manifest conflicts")
        return destination
    atomic_json(destination, payload)
    return destination


def register_legacy_gradient(
    *,
    repo: Path,
    registry_path: Path,
    iteration: int,
    gradient_dir: Path,
    source: str,
) -> Path:
    """Preserve the old registry only behind an explicit historical flag."""

    files = {name: gradient_dir / name for name in LEGACY_REQUIRED_FILES}
    missing = [str(path) for path in files.values() if not path.is_file()]
    if missing:
        raise RuntimeError("missing historical gradient files: " + ", ".join(missing))
    direction = _read_json(files["search_direction_summary.json"])
    if direction.get("result") not in {"PASS", "PASS_CERTIFIED_SEARCH_DIRECTION"}:
        raise RuntimeError("historical search-direction summary is not PASS")
    if int(direction.get("iteration", -1)) != int(iteration):
        raise RuntimeError("historical search-direction iteration mismatch")
    entry = {
        "iteration": int(iteration),
        "gradient_dir": str(gradient_dir),
        "source": str(source),
        "sha256": {name: sha256_file(path) for name, path in files.items()},
    }
    registry = (
        _read_json(registry_path)
        if registry_path.is_file()
        else {
            "schema_version": 1,
            "result": LEGACY_PASS_RESULT,
            "run": registry_path.parent.name,
            "iterations": {},
        }
    )
    if registry.get("result") != LEGACY_PASS_RESULT:
        raise RuntimeError("historical registry result mismatch")
    key = str(int(iteration))
    existing = registry.setdefault("iterations", {}).get(key)
    if existing is not None and existing != entry:
        raise RuntimeError("conflicting historical gradient registration")
    registry["iterations"][key] = entry
    atomic_json(registry_path, registry)
    return registry_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", default=".")
    parser.add_argument("--config")
    parser.add_argument("--engine-config")
    parser.add_argument("--iteration", type=int, required=True)
    parser.add_argument("--output")
    parser.add_argument("--historical-legacy", action="store_true")
    parser.add_argument("--registry")
    parser.add_argument("--gradient-dir")
    parser.add_argument("--source")
    args = parser.parse_args()
    if args.iteration < 0:
        parser.error("--iteration must be nonnegative")
    repo = Path(args.repo).expanduser().resolve()
    if args.historical_legacy:
        if not all((args.registry, args.gradient_dir, args.source)):
            parser.error("historical mode requires --registry, --gradient-dir, --source")
        output = register_legacy_gradient(
            repo=repo,
            registry_path=Path(args.registry).expanduser().resolve(),
            iteration=args.iteration,
            gradient_dir=Path(args.gradient_dir).expanduser().resolve(),
            source=args.source,
        )
        print(f"RESULT = {LEGACY_PASS_RESULT}")
        print(f"REGISTRY = {output}")
        return
    if not args.config:
        parser.error("CURRENT mode requires --config")
    runtime_path = Path(args.config).expanduser().resolve()
    runtime_config = _read_json(runtime_path)
    run_id = str(runtime_config["benchmark_name"])
    engine_path = (
        Path(args.engine_config).expanduser().resolve()
        if args.engine_config
        else repo / "configs" / f"{run_id}_iteration_engine.json"
    )
    engine_config = _read_json(engine_path)
    paths = _paths(
        repo=repo,
        runtime_config=runtime_config,
        engine_config=engine_config,
        iteration=args.iteration,
    )
    output = register_current_gradient(
        repo=repo, paths=paths, output_path=args.output
    )
    print(f"RESULT = {registered_gradient_result(args.iteration)}")
    print(f"MANIFEST = {output}")


if __name__ == "__main__":
    main()
