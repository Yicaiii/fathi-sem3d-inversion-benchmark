"""Build the durable CURRENT external-forward reference from frozen provenance.

This module does not run a forward, reverse, SEM3D, MPI, Mtilde, or optimizer.
It only binds already-certified operator assets, runtime configuration, source
operator, and TRUE receiver data into the run-level reference consumed by all
future CURRENT iterations.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping

from scripts.exact_adjoint.s43_external_forward import (
    load_certified_reference,
    sha256_arrays,
    sha256_file,
)

PASS_REFERENCE = "PASS_CERTIFIED_EXTERNAL_REFERENCE_CONTRACT"
PASS_REVERSE_PREFIX = "PASS_ITER"


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"JSON payload must be an object: {path}")
    return value


def _resolve(repo: Path, value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (repo / path).resolve()


def _record_path(repo: Path, record: Mapping[str, Any], label: str) -> Path:
    if not isinstance(record, Mapping):
        raise RuntimeError(f"{label} provenance is not an object")
    value = record.get("resolved_path") or record.get("path")
    if not value:
        raise RuntimeError(f"{label} provenance lacks path")
    path = _resolve(repo, str(value))
    if not path.exists():
        raise RuntimeError(f"{label} asset is missing: {path}")
    return path


def _relative(repo: Path, path: Path) -> str:
    try:
        return str(path.resolve().relative_to(repo))
    except ValueError as exc:
        raise RuntimeError(f"reference asset lies outside repository: {path}") from exc


def _directory_signature(path: Path) -> str:
    digest = hashlib.sha256()
    for item in sorted(value for value in path.rglob("*") if value.is_file()):
        relative = item.relative_to(path).as_posix()
        digest.update(f"{sha256_file(item)}  {relative}\n".encode("utf-8"))
    return digest.hexdigest()


def _source_path(repo: Path, forward: Mapping[str, Any], key: str) -> Path | None:
    value = forward.get(key)
    if not value:
        return None
    path = _resolve(repo, str(value))
    if not path.is_file():
        raise RuntimeError(f"configured source asset is missing: {path}")
    return path


def build_reference(
    *,
    repo: Path,
    runtime_config_path: Path,
    reverse_summary_path: Path,
) -> dict[str, Any]:
    runtime = _json(runtime_config_path)
    reverse = _json(reverse_summary_path)
    run = str(runtime["benchmark_name"])
    result = str(reverse.get("result", ""))
    if not (
        result.startswith(PASS_REVERSE_PREFIX)
        and result.endswith("_EXACT_REVERSE_MATERIAL_COVECTOR")
    ):
        raise RuntimeError("source exact-reverse summary is not CURRENT certified PASS")

    inputs = reverse.get("input_hashes")
    if not isinstance(inputs, Mapping):
        raise RuntimeError("exact-reverse summary lacks input_hashes")
    assets = inputs.get("driver_assets")
    if not isinstance(assets, Mapping):
        raise RuntimeError("exact-reverse summary lacks driver_assets")

    required = (
        "topology",
        "coefficients",
        "coupled_mass",
        "gll",
        "weights",
        "receiver",
        "stf",
        "config",
    )
    missing = [name for name in required if name not in assets]
    if missing:
        raise RuntimeError("driver provenance lacks: " + ", ".join(missing))

    paths = {name: _record_path(repo, assets[name], name) for name in required}
    true_record = inputs.get("true_external_receiver")
    true_external = _record_path(repo, true_record, "true_external_receiver")
    accepted_record = inputs.get("accepted_parent_summary")
    accepted_summary = _record_path(repo, accepted_record, "accepted_parent_summary")

    config_record = assets["config"]
    config_sha = str(config_record.get("sha256", ""))
    if paths["config"] != runtime_config_path.resolve():
        raise RuntimeError(
            "certified driver config does not resolve to CURRENT runtime config: "
            f"{paths['config']} != {runtime_config_path}"
        )
    if config_sha and sha256_file(runtime_config_path) != config_sha:
        raise RuntimeError("CURRENT runtime config SHA differs from certified driver")

    for name in ("gll", "weights", "stf"):
        expected = str(assets[name].get("sha256", ""))
        if expected and sha256_file(paths[name]) != expected:
            raise RuntimeError(f"{name} SHA differs from exact-reverse provenance")
    for name in ("topology", "coefficients", "coupled_mass", "receiver"):
        expected = str(assets[name].get("content_signature_sha256", ""))
        if expected and _directory_signature(paths[name]) != expected:
            raise RuntimeError(f"{name} content signature differs from provenance")

    true_sha = sha256_file(true_external)
    if str(true_record.get("sha256", "")) != true_sha:
        raise RuntimeError("TRUE receiver SHA differs from exact-reverse provenance")

    accepted = _json(accepted_summary)
    if str(accepted.get("run", accepted.get("run_id", ""))) != run:
        raise RuntimeError("accepted-parent run differs from CURRENT reference run")
    accepted_true_sha = str(accepted.get("true_external_sha256", ""))
    if accepted_true_sha and accepted_true_sha != true_sha:
        raise RuntimeError("accepted-parent TRUE SHA differs from certified TRUE")

    receiver_nodes = paths["receiver"] / "receiver_nodes.npy"
    receiver_weights = paths["receiver"] / "receiver_weights.npy"
    if not receiver_nodes.is_file() or not receiver_weights.is_file():
        raise RuntimeError("receiver operator lacks nodes/weights arrays")

    import numpy as np

    nodes = np.asarray(np.load(receiver_nodes), dtype=np.int64)
    weights = np.asarray(np.load(receiver_weights), dtype=np.float64)
    if nodes.shape != weights.shape:
        raise RuntimeError("receiver nodes/weights shape mismatch")

    forward = runtime["forward_operator"]
    sample_count = int(forward["expected_sample_count"])
    receiver_count = int(forward["physical_receiver_count"])
    source_count = int(forward["source_count"])
    component_count = int(forward["dimension"])
    dt = float(forward["effective_dt_s"])
    if nodes.shape[0] != receiver_count:
        raise RuntimeError("receiver operator count differs from runtime contract")

    source_coordinates = _source_path(repo, forward, "source_coordinates_path")
    source_amplitudes = _source_path(repo, forward, "source_amplitudes_path")
    if source_coordinates is None or source_amplitudes is None:
        raise RuntimeError("CURRENT runtime requires file-backed source coordinates/amplitudes")
    source_xyz = np.asarray(np.load(source_coordinates), dtype=np.float64)
    source_amp = np.asarray(np.load(source_amplitudes), dtype=np.float64)
    if source_xyz.shape != (source_count, 3):
        raise RuntimeError("source coordinate shape differs from CURRENT contract")
    if source_amp.shape != (source_count,):
        raise RuntimeError("source amplitude shape differs from CURRENT contract")

    manifest = {
        "schema_version": 2,
        "result": PASS_REFERENCE,
        "run": run,
        "reference_root": f"results/{run}",
        "operator_assets": {
            "topology": _relative(repo, paths["topology"]),
            "coefficients": _relative(repo, paths["coefficients"]),
            "coupled_mass": _relative(repo, paths["coupled_mass"]),
            "gll": _relative(repo, paths["gll"]),
            "weights": _relative(repo, paths["weights"]),
            "receiver": _relative(repo, paths["receiver"]),
        },
        "certification_assets": {
            "true_external": _relative(repo, true_external),
            "current_exact_reverse_summary": _relative(repo, reverse_summary_path),
            "accepted_parent_summary": _relative(repo, accepted_summary),
        },
        "immutable_input_assets": {
            "reference_stf": _relative(repo, paths["stf"]),
            "reference_stf_sha256": sha256_file(paths["stf"]),
            "runtime_config": _relative(repo, runtime_config_path),
            "runtime_config_sha256": sha256_file(runtime_config_path),
            "source_coordinates": _relative(repo, source_coordinates),
            "source_coordinates_sha256": sha256_file(source_coordinates),
            "source_amplitudes": _relative(repo, source_amplitudes),
            "source_amplitudes_sha256": sha256_file(source_amplitudes),
        },
        "contract": {
            "sample_count": sample_count,
            "receiver_count": receiver_count,
            "component_count": component_count,
            "dt": dt,
            "source_count": source_count,
            "source_direction": list(forward["source_direction"]),
            "assembled_peak_force_n": float(forward["assembled_peak_force_n"]),
            "residual_sign": "current_external - true_external (sim - obs)",
            "time_weighting": "native fixed-dt trapezoidal quadrature",
            "candidate_formula": "parent + alpha * biased physical direction",
            "normalization": "none",
        },
        "hashes": {
            "true_external_sha256": true_sha,
            "gll_coordinates_sha256": sha256_file(paths["gll"]),
            "gll_weights_sha256": sha256_file(paths["weights"]),
            "receiver_nodes_sha256": sha256_file(receiver_nodes),
            "receiver_weights_sha256": sha256_file(receiver_weights),
            "receiver_operator_sha256": sha256_arrays(nodes, weights),
            "source_coordinates_sha256": sha256_file(source_coordinates),
            "source_amplitudes_sha256": sha256_file(source_amplitudes),
            "reference_stf_sha256": sha256_file(paths["stf"]),
            "topology_content_signature_sha256": _directory_signature(paths["topology"]),
            "coefficients_content_signature_sha256": _directory_signature(paths["coefficients"]),
            "coupled_mass_content_signature_sha256": _directory_signature(paths["coupled_mass"]),
            "receiver_content_signature_sha256": _directory_signature(paths["receiver"]),
        },
        "iteration_policy": {
            "reference_operator_is_frozen": True,
            "reference_true_external_is_frozen": True,
            "current_material_is_iteration_specific": True,
            "current_external_receiver_is_iteration_specific": True,
            "gradient_is_iteration_specific": True,
            "lbfgs_history_is_iteration_specific": True,
            "ordinary_gpu_capteur_traces_required": False,
        },
        "provenance": {
            "classification": "CURRENT_CERTIFIED_REFERENCE_FROM_FROZEN_EXACT_REVERSE",
            "exact_reverse_result": result,
            "exact_reverse_summary": _relative(repo, reverse_summary_path),
            "exact_reverse_summary_sha256": sha256_file(reverse_summary_path),
            "runtime_config_sha256": sha256_file(runtime_config_path),
            "numerical_rerun": False,
        },
    }
    return manifest


def _core_identity(manifest: Mapping[str, Any]) -> tuple[str, str, str]:
    return (
        str(manifest.get("run", "")),
        str(manifest.get("hashes", {}).get("true_external_sha256", "")),
        str(manifest.get("provenance", {}).get("classification", "")),
    )


def persist_reference(path: Path, manifest: Mapping[str, Any]) -> str:
    action = "CREATED"
    if path.is_file():
        existing = _json(path)
        if existing == dict(manifest):
            return "REUSED"
        existing_run = str(existing.get("run", ""))
        existing_true = str(existing.get("hashes", {}).get("true_external_sha256", ""))
        new_run, new_true, _ = _core_identity(manifest)
        if existing.get("result") != PASS_REFERENCE or existing_run != new_run:
            raise RuntimeError("existing certified reference belongs to another contract")
        if existing_true and existing_true != new_true:
            raise RuntimeError("existing certified reference TRUE hash conflicts")
        action = "UPGRADED"
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps(dict(manifest), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)
    return action


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", default=".")
    parser.add_argument(
        "--runtime-config",
        default="configs/fathi_s43_repro_p20_t052_runtime.json",
    )
    parser.add_argument("--reverse-summary", required=True)
    parser.add_argument("--output")
    args = parser.parse_args()

    repo = Path(args.repo).expanduser().resolve()
    runtime_path = _resolve(repo, args.runtime_config)
    reverse_path = _resolve(repo, args.reverse_summary)
    runtime = _json(runtime_path)
    run = str(runtime["benchmark_name"])
    output = (
        _resolve(repo, args.output)
        if args.output
        else (repo / "results" / run / "certified_external_reference.json").resolve()
    )
    manifest = build_reference(
        repo=repo,
        runtime_config_path=runtime_path,
        reverse_summary_path=reverse_path,
    )
    action = persist_reference(output, manifest)
    loaded_path, loaded = load_certified_reference(repo, run, output)
    if loaded_path != output or loaded != manifest:
        raise RuntimeError("persisted CURRENT reference failed loader round-trip")
    print(f"RESULT = {PASS_REFERENCE}")
    print(f"ACTION = {action}")
    print(f"REFERENCE = {output}")
    print(f"REFERENCE_SHA256 = {sha256_file(output)}")
    print(f"RUN = {run}")
    print(f"SAMPLE_COUNT = {manifest['contract']['sample_count']}")
    print(f"RECEIVER_COUNT = {manifest['contract']['receiver_count']}")
    print(f"SOURCE_COUNT = {manifest['contract']['source_count']}")
    print(f"DT = {manifest['contract']['dt']:.17e}")
    print("NUMERICAL_RERUN = false")


if __name__ == "__main__":
    main()
