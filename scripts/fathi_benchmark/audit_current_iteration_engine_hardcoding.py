"""Deterministic static-literal audit for the generalized CURRENT route."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path


SEARCH_TERMS = (
    "iter_000",
    "iter_001",
    "iter_002",
    "fathi_s43_gpu_p40_np2_apow13p8155_pml6p25_t052",
    "/home/crellamaybe",
    "3249",
    "1040",
    "34596",
    "joint_maxabs",
)

PRODUCTION_ROUTE = {
    "scripts/fathi_benchmark/generic_iteration_runner.py",
    "scripts/fathi_benchmark/immutable_assets.py",
    "scripts/fathi_benchmark/iteration_context.py",
    "scripts/fathi_benchmark/lbfgs_history.py",
    "scripts/fathi_benchmark/optimizer_state.py",
    "scripts/fathi_benchmark/external_armijo.py",
    "scripts/fathi_benchmark/path_consistency.py",
    "scripts/fathi_benchmark/physical_space_optimizer.py",
    "scripts/fathi_benchmark/run_iteration_engine_structural_dry_run.py",
    "scripts/fathi_benchmark/runtime_paths.py",
    "configs/fathi_s43_repro_p20_t052_iteration_engine.json",
    "configs/fathi_s43_repro_p20_t052_immutable_assets.json",
    "configs/fathi_s43_repro_p20_t052_runtime.json",
}

FROZEN_COMPATIBILITY_PREFIXES = (
    "scripts/fathi_benchmark/run_current_t052_",
    "scripts/fathi_benchmark/finalize_current_t052_",
)

FROZEN_LEGACY_FILES = {
    "scripts/fathi_benchmark/accept_certified_external_candidate.py",
    "scripts/fathi_benchmark/audit_historical_operator_assets.py",
    "scripts/fathi_benchmark/run_certified_line_search.py",
}


def classification(path: str, term: str) -> tuple[str, str]:
    if path.startswith("tests/"):
        return "TEST_FIXTURE", "fixed path/name regression expectation"
    if path in PRODUCTION_ROUTE and not path.startswith("configs/"):
        return "BUG", "suspicious literal in reusable production route"
    if path.startswith(FROZEN_COMPATIBILITY_PREFIXES) or path in FROZEN_LEGACY_FILES:
        return (
            "HISTORICAL_ONLY",
            "frozen completed-transition or superseded-route compatibility source",
        )
    if path.startswith("configs/"):
        return (
            "CONFIG_DEFAULT",
            "scientific/runtime profile value retained as configuration data",
        )
    return "BUG", "unclassified occurrence in audited production-script tree"


def json_classification(
    path: str, pointer: str, term: str
) -> tuple[str, str, bool]:
    """Classify config values by semantic JSON key, not file extension."""

    if path.endswith("_immutable_assets.json"):
        return (
            "HISTORICAL_ONLY",
            "portable immutable certified-asset source provenance",
            False,
        )
    if path.endswith("_iteration_engine.json"):
        if pointer == "/historical_run_id":
            return (
                "JUSTIFIED_STATIC_REFERENCE",
                "explicit historical namespace isolation guard",
                False,
            )
        if pointer.startswith("/immutable_operator_assets"):
            return (
                "HISTORICAL_ONLY",
                "explicit immutable certified-asset manifest/provenance",
                False,
            )
        if pointer.startswith("/namespace") or pointer.startswith("/routes"):
            return (
                "BUG",
                "literal appears in an active mutable iteration route",
                True,
            )
        return (
            "CONFIG_DEFAULT",
            "fixed scientific/optimizer configuration, not a mutable path",
            False,
        )
    if path.endswith("_runtime.json"):
        if pointer.startswith("/runtime_layout"):
            return (
                "BUG",
                "literal appears in active reusable runtime_layout",
                True,
            )
        if pointer in {
            "/dt_stability_certification/workspace",
            "/production_timestep_certification/certification_workspace",
            "/material_h5_active_indices_path",
        }:
            return (
                "HISTORICAL_ONLY",
                "frozen certification or superseded completed-route provenance",
                False,
            )
        return (
            "CONFIG_DEFAULT",
            "scientific count/profile value consumed semantically from config",
            False,
        )
    return "BUG", "unexpected JSON file in audit", False


def _json_pointer_token(value: str) -> str:
    return value.replace("~", "~0").replace("/", "~1")


def config_findings(path: Path, relative: str) -> list[dict]:
    value = json.loads(path.read_text(encoding="utf-8"))
    findings = []

    def visit(item, pointer: str) -> None:
        if isinstance(item, dict):
            for key, child in item.items():
                visit(child, pointer + "/" + _json_pointer_token(str(key)))
            return
        if isinstance(item, list):
            for index, child in enumerate(item):
                visit(child, pointer + f"/{index}")
            return
        rendered = str(item)
        for term in SEARCH_TERMS:
            occurrence_count = rendered.count(term)
            if not occurrence_count:
                continue
            category, reason, active_mutable_route = json_classification(
                relative, pointer, term
            )
            for _ in range(occurrence_count):
                findings.append(
                    {
                        "path": relative,
                        "json_pointer": pointer,
                        "literal": term,
                        "classification": category,
                        "active_mutable_route": active_mutable_route,
                        "reason": reason,
                    }
                )

    visit(value, "")
    return findings


def audited_files(repo: Path, output: Path) -> list[Path]:
    values = []
    for path in (repo / "scripts" / "fathi_benchmark").iterdir():
        if not path.is_file():
            continue
        name = path.name
        if (
            path.resolve() == Path(__file__).resolve()
            or ".before_" in name
            or ".bak_" in name
            or name.endswith(".pyc")
        ):
            continue
        values.append(path)
    values.extend(
        [
            repo / "configs" / "fathi_s43_repro_p20_t052_runtime.json",
            repo / "configs" / "fathi_s43_repro_p20_t052_iteration_engine.json",
            repo / "configs" / "fathi_s43_repro_p20_t052_immutable_assets.json",
        ]
    )
    values.extend(sorted((repo / "tests").glob("test_*.py")))
    return sorted({path.resolve() for path in values if path.is_file() and path != output})


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    repo = Path(args.repo).expanduser().resolve()
    output = Path(args.output).expanduser().resolve()
    findings = []
    for path in audited_files(repo, output):
        relative = path.relative_to(repo).as_posix()
        if path.suffix == ".json":
            findings.extend(config_findings(path, relative))
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        for line_number, line in enumerate(text.splitlines(), start=1):
            for term in SEARCH_TERMS:
                start = 0
                while True:
                    column = line.find(term, start)
                    if column < 0:
                        break
                    category, reason = classification(relative, term)
                    findings.append(
                        {
                            "path": relative,
                            "line": line_number,
                            "column": column + 1,
                            "literal": term,
                            "classification": category,
                            "active_mutable_route": False,
                            "reason": reason,
                        }
                    )
                    start = column + len(term)

    counts = Counter(item["classification"] for item in findings)
    production_findings = [
        item for item in findings if item["path"] in PRODUCTION_ROUTE
    ]
    production_bug_count = sum(
        item["classification"] == "BUG" for item in production_findings
    )
    active_runtime_findings = [
        item for item in findings if item.get("active_mutable_route")
    ]
    payload = {
        "schema_version": 1,
        "result": (
            "PASS_CURRENT_ITERATION_ENGINE_HARDCODING_AUDIT"
            if production_bug_count == 0
            else "FAIL_CURRENT_ITERATION_ENGINE_HARDCODING_AUDIT"
        ),
        "scope": {
            "production_route": sorted(PRODUCTION_ROUTE),
            "compatibility_and_config_review": (
                "all ordinary files in scripts/fathi_benchmark, the current runtime "
                "and iteration-engine configs, and test_*.py; backup snapshots excluded"
            ),
        },
        "search_terms": list(SEARCH_TERMS),
        "classification_vocabulary": [
            "JUSTIFIED_STATIC_REFERENCE",
            "HISTORICAL_ONLY",
            "TEST_FIXTURE",
            "CONFIG_DEFAULT",
            "BUG",
        ],
        "summary": {
            "occurrence_count": len(findings),
            "classification_counts": dict(sorted(counts.items())),
            "production_occurrence_count": len(production_findings),
            "production_bug_count": production_bug_count,
            "active_runtime_iteration_literals": sum(
                item["literal"] in {"iter_000", "iter_001", "iter_002"}
                for item in active_runtime_findings
            ),
            "active_runtime_absolute_home_paths": sum(
                item["literal"] == "/home/crellamaybe"
                for item in active_runtime_findings
            ),
            "hardcoded_absolute_paths_in_production": sum(
                item["literal"] == "/home/crellamaybe"
                for item in production_findings
            ),
            "production_iteration_literal_count": sum(
                item["literal"] in {"iter_000", "iter_001", "iter_002"}
                for item in production_findings
            ),
            "joint_maxabs_in_production": sum(
                item["literal"] == "joint_maxabs"
                for item in production_findings
            ),
            "historical_run_literal_in_production": sum(
                item["literal"]
                == "fathi_s43_gpu_p40_np2_apow13p8155_pml6p25_t052"
                for item in production_findings
            ),
        },
        "findings": findings,
        "notes": [
            "Config values are classified by semantic JSON pointer; active mutable routes are not presumed harmless defaults.",
            "Frozen current_t052 and superseded certified-line-search sources remain non-destructive audit evidence, not future production entry points.",
            "Configured scientific counts are CONFIG_DEFAULT values and are consumed from config/manifests rather than duplicated in reusable execution code.",
        ],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(output)
    if production_bug_count:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
