#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import re
import stat
import subprocess
import sys
from pathlib import Path, PurePosixPath

DEFAULT_ROOTS = (
    "configs/fathi_s43_repro_p20_t052_runtime.json",
    "configs/fathi_s43_repro_p20_t052_iteration_engine.json",
    "configs/fathi_s43_repro_p20_t052_immutable_assets.json",
    "scripts/fathi_benchmark/run_current_iteration.sh",
    "scripts/fathi_benchmark/run_current_iteration.py",
    "scripts/fathi_benchmark/audit_current_iteration.py",
    "scripts/fathi_benchmark/audit_current_iteration.sh",
    "scripts/fathi_benchmark/current_pipeline_artifacts.py",
    "scripts/fathi_benchmark/build_current_certified_external_reference.py",
    "scripts/fathi_benchmark/run_certified_external_parent_forward.py",
    "scripts/fathi_benchmark/run_exact_reverse_gradient_generic.py",
    "scripts/fathi_benchmark/run_certified_external_exact_reverse.py",
    "scripts/fathi_benchmark/bridge_certified_external_gradient.py",
    "scripts/fathi_benchmark/register_certified_gradient.py",
    "scripts/fathi_benchmark/run_current_pipeline.py",
    "scripts/exact_adjoint/s43_external_forward.py",
    "scripts/exact_adjoint/s43_external_reverse_core.py",
    "scripts/exact_adjoint/real_s43_global_operator.py",
    "tests/test_current_iteration_routing.py",
    "tests/test_exact_reverse_gradient_generic.py",
    "tests/test_current_pipeline_contract_repairs.py",
    "tests/test_current_pipeline_integration_static.py",
    "tests/test_bridge_certified_external_gradient_current.py",
    "CURRENT_ITERATION_RUNBOOK.md",
    "FINAL_CURRENT_ITERATION_ENGINE_CERTIFICATION.md",
    "ITER002_TO_ITER003_FINAL_CLOSURE.md",
    "scripts/fathi_benchmark/build_production_source_manifest.py",
    "scripts/fathi_benchmark/build_clean_project.py",
    "scripts/fathi_benchmark/build_production_source_manifest.sh",
    "scripts/fathi_benchmark/build_clean_project.sh",
)

OPTIONAL_ROOTS = (
    "README.md",
    "LICENSE",
    "LICENSE.md",
    "pyproject.toml",
    "requirements.txt",
    "requirements-dev.txt",
    "setup.py",
    "setup.cfg",
    "pytest.ini",
)

FORBIDDEN_PREFIXES = ("results/", "data/reproduction/", ".git/")
FORBIDDEN_SUFFIXES = (".h5", ".hdf5", ".npy", ".npz", ".pyc", ".log", ".jsonl")
FORBIDDEN_PARTS = {"__pycache__", "checkpoint", "checkpoints", "replay_cache", "replay_caches"}
LEGACY_DENY_NAMES = {
    "bridge_stage5o_certified_gradient.py",
    "424B_compute_rhs_component_from_traces.py",
    "compute_search_direction.py",
    "prepare_gpu_adjoint_full.py",
    "run_gpu_adjoint_task.py",
    "solve_gpu_mtilde_gradient.py",
}
LEGACY_DENY_PREFIXES = ("run_current_t052_", "finalize_current_t052_")

LOCAL_PATH_RE = re.compile(
    r"""(?P<path>(?:scripts|configs|tests)/[A-Za-z0-9_./-]+\.(?:py|sh|json))"""
)


def git(repo: Path, *args: str) -> str:
    p = subprocess.run(
        ["git", "-C", str(repo), *args],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if p.returncode:
        raise RuntimeError(f"git {' '.join(args)} failed: {p.stderr.strip()}")
    return p.stdout


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def forbidden(rel: str) -> bool:
    p = PurePosixPath(rel)
    return (
        any(rel.startswith(x) for x in FORBIDDEN_PREFIXES)
        or p.suffix.lower() in FORBIDDEN_SUFFIXES
        or any(x in FORBIDDEN_PARTS for x in p.parts)
        or p.name in LEGACY_DENY_NAMES
        or any(p.name.startswith(x) for x in LEGACY_DENY_PREFIXES)
    )


def module_candidates(repo: Path, module: str) -> list[str]:
    base = module.replace(".", "/")
    out = []
    for rel in (f"{base}.py", f"{base}/__init__.py"):
        if (repo / rel).is_file():
            out.append(rel)
    return out


def python_import_refs(repo: Path, rel: str) -> set[str]:
    path = repo / rel
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    except UnicodeDecodeError:
        return set()

    refs = set()
    current_parts = list(PurePosixPath(rel).with_suffix("").parts)
    package_parts = current_parts[:-1]

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                refs.update(module_candidates(repo, alias.name))
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                base_parts = package_parts[:]
                if node.level > len(base_parts) + 1:
                    continue
                if node.level > 1:
                    base_parts = base_parts[: -(node.level - 1)]
                if node.module:
                    base_parts += node.module.split(".")
                module = ".".join(base_parts)
            else:
                module = node.module or ""

            if module:
                refs.update(module_candidates(repo, module))

            for alias in node.names:
                if alias.name == "*":
                    continue
                submodule = ".".join(x for x in (module, alias.name) if x)
                refs.update(module_candidates(repo, submodule))

    return refs


def textual_local_refs(repo: Path, rel: str) -> set[str]:
    path = repo / rel
    if path.suffix not in {".py", ".sh"}:
        return set()
    try:
        text = path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return set()
    return {
        m.group("path")
        for m in LOCAL_PATH_RE.finditer(text)
        if (repo / m.group("path")).is_file()
    }


def dependency_closure(repo: Path, roots: list[str]) -> list[str]:
    tracked = set(x for x in git(repo, "ls-files").splitlines() if x.strip())
    pending = list(dict.fromkeys(roots))
    seen = set()

    while pending:
        rel = pending.pop()
        if rel in seen:
            continue
        if rel not in tracked:
            raise RuntimeError(f"required production root/dependency is not tracked: {rel}")
        if forbidden(rel):
            raise RuntimeError(f"forbidden/legacy path entered production closure: {rel}")
        if not (repo / rel).is_file():
            raise RuntimeError(f"tracked production file missing: {rel}")

        seen.add(rel)
        refs = set()
        if rel.endswith(".py"):
            refs |= python_import_refs(repo, rel)
        refs |= textual_local_refs(repo, rel)

        for ref in sorted(refs):
            if ref not in seen:
                pending.append(ref)

    return sorted(seen)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default=".")
    ap.add_argument("--output", required=True)
    ap.add_argument("--expected-branch")
    ap.add_argument("--expected-commit")
    ap.add_argument("--root", action="append", default=[])
    ap.add_argument("--no-default-roots", action="store_true")
    args = ap.parse_args()

    repo = Path(args.repo).resolve()
    output = Path(args.output).resolve()

    branch = git(repo, "branch", "--show-current").strip()
    commit = git(repo, "rev-parse", "HEAD").strip()

    if args.expected_branch and branch != args.expected_branch:
        raise RuntimeError(f"branch mismatch: {branch} != {args.expected_branch}")
    if args.expected_commit and commit != args.expected_commit:
        raise RuntimeError(f"commit mismatch: {commit} != {args.expected_commit}")
    if git(repo, "status", "--porcelain").strip():
        raise RuntimeError("source Git worktree is not clean")

    roots = []
    if not args.no_default_roots:
        roots.extend(DEFAULT_ROOTS)
        roots.extend(x for x in OPTIONAL_ROOTS if (repo / x).is_file())
    roots.extend(args.root)

    closure = dependency_closure(repo, roots)

    rows = []
    for rel in closure:
        path = repo / rel
        rows.append(
            {
                "path": rel,
                "sha256": sha256_file(path),
                "bytes": path.stat().st_size,
                "executable": bool(path.stat().st_mode & stat.S_IXUSR),
            }
        )

    digest = hashlib.sha256()
    for row in rows:
        digest.update(f'{row["sha256"]}  {row["path"]}\n'.encode("utf-8"))

    payload = {
        "schema_version": 2,
        "result": "PASS_PRODUCTION_SOURCE_MANIFEST",
        "source": {
            "repo": str(repo),
            "branch": branch,
            "commit": commit,
            "clean_worktree": True,
        },
        "production_roots": sorted(set(roots)),
        "file_count": len(rows),
        "content_signature_sha256": digest.hexdigest(),
        "files": rows,
        "policy": {
            "source_only": True,
            "legacy_routes_excluded": True,
            "numerical_results_included": False,
        },
    }

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print("BRANCH =", branch)
    print("COMMIT =", commit)
    print("ROOT_COUNT =", len(set(roots)))
    print("FILE_COUNT =", len(rows))
    print("CONTENT_SIGNATURE_SHA256 =", payload["content_signature_sha256"])
    print("OUTPUT =", output)
    print("RESULT = PASS_PRODUCTION_SOURCE_MANIFEST")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"RESULT = BLOCK_PRODUCTION_SOURCE_MANIFEST: {exc}", file=sys.stderr)
        raise
