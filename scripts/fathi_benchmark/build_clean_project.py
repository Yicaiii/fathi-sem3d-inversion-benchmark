#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import stat
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path, PurePosixPath

FORBIDDEN_PREFIXES = ("results/", "data/reproduction/", ".git/")
FORBIDDEN_SUFFIXES = (".h5", ".hdf5", ".npy", ".npz", ".pyc", ".log", ".jsonl")
FORBIDDEN_PARTS = {"__pycache__", "checkpoint", "checkpoints", "replay_cache", "replay_caches"}


def git(repo: Path, *args: str, stdout=None) -> str:
    p = subprocess.run(
        ["git", "-C", str(repo), *args],
        stdout=subprocess.PIPE if stdout is None else stdout,
        stderr=subprocess.PIPE,
        text=stdout is None,
    )
    if p.returncode:
        err = p.stderr.decode() if isinstance(p.stderr, bytes) else p.stderr
        raise RuntimeError(f"git {' '.join(args)} failed: {err.strip()}")
    return p.stdout if isinstance(p.stdout, str) else ""


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
    )


def verify_tree(output: Path, manifest: dict) -> None:
    expected = {row["path"]: row for row in manifest["files"]}
    actual = {}

    for path in output.rglob("*"):
        if not path.is_file():
            continue
        rel = path.relative_to(output).as_posix()
        if rel in {"PRODUCTION_SOURCE_MANIFEST.json", "CLEAN_PROJECT_BUILD.json"}:
            continue
        if forbidden(rel):
            raise RuntimeError(f"forbidden artifact present in clean project: {rel}")
        actual[rel] = path

    missing = sorted(set(expected) - set(actual))
    extra = sorted(set(actual) - set(expected))
    if missing:
        raise RuntimeError(f"clean project missing manifest files: {missing[:20]}")
    if extra:
        raise RuntimeError(f"clean project contains non-manifest files: {extra[:20]}")

    for rel, row in expected.items():
        path = actual[rel]
        if sha256_file(path) != row["sha256"]:
            raise RuntimeError(f"SHA mismatch: {rel}")
        actual_exec = bool(path.stat().st_mode & stat.S_IXUSR)
        if actual_exec != bool(row.get("executable", False)):
            raise RuntimeError(f"executable-bit mismatch: {rel}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source-repo", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--archive")
    ap.add_argument("--replace-output", action="store_true")
    args = ap.parse_args()

    source_repo = Path(args.source_repo).resolve()
    manifest_path = Path(args.manifest).resolve()
    output = Path(args.output).resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    if manifest.get("result") != "PASS_PRODUCTION_SOURCE_MANIFEST":
        raise RuntimeError("manifest is not a PASS production manifest")

    commit = str(manifest["source"]["commit"])
    if git(source_repo, "rev-parse", commit).strip() != commit:
        raise RuntimeError(f"manifest commit is not resolvable exactly: {commit}")

    if output.exists():
        if not args.replace_output:
            raise RuntimeError(f"output already exists: {output}")
        shutil.rmtree(output)
    output.mkdir(parents=True)

    try:
        with tempfile.TemporaryDirectory(prefix="fathi_clean_project_") as td:
            td = Path(td)
            tar_path = td / "source.tar"
            with tar_path.open("wb") as fh:
                git(source_repo, "archive", "--format=tar", commit, stdout=fh)

            archive_root = td / "archive"
            archive_root.mkdir()
            with tarfile.open(tar_path, "r") as tf:
                tf.extractall(archive_root)

            for row in manifest["files"]:
                rel = row["path"]
                src = archive_root / rel
                if not src.is_file():
                    raise RuntimeError(f"manifest file absent from Git archive: {rel}")
                dst = output / rel
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, dst)
                mode = dst.stat().st_mode
                if row.get("executable", False):
                    dst.chmod(mode | stat.S_IXUSR)
                else:
                    dst.chmod(mode & ~stat.S_IXUSR)

        shutil.copy2(manifest_path, output / "PRODUCTION_SOURCE_MANIFEST.json")
        verify_tree(output, manifest)

        record = {
            "schema_version": 2,
            "result": "PASS_CLEAN_PROJECT_BUILD",
            "source_commit": commit,
            "source_branch": manifest["source"].get("branch"),
            "manifest_sha256": sha256_file(manifest_path),
            "manifest_content_signature_sha256": manifest["content_signature_sha256"],
            "file_count": manifest["file_count"],
            "output": str(output),
            "source_only": True,
            "numerical_results_included": False,
        }
        (output / "CLEAN_PROJECT_BUILD.json").write_text(
            json.dumps(record, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

        if args.archive:
            archive = Path(args.archive).resolve()
            archive.parent.mkdir(parents=True, exist_ok=True)
            with tarfile.open(archive, "w:gz") as tf:
                tf.add(output, arcname=output.name)
            print("ARCHIVE =", archive)
            print("ARCHIVE_SHA256 =", sha256_file(archive))

        print("OUTPUT =", output)
        print("SOURCE_COMMIT =", commit)
        print("FILE_COUNT =", manifest["file_count"])
        print("NUMERICAL_RESULTS_INCLUDED = false")
        print("RESULT = PASS_CLEAN_PROJECT_BUILD")

    except Exception:
        if output.exists():
            shutil.rmtree(output)
        raise


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"RESULT = BLOCK_CLEAN_PROJECT_BUILD: {exc}", file=sys.stderr)
        raise
