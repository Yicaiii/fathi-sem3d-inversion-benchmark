# Portability Audit

## Current problem

The repository still contains assumptions tied to the original development
machine. A fresh clone on another computer cannot yet be executed without
manually editing paths.

---

## 1. Executable configuration blockers

The following configuration files contain absolute Linux paths:

- `benchmark_fathi_strict/config/benchmark_config.json`
- `benchmark_fathi_strict/config/benchmark_config_baseline_iter008.json`
- `benchmark_fathi_strict/config/benchmark_config_fathi80_tv.json`
- `configs/example_iteration_context.json`

Typical examples include:

```text
/home/crellamaybe/sem3d_fathi_clean
/home/crellamaybe/fathi-sem3d-inversion-benchmark
/home/crellamaybe/SEM/build/SEM3D/sem3d.exe
```

These paths must be replaced by:

1. project-relative paths;
2. environment variables;
3. ignored local configuration files.

---

## 2. Script defaults tied to the old workspace

Many scripts use the environment variable `FATHI_BENCHMARK_ROOT`, but still
fall back to:

```text
~/sem3d_fathi_clean
```

This is better than a fully hard-coded absolute path, but it still assumes the
old workspace layout.

The new implementation should resolve the repository root from the
configuration or from the location of the installed package.

---

## 3. Fully hard-coded paths still present

The following files contain specific hard-coded paths that must be removed:

- `generic_from_legacy/450C_prepare_strict_full_forward_run_generic.py`
- `generic_from_legacy/455B_prepare_strict_adjoint_batches_from_residual_generic.py`
- `scripts/iteration_engine/run_candidate_forward.py`
- `scripts/regularization/08_generate_tv_candidates_from_mtilde_gradient.py`

These are direct portability blockers.

---

## 4. Historical reports

Files under `reports/` contain absolute paths from the original execution.

These paths do not necessarily prevent execution because the files are
historical evidence, but they should not be interpreted as portable runtime
configuration.

Recommended treatment:

- keep only small selected validation evidence;
- clearly label reports as historical;
- do not use report paths as runtime inputs;
- exclude automatically generated reports from future commits.

---

## 5. README

The README still includes commands and paths tied to the original workstation.

It should be reduced to a portable quick start:

```bash
git clone <repository>
cd <repository>
python -m venv .venv
pip install -e .
export SEM3D_EXE=/path/to/sem3d.exe
export FATHI_RUNTIME_ROOT=/path/to/runtime
python scripts/run_benchmark.py --config configs/fathi80_small.yaml --preflight
```

---

## 6. Target environment variables

The portable workflow should use at least:

```text
SEM3D_EXE
FATHI_RUNTIME_ROOT
```

A tracked `.env.example` may document these variables.

A real `.env` file must remain untracked.

---

## 7. Validation requirement

Portability is considered resolved only when a clean clone on another computer
can perform the following without editing Python source files:

1. install the Python package;
2. load a documented configuration;
3. locate SEM3D through an environment variable;
4. run preflight validation;
5. execute a small benchmark case;
6. resume an existing iteration.
