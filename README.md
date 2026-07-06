# Fathi / SEM3D Inversion Benchmark

This repository contains a context-driven local iteration engine for a SEM3D-based elastic inversion benchmark inspired by the Fathi adjoint-state workflow.

The purpose of this repository is to provide the reproducible orchestration layer, including context generation, staged execution, residual generation, adjoint batch preparation, gradient/Mtilde update, candidate generation, line-search acceptance and audit.

Large SEM3D simulation outputs are intentionally excluded from Git. HDF5 traces, protection files, runtime folders, candidate workspaces and accepted material directories should be restored separately on the local or shared filesystem.

## Current validated transition

The local transition `iter_008 -> iter_009` has been validated. The candidate `line_search_neg_mtilde_1p00MPa` produced a descent and was accepted.

## Main workflow

```text
state_k / accepted_k
    -> strict_forward
    -> residual_generation
    -> prepare_adjoint
    -> adjoint_batch x/y/z
    -> gradient / Mtilde solve
    -> candidates
    -> candidate forward + misfit
    -> acceptance
    -> state_{k+1} / accepted_{k+1}
    
    
    
    
##PyMoniK integration plan
The first PyMoniK MVP focuses on adjoint batch parallelization. After `prepare_adjoint`, the 30 adjoint batches are independent and can be submitted as distributed tasks.

Large HDF5 files are not transferred through task payloads. The payload should only contain small metadata such as context path, component and batch id.

