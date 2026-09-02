from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
from pathlib import Path

import h5py
import numpy as np

from scripts.fathi_benchmark.runtime_paths import (
    iteration_runtime_paths,
    resolve_path,
)


PASS_OBJECTIVE = (
    "PASS_CERTIFIED_EXTERNAL_CANDIDATE_OBJECTIVE"
)


def sha256_file(path):
    digest = hashlib.sha256()

    with Path(path).open("rb") as handle:
        for block in iter(
            lambda: handle.read(1024 * 1024),
            b"",
        ):
            digest.update(block)

    return digest.hexdigest()


def read_json(path):
    path = Path(path)

    if not path.is_file():
        raise RuntimeError(
            f"Missing JSON: {path}"
        )

    return json.loads(
        path.read_text(
            encoding="utf-8"
        )
    )


def atomic_json(path, payload):
    path = Path(path)

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary = path.with_name(
        path.name + ".tmp"
    )

    temporary.write_text(
        json.dumps(
            payload,
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    os.replace(
        temporary,
        path,
    )


def relative_or_absolute(
    root,
    path,
):
    root = Path(root).resolve()
    path = Path(path).resolve()

    try:
        return str(
            path.relative_to(root)
        )
    except ValueError:
        return str(path)


def load_h5_samples(path):
    with h5py.File(
        path,
        "r",
    ) as handle:
        if "samples" not in handle:
            raise RuntimeError(
                f"No samples dataset: {path}"
            )

        value = np.asarray(
            handle["samples"],
            dtype=np.float64,
        )

    if not np.all(
        np.isfinite(value)
    ):
        raise RuntimeError(
            f"Nonfinite material: {path}"
        )

    return value


def resolve_repo_path(
    root,
    value,
):
    path = Path(
        value
    ).expanduser()

    if not path.is_absolute():
        path = Path(root) / path

    return path.resolve()


def parent_objective(
    root,
    runtime,
    reference,
    iter_k,
):
    if int(iter_k) == 0:
        stage5n = read_json(
            resolve_repo_path(
                root,
                reference[
                    "certification_assets"
                ][
                    "stage5n_summary"
                ],
            )
        )

        return float(
            stage5n[
                "objective"
            ][
                "J_external"
            ]
        )

    parent_summary = (
        Path(
            runtime[
                "parent_workspace"
            ]
        )
        / "accepted_summary.json"
    )

    payload = read_json(
        parent_summary
    )

    return float(
        payload[
            "objective"
        ][
            "accepted"
        ]
    )


def copy_static_workspace(
    source,
    destination,
):
    source = Path(
        source
    ).resolve()

    destination = Path(
        destination
    ).resolve()

    excluded_parts = {
        "traces",
        "prot",
        "res",
        "snapshots",
        "snapshot",
        "__pycache__",
    }

    excluded_files = {
        "fin_sem",
        "output.solver",
        "output.err",
        "temps_sem.dat",
    }

    for path in source.rglob("*"):
        relative = path.relative_to(
            source
        )

        if (
            any(
                part in excluded_parts
                for part in relative.parts
            )
            or path.name in excluded_files
            or path.name.startswith(
                "output."
            )
        ):
            continue

        if relative.parts[:2] == (
            "mat",
            "h5",
        ):
            continue

        target = (
            destination
            / relative
        )

        if path.is_dir():
            target.mkdir(
                parents=True,
                exist_ok=True,
            )
            continue

        if path.is_file():
            target.parent.mkdir(
                parents=True,
                exist_ok=True,
            )

            shutil.copy2(
                path,
                target,
            )


def build_contract(args):
    root = Path(
        args.repo
    ).expanduser().resolve()

    config_path = resolve_path(
        args.config,
        base=root,
    )

    config = read_json(
        config_path
    )

    runtime = iteration_runtime_paths(
        config,
        args.iter_k,
        repo_root=root,
    )

    run = config_path.stem

    reference_path = (
        resolve_path(
            args.reference_manifest,
            base=root,
        )
        if args.reference_manifest
        else (
            root
            / "results"
            / run
            / "certified_external_reference.json"
        ).resolve()
    )

    reference = read_json(
        reference_path
    )

    if (
        reference.get("result")
        != "PASS_CERTIFIED_EXTERNAL_REFERENCE_CONTRACT"
    ):
        raise RuntimeError(
            "Reference manifest is not PASS"
        )

    transition_root = Path(
        runtime[
            "transition_root"
        ]
    ).resolve()

    candidate_root = (
        resolve_path(
            args.candidate_root,
            base=root,
        )
        if args.candidate_root
        else (
            transition_root
            / "certified_iteration"
            / "candidates"
        )
    ).resolve()

    objective_root = (
        resolve_path(
            args.candidate_objective_root,
            base=root,
        )
        if args.candidate_objective_root
        else (
            transition_root
            / "certified_iteration"
            / "candidate_external_objectives"
        )
    ).resolve()

    candidate_dir = (
        candidate_root
        / args.candidate
    ).resolve()

    material_dir = (
        candidate_dir
        / "mat"
        / "h5"
    )

    state_path = (
        candidate_dir
        / (
            f"{args.candidate}"
            "_state_candidate.npz"
        )
    )

    objective_summary = (
        objective_root
        / args.candidate
        / "summary.json"
    )

    accepted_dir = Path(
        runtime[
            "accepted_dir"
        ]
    ).resolve()

    state_out = Path(
        runtime[
            "state_out"
        ]
    ).resolve()

    audit_dir = (
        transition_root
        / "certified_iteration"
        / "promotion"
        / args.candidate
    ).resolve()

    return {
        "root": root,
        "run": run,
        "config": config,
        "config_path": config_path,
        "runtime": runtime,
        "reference_path": reference_path,
        "reference": reference,
        "transition_root": transition_root,
        "candidate_root": candidate_root,
        "objective_root": objective_root,
        "candidate_dir": candidate_dir,
        "material_dir": material_dir,
        "state_path": state_path,
        "objective_summary_path": objective_summary,
        "accepted_dir": accepted_dir,
        "state_out": state_out,
        "audit_dir": audit_dir,
    }


def audit(args, contract):
    root = contract[
        "root"
    ]

    runtime = contract[
        "runtime"
    ]

    reference = contract[
        "reference"
    ]

    objective_summary = read_json(
        contract[
            "objective_summary_path"
        ]
    )

    if (
        objective_summary.get(
            "result"
        )
        != PASS_OBJECTIVE
    ):
        raise RuntimeError(
            "Candidate certified objective "
            "is not PASS"
        )

    parent_j = parent_objective(
        root,
        runtime,
        reference,
        args.iter_k,
    )

    candidate_j = float(
        objective_summary[
            "objective"
        ][
            "objective"
        ]
    )

    if not (
        math.isfinite(
            parent_j
        )
        and math.isfinite(
            candidate_j
        )
    ):
        raise RuntimeError(
            "Nonfinite objective"
        )

    descent = (
        candidate_j
        < parent_j
    )

    if not descent:
        raise RuntimeError(
            "Candidate is not a strict descent"
        )

    material_dir = contract[
        "material_dir"
    ]

    material_files = {
        name: (
            material_dir
            / name
        )
        for name in (
            "Mat_0_Kappa.h5",
            "Mat_0_Mu.h5",
            "Mat_0_Density.h5",
        )
    }

    for path in material_files.values():
        if not path.is_file():
            raise RuntimeError(
                f"Missing candidate material: {path}"
            )

    expected_hashes = (
        objective_summary[
            "material"
        ][
            "material_sha256"
        ]
    )

    material_hashes = {
        name: sha256_file(
            path
        )
        for name, path
        in material_files.items()
    }

    if (
        material_hashes
        != expected_hashes
    ):
        raise RuntimeError(
            "Candidate material hashes differ "
            "from certified objective provenance"
        )

    state_path = contract[
        "state_path"
    ]

    if not state_path.is_file():
        raise RuntimeError(
            f"Missing candidate state: {state_path}"
        )

    with np.load(
        state_path,
        allow_pickle=False,
    ) as state:
        state_keys = set(
            state.files
        )

        required_state = {
            "lambda_",
            "lambda_field",
            "mu",
            "kappa",
            "density",
            "active_indices",
            "direction_coords",
        }

        missing = sorted(
            required_state
            - state_keys
        )

        if missing:
            raise RuntimeError(
                "Candidate state missing: "
                + ", ".join(missing)
            )

        lambda_state = np.asarray(
            state[
                "lambda_"
            ],
            dtype=np.float64,
        )

        mu_state = np.asarray(
            state[
                "mu"
            ],
            dtype=np.float64,
        )

        kappa_state = np.asarray(
            state[
                "kappa"
            ],
            dtype=np.float64,
        )

        density_state = np.asarray(
            state[
                "density"
            ],
            dtype=np.float64,
        )

    kappa_h5 = load_h5_samples(
        material_files[
            "Mat_0_Kappa.h5"
        ]
    )

    mu_h5 = load_h5_samples(
        material_files[
            "Mat_0_Mu.h5"
        ]
    )

    density_h5 = load_h5_samples(
        material_files[
            "Mat_0_Density.h5"
        ]
    )

    lambda_h5 = (
        kappa_h5
        - (2.0 / 3.0) * mu_h5
    )

    lambda_error = float(
        np.max(
            np.abs(
                lambda_h5
                - lambda_state
            )
        )
    )

    lambda_ulp = float(
        np.max(
            np.spacing(
                np.abs(
                    lambda_state
                )
            )
        )
    )

    checks = {
        "candidate_objective_pass": True,
        "strict_descent": True,
        "material_hashes": True,
        "candidate_state_exists": True,
        "mu_state_matches_h5": bool(
            np.array_equal(
                mu_state,
                mu_h5,
            )
        ),
        "kappa_state_matches_h5": bool(
            np.array_equal(
                kappa_state,
                kappa_h5,
            )
        ),
        "density_state_matches_h5": bool(
            np.array_equal(
                density_state,
                density_h5,
            )
        ),
        "lambda_roundtrip_one_ulp": bool(
            lambda_error
            <= max(
                lambda_ulp,
                np.finfo(
                    np.float64
                ).eps,
            )
        ),
        "true_external_sha256": (
            objective_summary[
                "objective"
            ][
                "true_external_sha256"
            ]
            == reference[
                "hashes"
            ][
                "true_external_sha256"
            ]
        ),
        "receiver_nodes_sha256": (
            objective_summary[
                "receiver_operator"
            ][
                "nodes_sha256"
            ]
            == reference[
                "hashes"
            ][
                "receiver_nodes_sha256"
            ]
        ),
        "receiver_weights_sha256": (
            objective_summary[
                "receiver_operator"
            ][
                "weights_sha256"
            ]
            == reference[
                "hashes"
            ][
                "receiver_weights_sha256"
            ]
        ),
    }

    if not all(
        checks.values()
    ):
        raise RuntimeError(
            "Certified promotion audit failed: "
            f"{checks}"
        )

    delta_j = (
        candidate_j
        - parent_j
    )

    relative_decrease = (
        parent_j
        - candidate_j
    ) / max(
        abs(parent_j),
        np.finfo(
            np.float64
        ).tiny,
    )

    summary = {
        "result": (
            "PASS_CERTIFIED_PROMOTION_AUDIT"
        ),
        "mode": args.mode,
        "run": contract[
            "run"
        ],
        "transition": runtime[
            "transition"
        ],
        "iter_k": int(
            args.iter_k
        ),
        "iter_kp1": (
            int(args.iter_k)
            + 1
        ),
        "candidate": args.candidate,
        "objective": {
            "parent": parent_j,
            "candidate": candidate_j,
            "delta_J": delta_j,
            "relative_decrease": (
                relative_decrease
            ),
            "descent": True,
        },
        "paths": {
            "parent_workspace": str(
                runtime[
                    "parent_workspace"
                ]
            ),
            "candidate": str(
                contract[
                    "candidate_dir"
                ]
            ),
            "candidate_objective": str(
                contract[
                    "objective_summary_path"
                ]
            ),
            "accepted_dir": str(
                contract[
                    "accepted_dir"
                ]
            ),
            "state_out": str(
                contract[
                    "state_out"
                ]
            ),
        },
        "material_sha256": (
            material_hashes
        ),
        "reference_manifest": str(
            contract[
                "reference_path"
            ]
        ),
        "reference_manifest_sha256": (
            sha256_file(
                contract[
                    "reference_path"
                ]
            )
        ),
        "true_external_sha256": (
            reference[
                "hashes"
            ][
                "true_external_sha256"
            ]
        ),
        "receiver_operator": {
            "nodes_sha256": (
                reference[
                    "hashes"
                ][
                    "receiver_nodes_sha256"
                ]
            ),
            "weights_sha256": (
                reference[
                    "hashes"
                ][
                    "receiver_weights_sha256"
                ]
            ),
        },
        "candidate_state": {
            "path": str(
                state_path
            ),
            "lambda_h5_roundtrip_max_abs_error": (
                lambda_error
            ),
            "lambda_h5_roundtrip_max_ulp": (
                lambda_ulp
            ),
        },
        "checks": checks,
    }

    contract[
        "audit_dir"
    ].mkdir(
        parents=True,
        exist_ok=True,
    )

    atomic_json(
        contract[
            "audit_dir"
        ]
        / "promotion_audit.json",
        summary,
    )

    return summary


def promote(
    args,
    contract,
    audit_summary,
):
    accepted_dir = contract[
        "accepted_dir"
    ]

    state_out = contract[
        "state_out"
    ]

    if accepted_dir.exists():
        existing_summary = (
            accepted_dir
            / "accepted_summary.json"
        )

        if existing_summary.is_file():
            existing = read_json(
                existing_summary
            )

            same = (
                existing.get(
                    "candidate"
                )
                == args.candidate
                and existing.get(
                    "material_sha256"
                )
                == audit_summary[
                    "material_sha256"
                ]
                and math.isclose(
                    float(
                        existing[
                            "objective"
                        ][
                            "accepted"
                        ]
                    ),
                    float(
                        audit_summary[
                            "objective"
                        ][
                            "candidate"
                        ]
                    ),
                    rel_tol=0.0,
                    abs_tol=0.0,
                )
            )

            if same:
                print(
                    "PROMOTION ALREADY PRESENT "
                    "WITH IDENTICAL PROVENANCE"
                )

                return existing

        if not args.force:
            raise RuntimeError(
                f"Accepted directory already exists: "
                f"{accepted_dir}"
            )

        shutil.rmtree(
            accepted_dir
        )

    temporary = accepted_dir.with_name(
        accepted_dir.name
        + f".tmp.{os.getpid()}"
    )

    if temporary.exists():
        shutil.rmtree(
            temporary
        )

    temporary.mkdir(
        parents=True,
        exist_ok=False,
    )

    copy_static_workspace(
        contract[
            "runtime"
        ][
            "parent_workspace"
        ],
        temporary,
    )

    destination_material = (
        temporary
        / "mat"
        / "h5"
    )

    destination_material.mkdir(
        parents=True,
        exist_ok=True,
    )

    for path in (
        contract[
            "material_dir"
        ]
    ).iterdir():
        if path.is_file():
            shutil.copy2(
                path,
                destination_material
                / path.name,
            )

    accepted_summary = {
        "schema_version": 1,
        "result": (
            "PASS_CERTIFIED_ACCEPTED_MODEL"
        ),
        "run": contract[
            "run"
        ],
        "transition": contract[
            "runtime"
        ][
            "transition"
        ],
        "iter_k": int(
            args.iter_k
        ),
        "iter": int(
            args.iter_k
        )
        + 1,
        "candidate": args.candidate,
        "accepted_from": str(
            contract[
                "candidate_dir"
            ]
        ),
        "accepted_dir": str(
            accepted_dir
        ),
        "objective": {
            "parent": audit_summary[
                "objective"
            ][
                "parent"
            ],
            "accepted": audit_summary[
                "objective"
            ][
                "candidate"
            ],
            "delta_J": audit_summary[
                "objective"
            ][
                "delta_J"
            ],
            "relative_decrease": (
                audit_summary[
                    "objective"
                ][
                    "relative_decrease"
                ]
            ),
            "descent": True,
        },
        "certified_candidate_objective_summary": (
            str(
                contract[
                    "objective_summary_path"
                ]
            )
        ),
        "reference_manifest": str(
            contract[
                "reference_path"
            ]
        ),
        "reference_manifest_sha256": (
            audit_summary[
                "reference_manifest_sha256"
            ]
        ),
        "true_external_sha256": (
            audit_summary[
                "true_external_sha256"
            ]
        ),
        "receiver_operator": (
            audit_summary[
                "receiver_operator"
            ]
        ),
        "material_sha256": (
            audit_summary[
                "material_sha256"
            ]
        ),
        "ordinary_capteur_traces_present": False,
        "ordinary_capteur_traces_required": False,
    }

    atomic_json(
        temporary
        / "accepted_summary.json",
        accepted_summary,
    )

    os.replace(
        temporary,
        accepted_dir,
    )

    with np.load(
        contract[
            "state_path"
        ],
        allow_pickle=False,
    ) as candidate_state:
        state = {
            key: candidate_state[
                key
            ]
            for key
            in candidate_state.files
        }

    state.update(
        {
            "J": np.asarray(
                audit_summary[
                    "objective"
                ][
                    "candidate"
                ],
                dtype=np.float64,
            ),
            "parent_J": np.asarray(
                audit_summary[
                    "objective"
                ][
                    "parent"
                ],
                dtype=np.float64,
            ),
            "delta_J": np.asarray(
                audit_summary[
                    "objective"
                ][
                    "delta_J"
                ],
                dtype=np.float64,
            ),
            "descent": np.asarray(
                True,
                dtype=np.bool_,
            ),
            "iter_k": np.asarray(
                int(args.iter_k),
                dtype=np.int64,
            ),
            "iter": np.asarray(
                int(args.iter_k) + 1,
                dtype=np.int64,
            ),
            "transition": np.asarray(
                contract[
                    "runtime"
                ][
                    "transition"
                ]
            ),
            "accepted_from": np.asarray(
                str(
                    contract[
                        "candidate_dir"
                    ]
                )
            ),
            "accepted_dir": np.asarray(
                str(
                    accepted_dir
                )
            ),
            "candidate_misfit_summary": np.asarray(
                str(
                    contract[
                        "objective_summary_path"
                    ]
                )
            ),
            "certified_candidate_objective_summary": np.asarray(
                str(
                    contract[
                        "objective_summary_path"
                    ]
                )
            ),
            "true_external_sha256": np.asarray(
                audit_summary[
                    "true_external_sha256"
                ]
            ),
            "receiver_operator_sha256": np.asarray(
                contract[
                    "reference"
                ][
                    "hashes"
                ][
                    "receiver_nodes_sha256"
                ]
                + ":"
                + contract[
                    "reference"
                ][
                    "hashes"
                ][
                    "receiver_weights_sha256"
                ]
            ),
            "objective_contract": np.asarray(
                json.dumps(
                    contract[
                        "reference"
                    ][
                        "contract"
                    ],
                    sort_keys=True,
                )
            ),
            "reference_manifest_sha256": np.asarray(
                audit_summary[
                    "reference_manifest_sha256"
                ]
            ),
            "material_kappa_sha256": np.asarray(
                audit_summary[
                    "material_sha256"
                ][
                    "Mat_0_Kappa.h5"
                ]
            ),
            "material_mu_sha256": np.asarray(
                audit_summary[
                    "material_sha256"
                ][
                    "Mat_0_Mu.h5"
                ]
            ),
            "material_density_sha256": np.asarray(
                audit_summary[
                    "material_sha256"
                ][
                    "Mat_0_Density.h5"
                ]
            ),
        }
    )

    state_out.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary_state = (
        state_out.with_name(
            state_out.name
            + ".tmp.npz"
        )
    )

    np.savez_compressed(
        temporary_state,
        **state,
    )

    os.replace(
        temporary_state,
        state_out,
    )

    return accepted_summary


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--mode",
        choices=(
            "audit",
            "promote",
        ),
        required=True,
    )

    parser.add_argument(
        "--repo",
        default=".",
    )

    parser.add_argument(
        "--config",
        required=True,
    )

    parser.add_argument(
        "--iter-k",
        type=int,
        required=True,
    )

    parser.add_argument(
        "--candidate",
        required=True,
    )

    parser.add_argument(
        "--candidate-root",
    )

    parser.add_argument(
        "--candidate-objective-root",
    )

    parser.add_argument(
        "--reference-manifest",
    )

    parser.add_argument(
        "--force",
        action="store_true",
    )

    args = parser.parse_args()

    contract = build_contract(
        args
    )

    summary = audit(
        args,
        contract,
    )

    print(
        json.dumps(
            summary,
            indent=2,
            ensure_ascii=False,
        )
    )

    if args.mode == "audit":
        print()
        print(
            "RESULT = "
            "PASS_CERTIFIED_PROMOTION_AUDIT"
        )
        return

    accepted = promote(
        args,
        contract,
        summary,
    )

    print()
    print(
        json.dumps(
            accepted,
            indent=2,
            ensure_ascii=False,
        )
    )

    print()
    print(
        "RESULT = "
        "PASS_CERTIFIED_MODEL_PROMOTION"
    )


if __name__ == "__main__":
    main()
