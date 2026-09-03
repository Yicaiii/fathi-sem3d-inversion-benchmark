"""Reusable external physical-displacement forward driver for S43.

This module evaluates the already validated benchmark-side global operator.  It
does not invoke SEM3D and does not read SEM3D snapshots during production runs.
"""

import gc
import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np

from scripts.exact_adjoint.audit_real_s43_pml_material_sensitivity import (
    pml_data_at_material,
    production_sample_xyz,
    sample_lambda_mu,
)
from scripts.exact_adjoint.real_s43_global_operator import (
    forward_step,
    load_global_data,
    material_tangent,
    state_norm,
)
from scripts.exact_adjoint.real_s43_solid_operator import load_solid_data
from scripts.exact_adjoint.run_real_s43_exact_material_gradient import zero_state


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_arrays(*arrays):
    digest = hashlib.sha256()
    for value in arrays:
        array = np.ascontiguousarray(value)
        digest.update(str(array.shape).encode("ascii"))
        digest.update(array.dtype.str.encode("ascii"))
        digest.update(array.view(np.uint8))
    return digest.hexdigest()


def atomic_save_npy(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp.npy")
    np.save(temporary, value)
    os.replace(temporary, path)


def atomic_save_npz(path, **values):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp.npz")
    np.savez(temporary, **values)
    os.replace(temporary, path)


def _reference_path(repo, value):
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = Path(repo) / path
    return path.resolve()


def load_certified_reference(repo, run, reference_manifest):
    repo = Path(repo).resolve()
    manifest_path = Path(reference_manifest).expanduser()
    if not manifest_path.is_absolute():
        manifest_path = repo / manifest_path
    manifest_path = manifest_path.resolve()

    if not manifest_path.is_file():
        raise RuntimeError(
            f"missing certified external reference manifest: {manifest_path}"
        )

    reference = json.loads(
        manifest_path.read_text(encoding="utf-8")
    )
    if not isinstance(reference, dict):
        raise RuntimeError(
            "certified external reference manifest must be a JSON object"
        )

    if (
        reference.get("result")
        != "PASS_CERTIFIED_EXTERNAL_REFERENCE_CONTRACT"
    ):
        raise RuntimeError(
            "certified external reference contract is not PASS"
        )

    if reference.get("run") != str(run):
        raise RuntimeError(
            "certified external reference run mismatch"
        )

    return manifest_path, reference


def common_paths(repo, run, reference_manifest=None):
    repo = Path(repo).resolve()

    if reference_manifest is None:
        result = repo / "results" / run / "iter_000_to_iter_001"
        return {
            "repo": repo,
            "run": run,
            "result": result,
            "reference_root": result,
            "reference_manifest": None,
            "config": repo / "configs" / f"{run}.json",
            "topology": result / "real_s43_compact_topology",
            "coefficients": result / "real_s43_pml_coefficients",
            "coupled_mass": result / "real_s43_coupled_mass",
            "gll": result / "exact_spatial_operator" / "gll_coordinates.npy",
            "weights": result / "exact_spatial_operator" / "gll_weights.npy",
            "receiver": result / "real_s43_receiver_spatial_operator",
            "baseline_material": repo
            / "data"
            / run
            / "iter_000"
            / "accepted"
            / "mat"
            / "h5",
            "stf": repo
            / "data"
            / run
            / "iter_000"
            / "accepted"
            / "gaussian_stf.txt",
        }

    manifest_path, reference = load_certified_reference(
        repo,
        run,
        reference_manifest,
    )

    operator = reference["operator_assets"]
    certification = reference["certification_assets"]
    immutable = reference["immutable_input_assets"]

    reference_root = _reference_path(
        repo,
        reference["reference_root"],
    )

    runtime_config_value = immutable.get("runtime_config")
    if runtime_config_value:
        runtime_config = _reference_path(repo, runtime_config_value)
    else:
        # Historical certified references predate explicit runtime-config
        # routing. Keep their frozen configs/<run>.json behavior unchanged.
        runtime_config = (repo / "configs" / f"{run}.json").resolve()

    paths = {
        "repo": repo,
        "run": run,
        "result": reference_root,
        "reference_root": reference_root,
        "reference_manifest": manifest_path,
        "config": runtime_config,
        "topology": _reference_path(
            repo,
            operator["topology"],
        ),
        "coefficients": _reference_path(
            repo,
            operator["coefficients"],
        ),
        "coupled_mass": _reference_path(
            repo,
            operator["coupled_mass"],
        ),
        "gll": _reference_path(
            repo,
            operator["gll"],
        ),
        "weights": _reference_path(
            repo,
            operator["weights"],
        ),
        "receiver": _reference_path(
            repo,
            operator["receiver"],
        ),
        "baseline_material": repo
        / "data"
        / run
        / "iter_000"
        / "accepted"
        / "mat"
        / "h5",
        "stf": _reference_path(
            repo,
            immutable["reference_stf"],
        ),
        "true_external": _reference_path(
            repo,
            certification["true_external"],
        ),
    }

    required = (
        "config",
        "topology",
        "coefficients",
        "coupled_mass",
        "gll",
        "weights",
        "receiver",
        "stf",
        "true_external",
    )

    missing = [
        f"{name}: {paths[name]}"
        for name in required
        if not Path(paths[name]).exists()
    ]

    if missing:
        raise RuntimeError(
            "missing certified external reference assets:\n"
            + "\n".join(missing)
        )

    return paths


def _material_files(material_dir):
    material_dir = Path(material_dir)
    kappa = material_dir / "Mat_0_Kappa.h5"
    mu = material_dir / "Mat_0_Mu.h5"
    if not kappa.is_file() or not mu.is_file():
        raise RuntimeError(f"missing Kappa/Mu material H5 under {material_dir}")
    return kappa, mu


def _rebuild_pml_for_material(data, paths, material_dir):
    """Apply the established production PML material sampling contract."""
    cfg = json.loads(paths["config"].read_text(encoding="utf-8"))
    topology = paths["topology"]
    pml = data["pml"]
    conn = np.asarray(
        np.load(topology / "pml_connectivity_compact.npy"), dtype=np.int64
    )
    xyz = np.asarray(np.load(topology / "pml_compact_xyz.npy"), dtype=np.float64)
    region = np.asarray(
        np.load(topology / "pml_element_region_code.npy"), dtype=np.uint8
    )
    sample_xyz = production_sample_xyz(xyz[conn], region, cfg)
    target_lam, target_mu = sample_lambda_mu(
        material_dir, sample_xyz, cfg, pml["lam"].shape
    )
    dlam = target_lam - pml["lam"]
    dmu = target_mu - pml["mu"]
    audit = {
        "lambda_support": int(np.count_nonzero(dlam)),
        "lambda_max_abs_change_pa": float(np.max(np.abs(dlam))),
        "mu_support": int(np.count_nonzero(dmu)),
        "mu_max_abs_change_pa": float(np.max(np.abs(dmu))),
        "rebuild_applied": bool(np.any(dlam) or np.any(dmu)),
    }
    if audit["rebuild_applied"]:
        data["pml"] = pml_data_at_material(pml, dlam, dmu, 1.0)
        if not np.array_equal(data["pml"]["lam"], target_lam):
            raise RuntimeError("PML lambda material rebuild mismatch")
        if not np.array_equal(data["pml"]["mu"], target_mu):
            raise RuntimeError("PML mu material rebuild mismatch")
    return audit


class ExternalForwardDriver:
    def __init__(
        self,
        repo,
        run,
        material_dir,
        batch_size=2048,
        reference_manifest=None,
    ):
        self.paths = common_paths(
            repo,
            run,
            reference_manifest=reference_manifest,
        )
        self.material_dir = Path(material_dir).resolve()
        self.batch_size = int(batch_size)
        kappa, mu = _material_files(self.material_dir)
        self.data = load_global_data(
            self.paths["config"],
            self.paths["topology"],
            self.paths["coefficients"],
            self.paths["coupled_mass"],
            self.paths["gll"],
            self.paths["weights"],
            kappa,
            mu,
            batch_size=self.batch_size,
        )
        self.pml_material_audit = _rebuild_pml_for_material(
            self.data, self.paths, self.material_dir
        )
        self.dt = float(self.data["solid"]["dt"])
        self.receiver_nodes = np.asarray(
            np.load(self.paths["receiver"] / "receiver_nodes.npy"), dtype=np.int64
        )
        self.receiver_weights = np.asarray(
            np.load(self.paths["receiver"] / "receiver_weights.npy"),
            dtype=np.float64,
        )
        if self.receiver_nodes.shape != self.receiver_weights.shape:
            raise RuntimeError("receiver node/weight shape mismatch")

        cfg = json.loads(self.paths["config"].read_text(encoding="utf-8"))
        forward = cfg["forward_operator"]

        # CURRENT_T052_WEIGHTED_SOURCE_BRIDGE
        source_count = int(
            forward["source_count"]
        )

        def configured_array(value, label):
            path = Path(value).expanduser()

            if not path.is_absolute():
                path = (
                    self.paths["repo"]
                    /
                    path
                )

            path = path.resolve()

            if not path.is_file():
                raise RuntimeError(
                    f"missing configured {label}: {path}"
                )

            return (
                np.asarray(
                    np.load(path),
                    dtype=np.float64,
                ),
                path,
            )

        coordinates_value = forward.get(
            "source_coordinates_path"
        )

        if coordinates_value:
            (
                source_xyz,
                source_coordinates_path,
            ) = configured_array(
                coordinates_value,
                "source coordinates",
            )
        else:
            source_xyz = np.asarray(
                forward["source_coordinates_m"],
                dtype=np.float64,
            )
            source_coordinates_path = None

        source_direction = np.asarray(
            forward["source_direction"],
            dtype=np.float64,
        )

        amplitudes_value = forward.get(
            "source_amplitudes_path"
        )

        if amplitudes_value:
            (
                source_amplitudes,
                source_amplitudes_path,
            ) = configured_array(
                amplitudes_value,
                "source amplitudes",
            )
        else:
            source_amplitudes = np.ones(
                source_count,
                dtype=np.float64,
            )
            source_amplitudes_path = None

        if source_xyz.shape != (
            source_count,
            3,
        ):
            raise RuntimeError(
                "configured source coordinate shape mismatch: "
                f"{source_xyz.shape}"
            )

        if source_amplitudes.shape != (
            source_count,
        ):
            raise RuntimeError(
                "configured source amplitude shape mismatch: "
                f"{source_amplitudes.shape}"
            )

        if source_direction.shape != (3,):
            raise RuntimeError(
                "source direction must have shape (3,)"
            )

        if not np.all(
            np.isfinite(
                source_amplitudes
            )
        ):
            raise RuntimeError(
                "nonfinite source amplitudes"
            )

        expected_force = forward.get(
            "assembled_peak_force_n"
        )

        if (
            expected_force is not None
            and not np.isclose(
                float(
                    np.sum(
                        source_amplitudes
                    )
                ),
                float(
                    expected_force
                ),
                rtol=1.0e-12,
                atol=1.0e-6,
            )
        ):
            raise RuntimeError(
                "configured assembled source force mismatch"
            )
        solid_xyz = self.data["solid"]["xyz"]
        source_nodes = []
        source_errors = []
        for position in source_xyz:
            distance = np.max(np.abs(solid_xyz - position), axis=1)
            node = int(np.argmin(distance))
            source_nodes.append(node)
            source_errors.append(float(distance[node]))
        self.source_nodes = np.asarray(source_nodes, dtype=np.int64)
        self.source_direction = source_direction
        self.source_amplitudes = source_amplitudes
        self.source_coordinates_path = source_coordinates_path
        self.source_amplitudes_path = source_amplitudes_path
        self.source_coordinate_error = float(max(source_errors, default=0.0))
        if self.source_coordinate_error > 1.0e-12:
            raise RuntimeError(
                f"source coordinate mapping error {self.source_coordinate_error}"
            )
        interface_nodes = set(
            np.asarray(self.data["interface"]["solid_idx"], dtype=np.int64).tolist()
        )
        self.source_interface_count = sum(
            int(node in interface_nodes) for node in self.source_nodes
        )
        if self.source_interface_count:
            raise RuntimeError("S43 source mapped to solid/PML interface")

        stf = np.asarray(np.loadtxt(self.paths["stf"]), dtype=np.float64)
        if stf.ndim != 2 or stf.shape[1] < 2 or not np.all(np.diff(stf[:, 0]) > 0):
            raise RuntimeError("invalid file STF")
        self.stf_time = stf[:, 0]
        self.stf_value = stf[:, 1]
        source_amplitude_signature = (
            sha256_arrays(
                self.source_amplitudes
            )
            if self.source_amplitudes_path
            is not None
            else ""
        )

        self.signature = hashlib.sha256(
            (
                sha256_file(kappa)
                + sha256_file(mu)
                + sha256_file(self.paths["stf"])
                + source_amplitude_signature
                + sha256_arrays(
                    self.receiver_nodes,
                    self.receiver_weights,
                    self.source_nodes,
                    self.source_direction,
                )
            ).encode("ascii")
        ).hexdigest()
        self.audit = {
            "material_dir": str(self.material_dir),
            "material_kappa_sha256": sha256_file(kappa),
            "material_mu_sha256": sha256_file(mu),
            "dt": self.dt,
            "receiver_count": int(len(self.receiver_nodes)),
            "source_count": int(len(self.source_nodes)),
            "source_amplitude_min": float(
                np.min(self.source_amplitudes)
            ),
            "source_amplitude_max": float(
                np.max(self.source_amplitudes)
            ),
            "source_amplitude_sum": float(
                np.sum(self.source_amplitudes)
            ),
            "source_coordinate_max_abs_error": self.source_coordinate_error,
            "source_interface_count": int(self.source_interface_count),
            "source_evaluation": "linear file-STF interpolation at t_n + dt/2",
            "first_source_evaluation_time": 0.5 * self.dt,
            "first_source_amplitude": self.source_amplitude(0),
            "pml_material": self.pml_material_audit,
            "signature_sha256": self.signature,
        }

    @property
    def receiver_count(self):
        return int(len(self.receiver_nodes))

    def source_amplitude(self, transition):
        return float(
            np.interp(
                (float(transition) + 0.5) * self.dt,
                self.stf_time,
                self.stf_value,
            )
        )

    def receiver(self, state):
        return np.sum(
            state[0][self.receiver_nodes] * self.receiver_weights[..., None], axis=1
        )

    def advance(self, state, transition):
        out = list(forward_step(self.data, state))
        amplitude = self.source_amplitude(transition)
        acceleration = (
            self.data["solid"]["invmass"][self.source_nodes, None]
            * amplitude
            * self.source_amplitudes[:, None]
            * self.source_direction[None, :]
        )
        np.add.at(out[1], self.source_nodes, self.dt * acceleration)
        np.add.at(out[0], self.source_nodes, self.dt * self.dt * acceleration)
        return tuple(out)


def load_material_direction(driver, manifest_path, component, half_step_pa):
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    workspaces = {
        (row["component"], row["sign"]): Path(row["workspace"])
        for row in manifest["cases"]
    }
    values = {}
    paths = driver.paths
    for sign in ("plus", "minus"):
        material = workspaces[(component, sign)] / "mat" / "h5"
        kappa, mu = _material_files(material)
        candidate = load_solid_data(
            paths["config"],
            paths["topology"],
            paths["coefficients"],
            paths["coupled_mass"],
            paths["gll"],
            paths["weights"],
            kappa,
            mu,
            batch_size=driver.batch_size,
        )
        values[sign] = (
            np.array(candidate["lam"], copy=True),
            np.array(candidate["mu"], copy=True),
        )
        del candidate
        gc.collect()
    denominator = 2.0 * float(half_step_pa)
    dlam = (values["plus"][0] - values["minus"][0]) / denominator
    dmu = (values["plus"][1] - values["minus"][1]) / denominator
    return {
        "solid_dlam": dlam,
        "solid_dmu": dmu,
        "pml_dlam": np.zeros_like(driver.data["pml"]["lam"]),
        "pml_dmu": np.zeros_like(driver.data["pml"]["mu"]),
        "audit": {
            "component": component,
            "half_step_pa": float(half_step_pa),
            "solid_dlambda_support": int(np.count_nonzero(dlam)),
            "solid_dlambda_max_abs": float(np.max(np.abs(dlam))),
            "solid_dmu_support": int(np.count_nonzero(dmu)),
            "solid_dmu_max_abs": float(np.max(np.abs(dmu))),
            "direction_sha256": sha256_arrays(dlam, dmu),
            "pml_direction_exactly_zero": True,
        },
    }


def _checkpoint_values(completed, signature, states):
    values = {
        "completed": np.asarray(completed, dtype=np.int64),
        "signature_sha256": np.asarray(signature),
        "labels": np.asarray(sorted(states)),
    }
    for label, state in states.items():
        for index, name in enumerate(("Us", "Vs", "Vp", "Sp")):
            values[f"{label}_{name}"] = state[index]
    return values


def _load_checkpoint(path, signature):
    with np.load(path) as saved:
        if str(saved["signature_sha256"].item()) != signature:
            raise RuntimeError("external-forward checkpoint signature mismatch")
        completed = int(saved["completed"])
        labels = [str(value) for value in saved["labels"]]
        states = {
            label: tuple(
                np.asarray(saved[f"{label}_{name}"])
                for name in ("Us", "Vs", "Vp", "Sp")
            )
            for label in labels
        }
    return completed, states


def run_external_forward(
    driver,
    target,
    receiver_paths,
    checkpoint_path,
    tangent_directions=None,
    checkpoint_interval=100,
    retained_primal_dir=None,
):
    """Run or resume a primal forward and optional forward-mode tangents."""
    target = int(target)
    tangent_directions = tangent_directions or {}
    labels = ["primal", *sorted(tangent_directions)]
    signature = hashlib.sha256(
        (
            driver.signature
            + "|"
            + "|".join(
                f"{name}:{tangent_directions[name]['audit']['direction_sha256']}"
                for name in sorted(tangent_directions)
            )
        ).encode("ascii")
    ).hexdigest()
    checkpoint_path = Path(checkpoint_path)
    if checkpoint_path.is_file():
        completed, states = _load_checkpoint(checkpoint_path, signature)
    else:
        completed = 0
        states = {label: zero_state(driver.data) for label in labels}
    if set(states) != set(labels):
        raise RuntimeError("checkpoint state labels mismatch")
    if completed > target:
        raise RuntimeError("checkpoint is beyond requested target")

    traces = {}
    for label in labels:
        path = Path(receiver_paths[label])
        if completed:
            prior = np.asarray(np.load(path), dtype=np.float64)
            if prior.shape != (completed, driver.receiver_count, 3):
                raise RuntimeError(f"{label} trace/checkpoint mismatch: {prior.shape}")
        else:
            prior = np.empty((0, driver.receiver_count, 3), dtype=np.float64)
        trace = np.empty((target, driver.receiver_count, 3), dtype=np.float64)
        trace[:completed] = prior
        traces[label] = trace

    started = time.time()
    zero_pml_verified = completed > 0
    for transition in range(completed, target):
        primal_pre = states["primal"]
        primal_post = driver.advance(primal_pre, transition)
        next_states = {"primal": primal_post}
        for label in sorted(tangent_directions):
            direction = tangent_directions[label]
            propagated = forward_step(driver.data, states[label])
            forcing = material_tangent(
                driver.data,
                primal_pre,
                direction["solid_dlam"],
                direction["solid_dmu"],
                direction["pml_dlam"],
                direction["pml_dmu"],
            )
            if not zero_pml_verified:
                if np.any(forcing[2]) or np.any(forcing[3]):
                    raise RuntimeError("zero PML direction produced PML forcing")
            next_states[label] = (
                propagated[0] + forcing[0],
                propagated[1] + forcing[1],
                propagated[2],
                propagated[3],
            )
            del propagated, forcing
        zero_pml_verified = True
        states = next_states
        for label in labels:
            traces[label][transition] = driver.receiver(states[label])

        count = transition + 1
        save_now = count % int(checkpoint_interval) == 0 or count == target
        if save_now:
            for label in labels:
                atomic_save_npy(receiver_paths[label], traces[label][:count])
            atomic_save_npz(
                checkpoint_path, **_checkpoint_values(count, signature, states)
            )
            if retained_primal_dir is not None:
                retained = Path(retained_primal_dir) / f"primal_{count:06d}.npz"
                if not retained.is_file():
                    atomic_save_npz(
                        retained,
                        **_checkpoint_values(
                            count, driver.signature, {"primal": states["primal"]}
                        ),
                    )
            elapsed = time.time() - started
            norms = ", ".join(
                f"{label}={state_norm(states[label]):.6e}" for label in labels
            )
            print(
                f"completed {count}/{target}; {norms}; elapsed={elapsed:.1f}s",
                flush=True,
            )

    return {
        "completed": target,
        "elapsed_seconds": float(time.time() - started),
        "signature_sha256": signature,
        "driver": driver.audit,
        "state_norms": {label: float(state_norm(states[label])) for label in labels},
        "receiver_norms": {
            label: float(np.linalg.norm(traces[label].reshape(-1))) for label in labels
        },
        "tangent_directions": {
            label: tangent_directions[label]["audit"] for label in tangent_directions
        },
    }
