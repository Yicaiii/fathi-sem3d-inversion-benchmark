# Canonical Fathi 80 MPa initialization

## Material model

The canonical initial model is homogeneous:

- lambda = 80 MPa
- mu = 80 MPa
- Kappa = lambda + 2 mu / 3 = 133.333333 MPa
- density = 2000 kg/m3
- material-array shape = (41, 33, 33)

The initialization is created once with:

```bash
python scripts/bootstrap/prepare_fathi80_initial.py \
  --config configs/fathi80_initial.json \
  --template /path/to/local/sem3d/template \
  --write
Forward operator

The canonical validation uses nine vertical impulse sources on a
3 x 3 surface grid:

x = -12.5, 0.0, 12.5 m
y = -12.5, 0.0, 12.5 m
z = 0.0 m

The source direction is (0, 0, 1), and all sources use
gaussian_stf.txt.

The observed objective uses 225 physical surface receivers.

The forward/adjoint gradient workflow uses 38,440 full-grid stations.
Receiver densification must not change the nine-source operator.

During strict-forward preparation:

mesh and full-grid stations come from the validated full-grid
template;
material HDF5 comes from the current accepted model;
the source operator comes from the current accepted input.spec;
dudx = 1 is enforced;
snapshots are disabled.
Runtime-data policy

Large runtime data are not committed to Git:

SEM3D trace HDF5 files;
snapshots;
adjoint workspaces;
candidate forward runs;
large inversion results.

They must be supplied through local data/ and results/ paths,
which may be symbolic links to external storage.
