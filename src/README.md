# Source Code

The scripts were written for research runs and are preserved with minimal
changes. Several scripts use Chinese filenames and identifiers because those
names match the frozen protocols and evidence files.

Use `scripts/prepare_local_workspace.py` to create the directory layout expected
by the HalfCheetah, Swimmer, and CarCircle scripts.

The BCQ-Lag scripts use:

- `BCQ_DATA_DIR` for the directory containing the CarGoal HDF5;
- `BCQ_CARGOAL_HDF5` to override the full HDF5 path.

The shared replay engine uses `DSRL_DATASET_DIR` when a local CarCircle dataset
directory is required.
