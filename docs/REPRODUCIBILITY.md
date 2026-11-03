# Reproduction Protocol

## Artifact levels

Use the smallest artifact that answers the question.

### Check a reported result

Use the episode-level JSON/CSV files and the released checkpoints in the
replication-data record. This level does not require retraining.

### Recompute an analysis

Use the source code in this repository and the released `artifacts/datasets`
directory. The fixed model weights and evaluation seeds must match the manifest.

### Retrain a policy

Use:

- the released dataset;
- the exact configuration;
- the recorded training seed;
- the pinned Python, PyTorch, Gymnasium, Safety-Gymnasium, and DSRL versions;
- one GPU with the recorded CUDA runtime.

## Nested statistical units

Evaluation episodes are nested within training seeds. Do not treat all
episodes as independent training runs. For multi-seed comparisons:

1. compute the episode mean within each training seed;
2. form paired differences across training seeds;
3. report confidence intervals at the training-seed level;
4. use the evaluation seed only for within-seed pairing.

BCQ-Lag uses evaluation seeds `149000` through `149099`. HalfCheetah uses the
sealed ranges recorded in the experiment protocols.

## Deterministic replay

The shared replay engine keeps the model batch size equal to the number of
active slots by padding completed episodes. This prevents floating-point
batch-shape changes from altering a policy trajectory when the number of live
environments changes.

## Safe checkpoint loading

Only load checkpoints obtained from this repository's official archive. PyTorch
checkpoint files can execute arbitrary code when loaded from an untrusted
source.
