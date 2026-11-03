# Reproduction Checks

Run date: 2026-11-03

## Dataset integrity

Both released datasets were loaded from the final replication directory and
checked independently:

- HalfCheetah Day21: 200,000 transitions, 200 episodes, observations `17`,
  actions `6`, SHA256 matched.
- SafetySwimmerVelocity: 20,000 transitions, 20 episodes, observations `8`,
  actions `2`, every `next_obs` aligned with the following observation, SHA256
  matched.

## Swimmer L1 diagnostic

`src/swimmer/工具_E4_0_风险代理有效性诊断.py` was rerun from the released code
and data. The rerun reproduced:

- D10 saturation rate: `0.9818333`;
- D10 ROC-AUC / PR-AUC: `0.5121435 / 0.2566627`;
- E100 ROC-AUC / PR-AUC: `0.7644392 / 0.5940540`;
- neighbor radius: `5.2887451`;
- empty-neighborhood fraction: `0.0`;
- mean neighbor count: `451.6763333`.

The runtime timestamp changed, but the reported numerical diagnostics matched
the archived result.

## Package integrity

The final replication archive contains 621 ZIP entries, including both
datasets, the complete checkpoint set, evidence files, generators, data card,
license, and SHA256 manifest.
