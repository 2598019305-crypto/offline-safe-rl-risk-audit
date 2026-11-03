# Source Map

The code is preserved under the experiment filenames where possible so that
claims, protocols, logs, and scripts can be matched directly.

## Diagnostic layers

| Paper layer | Main code |
| --- | --- |
| L1 risk-score resolution | `src/swimmer/工具_E4_0_风险代理有效性诊断.py`, `src/halfcheetah/工具_Day21_D组_直接邻域上尾统计消融.py`, `src/bcql/critic_calibration.py` |
| L2 finite-candidate feasibility | `src/carcircle/p2_gate_arms.py`, `src/carcircle/p2_train.py`, `src/bcql/l2_offline_candidates_v2.py`, `src/bcql/l2_rollout_v2.py` |
| L3 weight and gradient response | `src/halfcheetah/工具_Day21_P3_稳定IQL_DE端点复验_v1_1.py`, `src/halfcheetah/工具_Day21_P4_动作平方范数公平对照_扩种子v1.py`, `src/bcql/l3_static_probe.py` |
| L4 closed-loop cost | `src/halfcheetah/工具_P3_封存测试_仅评估.py`, `src/bcql/evaluate_bcql_checkpoint.py`, `src/carcircle/p3_sealed_eval.py` |

## Dataset generation

| Dataset | Generator |
| --- | --- |
| HalfCheetah Day21 | `src/halfcheetah/工具_路径二_异质数据_最小验证.py` |
| SafetySwimmerVelocity | `src/swimmer/工具_E3e_Swimmer离线数据pilot.py` |

The exact data hashes used by the paper are:

```text
e0850f2e929a3892806f99109937ca323a74f0d624f4fd5fcc027f7c111e8179  HalfCheetah Day21
4edffe8b0178c1220a3d158521cbeafccd2ca479dac07fa8e38430d6086202fa  SafetySwimmerVelocity
```
