#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""把"同状态动作效应"做成一个通用量，在所有基准上对照，看它能否替代 ρ_act。

为什么需要它
----------
Day 06 诊断发现：`ρ_act` 随近邻数 k 从 3.5%（k=1500）漂到 13.0%（k=100），
它衡量的是"动作解释了多少**总**方差"，而多步任务的总方差被状态吃掉了。
于是需要一个**对估计量参数不敏感、含义直接**的判据。

新判据：同状态动作可解释度（两个目标口径）
------------------------------------
  1. 只在**状态**空间里取每个测试点的 k 个最近邻（动作不参与匹配）；
  2. 在邻域内做线性回归，目标取两个：
       · **本步成本 c_t** —— 门控真正要压的东西；
       · **仅后续 F_t = G_t − c_t = γ·G_{t+1}** —— 检验"动作的影响会不会顺延"；
  3. 置换检验：打乱邻域内 (a, F) 的配对 400 次，给出 p 值。

读数：R² = 邻居状态几乎一样时，动作还能解释多少风险。
      以 **c_t** 口径为准：R² 大且显著 → 动作门控有作用对象；
      R² ≈ 0 → 风险由状态决定，门控没有着力点。
      （F_t 口径只作诊断用：Day 06 实测受控任务在 F_t 口径下 R² 也接近 0，
        说明"动作影响顺延到未来"是比门控本身更苛刻的要求。）

对照基准
------
  CarGoal1、PointGoal1（DSRL 自然基准）、路线 D（HalfCheetah 过载）、
  受控任务（动作决定风险，机制唯一显示效果的场景）。

运行：python 工具_判据对照_同状态动作可解释度.py
"""

import json
import os
import sys
import time

import numpy as np

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import importlib.util

HERE = os.path.dirname(os.path.abspath(__file__))


def _载(名, 文件):
    s = importlib.util.spec_from_file_location(名, os.path.join(HERE, 文件))
    m = importlib.util.module_from_spec(s)
    s.loader.exec_module(m)
    return m


折扣 = 0.99
近邻数 = 200
选数 = 300
置换次数 = 400


def 算G(cost, done):
    G = np.zeros(len(cost)); 跑 = 0.0
    for t in range(len(cost) - 1, -1, -1):
        if done[t] > 0.5:
            跑 = 0.0
        跑 = cost[t] + 折扣 * 跑
        G[t] = 跑
    return G


def 邻域R2(obs, act, F, 近邻_idx, 特征_选):
    """返回每个测试点的 R²（线性回归 F ~ [1,a]）。"""
    出 = []
    for i, idx in enumerate(近邻_idx):
        A = np.concatenate([np.ones((len(idx), 1)), act[idx]], axis=1)
        y = F[idx]
        # 最小二乘，奇异时退回均值预测（R²=0）
        try:
            参, *_ = np.linalg.lstsq(A, y, rcond=None)
        except np.linalg.LinAlgError:
            出.append(0.0); continue
        残 = y - A @ 参
        总 = y - y.mean()
        ss总 = float(总 @ 总)
        出.append(0.0 if ss总 < 1e-12 else float(1.0 - (残 @ 残) / ss总))
    return np.array(出)


def 评估(名, obs, act, cost, done, 危险说明=""):
    from scipy.spatial import cKDTree
    G = 算G(cost.astype(np.float64), done)
    F = G - cost.astype(np.float64)
    sm, ss = obs.mean(0), np.maximum(obs.std(0), 1e-6)
    特征 = (obs - sm) / ss
    rs = np.random.RandomState(0)
    选 = rs.choice(len(obs), size=选数, replace=False)
    掩 = np.ones(len(obs), dtype=bool); 掩[选] = False
    候选 = np.where(掩)[0]
    if len(候选) > 60000:
        候选 = rs.choice(候选, size=60000, replace=False)
    树 = cKDTree(特征[候选])
    距, 邻 = 树.query(特征[选], k=近邻数, workers=-1)
    print(f"\n{'='*100}\n{名}\n{'='*100}")
    if 危险说明:
        print(f"  {危险说明}")
    print(f"  {len(obs):,} 条转移；G 均值 {G.mean():.3f}，F 均值 {F.mean():.3f}")
    print(f"  邻域平均半径 {距[:, -1].mean():.3f}（状态空间，k={近邻数}）")
    出 = dict(名称=名, 半径=float(距[:, -1].mean()), 转移数=int(len(obs)),
              G均值=float(G.mean()), F均值=float(F.mean()), 口径={})
    for 口径, 目标 in (("本步成本 c_t", cost.astype(np.float64)),
                     ("仅后续 F_t", F)):
        参 = 目标[候选]
        r2 = float(邻域R2(obs, act[候选], 参, 邻, 特征[选]).mean())
        零 = np.empty(置换次数)
        for j in range(置换次数):
            零[j] = float(邻域R2(obs, act[候选], 参[rs.permutation(len(候选))],
                                邻, 特征[选]).mean())
        p = float((零 >= r2).mean())
        print(f"  [{口径}] 同状态动作可解释度 R² = {r2:.4f}"
              f"   零分布 {零.mean():.4f}±{零.std():.4f}   p = {p:.4f}"
              f" → {'显著' if p < 0.05 else '不显著'}")
        出["口径"][口径] = dict(R2=r2, p值=p, 零均值=float(零.mean()),
                             零标准差=float(零.std()), 显著=bool(p < 0.05))
    return 出


def 主函数():
    t0 = time.time()
    结果 = {}

    # 一、DSRL 自然基准
    体检模块 = _载("体检模块", "工具_基准体检.py")
    for 任务 in ("CarGoal1", "PointGoal1"):
        数据, 轨迹 = 体检模块.读基准(任务)
        # 按轨迹边界构造 done（读基准返回的是轨迹列表，没有全局 done）
        起 = 0; done = np.zeros(len(数据["obs"]))
        for t in 轨迹:
            n = len(t["obs"])
            done[起 + n - 1] = 1.0
            起 += n
        r = 评估(f"{任务}（DSRL 自然基准）", 数据["obs"], 数据["act"],
                 数据["cost"], done,
                 危险说明="代价来自 Safety Gym 的危险区，本质上是状态属性")
        结果[任务] = r

    # 二、路线 D（HalfCheetah + 执行器过载）
    z = np.load(os.path.join(HERE, "每日结果", "Day04_真实动力学新基准",
                             "数据_路线D_过载代价.npz"))
    r = 评估("路线 D（HalfCheetah + 执行器过载）", z["obs"], z["act"],
             z["cost"], z["done"],
             危险说明="代价 = max(0,‖a‖²−τ)，**定义上就是动作的函数**")
    结果["路线D"] = r

    # 三、受控任务（动作决定风险，机制唯一显示效果的场景）
    try:
        W2 = _载("W2", "阶段2W2_约束主干.py")
        W4 = _载("W4", "阶段2W4_动作决定风险任务.py")
        数据 = W4.重标号(W2.读转移(200_000), seed=0)
        done = 数据["done"] if "done" in 数据 else None
        if done is None:
            raise KeyError("读转移没有 done")
        r = 评估("受控任务（CarGoal 状态 + 按动作重标号的成本）",
                 数据["obs"], 数据["act"], 数据["cost"], done,
                 危险说明="侥幸模式的成本由 a[:,1]>0 直接决定，并带 0.5% 致命跳跃")
        结果["受控任务"] = r
    except Exception as e:
        print(f"\n  受控任务跳过：{type(e).__name__}: {e}")

    print(f"\n{'='*100}\n对照汇总\n{'='*100}")
    print(f"  {'基准':<38}{'c_t 口径 R²':>14}{'p':>9}{'F_t 口径 R²':>14}{'p':>9}")
    for k, v in 结果.items():
        a = v["口径"]["本步成本 c_t"]; b = v["口径"]["仅后续 F_t"]
        print(f"  {v['名称'][:36]:<38}{a['R2']:>14.4f}{a['p值']:>9.4f}"
              f"{b['R2']:>14.4f}{b['p值']:>9.4f}")
    print(f"\n  对照（原判据 ρ_act，k=1500）：CarGoal1 9.5% ｜ PointGoal1 5.0% ｜"
          f" 受控任务 25.3% ｜ 路线D 3.5%")
    print(f"\n  判读：新判据若能把'受控任务'与'自然基准/路线D'清晰分开，"
          f"就可以替代 ρ_act 作为基准适用性判据。")
    日 = os.path.join(HERE, "每日结果", "Day06_路线Dv2行为扫描")
    os.makedirs(日, exist_ok=True)
    with open(os.path.join(日, "结果_判据对照_同状态动作可解释度.json"), "w",
              encoding="utf-8") as f:
        json.dump(dict(近邻数=近邻数, 选数=选数, 置换次数=置换次数,
                       折扣=折扣, 结果=结果,
                       用时分钟=round((time.time() - t0) / 60, 1)),
                  f, ensure_ascii=False, indent=2, default=float)
    print(f"\n  已保存 每日结果\\Day06_路线Dv2行为扫描\\结果_判据对照_同状态动作可解释度.json")
    print(f"  用时 {(time.time()-t0)/60:.1f} 分钟")


if __name__ == "__main__":
    主函数()
