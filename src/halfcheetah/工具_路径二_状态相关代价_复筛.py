#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""路径二最小验证 · 第一步：状态相关代价 τ(s) 的复筛。

代价改成   c(s,a) = max(0, ‖a‖² − τ(s))，τ(s) 依赖躯干角度 θ = obs[2]。

三档 τ(s)（外加原口径对照），全部写死在这里，跑完之后不许再调：
    C0 原口径      τ(s) = 3.9079（= 行为策略 ‖a‖² 的 75 分位，常数）
    C1 温和收缩    τ(s) = 3.9079 · clip(1 − 0.6·|θ|/0.47, 0.4, 1)
    C2 线性扣减    τ(s) = 3.9079 − 12·|θ|      （可以变负 ⇒ 倾斜态即使动作=0 也要付代价）
    C3 强线性扣减  τ(s) = 3.9079 − 25·|θ|
其中 0.47 = 数据 |θ| 的 75 分位。

复筛口径（与协议完全一致，直接照抄 工具_新任务_HalfCheetah采数与体检.py）：
    判据一：同状态动作可解释度 ρ_loc，k=200、m=300、B=400，门槛 ρ_loc ≥ 0.10 且 p<0.001
    判据二：同状态下"冒险步(成本>0) vs 稳妥步"的前进奖励差，门槛 >0、p<0.05、为正邻域 ≥60%
    （另附协议里的对照：回合级 回报—成本 相关，旧闸一线 +0.20）

额外打印一件**决定成败**的事：按 ρ=1.0（每回合成本 ≤ 数据均值）筛选后，
安全子集与危险子集的回报各是多少。因为 BC-Safe 就是"筛掉高成本轨迹再做行为克隆"，
若安全子集的回报本来就更高，筛选等于"筛好学生"，BC-Safe 只会更占优。

运行（分析环境即可，不训练）：
    C:/Users/admin/dsrl-env/Scripts/python.exe 工具_路径二_状态相关代价_复筛.py
"""

import json
import os
import sys
import time

import numpy as np

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HERE = os.path.dirname(os.path.abspath(__file__))
数据路 = os.path.join(HERE, "每日结果", "Day17_奖励代价正交性", "数据_HalfCheetah_v4_双奖励.npz")
日目录 = os.path.join(HERE, "每日结果", "Day20_路径二状态相关代价")
τ0 = 3.9079
θ尺度 = 0.47
协议_k, 协议_m, 协议_B = 200, 300, 400
阈值_判据一, 阈值_判据一_p = 0.10, 0.001
阈值_正邻域比 = 0.60


def 候选代价表(obs, act):
    θ = np.abs(obs[:, 2])
    平方 = np.sum(act ** 2, axis=1)
    return {
        "C0_原口径": np.full_like(平方, τ0),
        "C1_温和收缩": τ0 * np.clip(1 - 0.6 * θ / θ尺度, 0.4, 1.0),
        "C2_线性扣减12": τ0 - 12.0 * θ,
        "C3_强扣减25": τ0 - 25.0 * θ,
    }, 平方


def 主函数():
    t0 = time.time()
    from scipy.spatial import cKDTree
    from scipy import stats
    os.makedirs(日目录, exist_ok=True)

    z = np.load(数据路)
    obs = z["obs"].astype(np.float64); act = z["act"].astype(np.float64)
    rew = z["rew前"].astype(np.float64); done = z["done"].astype(np.float64)
    τ表, 平方 = 候选代价表(obs, act)

    终 = np.where(done > 0.5)[0]
    边 = np.concatenate([[0], 终 + 1])
    轨 = [(a, b) for a, b in zip(边[:-1], 边[1:]) if b - a >= 2]

    sm, ss = obs.mean(0), np.maximum(obs.std(0), 1e-6)
    特征 = (obs - sm) / ss
    rs = np.random.RandomState(0)
    选 = rs.choice(len(obs), 协议_m, replace=False)
    掩 = np.ones(len(obs), bool); 掩[选] = False
    候 = np.where(掩)[0]
    if len(候) > 60000:
        候 = rs.choice(候, 60000, replace=False)
    树 = cKDTree(特征[候]); 动作候 = act[候]; 回报候 = rew[候]
    距, 邻 = 树.query(特征[选], k=协议_k, workers=-1)

    def 邻域R2(目):
        出 = []
        for idx in 邻:
            A = np.concatenate([np.ones((len(idx), 1)), 动作候[idx]], axis=1)
            y = 目[idx]
            try:
                参, *_ = np.linalg.lstsq(A, y, rcond=None)
            except np.linalg.LinAlgError:
                出.append(0.0); continue
            残 = y - A @ 参; 总 = y - y.mean(); s = float(总 @ 总)
            出.append(0.0 if s < 1e-12 else float(1.0 - (残 @ 残) / s))
        return float(np.mean(出))

    print("=" * 104)
    print("路径二 · 状态相关代价 τ(s) 复筛（口径与协议一致：k=200, m=300, B=400）")
    print("=" * 104, flush=True)

    结果 = {}
    for 名, τs in τ表.items():
        cost = np.maximum(0.0, 平方 - τs)
        print(f"\n【{名}】τ 范围 {τs.min():+.3f} ~ {τs.max():+.3f}；"
              f"每步代价均值 {cost.mean():.4f}", flush=True)

        成本候 = cost[候]
        ρloc = 邻域R2(成本候)
        零 = np.array([邻域R2(成本候[rs.permutation(len(候))]) for _ in range(协议_B)])
        p一 = float((零 >= ρloc).mean())
        过一 = (ρloc >= 阈值_判据一) and (p一 < 阈值_判据一_p)
        print(f"  判据一 ρ_loc = {ρloc:.4f}，置换 p = {p一:.4f}  → {'**过**' if 过一 else '**不过**'}")

        危候 = 成本候 > 0
        差R = []
        for idx in 邻:
            g = 回报候[idx]; m = 危候[idx]
            X, Y = g[m], g[~m]
            if len(X) < 20 or len(Y) < 20:
                continue
            差R.append(X.mean() - Y.mean())
        差R = np.array(差R); t2 = stats.ttest_1samp(差R, 0)
        过二 = 差R.mean() > 0 and t2.pvalue < 0.05 and (差R > 0).mean() >= 阈值_正邻域比
        print(f"  判据二 冒险−稳妥 = {差R.mean():+.6f}（为正邻域 {100*(差R > 0).mean():.1f}%，"
              f"p={t2.pvalue:.3g}）  → {'**过**' if 过二 else '**不过**'}")

        每条C = np.array([cost[a:b].sum() for a, b in 轨])
        每条R = np.array([rew[a:b].sum() for a, b in 轨])
        相关 = float(np.corrcoef(每条R, 每条C)[0, 1])
        C̄ = float(每条C.mean())
        达标 = 每条C <= C̄           # ρ = 1.0 的筛选线
        print(f"  回合级 回报—成本 相关 = {相关:+.4f}（旧闸一线 +0.20）")
        print(f"  每回合成本均值 C̄ = {C̄:.2f}；安全子集 {达标.sum()}/{len(轨)}"
              f"（{100*达标.mean():.1f}%）"
              f"  → 安全组回报 {每条R[达标].mean():.1f} vs 危险组回报 {每条R[~达标].mean():.1f}"
              f"  （差 {每条R[达标].mean()-每条R[~达标].mean():+.1f}）", flush=True)

        结果[名] = dict(tau_min=float(τs.min()), tau_max=float(τs.max()),
                      每步代价均值=float(cost.mean()),
                      判据一_rho=ρloc, 判据一_p=p一, 判据一过=bool(过一),
                      判据二_冒险减稳妥=float(差R.mean()),
                      判据二_为正邻域比=float((差R > 0).mean()),
                      判据二_p=float(t2.pvalue), 判据二过=bool(过二),
                      回合级相关=相关, 每回合成本均值=C̄,
                      安全轨迹数=int(达标.sum()),
                      安全组回报=float(每条R[达标].mean()),
                      危险组回报=float(每条R[~达标].mean()))

    print("\n" + "=" * 104)
    print(f"{'代价':<14}{'ρ_loc':>8}{'p':>9}{'判据一':>8}{'冒险−稳妥':>12}{'为正邻比':>9}"
          f"{'回合相关':>10}{'安全组回报':>11}{'危险组回报':>11}")
    for 名, v in 结果.items():
        print(f"{名:<14}{v['判据一_rho']:>8.4f}{v['判据一_p']:>9.4f}"
              f"{'过' if v['判据一过'] else '不过':>8}{v['判据二_冒险减稳妥']:>+12.5f}"
              f"{100*v['判据二_为正邻域比']:>8.1f}%{v['回合级相关']:>+10.3f}"
              f"{v['安全组回报']:>11.1f}{v['危险组回报']:>11.1f}")
    print("=" * 104)
    print("判读：安全组回报 ≥ 危险组回报 ⇒ 筛选等于「筛好学生」，BC-Safe 只会更占优（路径二大概率失败）")
    print("      安全组回报 ≪ 危险组回报 ⇒ 筛选要付回报代价，才有机会让 BC-Safe 掉下来")

    with open(os.path.join(日目录, "复筛_状态相关代价.json"), "w", encoding="utf-8") as f:
        json.dump(结果, f, ensure_ascii=False, indent=2, default=float)
    print(f"\n已保存 每日结果\\Day20_路径二状态相关代价\\复筛_状态相关代价.json；"
          f"用时 {(time.time()-t0)/60:.1f} 分钟")


if __name__ == "__main__":
    主函数()
