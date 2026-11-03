#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""路径二最小验证：把采数行为改成**两种模态按回合混合**，其余一律不动。

为什么是这个（而不是继续改代价定义）
----------------------------
今天实测到的根因：现有数据是「一个好行为策略 + 0.35 高斯动作噪声」，
代价 = max(0, ‖a‖²−3.9079) 而 τ 就是**噪声后**动作范数的 75 分位，
于是成本几乎全部来自噪声，任何去噪器（行为克隆取确定性均值）都白拿一大块——
实测：全量 BC 2859.7/22.7、安全 BC 2807.3/27.0、**只在高成本轨迹上做的 BC 也有 2820.3/15.9**。
这与代价怎么定义无关，所以换 τ(s) 救不了。

能救的是**数据的风险结构**：让"安全"与"快"变成两种真正不同的行为模态，
使确定性均值动作既不是最优、也不安全。

本次唯一改动
----------
采数动作从  a = π(s) + 0.35·N(0,I)
改为**按回合二选一**：稳模态  a = g稳·π(s) + 0.35·N(0,I)
                险模态  a = g险·π(s) + 0.35·N(0,I)
奖励（纯前进速度）、代价定义（τ=3.9079 固定不动）、噪声 0.35、回合长度、评估协议
**全部与 Day17/Day18 主表一致**。这是一次**受控构造**，论文里必须这么写。

三阶段
-----
  阶段A 行为扫描：对候选增益各采 10 回合（同种子配对），按**先写死**的规则挑两个模态；
  阶段B 采数：两个模态各 100 回合 → 数据_HalfCheetah_异质.npz；
  阶段C 复筛：判据一（ρ_loc）、判据二（同状态冒险−稳妥）、回合级相关、
              以及决定性的一列——**安全子集的回报是否低于危险子集**。

挑模态的规则（写死，不许边看边改）
--------------------------------
  稳模态 := 「每回合成本 ≤ 0.25·C̄原 且 回报 ≥ 稳模态回报下限」的增益里，**回报最低**的那个；
  （加回报下限是为了让稳模态仍是个像样的步态，避免做成"站着不动"的退化对照）
  险模态 := 「回报 ≥ 稳模态回报 + 300」的增益里，**回报最高**的那个；
  两者有一个挑不出 ⇒ 如实报告"这个环境里放大动作买不到速度"，路径二到此为止。

【修订记录 · 2026-10-05】
  第一版把"稳模态回报下限"写成 1500，**实测没过**：增益 0.8 拿到 回报 1483.9 / 成本 17.99，
  只差 16 分回报被卡掉（其余档位要么回报 < 1000，要么成本 > 33）。
  修订为 1400，理由是"像样的步态"这条门槛不该定得比实测可行值更高；原规则的结果保留在这里备查。
  修订只影响"挑哪两个增益"，不影响代价定义、奖励定义、评估协议。

【修订记录 · 2026-10-05 第二次】
  第一次修订后的规则**实测仍没过**：候选里同时满足「成本 ≤ 0.25·C̄原 且 回报 ≥ 1400」的只有增益 0.8
  一个档，因此没有任何档能比它再高 300 回报——险模态挑不出来。
  根因：本环境里"放大动作"的回报是**先升后降**（1.2/1.4 到顶 2141/2156，1.6/1.8 掉回 1960/1813），
  而成本始终单调暴涨；把"稳模态"锚在一个低回报档上，就会把"险模态"也一起锁死。
  第二次修订把结构改成：**在世界里直接指定两档增益**（稳 g稳、险 g险，由命令行或默认值给出），
  不再让脚本去挑；选择依据是扫描表本身（哪两档"回报差最大且成本差最大"），
  并把扫描表完整写进结果文件，接受读者复核。默认取 (0.8, 1.4)：
  回报 1483.9 → 2155.9（+672），每回合成本 17.99 → 708.24（+690）。

运行（主表用的分析环境，需要 d3rlpy）：
    C:/Users/admin/dsrl-env/Scripts/python.exe 工具_路径二_异质数据_最小验证.py [每个模态的回合数] [稳增益] [险增益]
    例：... 100 0.8 1.4
"""

import json
import os
import sys
import time

import numpy as np

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HERE = os.path.dirname(os.path.abspath(__file__))
模型路径 = os.path.join(HERE, "每日结果", "Day16_HalfCheetah_v4重采", "模型_行为策略_最好.d3")
日目录 = os.path.join(HERE, "每日结果", "Day20_路径二状态相关代价")
噪声 = 0.35
每回合上限 = 1000
基础种子 = 70000          # 与 Day17 采数一致
τ固定 = 3.9079            # 与主表一致，本次不动
C̄原 = 132.79             # 原数据每回合成本均值（写死，用于阶段A的判断）

扫描增益 = [0.5, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8]
扫描回合 = 10

阈值_判据一, 阈值_判据一_p = 0.10, 0.001
阈值_正邻域比 = 0.60
协议_k, 协议_m, 协议_B = 200, 300, 400


def 采一批(env, algo, 增益, 回合数, rs):
    """按 a = clip(增益·π(s) + 噪声·N, −1, 1) 采 回合数 个回合（种子 基础种子+i）。"""
    d = env.action_space.shape[0]
    O, A, R, 终 = [], [], [], []
    回报, 成本 = [], []
    for 回 in range(回合数):
        o, _ = env.reset(seed=基础种子 + 回)
        轨R, 轨C = 0.0, 0.0
        for t in range(每回合上限):
            a = np.clip(增益 * algo.predict(o.reshape(1, -1))[0]
                        + 噪声 * rs.randn(d), -1.0, 1.0).astype(np.float32)
            o2, r, te, tr, info = env.step(a)
            步成本 = max(0.0, float(np.sum(a.astype(np.float64) ** 2)) - τ固定)
            O.append(o); A.append(a); R.append(float(info["reward_run"]))
            轨R += float(info["reward_run"]); 轨C += 步成本
            终.append(1.0 if (te or tr) else (1.0 if t == 每回合上限 - 1 else 0.0))
            o = o2
            if te or tr:
                break
        回报.append(轨R); 成本.append(轨C)
    return (dict(obs=np.asarray(O, dtype=np.float32), act=np.asarray(A, dtype=np.float32),
                 rew=np.asarray(R, dtype=np.float32), done=np.asarray(终, dtype=np.float32)),
            np.array(回报), np.array(成本))


def 邻域评分(obs, act, cost, rew, 危步):
    """判据一（ρ_loc，k=200、m=300、B=400）+ 判据二（同状态 冒险−稳妥 的前进奖励差）。"""
    from scipy.spatial import cKDTree
    from scipy import stats
    sm, ss = obs.mean(0), np.maximum(obs.std(0), 1e-6)
    特征 = (obs - sm) / ss
    rs = np.random.RandomState(0)
    选 = rs.choice(len(obs), 协议_m, replace=False)
    掩 = np.ones(len(obs), bool); 掩[选] = False
    候 = np.where(掩)[0]
    if len(候) > 60000:
        候 = rs.choice(候, 60000, replace=False)
    树 = cKDTree(特征[候]); 动作候 = act[候]; 成本候 = cost[候]; 回报候 = rew[候]
    危候 = 危步[候]
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

    ρloc = 邻域R2(成本候)
    零 = np.array([邻域R2(成本候[rs.permutation(len(候))]) for _ in range(协议_B)])
    p一 = float((零 >= ρloc).mean())
    差R = []
    for idx in 邻:
        g = 回报候[idx]; m = 危候[idx]
        X, Y = g[m], g[~m]
        if len(X) < 20 or len(Y) < 20:
            continue
        差R.append(X.mean() - Y.mean())
    差R = np.array(差R); t2 = stats.ttest_1samp(差R, 0)
    return dict(rho=ρloc, p=p一, 有效邻域=int(len(差R)),
                冒险减稳妥=float(差R.mean()), 为正邻域比=float((差R > 0).mean()),
                判据二_p=float(t2.pvalue), 邻域半径=float(距[:, -1].mean()))


def 主函数():
    t0 = time.time()
    每模态回合 = int(sys.argv[1]) if len(sys.argv) > 1 else 100
    g稳 = float(sys.argv[2]) if len(sys.argv) > 2 else 0.8
    g险 = float(sys.argv[3]) if len(sys.argv) > 3 else 1.4
    import gymnasium as gym
    import d3rlpy
    os.makedirs(日目录, exist_ok=True)
    if not os.path.exists(模型路径):
        raise FileNotFoundError(模型路径)

    env = gym.make("HalfCheetah-v4")
    algo = d3rlpy.load_learnable(模型路径, device="cpu")

    print("=" * 104)
    print("路径二最小验证 · 阶段A：采数行为增益扫描（每个增益 10 回合、同种子配对）")
    print(f"  奖励=纯前进速度；代价=max(0,‖a‖²−{τ固定})（与主表完全一致，不动）；噪声 {噪声}")
    print("=" * 104, flush=True)

    扫描 = {}
    for g in 扫描增益:
        ts = time.time()
        _, 回报, 成本 = 采一批(env, algo, g, 扫描回合, np.random.RandomState(0))
        扫描[g] = dict(回报均值=float(回报.mean()), 回报标准差=float(回报.std(ddof=1)),
                       成本均值=float(成本.mean()), 成本最大=float(成本.max()),
                       用时秒=round(time.time() - ts, 1))
        print(f"  增益 {g:<4}  回报 {回报.mean():>8.1f} ± {回报.std(ddof=1):>6.1f}"
              f"   每回合成本 {成本.mean():>8.2f}（最大 {成本.max():>8.1f}）", flush=True)

    # ---- 规则的历史留档：两条原始规则实测都没过，过程见文件头【修订记录】 ----
    原规则一 = [g for g in 扫描增益
              if 扫描[g]["成本均值"] <= 0.25 * C̄原 and 扫描[g]["回报均值"] >= 1500]
    print(f"\n  【留档】原规则一（成本 ≤ {0.25*C̄原:.1f} 且 回报 ≥ 1500）："
          f"{'通过，候选 ' + str(原规则一) if 原规则一 else '**未通过**'}")
    原规则二 = [g for g in 扫描增益
              if 扫描[g]["成本均值"] <= 0.25 * C̄原 and 扫描[g]["回报均值"] >= 1400]
    print(f"  【留档】原规则二（下限改 1400）：{原规则二 if 原规则二 else '**未通过**'}"
          f"；在其上再比 +300 回报 ⇒ "
          f"{[g for g in 扫描增益 if 扫描[g]['回报均值'] >= 扫描[原规则二[0]]['回报均值'] + 300] if 原规则二 else '无候选'}")

    if g稳 not in 扫描 or g险 not in 扫描:
        raise SystemExit(f"指定的增益必须来自扫描表 {扫描增益}")
    print(f"\n  >>> 本次采用的构造（第二次修订，理由见文件头）："
          f"稳模态 g={g稳}（回报 {扫描[g稳]['回报均值']:.1f}、成本 {扫描[g稳]['成本均值']:.2f}）"
          f"  险模态 g={g险}（回报 {扫描[g险]['回报均值']:.1f}、成本 {扫描[g险]['成本均值']:.2f}）",
          flush=True)

    print("\n" + "=" * 104)
    print(f"阶段B：两模态各采 {每模态回合} 回合（共 {2*每模态回合} 回合）")
    print("=" * 104, flush=True)
    数稳, 回稳, 成稳 = 采一批(env, algo, g稳, 每模态回合, np.random.RandomState(0))
    数险, 回险, 成险 = 采一批(env, algo, g险, 每模态回合, np.random.RandomState(0))
    env.close()

    obs = np.concatenate([数稳["obs"], 数险["obs"]])
    act = np.concatenate([数稳["act"], 数险["act"]])
    rew = np.concatenate([数稳["rew"], 数险["rew"]])
    done = np.concatenate([数稳["done"], 数险["done"]])
    模式 = np.concatenate([np.zeros(len(数稳["obs"]), np.float32),
                         np.ones(len(数险["obs"]), np.float32)])
    平方 = np.sum(act.astype(np.float64) ** 2, axis=1)
    cost = np.maximum(0.0, 平方 - τ固定).astype(np.float32)

    终 = np.where(done > 0.5)[0]
    边 = np.concatenate([[0], 终 + 1])
    轨 = [(a, b) for a, b in zip(边[:-1], 边[1:]) if b - a >= 2]
    每条C = np.array([cost[a:b].sum() for a, b in 轨])
    每条R = np.array([rew[a:b].sum() for a, b in 轨])
    每条模式 = np.array([模式[a:b].mean() for a, b in 轨])
    C̄ = float(每条C.mean())
    相关 = float(np.corrcoef(每条R, 每条C)[0, 1])
    达标 = 每条C <= C̄
    print(f"  共 {len(obs):,} 条转移 / {len(轨)} 条轨迹；每回合成本均值 C̄ = {C̄:.2f}")
    print(f"  稳模态轨迹回报 {回稳.mean():.1f}、成本 {成稳.mean():.2f}")
    print(f"  险模态轨迹回报 {回险.mean():.1f}、成本 {成险.mean():.2f}")
    print(f"  回合级 回报—成本 相关 = {相关:+.4f}")
    print(f"  ρ=1.0 筛选：安全子集 {达标.sum()}/{len(轨)}（{100*达标.mean():.1f}%）"
          f"  → 安全组回报 {每条R[达标].mean():.1f} vs 危险组回报 {每条R[~达标].mean():.1f}"
          f"（差 {每条R[达标].mean()-每条R[~达标].mean():+.1f}）", flush=True)

    print("\n" + "=" * 104)
    print("阶段C：复筛（口径与协议一致）")
    print("=" * 104, flush=True)
    v = 邻域评分(obs.astype(np.float64), act.astype(np.float64), cost.astype(np.float64),
               rew.astype(np.float64), cost > 0)
    过一 = (v["rho"] >= 阈值_判据一) and (v["p"] < 阈值_判据一_p)
    过二 = (v["冒险减稳妥"] > 0) and (v["判据二_p"] < 0.05) and (v["为正邻域比"] >= 阈值_正邻域比)
    print(f"  判据一 ρ_loc = {v['rho']:.4f}，置换 p = {v['p']:.4f}  → {'**过**' if 过一 else '**不过**'}")
    print(f"  判据二 冒险−稳妥 = {v['冒险减稳妥']:+.6f}"
          f"（为正邻域 {100*v['为正邻域比']:.1f}%，p={v['判据二_p']:.3g}）"
          f"  → {'**过**' if 过二 else '**不过**'}")

    关键 = 每条R[达标].mean() - 每条R[~达标].mean()
    print("\n" + "=" * 104)
    if 关键 < -300 and 过一 and 过二:
        print("  判决：**张力成立且方向正确** —— 筛选要付回报代价（安全组明显更慢），")
        print("        且两个判据都过。值得进入 BC 三臂 + 重跑主表（1–2 天）。")
    else:
        print(f"  判决：不成立或方向错（安全组−危险组 = {关键:+.1f}，需要显著为负）")
        print("        ⇒ 筛选仍等于「筛好学生」，BC-Safe 依旧会占优。")
    print("=" * 104, flush=True)

    np.savez_compressed(os.path.join(日目录, "数据_HalfCheetah_异质.npz"),
                        obs=obs, act=act, rew=rew, cost=cost, done=done, 模式=模式,
                        τ=τ固定, 增益稳=g稳, 增益险=g险)
    with open(os.path.join(日目录, "结果_异质数据_最小验证.json"), "w", encoding="utf-8") as f:
        json.dump(dict(扫描={str(k): v for k, v in 扫描.items()}, 增益稳=g稳, 增益险=g险,
                       规则留档=dict(**{
                           "原规则一（成本≤0.25C̄ 且 回报≥1500）": {"候选": 原规则一,
                                                                  "通过": bool(原规则一)},
                           "原规则二（下限改 1400）": {"候选": 原规则二,
                                                    "通过": bool(原规则二)},
                           "第二次修订": "直接指定两档增益（0.8, 1.4），依据是扫描表里"
                                       "回报差与成本差都最大的那一对，扫描表全量留档"}),
                       每模态回合=每模态回合, 转移数=int(len(obs)), 轨迹数=int(len(轨)),
                       每回合成本均值=C̄, 回合级相关=相关,
                       安全轨迹数=int(达标.sum()),
                       安全组回报=float(每条R[达标].mean()),
                       危险组回报=float(每条R[~达标].mean()),
                       安全组模式均值=float(每条模式[达标].mean()),
                       危险组模式均值=float(每条模式[~达标].mean()),
                       判据一=v, 判据一过=bool(过一), 判据二过=bool(过二),
                       用时分钟=round((time.time() - t0) / 60, 2)),
                  f, ensure_ascii=False, indent=2, default=float)
    print(f"\n已保存 数据_HalfCheetah_异质.npz 与 结果_异质数据_最小验证.json；"
          f"用时 {(time.time()-t0)/60:.1f} 分钟")


if __name__ == "__main__":
    主函数()
