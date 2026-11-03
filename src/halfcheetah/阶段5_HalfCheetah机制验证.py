#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""机制验证：在**改造后的 HalfCheetah-v4 基准**上验证尾部风险门控机制。

任务定义（Day17 验证出来的唯一通过组合，写死）
----------------------------------------
    环境：HalfCheetah-v4（标准 MuJoCo 动力学）
    奖励：**纯前进速度** `info["reward_run"]`
        —— 把标准奖励里的 `−0.1Σa²` 剥离出去，只留在约束通道
        （标准奖励把控制代价揉进奖励，会让"冒险买回报"的张力反向，实测 33.6% vs 61.1%）
    代价：`max(0, ‖a‖² − τ)`，τ = 数据 ‖a‖² 的 75 分位
    数据：自采 v4 数据（Day16 行为策略 + 0.35 噪声，200 回合 20 万条）

两个臂
-----
    无约束    ：只学奖励
    尾部约束  ：在优势里减去 λ·(尾部风险信号)，λ 由对偶在线更新
                （风险信号 = 折扣成本回报的上尾 CVaR@10，经池化邻域蒸馏而成）

评估：定批回放引擎 + 固定随机数（按回合播种、每步抽一次），
      所以三个臂之间是**配对比较**，与前面所有实验同一协议。

运行：
    python 阶段5_HalfCheetah机制验证.py 冒烟
    python 阶段5_HalfCheetah机制验证.py 正式 <主干> <臂> <种子> [步数]
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


W2 = _载("W2", "阶段2W2_约束主干.py")
W4 = _载("W4", "阶段2W4_动作决定风险任务.py")

# —— 任务维度（换任务只改这两行）——
W2.观察维, W2.动作维 = 17, 6
# 注意：阶段2W4 内部**又独立加载了一份 W2**（importlib 的 _载 不会把模块放进 sys.modules，
# 所以每次加载都是新实例）。只改上面那份没用——风险蒸馏网是按 W4 自己那份 W2 的维度建的，
# 结果就是 mat1/mat2 形状不匹配（256x23 vs 74x128）。这里把它那份也改掉。
W4.W2.观察维, W4.W2.动作维 = 17, 6

固定视界 = 1000          # HalfCheetah-v4 的自然回合长度
评估回合数 = 100
评估初始种子 = 95000
折扣 = 0.99
成本折扣 = 0.00          # 风险标签只看本步成本（见 造风险critic 的说明）
预算 = 0.75              # 归一化风险预算（数据平均风险记为 1.0）
数据路 = os.path.join(HERE, "每日结果", "Day17_奖励代价正交性", "数据_HalfCheetah_v4_双奖励.npz")
日目录 = os.path.join(HERE, "每日结果", "Day18_HalfCheetah机制验证")
模型目录 = 日目录


def 读并标号():
    z = np.load(数据路)
    obs = z["obs"].astype(np.float32); act = z["act"].astype(np.float32)
    rew = z["rew前"].astype(np.float32)      # 纯前进速度
    done = z["done"].astype(np.float32)
    平方 = np.sum(act.astype(np.float64) ** 2, axis=1)
    τ = float(np.quantile(平方, 0.75))
    cost = np.maximum(0.0, 平方 - τ).astype(np.float32)
    # 下一观测
    obs2 = np.zeros_like(obs); obs2[:-1] = obs[1:]
    末 = done[:-1] > 0.5
    obs2[:-1][末] = 0.0
    return dict(obs=obs, act=act, rew=rew, cost=cost, done=done, obs2=obs2), τ


def 建环境():
    import gymnasium as gym

    class 改造HalfCheetah(gym.Wrapper):
        """奖励换成纯前进速度；代价 = max(0, ‖a‖²−τ)。"""

        def __init__(self):
            super().__init__(gym.make("HalfCheetah-v4"))
            self.随机数 = np.random.RandomState(0)

        def reset(self, **kw):
            种子 = kw.get("seed")
            # 与 CarGoal 任务同一约定：按回合播种，保证同一回合内可复现；
            # 代价在本任务里是确定性的，随机数只用于保持接口一致。
            self.随机数 = np.random.RandomState(0 if 种子 is None else int(种子) + 7777)
            self.步数 = 0
            o, info = self.env.reset(**kw)
            return o, info

        def step(self, a):
            o, r, te, tr, info = self.env.step(a)
            self.步数 += 1
            奖励 = float(info["reward_run"])           # ← 纯前进速度，剥离控制代价
            平方 = float(np.sum(np.asarray(a, dtype=np.float64) ** 2))
            成本 = max(0.0, 平方 - 数据阈值[0])
            结束 = bool(te or tr) or (self.步数 >= 固定视界)
            info = dict(info); info["cost"] = 成本
            return o, 奖励, 结束, False, info

    return 改造HalfCheetah()


数据阈值 = [0.0]     # 由 准备() 填；用列表包一层便于闭包读取


def 监督训练均值critic(网, 数据, 步数=20000, 批量=256, 打印=True):
    """把均值成本 critic 直接监督回归到**本步成本**上，不做 TD 自举。

    为什么换掉 TD 版（2026-10-04 实测）
    --------------------------------
    原来的 `W2.训练成本critic` 是 TD 自举：qc ← c + γ·qc(s',a')。
    在本任务上它收敛到了**错误的固定点**：
        数据里真实的折扣成本回报 G 均值 = **11.96**
        但学出来的 qc 均值只有 **0.23**
    于是归一化（惩罚项 = qc / 均值尺度）把约束量放大到 **−10⁶** 量级，
    对偶更新看到"约束量远小于预算"就把 λ 压到 0 ——
    **整个均值约束臂退化成无约束**，跑出来的结果全部无效（已作废重跑）。

    本步成本是可直接监督的量，且天然有界（≥0），不存在自举固定点的问题；
    这也与尾部 risk critic 的构造对称（后者也是监督式的池化标签回归）。
    """
    import torch
    s = torch.tensor(数据["obs"]); a = torch.tensor(数据["act"])
    c = torch.tensor(数据["cost"]).reshape(-1, 1)
    n = len(s)
    优 = torch.optim.Adam(网.qc.parameters(), lr=3e-4)
    t0 = time.time()
    for 步 in range(1, 步数 + 1):
        j = torch.randint(0, n, (批量,))
        损 = ((网.qc(torch.cat([s[j], a[j]], -1)) - c[j]) ** 2).mean()
        优.zero_grad(); 损.backward(); 优.step()
        if 打印 and 步 % 5000 == 0:
            print(f"    [均值成本 critic·监督] {步:>6} 步 {time.time()-t0:5.0f}s "
                  f"损={损.item():.6f}", flush=True)
    return 网


def 评估(网, 回合=None, 并行=6, 种子=None):
    定批 = _载("定批", "公共_定批回放引擎.py")

    class 零主干:
        def predict(self, 批):
            return np.zeros((len(批), W2.动作维), dtype=np.float32)

    def 决策(批, _):
        import torch
        with torch.no_grad():
            return 网.均值动作(torch.tensor(批, dtype=torch.float32)).numpy()

    return 定批.定批回放(零主干(), 决策,
                       回合数=(评估回合数 if 回合 is None else 回合),
                       并行数=并行,
                       起始种子=(评估初始种子 if 种子 is None else 种子),
                       环境工厂=建环境, 最长步数=固定视界,
                       观察维=W2.观察维, 动作维=W2.动作维)


def 准备(打印=True):
    os.makedirs(模型目录, exist_ok=True)
    数据, τ = 读并标号()
    数据阈值[0] = τ
    if 打印:
        print(f"  数据 {len(数据['obs']):,} 条；动作维 {W2.动作维}；"
              f"代价 τ = {τ:.4f}（冒险步占比 {100*(数据['cost']>0).mean():.1f}%）", flush=True)
        print(f"  每步奖励 均值 {数据['rew'].mean():.4f}；每步代价 均值 {数据['cost'].mean():.5f}",
              flush=True)
    风险缓存 = os.path.join(模型目录, "模型_HalfCheetah_风险critic.pt")
    if os.path.exists(风险缓存):
        import torch
        import torch.nn as nn
        检查点 = torch.load(风险缓存, map_location="cpu", weights_only=False)
        网 = nn.Sequential(nn.Linear(W2.观察维 + W2.动作维, 128), nn.ReLU(),
                          nn.Linear(128, 128), nn.ReLU(), nn.Linear(128, 1))
        网.load_state_dict(检查点["net"]); 网.eval()
        风险 = W4.新风险模块(检查点["均值"], 检查点["标准差"], 网)
        if 打印:
            print("  已复用风险 critic", flush=True)
    else:
        风险 = W4.造风险critic(数据, 打印=打印, 成本折扣=成本折扣)
        import torch
        torch.save({"net": 风险.网.state_dict(), "均值": 风险.均值.numpy(),
                    "标准差": 风险.标准差.numpy()}, 风险缓存)
    均值网 = W2.均值成本critic()
    成本缓存 = os.path.join(模型目录, "模型_HalfCheetah_均值成本critic.pt")
    if os.path.exists(成本缓存):
        import torch
        均值网.load_state_dict(torch.load(成本缓存, map_location="cpu", weights_only=False))
        if 打印:
            print("  已复用均值成本 critic", flush=True)
    else:
        监督训练均值critic(均值网, 数据, 步数=20000, 打印=打印)
        import torch
        torch.save(均值网.state_dict(), 成本缓存)
    # 自检：学到的均值 critic 必须落在真值量级里（本步成本 ∈ [0, ~2]）。
    # 这一步是为了防止再次出现"critic 收敛到错误量级却没人发现"。
    import torch
    抽 = np.random.RandomState(0).choice(len(数据["obs"]), 5000, replace=False)
    with torch.no_grad():
        预 = 均值网.qc(torch.cat([torch.tensor(数据["obs"][抽], dtype=torch.float32),
                                torch.tensor(数据["act"][抽], dtype=torch.float32)],
                              -1)).reshape(-1)
    真 = float(数据["cost"][抽].mean())
    if 打印:
        print(f"  【自检】均值 critic 预测均值 {预.mean():.5f} vs 本步成本真值 {真:.5f}"
              f"  → {'量级正常' if abs(预.mean()-真) < 10*max(真,1e-6) else '**量级异常，需检查**'}",
              flush=True)
    均值网.eval()
    return 数据, 风险, 均值网


def 正式(主干, 臂, 种子, 步数=40000):
    t0 = time.time()
    print("=" * 96)
    print(f"HalfCheetah 机制验证：主干 {主干}，臂 {臂}，种子 {种子}，{步数:,} 步")
    print("=" * 96, flush=True)
    数据, 风险, 均值网 = 准备()
    网, lam, 日志 = W2.训练一臂(主干, 臂, 数据, 风险, 均值网,
                             步数=步数, 种子=种子, 打印间隔=5000, 预算=预算)
    评 = 评估(网)
    出 = dict(任务="HalfCheetah改造基准", 主干=主干, 臂=臂, 种子=种子, 步数=步数,
              预算=预算, 评估回合数=评估回合数,
              奖励定义="纯前进速度（剥离标准奖励的控制代价）",
              代价定义=f"max(0,‖a‖²−τ)，τ={数据阈值[0]:.4f}",
              最终对偶变量=lam.值, 对偶轨迹=lam.轨迹,
              评估=dict(回报均值=评["回报均值"], 成本均值=评["成本均值"],
                      成本CVaR25=评["成本CVaR25"]),
              逐回合回报=评["逐回合回报"].tolist(),
              逐回合成本=评["逐回合成本"].tolist(),
              训练日志=日志, 用时分钟=round((time.time() - t0) / 60, 1))
    名 = f"结果_HC_{主干}_{臂}_种子{种子}.json"
    with open(os.path.join(日目录, 名), "w", encoding="utf-8") as f:
        json.dump(出, f, ensure_ascii=False, indent=2, default=float)
    import torch
    torch.save(网.state_dict(), os.path.join(日目录, f"模型_HC_{主干}_{臂}_种子{种子}.pt"))
    print(f"\n评估：回报 {评['回报均值']:.1f}  成本 {评['成本均值']:.4f}  "
          f"CVaR25 {评['成本CVaR25']:.4f}  λ={lam.值:.3f}", flush=True)
    print(f"已保存 {名}；用时 {出['用时分钟']:.1f} 分钟", flush=True)


def 冒烟():
    数据, 风险, 均值网 = 准备()
    import torch
    s = torch.tensor(数据["obs"][:2000]); a = torch.tensor(数据["act"][:2000])
    with torch.no_grad():
        风 = 风险.风险值(s, a).numpy()
        均 = 均值网.qc(torch.cat([s, a], -1)).reshape(-1).numpy()
    危 = 数据["cost"][:2000] > 0
    print(f"\n  风险信号：冒险步 {风[危].mean():.4f}  稳妥步 {风[~危].mean():.4f}"
          f"  → 差 {风[危].mean()-风[~危].mean():+.4f}", flush=True)
    print(f"  均值成本：冒险步 {均[危].mean():.4f}  稳妥步 {均[~危].mean():.4f}"
          f"  → 差 {均[危].mean()-均[~危].mean():+.4f}", flush=True)


def 主函数():
    if len(sys.argv) > 1 and sys.argv[1] == "冒烟":
        return 冒烟()
    if len(sys.argv) > 1 and sys.argv[1] == "正式":
        主干 = sys.argv[2]; 臂 = sys.argv[3]; 种子 = int(sys.argv[4])
        步数 = int(sys.argv[5]) if len(sys.argv) > 5 else 40000
        return 正式(主干, 臂, 种子, 步数)
    print("用法：冒烟 | 正式 <主干> <臂> <种子> [步数]")


if __name__ == "__main__":
    主函数()
