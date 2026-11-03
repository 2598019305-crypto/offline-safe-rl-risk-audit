#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""定批回放引擎：不管有多少个回合在跑，每一轮送进网络的批次大小永远是固定的。

为什么必须这么做
--------------
昨天验证出来一件要命的事：同一配置重复跑逐位一致，但只要改并行环境个数，
轨迹就完全不同（单回合成本最大差到 60）。根因是**批量推理的批次大小会变**——
每有环境跑完，存活个数就少一个，批次从 16 掉到 15、14……于是矩阵乘法的分块方式变了，
浮点结果的最后几位变了，再被动力学放大成完全不同的轨迹。

后果：不同配置的"存活个数随时间的变化"本来就不一样，所以逐回合配对比较其实不成立，
只能退而用功效低得多的非配对检验。

这个引擎的修法是：**批次大小永远等于 并行数**。已经跑完的槽位填一段固定的假观测
（全零），它们算出来的动作直接丢掉。这样每一行的输出只取决于这一行自己的输入，
和别的槽位在不在、是哪个回合都无关。于是：
  · 同一个回合（同一个种子）在任何调度下都会走出完全相同的轨迹；
  · 配对比较重新成立，统计功效回到应有的水平。

代价只是末尾几个空转的槽位，可以忽略。

设计上把"怎么根据观测改动作"抽成一个回调函数传入，这样将来把干预从"切换动作"
换成"Q 值调节"时，不用再动引擎。
"""

import sys
import time

import numpy as np

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# 默认维度（CarGoal）。换任务时由调用方通过参数覆盖，**不要在别处再写死维度**：
# 之前 CarGoal 的 72/2 是写死的，换到 HalfCheetah（观测 17 / 动作 6）就必须改这里。
OBS_DIM, ACT_DIM = 72, 2


def 定批回放(algo, 决策函数, 回合数, 并行数=16, 起始种子=0, 环境名="SafetyCarGoal1Gymnasium-v0",
            打印间隔=0, 最长步数=1000, 环境工厂=None, 观察维=None, 动作维=None):
    """跑完 回合数 个回合，返回逐回合的回报与成本。

    决策函数(观测批, 主干动作批) -> 执行动作批，形状都是 (并行数, 2)。
    它在每一轮都会被调用一次，收到的始终是固定的 并行数 行。
    """
    import gymnasium as gym
    OBS = OBS_DIM if 观察维 is None else int(观察维)
    ACT = ACT_DIM if 动作维 is None else int(动作维)
    try:
        import dsrl  # noqa: F401  （注册 DSRL 的环境名）
    except ImportError:
        # 隔离的 OSRL 环境里没有 dsrl：PyPI 版 dsrl 会拉 pybullet，
        # 而 pybullet 在 Windows 上没有任何轮子、编译也不通。
        # Safety-Gymnasium 自己就注册了同名的 SafetyCarGoal1Gymnasium-v0，
        # 且实测两者动力学**逐位一致**（同种子同动作序列，200 步观测哈希同为 1935.036578），
        # 所以在这里退回用 safety_gymnasium 注册，评估结果与主表可比。
        import safety_gymnasium  # noqa: F401

    W = min(并行数, 回合数)
    # 环境工厂：允许传入自定义包装（例如把奖励与成本换成按动作定义的合成版本）。
    # 不传时按环境名直接创建，保持与原有一致。
    建 = 环境工厂 if 环境工厂 is not None else (lambda: gym.make(环境名))
    环境表 = [建() for _ in range(W)]
    种子 = [起始种子 + i for i in range(回合数)]

    观测 = np.zeros((W, OBS), dtype=np.float32)
    存活 = np.zeros(W, dtype=bool)
    归属 = np.full(W, -1, dtype=int)
    下一个发 = 0
    for i in range(W):
        观测[i], _ = 环境表[i].reset(seed=种子[下一个发])
        归属[i] = 下一个发
        存活[i] = True
        下一个发 += 1

    回报 = np.zeros(回合数)
    成本 = np.zeros(回合数)
    当前回报 = np.zeros(W)
    当前成本 = np.zeros(W)
    轮次 = 0
    t0 = time.time()

    while 存活.any():
        # 关键：永远喂满 W 行。死掉的槽位是固定的全零观测，不影响还活着的行。
        批 = np.zeros((W, OBS), dtype=np.float32)
        批[存活] = 观测[存活]
        a主干 = np.asarray(algo.predict(批), dtype=np.float32)
        a执行 = np.asarray(决策函数(批, a主干), dtype=np.float32)

        for i in range(W):
            if not 存活[i]:
                continue
            o, rr, te, tr, info = 环境表[i].step(a执行[i])
            当前回报[i] += float(rr)
            当前成本[i] += float(info.get("cost", 0.0))
            观测[i] = o
            if te or tr:
                回报[归属[i]] = 当前回报[i]
                成本[归属[i]] = 当前成本[i]
                当前回报[i] = 0.0
                当前成本[i] = 0.0
                if 下一个发 < 回合数:
                    观测[i], _ = 环境表[i].reset(seed=种子[下一个发])
                    归属[i] = 下一个发
                    下一个发 += 1
                else:
                    存活[i] = False
                    归属[i] = -1
        轮次 += 1
        if 打印间隔 and 轮次 % 打印间隔 == 0:
            print(f"    ... {轮次} 轮，{time.time()-t0:.0f}s", flush=True)

    for e in 环境表:
        e.close()
    return dict(逐回合回报=回报, 逐回合成本=成本,
                回报均值=float(回报.mean()), 成本均值=float(成本.mean()),
                成本CVaR25=float(np.sort(成本)[-max(1, 回合数 // 4):].mean()),
                秒数=round(time.time() - t0, 1), 并行数=W, 回合数=回合数)
