#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""在**异质数据**上重跑 HalfCheetah 主表（2 主干 × 3 臂 × 10 种子 = 60 格）。

为什么用"导入覆盖"而不是复制 / 改原脚本
------------------------------------
`阶段5_HalfCheetah机制验证.py` 是产出 Day18 主表的那份脚本，**必须保持可复现**。
本脚本只 import 它，然后覆盖四样东西（其余一行不动）：
    1. 数据路   → 每日结果\\Day20_路径二状态相关代价\\数据_HalfCheetah_异质.npz
    2. 日目录 / 模型目录 → 每日结果\\Day21_异质数据主表（新目录，避免复用旧的风险 critic 缓存）
    3. 读并标号 → 换成"固定 τ"版本（见下）
    4. 预算     → 命令行给出

为什么 τ 必须固定成 3.9079
------------------------
原脚本的 `读并标号` 用 `τ = quantile(‖a‖², 0.75)` **从数据里现算**。
异质数据里有一半回合是 1.4 倍放大的动作，现算出来的 τ 会大得多，
那样代价量纲就和我们刚跑完的 BC 三臂（τ 固定 3.9079）不是一回事，比较直接作废。
所以这里把 τ **钉死在 3.9079**（= 原 Day17 数据的 75 分位），与 BC 臂、与 Day18 主表同一定义。
奖励仍是纯前进速度，其余协议（40,000 步、10 种子、100 回合评估、种子 95000）完全不变。

预算语义（本次要报告的两个锚点）
----------------------------
约束量归一化成"数据平均风险 = 1.0"，所以：
    ρ = 1.00 → 不恶化（政策平均风险 = 数据平均，358.15/回合）
    ρ ≈ 0.05 → 绝对安全（政策平均风险 = 稳模态自身，18.28/回合）
主表按原协议先跑 ρ=0.75，再对受约束的两臂扫 {0.05, 0.25, 0.50, 1.00, 1.33} 画前沿。

断点续跑：只看本轮目录里有没有对应结果文件。
**注意**：结果文件名里不含 ρ，所以不同预算必须写进不同目录——用第 5 个参数给子目录名，
否则跑 ρ=1.0 时会把 ρ=0.75 的结果当成"已完成"而全部跳过（第一天就踩到的坑）。

用法：python 工具_HC_异质驱动.py <分片> <分片数> <预算ρ> [步数] [子目录] [臂表]
      python 工具_HC_异质驱动.py 冒烟        # 1 格 200 步，结果写到临时目录，不污染正式结果
例：0 3 0.75 40000                → 主表，写 Day21_异质数据主表\
    0 3 0.05 40000 前沿_ρ0.05 尾部约束 → 前沿扫描，写 Day21_异质数据主表\前沿_ρ0.05\
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
固定τ = 3.9079


def _载(名, 文件):
    s = importlib.util.spec_from_file_location(名, os.path.join(HERE, 文件))
    m = importlib.util.module_from_spec(s)
    s.loader.exec_module(m)
    return m


主干表 = ["IQL", "TD3BC"]
臂表_全部 = ["无约束", "均值约束", "尾部约束"]
种子表 = list(range(10))


def 装好(预算, 子目录="", 臂表=None):
    HC = _载("HC", "阶段5_HalfCheetah机制验证.py")
    HC.数据路 = os.path.join(HERE, "每日结果", "Day20_路径二状态相关代价",
                           "数据_HalfCheetah_异质.npz")
    HC.日目录 = os.path.join(HERE, "每日结果", "Day21_异质数据主表", 子目录) if 子目录 \
        else os.path.join(HERE, "每日结果", "Day21_异质数据主表")
    # 风险 critic 与均值 critic 只依赖数据（对所有 ρ 都一样），所以**共用**主目录里的缓存：
    # 既不重复训练（每个子目录省约 4 分钟），也避免三条分片同时往同一个新目录写缓存而互相覆盖。
    HC.模型目录 = os.path.join(HERE, "每日结果", "Day21_异质数据主表")
    HC.预算 = float(预算)

    def 读并标号_固定τ():
        z = np.load(HC.数据路)
        obs = z["obs"].astype(np.float32); act = z["act"].astype(np.float32)
        rew = z["rew"].astype(np.float32)        # 异质数据里键名是 rew（纯前进速度）
        done = z["done"].astype(np.float32)
        平方 = np.sum(act.astype(np.float64) ** 2, axis=1)
        τ = 固定τ
        cost = np.maximum(0.0, 平方 - τ).astype(np.float32)
        obs2 = np.zeros_like(obs); obs2[:-1] = obs[1:]
        末 = done[:-1] > 0.5
        obs2[:-1][末] = 0.0
        return dict(obs=obs, act=act, rew=rew, cost=cost, done=done, obs2=obs2), τ

    HC.读并标号 = 读并标号_固定τ
    return HC


def 主函数():
    t0 = time.time()
    if len(sys.argv) > 1 and sys.argv[1] == "冒烟":
        return 冒烟()
    分片 = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    分片数 = int(sys.argv[2]) if len(sys.argv) > 2 else 1
    预算 = float(sys.argv[3]) if len(sys.argv) > 3 else 0.75
    步数 = int(sys.argv[4]) if len(sys.argv) > 4 else 40000
    子目录 = sys.argv[5] if len(sys.argv) > 5 else ""
    臂表 = (sys.argv[6].split(",") if len(sys.argv) > 6 else 臂表_全部)
    缺臂 = [a for a in 臂表 if a not in 臂表_全部]
    if 缺臂:
        raise SystemExit(f"臂名不认识：{缺臂}；只能是 {臂表_全部}")

    HC = 装好(预算, 子目录, 臂表)
    os.makedirs(HC.日目录, exist_ok=True)
    全 = [(m, a, s) for m in 主干表 for a in 臂表 for s in 种子表]
    待办 = [x for i, x in enumerate(全) if i % 分片数 == 分片]

    print("=" * 96)
    print(f"异质数据主表：分片 {分片}/{分片数}，本片 {len(待办)} 次，"
          f"每次 {步数:,} 步，预算 ρ={预算}，τ 固定 {固定τ}，臂 {臂表}"
          f"{'，目录 ' + 子目录 if 子目录 else ''}")
    print("=" * 96, flush=True)

    失败 = []
    for i, (主干, 臂, 种子) in enumerate(待办, 1):
        名 = f"结果_HC_{主干}_{臂}_种子{种子}.json"
        if os.path.exists(os.path.join(HC.日目录, 名)):
            print(f"  [{i}/{len(待办)}] {主干}/{臂}/种子{种子} 已有，跳过", flush=True)
            continue
        print(f"\n  [{i}/{len(待办)}] 开始 {主干}/{臂}/种子{种子}"
              f"（已用 {(time.time()-t0)/60:.1f} 分钟）", flush=True)
        try:
            HC.正式(主干, 臂, 种子, 步数=步数)
        except Exception as e:
            失败.append(dict(主干=主干, 臂=臂, 种子=种子, 错误=f"{type(e).__name__}: {e}"))
            print(f"    !! 失败 {type(e).__name__}: {e}", flush=True)
    print(f"\n本片结束，用时 {(time.time()-t0)/60:.1f} 分钟，失败 {len(失败)}：{失败}", flush=True)

    # 记一笔本轮配置，防止以后分不清哪批结果用了哪个预算
    with open(os.path.join(HC.日目录, f"配置_ρ{预算}.json"), "w", encoding="utf-8") as f:
        json.dump(dict(数据路=HC.数据路, 固定τ=固定τ, 预算=预算, 步数=步数,
                       主干表=主干表, 臂表=臂表, 种子表=种子表,
                       奖励定义="纯前进速度", 评估回合数=HC.评估回合数,
                       评估初始种子=HC.评估初始种子,
                       两个锚点=dict(绝对安全_稳模态=18.28, 不恶化_数据均值=358.15)),
                  f, ensure_ascii=False, indent=2, default=float)


def 冒烟():
    """跑 1 格 200 步，验证数据/τ/缓存/评估都接通；结果写到临时目录，不影响正式跑。"""
    HC = 装好(0.75)
    正式目录 = HC.日目录
    HC.日目录 = os.path.join(正式目录, "_冒烟")
    os.makedirs(HC.日目录, exist_ok=True)
    print("冒烟：IQL/尾部约束/种子 99，200 步；风险 critic 与均值 critic 缓存写到正式目录")
    print(f"  数据路 {HC.数据路}")
    print(f"  正式目录 {正式目录} ｜ 冒烟输出 {HC.日目录}", flush=True)
    HC.正式("IQL", "尾部约束", 99, 步数=200)
    print("\n冒烟通过：数据/τ/缓存/评估全部接通。", flush=True)


if __name__ == "__main__":
    主函数()
