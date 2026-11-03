#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""HalfCheetah 改造基准上的**完整主表**分析：2 主干 × 3 臂 × N 种子。

输出的三样（论文主表直接用）：
  1. 每个臂的 回报 / 平均成本 / CVaR@25（均值 ± 标准差）；
  2. **配对检验**：尾部−均值、均值−无约束（同一批种子逐种子配对，
     配对 t 检验 + 自助法 95%CI；n≥8 时 Wilcoxon 才可能有判定力，一并报出）；
  3. **约束满足率**（对偶轨迹里 约束量 ≤ 预算 的比例）——这是"约束到底有没有生效"的证据，
     必须与主表一起报，不能藏。

运行：python 工具_HC_主表分析.py
"""

import json
import os
import sys

import numpy as np

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HERE = os.path.dirname(os.path.abspath(__file__))
日目录 = os.path.join(HERE, "每日结果", "Day18_HalfCheetah机制验证")
主干表 = ("IQL", "TD3BC")
臂表 = ("无约束", "均值约束", "尾部约束")
指标 = (("成本CVaR25", "CVaR@25（越小越安全）"), ("成本均值", "平均成本"),
       ("回报均值", "回报（越大越好）"))


def 读(主干, 臂):
    出 = {}
    if not os.path.isdir(日目录):
        return 出
    for f in os.listdir(日目录):
        前 = f"结果_HC_{主干}_{臂}_种子"
        if f.startswith(前) and f.endswith(".json"):
            try:
                s = int(f[len(前):-5])
            except Exception:
                continue
            出[s] = json.load(open(os.path.join(日目录, f), encoding="utf-8"))
    return 出


def 配对(名, 甲, 乙, 种子, 键):
    from scipy import stats
    a = np.array([甲[s]["评估"][键] for s in 种子], float)
    b = np.array([乙[s]["评估"][键] for s in 种子], float)
    d = a - b
    t = stats.ttest_rel(a, b)
    try:
        w = stats.wilcoxon(d) if np.any(d != 0) else None
    except Exception:
        w = None
    rs = np.random.RandomState(0)
    重 = d[rs.randint(0, len(d), size=(20000, len(d)))].mean(axis=1)
    低, 高 = np.percentile(重, [2.5, 97.5])
    显 = bool(t.pvalue < 0.05 and not (低 <= 0 <= 高))
    return dict(名称=名, 甲=float(a.mean()), 乙=float(b.mean()), 差=float(d.mean()),
                相对=float(d.mean() / (abs(b.mean()) + 1e-12)), p=float(t.pvalue),
                wilcoxon_p=(float(w.pvalue) if w is not None else None),
                配对d=float(d.mean() / (d.std(ddof=1) + 1e-12)),
                CI=(float(低), float(高)), 含零=bool(低 <= 0 <= 高), 显著=显,
                逐种子差=d.tolist())


def 主函数():
    数据 = {m: {a: 读(m, a) for a in 臂表} for m in 主干表}
    种子 = {m: sorted(set.intersection(*[set(数据[m][a]) for a in 臂表]))
           if all(数据[m][a] for a in 臂表) else []
           for m in 主干表}
    print("=" * 104)
    print("HalfCheetah 改造基准 主表")
    print("  奖励 = 纯前进速度（剥离标准奖励的控制代价）｜代价 = max(0,‖a‖²−τ)")
    print("=" * 104)
    for m in 主干表:
        print(f"  {m}: 可用种子 {len(种子[m])} 个 {种子[m] if len(种子[m]) <= 12 else ''}")
    可分析 = [m for m in 主干表 if len(种子[m]) >= 2]
    if not 可分析:
        print("  还没有任何一个主干凑齐三个臂，先等训练跑完。"); return

    print(f"\n{'─'*104}\n主表（均值 ± 标准差）\n{'─'*104}")
    print(f"  {'主干':<7}{'臂':<7}{'回报':>20}{'平均成本':>20}{'CVaR@25':>20}")
    主表 = {}
    for m in 可分析:
        for a in 臂表:
            s = 种子[m]
            R = np.array([数据[m][a][i]["评估"]["回报均值"] for i in s])
            C = np.array([数据[m][a][i]["评估"]["成本均值"] for i in s])
            V = np.array([数据[m][a][i]["评估"]["成本CVaR25"] for i in s])
            f = lambda x: f"{x.mean():.1f} ± {x.std(ddof=1):.1f}"   # noqa: E731
            print(f"  {m:<7}{a:<7}{f(R):>20}{f(C):>20}{f(V):>20}")
            主表[f"{m}/{a}"] = dict(回报=R.tolist(), 成本=C.tolist(), CVaR=V.tolist())

    汇 = {}
    for m in 可分析:
        s = 种子[m]
        print(f"\n{'─'*104}\n{m}：配对检验（{len(s)} 个种子；差值为「甲 − 乙」，负=甲更小）\n{'─'*104}")
        子 = {}
        for 键, 名 in 指标:
            尾 = 数据[m]["尾部约束"]; 均 = 数据[m]["均值约束"]; 无 = 数据[m]["无约束"]
            r1 = 配对(f"{名}：尾部 − 均值", 尾, 均, s, 键)
            r2 = 配对(f"{名}：均值 − 无约束", 均, 无, s, 键)
            子[名] = dict(尾部减均值=r1, 均值减无约束=r2)
            for r in (r1, r2):
                wp = r["wilcoxon_p"]
                print(f"    {r['名称']:<26} 差 {r['差']:>+12.3f}（{r['相对']*100:>+7.1f}%）"
                      f" d={r['配对d']:>+6.2f} t检验 p={r['p']:.4f}"
                      f" Wilcoxon p={(f'{wp:.4f}' if wp is not None else '—')}"
                      f" 95%CI[{r['CI'][0]:+.3f},{r['CI'][1]:+.3f}]"
                      f" {'**显著**' if r['显著'] else '未达显著'}")
        汇[m] = 子

    print(f"\n{'─'*104}\n约束满足率（对偶轨迹里 约束量 ≤ 预算 的比例）\n{'─'*104}")
    满足 = {}
    for m in 可分析:
        for a in ("均值约束", "尾部约束"):
            全 = []
            for i in 种子[m]:
                轨 = 数据[m][a][i].get("对偶轨迹") or []
                if 轨:
                    量 = np.array([x[1] for x in 轨], float)
                    预 = 数据[m][a][i].get("预算", 0.75)
                    全.append((量 <= 预))
            if 全:
                比 = float(np.concatenate(全).mean()) * 100
                print(f"  {m}/{a:<6} 满足率 {比:5.1f}%"
                      f"  → {'达标' if 比 > 0 else '**从未达标**（方向对但没到预算）'}")
                满足[f"{m}/{a}"] = 比

    print(f"\n{'='*104}\n结论\n{'='*104}")
    for m in 可分析:
        r = 汇[m]["CVaR@25（越小越安全）"]["尾部减均值"]
        q = 汇[m]["CVaR@25（越小越安全）"]["均值减无约束"]
        rr = 汇[m]["回报（越大越好）"]["尾部减均值"]
        print(f"  {m}：CVaR 尾部 − 均值 {r['相对']*100:+.1f}%（p={r['p']:.4f}，"
              f"{'显著' if r['显著'] else '未达显著'}）"
              f"；均值 − 无约束 {q['相对']*100:+.1f}%（p={q['p']:.4f}）"
              f"；回报 尾部 − 均值 {rr['相对']*100:+.1f}%")
    if len(可分析) == 2:
        a = 汇["IQL"]["CVaR@25（越小越安全）"]["尾部减均值"]["差"]
        b = 汇["TD3BC"]["CVaR@25（越小越安全）"]["尾部减均值"]["差"]
        print(f"\n  跨主干方向：{'一致（都是尾部更低）' if (a<0)==(b<0) else '不一致'}")
    print(f"\n  提醒：n≥8 时 Wilcoxon 才可能有判定力（n=8 时最小 p=0.0078，n=10 时 0.00195）。")

    with open(os.path.join(日目录, "结果_HC主表分析.json"), "w", encoding="utf-8") as f:
        json.dump(dict(主表=主表, 配对检验=汇, 约束满足率=满足, 种子=种子),
                  f, ensure_ascii=False, indent=2, default=float)
    print(f"\n  已保存 每日结果\\Day18_HalfCheetah机制验证\\结果_HC主表分析.json")


if __name__ == "__main__":
    主函数()
