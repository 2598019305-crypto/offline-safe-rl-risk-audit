#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""COSTPRED_V1 图：真实 vs 预测的平均原生成本（测试策略，按家族着色）。"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'carcircle'))
import night_lib as L  # noqa: E402

OUT = L.WORK / 'costpred'
FIG = OUT / 'figures'


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    d = pd.read_csv(OUT / 'costpred_predictions.csv')
    res = json.loads((OUT / 'costpred_results.json').read_text(encoding='utf-8'))
    fig, ax = plt.subplots(figsize=(5.6, 5.0))
    arms = sorted(d.arm.unique())
    cmap = plt.get_cmap('tab10')
    for i, a in enumerate(arms):
        g = d[d.arm == a]
        ax.scatter(g.mean_cost, g.pred_FULL_3, s=46, color=cmap(i % 10), label=a,
                   edgecolor='k', linewidth=0.4)
    lo = float(min(d.mean_cost.min(), d.pred_FULL_3.min())) - 10
    hi = float(max(d.mean_cost.max(), d.pred_FULL_3.max())) + 10
    ax.plot([lo, hi], [lo, hi], color='0.4', ls='--', lw=1)
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
    ax.set_xlabel('True mean native cost (32 unused env seeds 600-631)')
    ax.set_ylabel('Predicted (offline features + Ridge)')
    ax.set_title('COSTPRED_V1: unseen policy families\nSpearman rho=%.3f  MAE=%.1f  (target sd=%.1f)'
                 % (res['models']['FULL_3']['spearman'], res['models']['FULL_3']['mae'],
                    res['target_sd_test']), fontsize=10)
    ax.grid(alpha=0.25); ax.legend(fontsize=7, loc='upper left')
    for ext in ('png', 'svg'):
        fig.savefig(FIG / ('costpred_scatter.%s' % ext), dpi=300, bbox_inches='tight')
    plt.close(fig)
    # 条形对比：各模型 Spearman
    fig, ax = plt.subplots(figsize=(6.4, 3.2))
    names = list(res['models'].keys())
    vals = [res['models'][n]['spearman'] for n in names]
    ax.bar(np.arange(len(names)), vals, color='#2166ac')
    for i, v in enumerate(vals):
        ax.text(i, v, ' %.3f' % v, ha='center', va='bottom', fontsize=8)
    ax.axhline(0.6, color='g', ls=':', lw=1, label='usable band (0.60)')
    ax.axhline(0.3, color='orange', ls=':', lw=1, label='weak band (0.30)')
    ax.set_xticks(np.arange(len(names))); ax.set_xticklabels(names, rotation=20, fontsize=8)
    ax.set_ylabel('Spearman rho (test)'); ax.grid(alpha=0.25, axis='y'); ax.legend(fontsize=7)
    ax.set_title('COSTPRED_V1: model comparison on 19 unseen-family policies', fontsize=10)
    for ext in ('png', 'svg'):
        fig.savefig(FIG / ('costpred_models.%s' % ext), dpi=300, bbox_inches='tight')
    plt.close(fig)
    d.to_csv(FIG / 'costpred_scatter_data.csv', index=False, encoding='utf-8-sig')
    print('图已写出:', FIG)


if __name__ == '__main__':
    main()
