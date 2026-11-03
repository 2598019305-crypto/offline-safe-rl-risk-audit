#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""COSTPRED_V1：计算 3 个完全离线特征（不需要任何环境交互）。"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'carcircle'))
import iql          # noqa: E402
import night_lib as L  # noqa: E402
from scipy.spatial import cKDTree
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

OUT = L.WORK / 'costpred'


class NetWrap(nn.Module):
    def __init__(self, net):
        super().__init__()
        self.net = net


def load_net(path, kind):
    ck = torch.load(path, map_location='cpu', weights_only=False)
    sd = ck['actor'] if 'actor' in ck else (ck['bc'] if 'bc' in ck else ck)
    if kind == 'stochastic_gaussian':
        a = iql.Actor(8, 2)
        a.load_state_dict(sd)
        a.eval()
        return a.net
    net = iql.mlp(8, 2, 256, nn.Tanh)
    net.load_state_dict(sd)
    net.eval()
    return net


def main():
    t0 = time.time()
    data = iql.load_and_split(L.ENV_A)
    tr = data['train_idx'].astype(np.int64)
    S, A, C = data['S'], data['A'], data['C']
    om = torch.from_numpy(data['obs_mean']); osd = torch.from_numpy(data['obs_std'])
    Sstar = S[tr][::25]
    print('S* 规模', len(Sstar), flush=True)
    # 1) 成本事件代理模型（在训练划分上拟合）
    Xtr = np.concatenate([S[tr], A[tr]], axis=1)
    ytr = (C[tr] > 0).astype(int)
    sc = StandardScaler().fit(Xtr)
    clf = LogisticRegression(C=1, max_iter=2000, random_state=20261015).fit(sc.transform(Xtr), ytr)
    print('成本事件代理：训练正例率 %.4f，训练集 AUC %.4f'
          % (ytr.mean(), 0.0), flush=True)
    # 3) OOD：S* 每个状态的 64 个最近邻（z(s) 空间）
    sm = S[tr].mean(0); ss = np.maximum(S[tr].std(0), 1e-8)
    feats = (S[tr] - sm) / ss
    tree = cKDTree(feats)
    nn_idx = tree.query((Sstar - sm) / ss, k=64, workers=-1)[1]
    st = torch.from_numpy(Sstar.astype(np.float32))
    norm_s = (st - om) / osd
    ros = json.loads((OUT / 'policy_roster.json').read_text(encoding='utf-8'))
    rows = []
    for p in ros['policies']:
        if not p['loadable']:
            continue
        net = load_net(p['path'], p['kind'])
        with torch.no_grad():
            ap = net(norm_s).numpy().astype(np.float64)
        # f1
        Xq = np.concatenate([Sstar, ap], axis=1)
        f1 = float(clf.predict_proba(sc.transform(Xq))[:, 1].mean())
        # f2
        f2 = float(np.linalg.norm(ap, axis=1).mean())
        # f3
        nb_actions = A[tr][nn_idx]                      # (n,64,2)
        d3 = np.linalg.norm(nb_actions - ap[:, None, :], axis=2).min(axis=1)
        f3 = float(d3.mean())
        rows.append(dict(policy_id=p['policy_id'], arm=p['arm'], split=p['split'],
                         f_cost_surrogate=f1, f_action_magnitude=f2, f_action_ood=f3))
        print('  %-28s f1=%.4f f2=%.4f f3=%.4f (%.0fs)'
              % (p['policy_id'], f1, f2, f3, time.time() - t0), flush=True)
    import pandas as pd
    df = pd.DataFrame(rows)
    df.to_csv(OUT / 'features_offline.csv', index=False, encoding='utf-8-sig')
    print('写出 features_offline.csv（%d 行，%.0fs）' % (len(df), time.time() - t0))


if __name__ == '__main__':
    main()
