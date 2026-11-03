#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""用已保存的 CPQ 检查点验证修复后的评估路径（8 个开发回合，约 10 秒）。"""
from __future__ import annotations

import os
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
VENDOR = SCRIPT_DIR / 'vendor'
sys.path.insert(0, str(VENDOR / 'fsrl_stub'))
sys.path.insert(0, str(VENDOR / 'OSRL'))
sys.path.insert(0, str(SCRIPT_DIR))

import numpy as np            # noqa: E402
import torch                  # noqa: E402
import bullet_safety_gym      # noqa: E402,F401
import dsrl                   # noqa: E402,F401
import gymnasium as gym       # noqa: E402
from osrl.algorithms import CPQ  # noqa: E402
import night_lib as L         # noqa: E402


def main():
    checkpoint = Path(
        os.environ.get(
            'CPQ_CHECKPOINT',
            SCRIPT_DIR.parents[1]
            / 'artifacts'
            / 'checkpoints'
            / 'carcircle'
            / 'cpq_full'
            / 'cpq_seed0.pt',
        )
    )
    ck = torch.load(checkpoint, map_location='cpu')
    env = gym.make('OfflineCarCircle-v0')
    m = CPQ(state_dim=8, action_dim=2, max_action=float(env.action_space.high[0]),
            a_hidden_sizes=[256, 256], c_hidden_sizes=[256, 256], vae_hidden_sizes=400,
            sample_action_num=10, gamma=0.99, tau=0.005, beta=0.5, num_q=2, num_qc=2,
            qc_scalar=1.5, cost_limit=10, episode_len=300, device='cpu')
    m.load_state_dict(ck['model_state'])
    print('已载入 step=%s 的检查点' % ck.get('step'))
    for es in (510, 511):
        e = gym.make('OfflineCarCircle-v0')
        try:
            np.random.seed(es)
            obs, _ = e.reset(seed=es)
            c_tot = r_tot = 0.0
            t, done = 0, False
            while not done:
                a, _ = m.act(obs.astype(np.float32), deterministic=True, with_logprob=True)
                a = np.clip(a, e.action_space.low, e.action_space.high).astype(np.float32)
                obs, r, c, done, _ = L.step_env(e, a)
                r_tot += r * 0.1
                c_tot += c
                t += 1
            print('  seed%d: steps=%d cost=%.1f reward(缩放)=%.2f' % (es, t, c_tot, r_tot))
        finally:
            e.close()
    print('EVAL_PATH_OK')


if __name__ == '__main__':
    main()
