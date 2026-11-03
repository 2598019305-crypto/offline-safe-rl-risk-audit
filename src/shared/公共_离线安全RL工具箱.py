#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""公共工具箱：三个判据实验共用的一套代码。

这里集中放四样东西，保证三个实验用的是同一套标准，不会互相打架：
  1) 数据读取：DSRL CarGoal1 的离线数据（按轨迹切开，保留成本通道）；
  2) 风险与行为网络：成本评论家（上尾 expectile）+ 行为克隆（高斯），
     前者给出"尾部风险"信号，后者给出"支撑度"信号；
  3) 状态支撑度工具：用 k 近邻距离衡量"这个状态在数据里常不常见"，
     并把距离换算成"在数据自身分布里的百分位"，这样数字才有尺度感；
  4) 批量回放引擎：一次推进多个环境，每一步只做一次批量网络前向，
     支持四种门控模式和四种干预形式。

所有阈值都只由数据本身决定（分位数），不引入任何"调出来的"魔法数字。
"""

import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

OBS_DIM, ACT_DIM, GAMMA = 72, 2, 0.99
REPO_ROOT = Path(__file__).resolve().parents[2]
数据集目录 = os.environ.setdefault(
    "DSRL_DATASET_DIR",
    str(REPO_ROOT / "data" / "raw"),
)

门控模式列表 = ["无", "均匀", "只支撑", "只风险", "联合"]
干预形式列表 = ["凸混合", "硬切换", "随机切换", "风险最小"]


# ----------------------------------------------------------------------------
# 一、数据读取
# ----------------------------------------------------------------------------
def 读取CarGoal1轨迹(limit=200_000):
    """按轨迹读出 CarGoal1 离线数据。返回 (完整字典, 轨迹列表)。

    轨迹列表里每条是 dict(obs, act, rew, cost)，方便做"按轨迹筛选"这类操作。
    """
    import gymnasium as gym
    import dsrl  # noqa: F401

    env = gym.make("OfflineCarGoal1Gymnasium-v0")
    ds = env.get_dataset()
    env.close()
    ends = np.where((ds["terminals"].reshape(-1) + ds["timeouts"].reshape(-1)) > 0.5)[0] + 1
    bounds = np.concatenate([[0], ends])

    轨迹 = []
    总数 = 0
    for a, b in zip(bounds[:-1], bounds[1:]):
        if b - a < 2:
            continue
        轨迹.append(dict(
            obs=ds["observations"][a:b].astype(np.float32),
            act=ds["actions"][a:b].astype(np.float32),
            rew=ds["rewards"][a:b].reshape(-1).astype(np.float32),
            cost=ds["costs"][a:b].reshape(-1).astype(np.float32),
        ))
        总数 += b - a
        if 总数 >= limit:
            break

    拼 = lambda k: np.concatenate([t[k] for t in 轨迹])  # noqa: E731
    合并 = dict(obs=拼("obs"), act=拼("act"), rew=拼("rew"), cost=拼("cost"))
    return 合并, 轨迹


def 轨迹合成为d3rlpy数据集(轨迹, 步长上限=None):
    """把轨迹列表打包成 d3rlpy 的 MDPDataset（terminals 全 False、按 timeout 截断）。

    离线数据里每条轨迹最后一个 transition 的 terminals 或 timeouts 为 1，
    这里统一按 timeouts 处理，与 DSRL 官方 d3rlpy 转换脚本保持一致。
    """
    from d3rlpy.dataset import MDPDataset

    obs, act, rew, tos = [], [], [], []
    for t in 轨迹:
        n = len(t["obs"])
        if 步长上限 is not None:
            n = min(n, 步长上限)
        obs.append(t["obs"][:n])
        act.append(t["act"][:n])
        rew.append(t["rew"][:n])
        t_out = np.zeros(n, dtype=np.float32)
        t_out[-1] = 1.0
        tos.append(t_out)
    cat = lambda x: np.concatenate(x).astype(np.float32)  # noqa: E731
    return MDPDataset(observations=cat(obs), actions=cat(act), rewards=cat(rew),
                      terminals=np.zeros_like(cat(tos)), timeouts=cat(tos))


# ----------------------------------------------------------------------------
# 二、风险与行为网络
# ----------------------------------------------------------------------------
def _mlp(i, o, h=256):
    return nn.Sequential(nn.Linear(i, h), nn.ReLU(), nn.Linear(h, h), nn.ReLU(), nn.Linear(h, o))


class 风险与行为网络(nn.Module):
    """qc: 成本动作价值；vc: 成本状态价值（上尾 expectile）；bc: 行为克隆高斯。

    两个信号的分工：
      · vc 越大 = 从该状态出发的尾部成本越高（风险信号）；
      · bc_logp 越小 = 这个动作越不像数据里的动作（支撑度信号）。
    """

    def __init__(self):
        super().__init__()
        self.qc = _mlp(OBS_DIM + ACT_DIM, 1)
        self.vc = _mlp(OBS_DIM, 1)
        self.bc = _mlp(OBS_DIM, 2 * ACT_DIM)

    def bc_logp(self, s, a):
        out = self.bc(s)
        mu, ls = out[:, :ACT_DIM], out[:, ACT_DIM:].clamp(-5, 2)
        return (-0.5 * ((a - mu) / ls.exp()).pow(2) - ls - 0.9189).sum(-1, keepdim=True)

    def bc_mean(self, s):
        return torch.tanh(self.bc(s)[:, :ACT_DIM])


def 训练风险与行为网络(obs_np, act_np, cost_np, steps=8000, batch=256, 静默=False):
    """同一份数据上联合训练成本评论家、成本价值与行为克隆，和之前实验完全一致。"""
    obs = torch.tensor(obs_np, dtype=torch.float32)
    act = torch.tensor(act_np, dtype=torch.float32)
    cst = torch.tensor(cost_np, dtype=torch.float32).reshape(-1, 1)
    n = len(obs)
    net = 风险与行为网络()
    opt = torch.optim.Adam(net.parameters(), lr=3e-4)
    t0 = time.time()
    for step in range(1, steps + 1):
        j = torch.randint(0, n, (batch,))
        s, a, c = obs[j], act[j], cst[j]
        qc = net.qc(torch.cat([s, a], -1))
        loss_q = ((qc - c) ** 2).mean()
        d = qc.detach() - net.vc(s)
        loss_v = (torch.where(d > 0, 0.9, 0.1) * d.pow(2)).mean()
        loss_bc = -net.bc_logp(s, a).mean()
        opt.zero_grad()
        (loss_q + loss_v + loss_bc).backward()
        nn.utils.clip_grad_norm_(net.parameters(), 5.0)
        opt.step()
        if (not 静默) and step % 2000 == 0:
            print(f"  [风险评论家/BC] step {step:>5} {time.time()-t0:5.0f}s "
                  f"loss_q={loss_q.item():.4f} loss_v={loss_v.item():.4f} "
                  f"loss_bc={loss_bc.item():.3f}", flush=True)
    net.eval()
    return net


def 门控阈值(net, obs_np, act_np, 抽样=50_000):
    """两个阈值都取数据自身的分位数：风险取 75 分位，支撑度取 25 分位。"""
    idx = np.random.RandomState(0).choice(len(obs_np), size=min(抽样, len(obs_np)), replace=False)
    obs = torch.tensor(obs_np[idx], dtype=torch.float32)
    act = torch.tensor(act_np[idx], dtype=torch.float32)
    with torch.no_grad():
        vc = net.vc(obs).reshape(-1).numpy()
        ll = net.bc_logp(obs, act).reshape(-1).numpy()
    return float(np.quantile(vc, 0.75)), float(np.quantile(ll, 0.25)), vc, ll


# ----------------------------------------------------------------------------
# 三、状态支撑度（k 近邻 + 数据自身百分位）
# ----------------------------------------------------------------------------
class 状态支撑度工具:
    """把"状态在不在数据里"变成一个可比较的数字。

    做法：用数据自己的状态做参照集，算每个状态的 k 近邻平均距离；
    再把新状态的这个距离映射成"在数据自身分布里的百分位"。
    百分位高 = 这个状态在数据里很罕见（靠近 100 就是数据外）。
    """

    def __init__(self, 数据状态, k=10, 参照数=20_000, 查询数=2_000, seed=0):
        from scipy.spatial import cKDTree

        rs = np.random.RandomState(seed)
        n = len(数据状态)
        idx = rs.choice(n, size=min(参照数 + 查询数, n), replace=False)
        参照 = 数据状态[idx[:参照数]]
        查询 = 数据状态[idx[参照数:]]

        self.均值 = 数据状态.mean(0)
        self.标准差 = np.maximum(数据状态.std(0), 1e-6)
        self.k = k
        self.树 = cKDTree((参照 - self.均值) / self.标准差)
        # 用数据自己的另一批状态当查询，得到"数据内状态"的距离分布
        自身距离 = self._原始距离(查询)
        self.自身距离分位 = np.sort(自身距离)

    def _原始距离(self, 状态):
        z = (状态 - self.均值) / self.标准差
        d, _ = self.树.query(z, k=self.k, workers=-1)
        return d.mean(1)

    def 查询(self, 状态, 每步抽样=1):
        """返回 (k近邻平均距离, 在数据自身分布里的百分位)。"""
        if 每步抽样 > 1:
            状态 = 状态[::每步抽样]
        距离 = self._原始距离(状态)
        百分位 = np.searchsorted(self.自身距离分位, 距离) / len(self.自身距离分位) * 100.0
        return 距离, 百分位


# ----------------------------------------------------------------------------
# 四、批量回放引擎
# ----------------------------------------------------------------------------
def 算门控权重(模式, vc, ll, vc_thr, ll_thr):
    """四种门控模式，全部返回 0/1 权重（硬门控，先不引入软门控的温度参数）。"""
    风险 = (vc >= vc_thr).astype(float)
    支撑 = (ll >= ll_thr).astype(float)
    return 算门控权重_通用(模式, 风险, 支撑)


def 算门控权重_通用(模式, 风险, 支撑):
    """门控权重的通用形式：输入已经是 0/1 的「风险高」「支撑足」两个判据。

    这样写的好处是：风险判据既可以是状态级的 vc(s)，也可以是动作级的 qc(s,a)，
    下游四象限与门控逻辑完全不用改，两套方案可以直接对比。
    """
    风险 = np.asarray(风险, dtype=float)
    支撑 = np.asarray(支撑, dtype=float)
    if 模式 == "无":
        return np.zeros_like(风险)
    if 模式 == "均匀":
        return np.ones_like(风险)
    if 模式 == "只支撑":
        return 支撑
    if 模式 == "只风险":
        return 风险
    if 模式 == "联合":
        return 风险 * 支撑
    raise ValueError(模式)


def _一步决策(模式, 干预形式, kappa, 风险级别, qc阈值, vc_thr, ll_thr,
              net, 参照网络, s_batch, a_pi, 随机种子):
    """纯函数式的"一步决策"：给定观测，算出要执行的动作与门控权重。

    单进程引擎和多进程引擎共用这一段代码，保证两种跑法行为完全一致
    （这是后面做"一致性校验"的前提）。
    """
    a_pi_t = torch.tensor(a_pi, dtype=torch.float32)
    with torch.no_grad():
        vc = net.vc(s_batch).reshape(-1).numpy()
        ll = net.bc_logp(s_batch, a_pi_t).reshape(-1).numpy()
        a_ref = 参照网络.bc_mean(s_batch).numpy().astype(np.float32)
        q_pi = net.qc(torch.cat([s_batch, a_pi_t], -1)).reshape(-1).numpy()
    风险 = (q_pi >= qc阈值) if 风险级别 == "动作" else (vc >= vc_thr)
    支撑 = (ll >= ll_thr)
    w = 算门控权重_通用(模式, 风险.astype(float), 支撑.astype(float))
    if 模式 == "无" or kappa == 0.0:
        a_exec = a_pi
    elif 干预形式 == "凸混合":
        alpha = np.clip(kappa * w, 0.0, 1.0)[:, None]
        a_exec = (1 - alpha) * a_pi + alpha * a_ref
    elif 干预形式 == "硬切换":
        a_exec = np.where((w > 0.5)[:, None], a_ref, a_pi)
    elif 干预形式 == "随机切换":
        r = np.random.RandomState(随机种子).rand(len(a_pi))
        a_exec = np.where((r < w)[:, None], a_ref, a_pi)
    elif 干预形式 == "风险最小":
        with torch.no_grad():
            q_ref = net.qc(torch.cat([s_batch, torch.tensor(a_ref)], -1)).reshape(-1).numpy()
        更安全 = (q_ref < q_pi) & (w > 0.5)
        a_exec = np.where(更安全[:, None], a_ref, a_pi)
    else:
        raise ValueError(干预形式)
    return np.clip(a_exec, -1.0, 1.0).astype(np.float32), w, vc, ll, q_pi


def _打包结果(回报, 成本, 轨迹记录, 回合数, 秒数):
    结果 = dict(逐回合回报=np.asarray(回报).tolist(), 逐回合成本=np.asarray(成本).tolist(),
               回报均值=float(np.mean(回报)), 成本均值=float(np.mean(成本)),
               成本CVaR25=float(np.sort(成本)[-max(1, 回合数 // 4):].mean()),
               成本标准差=float(np.std(成本, ddof=1)), 回合数=回合数, 秒数=round(秒数, 1))
    if 轨迹记录 is not None:
        结果["轨迹"] = {k: np.asarray(v) for k, v in 轨迹记录.items()}
    return 结果


def _新建轨迹记录(记录轨迹):
    return (dict(成本=[], 门控权重=[], vc=[], ll=[], qc=[], 状态=[], 状态下标=[],
                 奖励=[], 回合=[], 动作=[])
            if 记录轨迹 else None)


def _记一步(轨迹记录, w_k, vc_k, ll_k, qc_k, 回合号, 步计数, 状态抽样, 观测, 动作=None):
    轨迹记录["成本"].append(0.0)      # 先占位，环境推进一步后回填
    轨迹记录["奖励"].append(0.0)
    轨迹记录["门控权重"].append(w_k)
    轨迹记录["vc"].append(vc_k)
    轨迹记录["ll"].append(ll_k)
    轨迹记录["qc"].append(qc_k)
    轨迹记录["回合"].append(回合号)
    if 动作 is not None:
        轨迹记录["动作"].append(np.array(动作, copy=True))
    # 用"扁平步号"决定要不要存状态，并把它对应的下标记下来，
    # 这样后面做 k 近邻分析时能精确对齐，不会因为每轮推进的环境数不同而错位。
    扁平下标 = len(轨迹记录["成本"]) - 1
    if 扁平下标 % 状态抽样 == 0:
        轨迹记录["状态"].append(np.array(观测, copy=True))
        轨迹记录["状态下标"].append(扁平下标)


def 批量回放(algo, net, vc_thr, ll_thr, 模式="无", 干预形式="凸混合", kappa=0.0,
             回合数=200, 并行数=16, 起始种子=7000, 记录轨迹=False, 状态抽样=4,
             打印间隔=0, 参考策略="BC", 安全网络=None,
             风险级别="状态", qc阈值=None):
    """单进程引擎：一次推进 并行数 个环境，每一步只做一次批量网络前向。

    超过 16 个 MuJoCo 环境会吃掉好几个 GB 内存，16 GB 的机器上会开始内存交换，
    反而变慢；那种情况下请用 批量回放_多进程。
    """
    import gymnasium as gym
    import dsrl  # noqa: F401

    envs = [gym.make("SafetyCarGoal1Gymnasium-v0") for _ in range(并行数)]
    种子 = [起始种子 + i for i in range(回合数)]
    下一个发 = min(并行数, 回合数)
    obs = np.zeros((并行数, OBS_DIM), dtype=np.float32)
    在跑 = np.zeros(并行数, dtype=bool)
    for i in range(下一个发):
        obs[i], _ = envs[i].reset(seed=种子[i])
        在跑[i] = True

    回报 = np.zeros(回合数)
    成本 = np.zeros(回合数)
    当前回报 = np.zeros(并行数)
    当前成本 = np.zeros(并行数)
    归属 = np.full(并行数, -1, dtype=int)
    for i in range(并行数):
        if 在跑[i]:
            归属[i] = i

    轨迹记录 = _新建轨迹记录(记录轨迹)
    参照网络 = net if (参考策略 == "BC" or 安全网络 is None) else 安全网络
    步计数 = 0
    t0 = time.time()
    while 在跑.any():
        idx = np.where(在跑)[0]
        s_batch = torch.tensor(obs[idx], dtype=torch.float32)
        a_pi = np.asarray(algo.predict(obs[idx]), dtype=np.float32)
        a_exec, w, vc, ll, q_pi = _一步决策(模式, 干预形式, kappa, 风险级别, qc阈值,
                                          vc_thr, ll_thr, net, 参照网络, s_batch, a_pi,
                                          1000 + 步计数)
        if 记录轨迹:
            for k, i in enumerate(idx):
                _记一步(轨迹记录, w[k], vc[k], ll[k], q_pi[k], 归属[i], 步计数,
                        状态抽样, obs[i], a_exec[k])
        for k, i in enumerate(idx):
            o, rr, te, tr, info = envs[i].step(a_exec[k])
            c = float(info.get("cost", 0.0))
            当前回报[i] += float(rr)
            当前成本[i] += c
            obs[i] = o
            if 记录轨迹:
                轨迹记录["成本"][-len(idx) + k] = c
                轨迹记录["奖励"][-len(idx) + k] = float(rr)
            if te or tr:
                回报[归属[i]] = 当前回报[i]
                成本[归属[i]] = 当前成本[i]
                当前回报[i] = 0.0
                当前成本[i] = 0.0
                if 下一个发 < 回合数:
                    obs[i], _ = envs[i].reset(seed=种子[下一个发])
                    归属[i] = 下一个发
                    下一个发 += 1
                else:
                    在跑[i] = False
                    归属[i] = -1
        步计数 += 1
        if 打印间隔 and 步计数 % 打印间隔 == 0:
            print(f"    ... 已推进 {步计数} 轮，共 {步计数 * len(idx)} 次交互，"
                  f"用时 {time.time()-t0:.0f}s，"
                  f"约 {步计数 * len(idx) / max(time.time()-t0, 1e-9):.0f} 步/秒", flush=True)
    for e in envs:
        e.close()
    return _打包结果(回报, 成本, 轨迹记录, 回合数, time.time() - t0)


def 批量回放_多进程(algo, net, vc_thr, ll_thr, 模式="无", 干预形式="凸混合", kappa=0.0,
                   回合数=200, 进程数=8, 起始种子=7000, 记录轨迹=False, 状态抽样=4,
                   打印间隔=0, 参考策略="BC", 安全网络=None,
                   风险级别="状态", qc阈值=None):
    """多进程引擎：每个子进程独占一个环境，8 个进程合计能到几千步/秒。

    与单进程引擎共用同一段 _一步决策，所以策略行为逐位一致；
    唯一差别是"环境推进"被搬进了子进程。

    这里刻意**不用 multiprocessing.Pool**：Pool 把任务派给哪个工人是不确定的，
    同一步里"给第 k 号回合的动作"可能被送进持有别的回合状态的环境，状态会串。
    改成"第 k 根管道 ↔ 第 k 个环境"的固定绑定后，两个引擎才能真正逐位一致。
    """
    import multiprocessing as mp

    HERE = os.path.dirname(os.path.abspath(__file__))
    if HERE not in sys.path:
        sys.path.insert(0, HERE)
    import 并行环境工人 as 工人

    种子 = [起始种子 + i for i in range(回合数)]
    W = min(进程数, 回合数)
    回报 = np.zeros(回合数)
    成本 = np.zeros(回合数)
    当前回报 = np.zeros(W)
    当前成本 = np.zeros(W)
    归属 = np.full(W, -1, dtype=int)
    obs = np.zeros((W, OBS_DIM), dtype=np.float32)
    在跑 = np.zeros(W, dtype=bool)
    轨迹记录 = _新建轨迹记录(记录轨迹)
    参照网络 = net if (参考策略 == "BC" or 安全网络 is None) else 安全网络
    步计数 = 0
    t0 = time.time()

    父端, 子端, 进程表 = [], [], []
    for _ in range(W):
        甲, 乙 = mp.Pipe()
        父端.append(甲)
        子端.append(乙)
    for 乙 in 子端:
        p = mp.Process(target=工人.工人主循环, args=(乙,), daemon=True)
        p.start()
        进程表.append(p)
    for 乙 in 子端:
        乙.close()
    for 甲 in 父端:
        if not 甲.poll(240):
            raise RuntimeError("子进程启动超时")
        状态, _ = 甲.recv()
        if 状态 != "就绪":
            raise RuntimeError(f"子进程启动失败：{状态}")

    def 收(槽):
        if not 父端[槽].poll(240):
            raise RuntimeError(f"第 {槽} 号环境子进程超过 240 秒没有回应，可能已崩溃")
        return 父端[槽].recv()

    def 重置一批(任务):
        """任务 = [(槽, 种子), ...]；返回后 obs/在跑/归属 都已更新。"""
        for 槽, 种 in 任务:
            父端[槽].send(("重置", int(种)))
        for 槽, 种 in 任务:
            状态, 观测 = 收(槽)
            obs[槽] = 观测
            在跑[槽] = True
            归属[槽] = int(种) - 起始种子

    try:
        下一个发 = W
        重置一批([(i, 种子[i]) for i in range(W)])
        while 在跑.any():
            idx = np.where(在跑)[0]
            s_batch = torch.tensor(obs[idx], dtype=torch.float32)
            a_pi = np.asarray(algo.predict(obs[idx]), dtype=np.float32)
            a_exec, w, vc, ll, q_pi = _一步决策(模式, 干预形式, kappa, 风险级别, qc阈值,
                                              vc_thr, ll_thr, net, 参照网络, s_batch, a_pi,
                                              1000 + 步计数)
            if 记录轨迹:
                for k, i in enumerate(idx):
                    _记一步(轨迹记录, w[k], vc[k], ll[k], q_pi[k], 归属[i], 步计数,
                            状态抽样, obs[i], a_exec[k])
            for k, i in enumerate(idx):
                父端[i].send(("步进", a_exec[k]))
            结束槽 = []
            for k, i in enumerate(idx):
                状态, 观测, rr, c, 完 = 收(i)
                当前回报[i] += rr
                当前成本[i] += c
                obs[i] = 观测
                if 记录轨迹:
                    轨迹记录["成本"][-len(idx) + k] = c
                    轨迹记录["奖励"][-len(idx) + k] = rr
                if 完:
                    回报[归属[i]] = 当前回报[i]
                    成本[归属[i]] = 当前成本[i]
                    当前回报[i] = 0.0
                    当前成本[i] = 0.0
                    # 必须把"本轮已经排队要补的槽"也算进去，否则多个环境在同一轮
                    # 同时结束时，末段会取到超出种子表的回合编号（会直接报越界）。
                    if 下一个发 + len(结束槽) < 回合数:
                        结束槽.append(int(i))
                    else:
                        在跑[i] = False
                        归属[i] = -1
            if 结束槽:
                新任务 = [(i, 种子[下一个发 + j]) for j, i in enumerate(结束槽)]
                重置一批(新任务)
                下一个发 += len(结束槽)
            步计数 += 1
            if 打印间隔 and 步计数 % 打印间隔 == 0:
                print(f"    ... 已推进 {步计数} 轮，共 {步计数 * len(idx)} 次交互，"
                      f"用时 {time.time()-t0:.0f}s，"
                      f"约 {步计数 * len(idx) / max(time.time()-t0, 1e-9):.0f} 步/秒", flush=True)
    finally:
        for 甲 in 父端:
            try:
                甲.send(("结束", None))
            except Exception:
                pass
        for p in 进程表:
            p.join(timeout=10)
            if p.is_alive():
                p.terminate()
        for 甲 in 父端:
            try:
                甲.close()
            except Exception:
                pass
    return _打包结果(回报, 成本, 轨迹记录, 回合数, time.time() - t0)


def 分组统计(逐回合成本, 逐回合回报, 组数=4):
    """把逐回合结果切成若干等份，用来给出均值的波动范围（跨组标准差）。"""
    c = np.asarray(逐回合成本)
    r = np.asarray(逐回合回报)
    n = len(c) // 组数
    各成本 = [float(np.sort(c[i * n:(i + 1) * n])[-max(1, n // 4):].mean()) for i in range(组数)]
    return dict(回报均值=float(r.mean()), 回报组间标准差=float(np.std([r[i*n:(i+1)*n].mean()
                                                                     for i in range(组数)], ddof=1)),
                成本均值=float(c.mean()), 成本组间标准差=float(np.std([c[i*n:(i+1)*n].mean()
                                                                     for i in range(组数)], ddof=1)),
                成本CVaR25=float(np.sort(c)[-max(1, len(c) // 4):].mean()),
                CVaR各组=各成本,
                CVaR组间标准差=float(np.std(各成本, ddof=1)))
