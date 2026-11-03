#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""P0-2 夜间无人值守实验：公共库。

约定：
  - 训练/临时文件放 WSL ext4（WORK）；阶段产物与状态文件同步复制到 Windows 输出目录（OUT）。
  - 绝不触碰封存评估种子 125000-125099。
  - 所有随机性显式播种；数据只从本地 HDF5 读取（h5path=），禁止隐式联网下载。
"""
from __future__ import annotations
import csv, hashlib, json, os, platform, shutil, subprocess, sys, threading, time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
WORK = Path(os.environ.get('REPRO_WORK_DIR', REPO_ROOT / 'outputs'))
OUT = Path(os.environ.get('REPRO_OUTPUT_DIR', WORK))
DATA_DIR = Path(os.environ.get('CAR_CIRCLE_DATA_DIR', REPO_ROOT / 'data' / 'raw' / 'car_circle'))
P01_DIR = Path(os.environ.get('CAR_CIRCLE_P01_DIR', REPO_ROOT / 'artifacts' / 'car_circle' / 'p01'))

ENV_A = 'OfflineCarCircle-v0'
ENV_B = 'OfflineBallCircle-v0'
DATASET = {ENV_A: DATA_DIR / 'SafetyCarCircle-v0-100-1450.hdf5',
           ENV_B: DATA_DIR / 'SafetyBallCircle-v0-80-886.hdf5'}
HF_REPO = 'YYY-45/DSRL'
HF_COMMIT = 'dbe74cbc3a3435558b182166f238c677de3a8a33'

DEV_SEEDS = list(range(510, 518))          # P0-2 开发评估种子
RNG_SEED = 20261013                        # 预注册随机种子（抽样/分组）
SEALED = (125000, 125100)                  # 封存评估种子区间（禁用）
ACTOR_SEEDS = (0, 1, 2)

GAMMA = 0.99
BATCH = 256
LR = 3e-4
CRITIC_UPDATES = 40000
ACTOR_UPDATES = 10000
EXPECTILE_TAU = 0.7                        # 标准 IQL 官方默认
POLYAK = 0.005
IQL_TEMPERATURE = 3.0                      # 标准 IQL actor 权重温度（官方默认）
MECH_BETA = 0.5                            # D10 论文算子（机制三臂）
MECH_LAMBDA = 1.0
WEIGHT_CAP = 100.0
BC_UPDATES = 2000
EXIT_GATE_FAIL = 0

_LOCK = threading.Lock()

# 线程数：实测 8 线程对 batch=256 的小网络严重过度订阅（79ms/步 vs 4 线程 3.45ms/步）
try:
    import torch as _torch
    _torch.set_num_threads(int(os.environ.get('NIGHT_TORCH_THREADS', '4')))
    _torch.set_num_interop_threads(1)
except Exception:
    pass


def now() -> str:
    return time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())


class Log:
    def __init__(self, name: str):
        (WORK / 'logs').mkdir(parents=True, exist_ok=True)
        (OUT / 'logs').mkdir(parents=True, exist_ok=True)
        self.p = WORK / 'logs' / f'{name}.log'
        self.win = OUT / 'logs' / f'{name}.log'
        self.f = open(self.p, 'a', encoding='utf-8')

    def __call__(self, msg: str):
        line = f'[{now()}] {msg}'
        with _LOCK:
            print(line, flush=True)
            self.f.write(line + '\n')
            self.f.flush()

    def sync(self):
        try:
            shutil.copy2(self.p, self.win)
        except Exception:
            pass


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for c in iter(lambda: f.read(1 << 20), b''):
            h.update(c)
    return h.hexdigest()


def write_json_atomic(path: Path, obj, indent=2):
    path.parent.mkdir(parents=True, exist_ok=True)
    # 唯一临时名 + 加锁：心跳线程与主线程可能同时写，共用 .tmp 会互相抢走文件
    tmp = path.with_name('%s.%d.%d.tmp' % (path.name, os.getpid(), threading.get_ident()))
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=indent, default=float), encoding='utf-8')
    os.replace(tmp, path)


def copy_to_win(rel: str):
    s = WORK / rel
    d = OUT / rel
    if not s.exists():
        return
    d.parent.mkdir(parents=True, exist_ok=True)
    if s.is_dir():
        shutil.copytree(s, d, dirs_exist_ok=True)
    else:
        shutil.copy2(s, d)


def dump_csv(path: Path, rows, fieldnames=None):
    if not rows:
        path.write_text('', encoding='utf-8-sig')
        return
    fields = fieldnames or []
    if not fields:
        for r in rows:
            for k in r:
                if k not in fields:
                    fields.append(k)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8-sig', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields, restval='')
        w.writeheader()
        w.writerows(rows)


class Status:
    """阶段状态 + 心跳；原子写，同时写到 Windows 输出目录。"""

    def __init__(self, path: Path):
        self.path = path
        self.obj = {'started': now(), 'stage': 'init', 'stages': {}, 'notes': [],
                    'sealed_seeds_untouched': True, 'finished': False}
        if path.exists():
            try:
                self.obj.update(json.loads(path.read_text(encoding='utf-8')))
            except Exception:
                pass

    def set_stage(self, stage: str, **kw):
        self.obj['stage'] = stage
        self.obj['stage_started'] = now()
        self.obj[f'{stage}_info'] = kw
        self.save()

    def note(self, msg: str):
        self.obj.setdefault('notes', []).append(f'[{now()}] {msg}')
        self.save()

    def stage_done(self, stage: str, result):
        self.obj['stages'][stage] = {'status': 'done', 'at': now(), 'result': result}
        self.save()

    def stage_fail(self, stage: str, err: str):
        self.obj['stages'][stage] = {'status': 'fail', 'at': now(), 'error': err}
        self.save()

    def done(self, stage: str) -> bool:
        return self.obj['stages'].get(stage, {}).get('status') == 'done'

    def save(self):
        try:
            with _LOCK:
                self.obj['updated'] = now()
                write_json_atomic(self.path, self.obj)
            try:
                shutil.copy2(self.path, OUT / 'STATUS.json')
            except Exception:
                pass
        except Exception as e:
            # 状态文件写入失败绝不能中断实验
            try:
                print('[warn] STATUS 写入失败: %s' % e, flush=True)
            except Exception:
                pass


class Heartbeat(threading.Thread):
    def __init__(self, status: Status, log: Log, interval=300):
        super().__init__(daemon=True)
        self.status, self.log, self.interval = status, log, interval

    def run(self):
        while True:
            try:
                self.status.save()
                write_json_atomic(OUT / 'HEARTBEAT.json', {
                    'at': now(), 'pid': os.getpid(), 'stage': self.status.obj.get('stage'),
                    'stages': {k: v.get('status') for k, v in self.status.obj.get('stages', {}).items()},
                    'loadavg': list(os.getloadavg()), 'disk_free_gb': round(
                        shutil.disk_usage('/').free / 1e9, 1)})
                self.log.sync()
            except Exception as e:
                self.log(f'心跳写入异常: {type(e).__name__}: {e}')
            time.sleep(self.interval)


def env_versions() -> dict:
    v = {'python': sys.version.split()[0], 'platform': platform.platform()}
    for m in ['numpy', 'torch', 'scipy', 'h5py', 'gym', 'gymnasium', 'pybullet',
              'bullet_safety_gym', 'dsrl']:
        try:
            mod = __import__(m)
            v[m] = getattr(mod, '__version__', '(无版本号)')
        except Exception as e:
            v[m] = f'导入失败: {type(e).__name__}'
    try:
        v['pip_freeze'] = subprocess.run(['pip3', 'freeze'], capture_output=True, text=True,
                                         timeout=120).stdout
    except Exception as e:
        v['pip_freeze'] = f'失败: {e}'
    return v


def make_env(env_id: str, seed: int = 0):
    """创建原生环境（DIRECT 无 GUI），并核对 API 约定。"""
    import warnings
    warnings.filterwarnings('ignore')
    import gymnasium as gym
    import dsrl.offline_bullet_safety_gym  # noqa: F401 注册 Offline*.v0
    env = gym.make(env_id)
    env.reset(seed=seed)
    return env


def step_env(env, action):
    """统一的 step 适配。

    Bullet-Safety-Gym / DSRL 这一族使用 Gymnasium 标准五元组
    `(obs, reward, terminated, truncated, info)`，原生成本在 `info['cost']`。
    返回 (obs, reward, cost, done, raw)。
    """
    out = env.step(action)
    if len(out) == 5:
        obs, rew, third, fourth, info = out
        if isinstance(third, dict):          # 非标准：第三位是 info
            cost = float(third.get('cost', 0.0))
            done = bool(fourth)
            return obs, float(rew), cost, done, out
        cost = float(info.get('cost', 0.0)) if isinstance(info, dict) else 0.0
        done = bool(third) or bool(fourth)
        return obs, float(rew), cost, done, out
    raise RuntimeError('未预期的 step 返回长度: %d' % len(out))


def load_trajectories(d: dict):
    """按 terminals/timeouts 切出完整轨迹；返回 [(start, end_exclusive)] 与统计。"""
    import numpy as np
    done = np.where((d['terminals'] == 1) | (d['timeouts'] == 1))[0]
    bounds, start = [], 0
    for i in done:
        bounds.append((start, int(i) + 1))
        start = int(i) + 1
    if start < len(d['terminals']):
        bounds.append((start, len(d['terminals'])))   # 尾段（理论上不应存在）
    return bounds


def traj_hash(seed: int, tid: int) -> int:
    return int(hashlib.sha256(f'{seed}-{tid}'.encode()).hexdigest(), 16) % 1000
