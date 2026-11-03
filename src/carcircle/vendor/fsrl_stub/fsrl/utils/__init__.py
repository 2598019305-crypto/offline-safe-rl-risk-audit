"""fsrl.utils 的最小等价替身（仅日志接口）。

登记说明：官方 FSRL 仓库（github.com/liuzuxin/FSRL）在本机无法完整获取（30.5 MB，
多次下载均在 7.2 MB 处断流且不支持续传），而 OSRL 只从 FSRL 使用
`DummyLogger` / `WandbLogger` 两个**日志**类（不参与任何算法计算）。
因此这里提供语义等价的空实现，**OSRL 算法模块保持逐字节未修改**（SHA 见 manifest）。
"""
from __future__ import annotations

import contextlib
import io
import sys


class _LoggerBase:
    """与 fsrl.utils.Logger 的接口对齐：全部为记录动作，不影响计算。"""

    def __init__(self, *a, **k):
        self._store = {}
        self._ckpt_fn = None
        self._steps = 0

    def setup_checkpoint_fn(self, fn):
        self._ckpt_fn = fn

    def store(self, tab=None, **kwargs):
        bucket = self._store.setdefault(tab or 'train', {})
        bucket.update(kwargs)

    def write(self, step=None, display=False, **kwargs):
        self._steps = step if step is not None else self._steps + 1
        self._store.clear()

    def write_without_reset(self, step=None, **kwargs):
        self._steps = step if step is not None else self._steps + 1

    def log(self, *a, **k):
        pass

    def print(self, *a, **k):
        print(*a, **k)

    def save_checkpoint(self, *a, **k):
        return None

    def finish(self, *a, **k):
        pass

    def __getattr__(self, item):
        # 未列出的方法一律退化为无副作用的空操作
        return lambda *a, **k: None


class DummyLogger(_LoggerBase):
    pass


class WandbLogger(_LoggerBase):
    """不做任何联网上传；仅保留接口。"""

    def __init__(self, cfg=None, project=None, group=None, name=None, logdir=None):
        super().__init__()
        self.cfg, self.project, self.group, self.name, self.logdir = cfg, project, group, name, logdir

    def save_config(self, cfg, verbose=False):
        self.cfg = cfg
