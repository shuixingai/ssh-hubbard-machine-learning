#!/usr/bin/env python3
"""
pilot_part_c.py — 闸门 1 的 Part C 独立运行版（shot 读出链路验证）
=================================================================
pilot_le.py 的 Part C 此前因缺 qiskit-aer 被 skip；aer 0.17.2 已装后，
此脚本只跑 Part C，不必重跑 Part A/B。逻辑与 pilot_le.part_c_shot 完全
一致（直接复用其函数）：

    C1 — topo_M4 组 6 个随机 DQAP 态，取 K 最接近 0.5 的 12 对，在
         SHOT_N=20000 shots 下 |K_shot − K_direct| < 3σ + slack？
    C2 — K≈0.5 那一对在 [100, 1e3, 1e4, 1e5] shots 下的误差 vs 3σ
         （shot 预算预览，喂闸门 5 定 shots 用）

通过 → 证明 L-e 线路 + shot 测量链路在采样误差内恢复 fidelity kernel，
即"真实量子机的 shot 读出"这一层成立。

用法：
    python pilot_part_c.py
"""

import os
import sys

try:
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
except Exception:
    pass

_HERE = os.path.dirname(os.path.abspath(__file__))
for p in (_HERE, os.path.dirname(_HERE)):
    if p not in sys.path:
        sys.path.insert(0, p)

from pilot_le import build_part_a_states, part_c_shot, SLACK


def main():
    print('=' * 68)
    print('Part C（独立）：DQAP L-e shot 读出链路验证')
    print('=' * 68)
    groups = build_part_a_states()
    c1_ok, c1_dev, c1_sig, c2_rows = part_c_shot(groups)
    if c1_ok is None:
        print('\n  [skip] qiskit-aer 未安装：pip install qiskit-aer')
        return 1
    print()
    if c1_ok:
        print('  Part C 判定：PASS ✅  shot 读出在 3σ+slack 内复现 fidelity')
    else:
        print(f'  Part C 判定：FAIL ❌  max|K_shot−K_direct|={c1_dev:.4f} '
              f'> 3σ+slack={c1_sig + SLACK:.4f}')
    return 0 if c1_ok else 1


if __name__ == '__main__':
    sys.exit(main())
