#!/usr/bin/env python3
"""
pilot_gate5.py — 闸门 5：shot/NISQ QKM（AerSimulator 采样 L-e fidelity kernel）
============================================================================
闸门 4（pilot_gate4.py）已证理想（Statevector）DQAP kernel 判别两相不翻决策。
本脚本把 kernel 读出换成 shot 采样（真实量子机行为）：

    K_shot(i,j) = P(all-zero | U_i^† U_j |0⟩)   ← L-e 线路 + 测量 + N shots

用 AerSimulator 模拟（qiskit-aer 0.17.2）。判据：
    ① 逐 pair |K_shot − K_ideal| < 3σ + slack 的**超限比例 ≤ 5%**
       （σ = sqrt(K(1−K)/shots)；1300 对下 ~3σ 本就有统计漂移，按比例判不按 0 对）
       另报 max 偏差做全局 sanity（broken 读出会整体超限）
    ② shot kernel 喂 SVM 后 acc ≥ 0.8（shot 噪声下判别力不塌）
    ③ 与理想 kernel 的 SVM 决策一致率 ≥ 0.98（不翻决策）
结论 → 论文"真实量子机可测此 fidelity kernel"声明的前提。

约束（pilot_le.py 已钉死）：shot 只对 DQAP 态做（其线路逆原生可逆）；
ED 态含 initialize/multiplexer，AerSimulator 不认。态来源 = pilot_gate4.npz
（含 t1_vals/t2_vals/idx/y/params，此处按同序重建线路 → K_ideal=gate4.K_DQAP）。

用法：
    python pilot_gate5.py --quick      # 若 gate4 用 --quick 跑过 → 同 52 点
    python pilot_gate5.py              # 全 156 点 → 12090 对 × 5000 shots（分钟级）
    python pilot_gate5.py --shots 20000 --max-pairs 500   # 诊断：只前 500 对
"""

import argparse
import os
import sys
import time

try:
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
except Exception:
    pass

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
for p in (_HERE, os.path.dirname(_HERE)):
    if p not in sys.path:
        sys.path.insert(0, p)

from dqap_ssh_reproduce import build_dqap_circuit
from pilot_le import (L, BOUNDARY, N_TARGET,
                      build_loschmidt_echo, _transpile_for_aer)
from pilot_gate4 import kernel_matrix, svm_preds
from pilot_gate2 import cv_svm_precomputed, cv_kernel_nn

GATE4_NPZ = os.path.join(_HERE, 'pilot_gate4.npz')
OUT = 'pilot_gate5.npz'
SIGMA = 3.0
SLACK = 0.005
ACC_MIN = 0.80
AGREE_MIN = 0.98
FRAC_VIOL_MAX = 0.05


def shot_kernel(qcs, shots, max_pairs=None, verbose=True):
    """逐 pair 采样 L-e all-zero 概率 → K_shot（对称填满）。缺 aer 返回 None。"""
    try:
        from qiskit_aer import AerSimulator
    except ImportError:
        print('  [skip] qiskit-aer 未安装：pip install qiskit-aer')
        return None, None
    sim = AerSimulator()
    n = qcs[0].num_qubits
    m = len(qcs)
    K = np.zeros((m, m))
    pairs = [(i, j) for i in range(m) for j in range(i + 1, m)]
    if max_pairs is not None:
        pairs = pairs[:int(max_pairs)]
    t0 = time.time()
    for k, (i, j) in enumerate(pairs):
        qc_le = build_loschmidt_echo(qcs[j], qcs[i], with_measurement=True)
        qc_t = _transpile_for_aer(qc_le)
        counts = sim.run(qc_t, shots=shots).result().get_counts()
        p = counts.get('0' * n, 0) / shots
        K[i, j] = K[j, i] = p
        if verbose and (k % 200 == 0 or k == len(pairs) - 1):
            print(f'  pair {k+1}/{len(pairs)} (i={i},j={j}) K_shot={p:.4f} '
                  f'el={time.time()-t0:.0f}s', flush=True)
    np.fill_diagonal(K, 1.0)
    return K, pairs


def main(shots=5000, max_pairs=None, seed=42, verbose=True):
    print('=' * 72)
    print(f'闸门 5：shot/NISQ QKM  |  {BOUNDARY} N={N_TARGET}  |  '
          f'AerSimulator, shots={shots}')
    print('=' * 72)

    print('[1/4] 载入 gate4（态参数 + 理想 kernel）…')
    g4 = np.load(GATE4_NPZ)
    idx, y = g4['idx'], g4['y']
    params = g4['params']
    t1, t2 = g4['t1_vals'], g4['t2_vals']
    K_ideal = g4['K_DQAP']
    n2 = len(t2)
    print(f'  {len(idx)} 点（stride={g4["stride"]}，从 gate4 继承）')

    print('[2/4] 重建 DQAP 线路…')
    qcs = []
    for k, fl in enumerate(idx):
        i, j = divmod(int(fl), n2)
        v, w = float(t1[i]), float(t2[j])
        qcs.append(build_dqap_circuit(
            L, np.asarray(params[k], dtype=float), v, w, BOUNDARY))

    n_pairs_all = len(idx) * (len(idx) - 1) // 2
    n_pairs = n_pairs_all if max_pairs is None else int(max_pairs)
    print(f'[3/4] shot kernel（{n_pairs}/{n_pairs_all} 对 × {shots} shots）…')
    K_shot, pairs = shot_kernel(qcs, shots, max_pairs=max_pairs,
                                verbose=verbose)
    if K_shot is None:
        return 1

    print('[4/4] 判据…')
    # ① 逐对 3σ + slack：统计漂移内应能复现理想 kernel
    devs, sigs = [], []
    for (i, j) in pairs:
        kid = float(K_ideal[i, j])
        sig = SIGMA * np.sqrt(kid * (1 - kid) / shots)
        devs.append(abs(K_shot[i, j] - kid))
        sigs.append(sig)
    devs, sigs = np.array(devs), np.array(sigs)
    viol = devs > sigs + SLACK
    n_viol = int(viol.sum())
    frac_viol = n_viol / len(pairs)
    max_dev = float(devs.max())
    print(f'  ① |K_shot−K_ideal| max={max_dev:.4f}，超 3σ+slack {n_viol} 对 '
          f'({frac_viol:.1%})')

    # ② SVM on shot kernel（只判已采到的行；max_pairs 截断时行数可能更少）
    y_used = y[:K_shot.shape[0]]
    acc_shot, std_shot, _ = cv_svm_precomputed(K_shot, y_used)
    acc_nn, std_nn = cv_kernel_nn(K_shot, y_used)
    print(f'  ② SVM(shot kernel) acc = {acc_shot:.4f}±{std_shot:.4f}  '
          f'| 最近邻 {acc_nn:.4f}±{std_nn:.4f}')

    # ③ 决策一致性 vs 理想（同一选点子集上）
    K_ideal_used = K_ideal[:K_shot.shape[0], :K_shot.shape[0]]
    pred_ideal = svm_preds(K_ideal_used, y_used, seed=seed)
    pred_shot = svm_preds(K_shot, y_used, seed=seed)
    agree = float(np.mean(pred_ideal == pred_shot))
    n_flip = int((pred_ideal != pred_shot).sum())
    print(f'  ③ 决策一致率 = {agree:.4f}（翻转 {n_flip}/{len(y_used)}）')

    ok_1 = frac_viol <= FRAC_VIOL_MAX
    ok_2 = acc_shot >= ACC_MIN
    ok_3 = agree >= AGREE_MIN
    passed = ok_1 and ok_2 and ok_3
    print('\n' + '═' * 72)
    print('闸门 5 判定')
    print('═' * 72)
    print(f'  ① 3σ+slack 超限比例 ≤ {FRAC_VIOL_MAX:.0%}: '
          f'{frac_viol:.1%}  {"PASS" if ok_1 else "FAIL"}'
          f'（max dev {max_dev:.4f}）')
    print(f'  ② SVM acc ≥ {ACC_MIN}: {acc_shot:.4f}  '
          f'{"PASS" if ok_2 else "FAIL"}')
    print(f'  ③ 决策一致率 ≥ {AGREE_MIN}: {agree:.4f}  '
          f'{"PASS" if ok_3 else "FAIL"}')
    print('\n  总判定：' + ('PASS ✅ shot 读出可复现 QKM 判别 → NISQ 声明有据'
                          if passed else 'FAIL ❌（见上方哪一项未过）'))

    np.savez(OUT, gate='pilot_gate5', passed=bool(passed),
             shots=shots, stride=int(g4['stride']), n_pairs=len(pairs),
             K_shot=K_shot, K_ideal=K_ideal_used, devs=devs, sigs=sigs,
             max_dev=max_dev, n_viol=n_viol, frac_viol=frac_viol,
             acc_shot=acc_shot, acc_nn=acc_nn, agree=agree, n_flip=n_flip)
    print(f'\n结果已存 {os.path.abspath(OUT)}')
    return 0 if passed else 1


if __name__ == '__main__':
    p = argparse.ArgumentParser(description='闸门 5：shot/NISQ QKM')
    p.add_argument('--shots', type=int, default=5000)
    p.add_argument('--max-pairs', type=int, default=None,
                   help='只采前 N 对（诊断/冒烟）')
    p.add_argument('--seed', type=int, default=42)
    args = p.parse_args()
    sys.exit(main(shots=args.shots, max_pairs=args.max_pairs, seed=args.seed))
