#!/usr/bin/env python3
"""
pilot_gate4.py — 闸门 4：DQAP 态污染预算（U=0 pilot, spinless，决策翻转检验）
============================================================================
闸门 2（pilot_gate2.py）已证 ED 精确态 fidelity kernel 判别两相（SVM 100%，
无 0/1 分块）。本脚本把态制备换成 DQAP-VQE（真实态制备引擎，含 ansatz 误差），
测污染是否翻转 SVM 决策：

    K_DQAP(i,j) = |⟨ψ_DQAP(i)|ψ_DQAP(j)⟩|²,   ψ_DQAP = DQAP-VQE(M 层) 近似基态
    对照 K_ED（pilot_gate2.npz 已存 ED 态，同选点子集）。

判据：
    ① SVM(precomputed) acc ≥ 0.8         —— 判别力不塌
    ② 决策不翻转：K_DQAP-SVM 与 K_ED-SVM 同 5-fold 测试集预测一致率 ≥ 0.98
    ③ 态品质 Q = |⟨ψ_DQAP|ψ_ED⟩|² 均值/最小（线路层 target 佐证，报告用）

U=0 无粒子数漂移（spinless L=4 APBC N=4，169 全唯一 GS），参照干净；
L-e 读出已由闸门 1（pilot_le.py）对账到 1e-14，此处直接用内积等价。
本脚本只用 Statevector 精确 fidelity —— 不引入 shot 噪声（aer 政策：shot
层留给 Part C 与闸门 5，独立误差预算）。

用法：
    python pilot_gate4.py                # 全 13×13 网格 off-diagonal 156 点（M=3）
    python pilot_gate4.py --quick        # stride=3 → ~52 点冒烟
    python pilot_gate4.py --M 4 --maxiter 3000 --seed 0

输出：pilot_gate4.npz（含 t1_vals/t2_vals/idx/y/params → 闸门 5 复用重建线路）
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
from qiskit.quantum_info import Statevector

_HERE = os.path.dirname(os.path.abspath(__file__))
for p in (_HERE, os.path.dirname(_HERE)):
    if p not in sys.path:
        sys.path.insert(0, p)

from dqap_ssh_reproduce import build_ssh_hamiltonian, build_dqap_circuit, run_vqe
from pilot_le import L, BOUNDARY, N_TARGET
from pilot_gate2 import cv_svm_precomputed, cv_kernel_nn

GATE2_NPZ = os.path.join(_HERE, 'pilot_gate2.npz')
OUT = 'pilot_gate4.npz'
ACC_MIN = 0.80
AGREE_MIN = 0.98


def select_offdiag(t1, t2, lab, stride=1):
    """off-diagonal（非临界、唯一 GS）扁平索引的 stride 子集 + 标签。"""
    n1, n2 = len(t1), len(t2)
    idx_all = [i * n2 + j for i in range(n1) for j in range(n2)
               if lab[i, j] != 2]
    idx = idx_all[::stride]
    y = np.array([int(lab[i // n2, i % n2]) for i in idx])
    return np.array(idx), y


def dqap_states_at(idx, t1, t2, states_ed, M, maxiter, verbose=True):
    """逐点 DQAP-VQE → statevector + 态品质 Q；返回 (svs, qs, params)。"""
    n2 = len(t2)
    nq = 2 * L
    svs = np.zeros((len(idx), 1 << nq), dtype=complex)
    qs = np.zeros(len(idx))
    params = np.zeros((len(idx), 2 * M))
    t0 = time.time()
    for k, fl in enumerate(idx):
        i, j = divmod(int(fl), n2)
        v, w = float(t1[i]), float(t2[j])
        h = build_ssh_hamiltonian(L, v, w, BOUNDARY)
        res = run_vqe(L, v, w, BOUNDARY, M, h, maxiter=maxiter, verbose=False)
        x = np.asarray(res.x, dtype=float)
        qc = build_dqap_circuit(L, x, v, w, BOUNDARY)
        sv = np.asarray(Statevector(qc).data, dtype=complex)
        svs[k] = sv
        params[k] = x
        qs[k] = float(abs(np.vdot(sv, states_ed[i, j])) ** 2)
        if verbose and (k % 20 == 0 or k == len(idx) - 1):
            print(f'  [{k+1:3d}/{len(idx)}] (t1,t2)=({v:4.2f},{w:4.2f}) '
                  f'Q={qs[k]:.4f} E={res.fun:10.6f}  el={time.time()-t0:5.0f}s',
                  flush=True)
    return svs, qs, params


def kernel_matrix(states):
    """K(i,j) = |⟨ψ_i|ψ_j⟩|² = |S S^†|² 逐元素。"""
    G = states @ states.conj().T
    return np.abs(G) ** 2


def svm_preds(K, y, folds=5, seed=42):
    """固定 5-fold StratifiedKFold 的逐点预测（用于跨 kernel 决策一致性）。"""
    from sklearn.svm import SVC
    from sklearn.model_selection import StratifiedKFold
    n = len(y)
    pred = np.empty(n, dtype=int)
    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    for tr, te in skf.split(np.arange(n), y):
        clf = SVC(kernel='precomputed', C=1.0)
        clf.fit(K[np.ix_(tr, tr)], y[tr])
        pred[te] = clf.predict(K[np.ix_(te, tr)])
    return pred


def main(stride=1, M=3, maxiter=2000, seed=0, verbose=True):
    print('=' * 72)
    print(f'闸门 4：DQAP 态污染预算（U=0 pilot, spinless {BOUNDARY} N={N_TARGET}）'
          f'  |  态制备 = DQAP-VQE M={M}')
    print('=' * 72)

    print(f'[1/4] 载入 gate2 网格 + 选点（stride={stride}）…')
    g2 = np.load(GATE2_NPZ)
    t1, t2, lab = g2['t1_vals'], g2['t2_vals'], g2['label_u0']
    states_ed = g2['states']
    idx, y = select_offdiag(t1, t2, lab, stride)
    n_triv, n_topo = int((y == 0).sum()), int((y == 1).sum())
    assert n_triv > 0 and n_topo > 0, '两类都需保留（stride 太大则减小）'
    print(f'  选点 {len(idx)}：平凡={n_triv} 拓扑={n_topo}')

    print(f'[2/4] DQAP-VQE 态制备（{len(idx)} 点 × M={M}, maxiter={maxiter}）…')
    np.random.seed(seed)   # run_vqe 用全局 RNG 取 x0，锁定复现
    svs, qs, params = dqap_states_at(idx, t1, t2, states_ed, M, maxiter,
                                     verbose=verbose)
    print(f'  态品质 Q = |⟨ψ_DQAP|ψ_ED⟩|² : mean={qs.mean():.4f}  '
          f'min={qs.min():.4f}')

    print('[3/4] K_DQAP vs K_ED（同选点子集）…')
    states_ed_sel = states_ed.reshape(-1, states_ed.shape[-1])[idx]
    K_dqap = kernel_matrix(svs)
    K_ed = kernel_matrix(states_ed_sel)
    err_poll = float(np.abs(K_dqap - K_ed).max())
    print(f'  max|K_DQAP − K_ED| = {err_poll:.4f}  （K_ED 内平滑，作污染参照）')

    print('[4/4] SVM 判别 + 决策翻转检验…')
    acc_ed, std_ed, _ = cv_svm_precomputed(K_ed, y)
    acc_dqap, std_dqap, _ = cv_svm_precomputed(K_dqap, y)
    acc_nn, std_nn = cv_kernel_nn(K_dqap, y)
    pred_ed = svm_preds(K_ed, y)
    pred_dqap = svm_preds(K_dqap, y)
    agree = float(np.mean(pred_ed == pred_dqap))
    n_flip = int((pred_ed != pred_dqap).sum())
    print(f'  SVM(precomputed) CV: K_ED={acc_ed:.4f}±{std_ed:.4f}  '
          f'K_DQAP={acc_dqap:.4f}±{std_dqap:.4f}')
    print(f'  kernel 最近邻 (K_DQAP): {acc_nn:.4f}±{std_nn:.4f}')
    print(f'  决策一致率 = {agree:.4f}（翻转 {n_flip}/{len(y)} 点）')

    ok_acc = acc_dqap >= ACC_MIN
    ok_agree = agree >= AGREE_MIN
    passed = ok_acc and ok_agree
    print('\n' + '═' * 72)
    print('闸门 4 判定')
    print('═' * 72)
    print(f'  ① SVM acc ≥ {ACC_MIN}: {acc_dqap:.4f}  '
          f'{"PASS" if ok_acc else "FAIL"}')
    print(f'  ② 决策一致率 ≥ {AGREE_MIN}: {agree:.4f}  '
          f'{"PASS" if ok_agree else "FAIL"}')
    print(f'  ③ 态品质 Q: mean={qs.mean():.4f} min={qs.min():.4f}（报告用）')
    print('\n  总判定：' + ('PASS ✅ DQAP 污染不塌判别、不翻决策 → QKM 端到端可行'
                          if passed else 'FAIL ❌（见上方哪一项未过）'))

    np.savez(OUT, gate='pilot_gate4', passed=bool(passed),
             stride=stride, M=M, L=L, boundary=BOUNDARY, n_target=N_TARGET,
             t1_vals=t1, t2_vals=t2, idx=idx, y=y, params=params, qs=qs,
             K_DQAP=K_dqap, K_ED=K_ed, err_poll=err_poll,
             acc_ed=acc_ed, acc_dqap=acc_dqap, acc_nn=acc_nn,
             agree=agree, n_flip=n_flip)
    print(f'\n结果已存 {os.path.abspath(OUT)}')
    return 0 if passed else 1


if __name__ == '__main__':
    p = argparse.ArgumentParser(description='闸门 4：DQAP 态污染预算')
    p.add_argument('--quick', action='store_true', help='stride=3 冒烟')
    p.add_argument('--stride', type=int, default=1)
    p.add_argument('--M', type=int, default=3)
    p.add_argument('--maxiter', type=int, default=2000)
    p.add_argument('--seed', type=int, default=0)
    args = p.parse_args()
    sys.exit(main(stride=3 if args.quick else args.stride,
                  M=args.M, maxiter=args.maxiter, seed=args.seed))
