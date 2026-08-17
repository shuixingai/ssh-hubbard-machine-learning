#!/usr/bin/env python3
"""
qkm_grid_features.py — 全网格线路 feature 构建（QKM vs 经典 gap4 同规格对决）
============================================================================
经典 gap4 baseline（baseline_ml.py）参照线 = 69.5%（memory: gap4-classical-baseline）。
本脚本构建线路侧 feature：在 topo_dataset_full.npz 的**同一** (t1,t2,U) 网格上，
逐点 DQAP-VQE（spinful, L=4, APBC, N=8 半满扇区）制备态 ψ_DQAP，存
fidelity kernel  K(i,j) = |⟨ψ_i|ψ_j⟩|²  供下游 ML（qkm_ml.py）喂 SVM 判别。

网格规格（与经典 gap4 逐点一致，轴直接读自标签 npz → 口径零漂移）：
    t1_vals/t2_vals = arcsin 加密 13 点 [0.25, 4.0]
    U_vals          = linspace(0, 4, 7)
    → 13×13×7 = 1183 点；剔除临界(2)/未定(3) → 与 baseline 同口径 1092 点

设计事实（无污染对照，勿破坏）：
    - 特征 = DQAP 电路态（量子线路层）   标签 = topo γ_up（PBC/TBC Wilson loop）
    - 两次计算完全独立，只共享 (t1,t2,U) 坐标；标签唯一来源 = topo_dataset_full.npz
    - 有污染是设计事实而非 bug：DQAP M 层近似误差 → Q<1 → K 偏离理想 kernel，
      这正是"量子 vs 经典"对决的内容（QKM 要跨 69.5% 栏，参考 gap4-classical-baseline）

用法：
    python qkm_grid_features.py                 # 全量 1092 点, M=3（重活，小时级）
    python qkm_grid_features.py --M 4 --maxiter 3000
    python qkm_grid_features.py --u-stride 2    # 只扫 U 偶数层（快测 U 依赖）
    python qkm_grid_features.py --max-pts 40    # 冒烟：固定随机 40 点子集
    python qkm_grid_features.py --save-states   # 额外存 ψ（+1+ GB，谨慎）

输出：qkm_grid_M{M}.npz
    t1_vals/t2_vals/U_vals  网格轴（= 标签 npz 原轴，逐点对账用）
    idx      扁平索引（C 序 13×13×7，同 baseline_ml mask 口径）
    t1/t2/U   逐点参数；lab 逐点标签（0=triv / 1=topo，复制自 topo）
    K_DQAP   (n,n) fidelity kernel（特征本身）
    params   (n, 3M)  VQE 最优参数（复现线路用）
    E        (n,) VQE 能量；E_ed (n,) ED N=8 能量；Q (n,) 态品质
    M, maxiter, boundary, n_target
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
_DATA = os.path.dirname(_HERE)
for p in (_HERE, _DATA):
    if p not in sys.path:
        sys.path.insert(0, p)

from dqap_ssh_hubbard import (
    build_ssh_hubbard_hamiltonian,
    build_dqap_circuit_spinful,
    run_vqe,
)
from dqap_ssh_hubbard_tbc_berry import sector_gs, N_TARGET, DEGEN_TOL

L = 4
BOUNDARY = 'APBC'          # 与 pilot / 标签计算口径一致
LABEL_NPZ = os.path.join(_DATA, 'topo_dataset_full.npz')
OUT_TMPL = os.path.join(_HERE, 'qkm_grid_M{M}.npz')


def load_grid():
    """读标签 npz 的网格轴 + 全标签（轴 = 唯一规格来源，杜绝口径漂移）。"""
    d = np.load(LABEL_NPZ)
    return (np.asarray(d['t1_vals'], dtype=float),
            np.asarray(d['t2_vals'], dtype=float),
            np.asarray(d['U_vals'], dtype=float),
            np.asarray(d['label'], dtype=np.int8))


def select_binary(t1, t2, U, label, u_stride=1, max_pts=None):
    """C 序展平后取 label∈{0,1} 的点（剔除 2/3，同 baseline_ml 口径）。

    u_stride>1 时只取 U 轴每隔 stride 层（快测用，标签仍逐点有效）。
    返回 (idx, t1p, t2p, Up, lab) —— idx 是全网格扁平索引。
    """
    n1, n2, nu = len(t1), len(t2), len(U)
    idx_all = [i * n2 * nu + j * nu + k
               for i in range(n1) for j in range(n2)
               for k in range(nu) if (k % u_stride) == 0
               and label[i, j, k] in (0, 1)]
    if max_pts:
        rng = np.random.default_rng(42)
        pick = rng.choice(len(idx_all), size=max_pts, replace=False)
        pick.sort()
        idx_all = [idx_all[i] for i in pick]
    idx = np.array(idx_all, dtype=np.int64)
    t1p = np.array([t1[i // (n2 * nu)] for i in idx])
    t2p = np.array([t2[(i // nu) % n2] for i in idx])
    Up = np.array([U[i % nu] for i in idx])
    lab = np.array([int(label[i // (n2 * nu), (i // nu) % n2, i % nu])
                    for i in idx], dtype=np.int8)
    return idx, t1p, t2p, Up, lab


def dqap_point(v, w, U, M, maxiter, seed):
    """单点：ED N=8 流形（Q 参照）→ DQAP-VQE → ψ + 能量 + 品质 Q。

    Q = Σ|⟨ψ_DQAP|φ_ED,N8⟩|²（简并流形上投影）—— 态品质，报告用。
    """
    h = build_ssh_hubbard_hamiltonian(L, v, w, U, BOUNDARY)
    Mh = h.to_matrix(sparse=True).tocsr()
    E0_ed, _, vecs_ed, _ = sector_gs(Mh, N=N_TARGET, tol=DEGEN_TOL)

    res = run_vqe(L, v, w, U, BOUNDARY, M, h,
                  maxiter=maxiter, verbose=False, seed=seed)
    x = np.asarray(res.x, dtype=float)
    qc = build_dqap_circuit_spinful(L, x, v, w, U, BOUNDARY)
    sv = np.asarray(Statevector(qc).data, dtype=complex)
    ov = vecs_ed @ sv.conj()
    Q = float((np.abs(ov) ** 2).sum())
    return x, sv, float(res.fun), Q, float(E0_ed)


def kernel_matrix(states):
    """K(i,j) = |⟨ψ_i|ψ_j⟩|² = |S S^†|² 逐元素（pilot_gate2/gate4 同式）。"""
    G = states @ states.conj().T
    return np.abs(G) ** 2


def main(M=3, maxiter=2000, seed=0, u_stride=1, max_pts=None,
         save_states=False, verbose=True):
    print('=' * 74)
    print(f'全网格线路 feature 构建 — spinful L={L} {BOUNDARY} N={N_TARGET} 半满'
          f'  |  M={M}, maxiter={maxiter}')
    print('=' * 74)

    t1, t2, U, label = load_grid()
    n1, n2, nu = len(t1), len(t2), len(U)
    print('[1/3] 网格（读自 topo_dataset_full.npz，= 经典 gap4 同轴，无污染设计）')
    print(f'      t1 {n1} 点 arcsin[0.25,4]   t2 {n2} 点 arcsin[0.25,4]'
          f'   U {nu} 点 linspace(0,4)')

    idx, t1p, t2p, Up, lab = select_binary(t1, t2, U, label,
                                           u_stride=u_stride, max_pts=max_pts)
    n_triv = int((lab == 0).sum())
    n_topo = int((lab == 1).sum())
    print(f'[2/3] 选点（剔除临界/未定，u_stride={u_stride}, max_pts={max_pts}）')
    print(f'      {len(idx)} 点：平凡={n_triv} 拓扑={n_topo}')

    print(f'[3/3] 逐点 DQAP-VQE（{len(idx)} 点 × M={M}, maxiter={maxiter}）…  '
          f'（每点含 ED N=8 流形作 Q 参照）', flush=True)
    nq = 4 * L
    n_params = 3 * M
    states = np.zeros((len(idx), 1 << nq), dtype=complex)
    params = np.zeros((len(idx), n_params))
    E = np.zeros(len(idx))
    Q = np.zeros(len(idx))
    E_ed = np.zeros(len(idx))
    t0 = time.time()
    for k, (v, w, uu) in enumerate(zip(t1p, t2p, Up)):
        x, sv, e, q, e0 = dqap_point(float(v), float(w), float(uu),
                                     M, maxiter, seed + k)
        states[k] = sv
        params[k] = x
        E[k] = e
        Q[k] = q
        E_ed[k] = e0
        if verbose and (k % 10 == 0 or k == len(idx) - 1):
            el = time.time() - t0
            eta = el / (k + 1) * (len(idx) - k - 1)
            print(f'  [{k+1:4d}/{len(idx)}] (t1,t2,U)=({v:5.2f},{w:5.2f},'
                  f'{uu:4.2f})  Q={q:.4f}  E={e:9.6f}  ΔE_vs_ED={e-e0:9.3e}'
                  f'  el={el:5.0f}s  eta≈{eta:5.0f}s', flush=True)

    print('\n态品质 Q = |⟨ψ_DQAP|ψ_ED,N8⟩|² : '
          f'mean={Q.mean():.4f}  min={Q.min():.4f}')
    print(f'  能量误差 ΔE = E_DQAP − E_ED   : mean={np.mean(E-E_ed):9.3e}  '
          f'max={np.max(np.abs(E-E_ed)):9.3e}')

    print('[收尾] 计算 K_DQAP…')
    K = kernel_matrix(states)

    out = dict(t1_vals=t1, t2_vals=t2, U_vals=U, idx=idx,
               t1=t1p, t2=t2p, U=Up, lab=lab,
               K_DQAP=K, params=params, E=E, Q=Q, E_ed=E_ed,
               M=M, maxiter=maxiter, boundary=BOUNDARY, n_target=N_TARGET,
               n_triv=n_triv, n_topo=n_topo)
    if save_states:
        out['states'] = states
    out_npz = OUT_TMPL.format(M=M)
    np.savez(out_npz, **out)
    print(f'\n结果已存 {os.path.abspath(out_npz)}')
    return out_npz


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    p.add_argument('--M', type=int, default=3)
    p.add_argument('--maxiter', type=int, default=2000)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--u-stride', type=int, default=1,
                   help='U 轴抽样步长（快测：2 → 只扫偶数层）')
    p.add_argument('--max-pts', type=int, default=None,
                   help='冒烟：固定随机子集点数（seed=42 确定）')
    p.add_argument('--save-states', action='store_true',
                   help='额外存 ψ（+1+ GB，默认不存）')
    args = p.parse_args()
    main(M=args.M, maxiter=args.maxiter, seed=args.seed,
         u_stride=args.u_stride, max_pts=args.max_pts,
         save_states=args.save_states)
