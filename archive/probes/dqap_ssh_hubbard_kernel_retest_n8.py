#!/usr/bin/env python3
"""
QKM kernel 决定性检验 — N=8 参照 + U=4 M=4 态重测 K 双矩阵（免重跑 VQE）
========================================================================
旧结论（fidelity_probe.py Part C）：max|K_ED − K_DQAP| = 0.9569 →
"态制备污染灾难性破坏 QKM kernel"。但旧测量同时含两个伪影：
  (a) K_ED 参照 = 全空间 GS 流形 → U=4 全空间 GS 漂到 N=6（粒子数漂移
      伪影），与 U≤2 的 N=8 GS 正交 → 旧 K_ED 的"分块对角"本身就是
      粒子数正交伪影，不是物理分块
  (b) U=4 用的 DQAP 态是 M=3（N=8 扇区内 Q=0.871），M=4 才是 Q=0.997

修正（memory: dqap-polarization-hypotheses N=8 重跑落地）：
  参照投影 N=8 扇区 + U=4 取 U4_M4 态。二者 u4_probe.npz 都在手 → 纯后处理。

预测：
  - K_ED_N8 沿 U ≈ 全 1（N=8 扇区内无相变、GS 平滑）→ 与阶段1
    "U 是纯干扰维度"（label 对 U 鲁棒）自洽：分类信息在 (t1,t2) 不在 U
  - 污染 max|K_ED_N8 − K_DQAP| 从 0.9569 跌到 ~0.03（态品质量级）

通过判据：污染 < 0.05 且 K_ED_N8 无 0/1 陡峭分块 → QKM 路线坐实，
0.9569 归因为参照伪影 + M=3 态质量，非 QKM 本身。

用法：
  python dqap_ssh_hubbard_kernel_retest_n8.py [u4_probe.npz 路径]
"""

import os
import sys

try:
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
except Exception:
    pass

import numpy as np
from qiskit.quantum_info import Statevector

from dqap_ssh_hubbard import (
    build_ssh_hubbard_hamiltonian,
    build_dqap_circuit_spinful,
    diagonalize_hamiltonian,
)
from dqap_ssh_hubbard_tbc_berry import sector_gs, N_TARGET

L = 4
V, W, BOUNDARY = 1.0, 2.0, 'APBC'      # 拓扑区（与 u4_probe.npz 一致）
K_ED = 8
DEGEN_TOL = 1e-4


def gs_manifold_full(h, k=K_ED, tol=DEGEN_TOL):
    """全空间 GS 流形（旧 fidelity_probe 的参照口径，仅用于演示伪影）。"""
    eigvals, eigvecs = diagonalize_hamiltonian(h, k=k)
    eigvals = np.asarray(eigvals, dtype=float).real
    order = np.argsort(eigvals)
    eigvals = eigvals[order]
    eigvecs = np.asarray(eigvecs, dtype=complex)[order]
    E0 = eigvals[0]
    n = int(np.sum((eigvals - E0) < tol))
    return eigvecs[:n]


def subspace_fidelity(vecs_a, vecs_b):
    """F_sub = Σ|⟨a|b⟩|² / min(Da,Db) ∈ [0,1]（简并鲁棒，跨扇区正交=0）。"""
    if vecs_a.shape[0] == 0 or vecs_b.shape[0] == 0:
        return 0.0
    O = vecs_a @ vecs_b.conj().T
    D = min(vecs_a.shape[0], vecs_b.shape[0])
    return float((np.abs(O) ** 2).sum() / D)


def _parse_key(key):
    """'U0.5' → (0.5, None)；'U4_M3' → (4.0, 3)。"""
    body = key[1:]
    if '_M' in body:
        u_str, m_str = body.split('_M')
        return float(u_str), int(m_str)
    return float(body), None


def main(npz_path=None, save_path='kernel_retest_n8.npz'):
    if npz_path is None:
        npz_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                'u4_probe.npz')
    data = np.load(npz_path, allow_pickle=True)['results'].item()

    # 每组（每个 U 的每个 M）→ DQAP 态 + N=8 流形 + 全空间流形
    U_uniq = sorted({_parse_key(k)[0] for k in data})
    man_n8 = {}        # U → N=8 流形
    man_full = {}      # U → 全空间流形（伪影演示）
    for U in U_uniq:
        h = build_ssh_hubbard_hamiltonian(L, V, W, U, BOUNDARY)
        M = h.to_matrix(sparse=True).tocsr()
        _, _, vecs_n8, nd_n8 = sector_gs(M, N=N_TARGET)
        man_n8[U] = vecs_n8
        man_full[U] = gs_manifold_full(h)
        print(f"  U={U:4.1f}   N=8 扇区 degen={nd_n8}   "
              f"全空间 GS degen={man_full[U].shape[0]}")

    # DQAP 态
    svs = {}
    for key, rec in data.items():
        params = np.asarray(rec['params'], dtype=float)
        qc = build_dqap_circuit_spinful(L, params, V, W, _parse_key(key)[0],
                                        BOUNDARY)
        svs[key] = np.asarray(Statevector(qc).data, dtype=complex)

    # ── 三组矩阵：全空间参照（旧口径）vs N=8 参照（修正）──
    def build_K(keys, mode):
        n = len(keys)
        K = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                ui = _parse_key(keys[i])[0]
                uj = _parse_key(keys[j])[0]
                if mode == 'full':
                    K[i, j] = subspace_fidelity(man_full[ui], man_full[uj])
                elif mode == 'n8':
                    K[i, j] = subspace_fidelity(man_n8[ui], man_n8[uj])
                else:  # dqap
                    K[i, j] = float(np.abs(
                        np.vdot(svs[keys[i]], svs[keys[j]])) ** 2)
        return K

    # 修正集：U≤2 用 M=3（Q≈1），U=4 用 M=4（Q=0.997）
    corr_keys = ['U0.0', 'U0.5', 'U1.0', 'U2.0', 'U4_M4']
    # 对照集：U=4 仍用 M=3（隔离效应 (b)：纯 M 的贡献）
    old_keys = ['U0.0', 'U0.5', 'U1.0', 'U2.0', 'U4_M3']

    print("\n" + "=" * 74)
    print("旧口径：全空间 GS 参照（演示分块伪影来源）")
    print("=" * 74)
    K_full = build_K(old_keys, 'full')
    _dump(K_full, old_keys, "K_ED_full（旧）")

    print("\n" + "=" * 74)
    print("修正口径：N=8 扇区参照")
    print("=" * 74)
    K_n8 = build_K(corr_keys, 'n8')
    _dump(K_n8, corr_keys, "K_ED_N8（理想，修正）")

    print("\n" + "=" * 74)
    print("DQAP kernel（修正集：U≤2 M=3，U=4 M=4）")
    print("=" * 74)
    K_dqap_corr = build_K(corr_keys, 'dqap')
    _dump(K_dqap_corr, corr_keys, "K_DQAP_M4（修正）")

    print("\n" + "=" * 74)
    print("DQAP kernel（对照：U=4 仍用 M=3，隔离 M 的贡献）")
    print("=" * 74)
    K_dqap_old = build_K(old_keys, 'dqap')
    _dump(K_dqap_old, old_keys, "K_DQAP_M3（对照）")

    # ── 判据 ──
    err_old = float(np.abs(K_full - K_dqap_old).max())      # 应≈0.9569（复现旧结论）
    err_corr = float(np.abs(K_n8 - K_dqap_corr).max())      # 修正后污染
    err_m3 = float(np.abs(K_n8 - K_dqap_old).max())         # 只修参照、仍 M=3
    diag_flat = float(np.max(np.abs(K_n8 - 1.0)))           # K_ED_N8 离全 1 多远

    print("\n" + "=" * 74)
    print("污染判定")
    print("=" * 74)
    print(f"  旧口径（全空间参照 + M=3）：max|K_ED_full − K_DQAP_M3| = {err_old:.4f}"
          f"   （应复现旧 0.9569）")
    print(f"  只修参照（N=8，仍 M=3）   ：max|K_ED_N8 − K_DQAP_M3| = {err_m3:.4f}"
          f"   （隔离效应 a）")
    print(f"  修正（N=8 参照 + M=4）     ：max|K_ED_N8 − K_DQAP_M4| = {err_corr:.4f}"
          f"   （应 < 0.05）")
    print(f"  K_ED_N8 离全 1 的最大偏差   ：max|K_ED_N8 − 1| = {diag_flat:.4f}"
          f"   （N=8 扇区内应无 0/1 陡峭分块）")
    print()
    if err_corr < 0.05 and diag_flat < 0.05:
        print("  ✓ 通过判据：污染归因参照伪影 + M=3 态质量，QKM 路线坐实。")
    else:
        print("  ✗ 未通过：污染仍有残留或 K_ED_N8 非全 1，需进一步归因。")

    np.savez(save_path,
             keys=corr_keys, K_ED_full=K_full, K_ED_N8=K_n8,
             K_DQAP_M4=K_dqap_corr, K_DQAP_M3=K_dqap_old,
             err_old=err_old, err_m3=err_m3, err_corr=err_corr,
             diag_flat=diag_flat)
    print(f"\n结果已存 {save_path}")


def _dump(K, keys, title):
    print(f"\n  {title}（列=行=U 轴）")
    print("      " + " ".join(f"{k:>7}" for k in keys))
    for i in range(len(keys)):
        print(f"  {keys[i]:>5}  " + " ".join(f"{K[i, j]:7.3f}" for j in range(len(keys))))


if __name__ == '__main__':
    npz = sys.argv[1] if len(sys.argv) > 1 else None
    main(npz_path=npz)
