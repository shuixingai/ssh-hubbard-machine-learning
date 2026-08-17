#!/usr/bin/env python3
"""
自旋扇区探针 — 判定 U∈[3.5,6] 简并基态流形的自旋结构（S=1 三重态假设？）
========================================================================
背景（memory: dqap-polarization-hypotheses 2026-08-10）：
    fidelity probe 发现 U_c≈3.5 基态简并（2~3 重），U=4 DQAP Q=0.0000
    → 假说：相变 = 单态(S=0)→三重态(S=1) 跨自旋扇区跃迁，DQAP 被对称性锁在单态侧。
    但 Lieb 定理：半满、U>0、二部格子 → 基态唯一单态。APBC 环（一条负键 = π 通量）
    是否破坏定理前提，数值判定。

本探针在**全 2^{4L} 空间**（与 fidelity probe 同一 ED，L=4 → 16 qubit）对
每个简并基态算：
    N     粒子数        （检查是否真的在半满扇区；H 未固定粒子数）
    S²    总自旋平方    （0=单态, 2=三重态, 6=五重态）
    S_z   自旋投影
    nA,nB 每 cell 密度  （CDW 诊断，Wang 的 U≠0 判据之一）

关键约定：JW 排序每 cell = (A↑,A↓,B↑,B↓)，同格内 ↑↓ 相邻 → 自旋翻转算符的
JW 弦坍缩为 Z_{2p} 且作用在必为 0 的位上，即 **同格自旋翻转无 JW 相因子**：
    c†_{2p} c_{2p+1} = Z_{2p} σ⁺_{2p} σ⁻_{2p+1}，而 b_{2p}=0 → 相位 +1。
因此 S⁺ = Σ_sites c†_↑ c_↓ 可直接按占据数基构造（+1 系数翻转 (0,1)→(1,0)）。

判据：
    简并基态全 ⟨S²⟩=0 → 三重态假设死；简并 = 不同动量单态交叉（有限尺寸）
    简并基态有 ⟨S²⟩=2 → 三重态成立（或跨粒子扇区，看 N）
    简并基态 N≠8      → 简并是粒子数扇区切换，与自旋无关

用法：
    python spin_sector_probe.py
"""
import os
import sys

try:
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
except Exception:
    pass

import numpy as np
from scipy.sparse import csc_matrix, diags

from dqap_ssh_hubbard import build_ssh_hubbard_hamiltonian, diagonalize_hamiltonian

DEGEN_TOL = 1e-4      # 与 fidelity probe 一致
K_ED = 8              # eigsh 最低本征值个数
N_QUBITS = 16         # L=4 × 4 qubits/cell


# ============================================================================
# 自旋算符（占据数基，LSB = qubit 0）
# ============================================================================

def build_spin_ops(n_q):
    """返回 (Splus_csc, Sz_diag, N_diag, nA_list, nB_list)。
    基态向量 v（行向量）下：⟨S²⟩ = ⟨v|S_z²|v⟩ + ⟨v|S_z|v⟩ + ‖S⁺|v⟩‖²
    """
    dim = 1 << n_q
    n_cells = n_q // 4
    sites = n_q // 2                      # 空间位点数（每 cell 2 个）

    # ── 对角量 ──
    sz = np.zeros(dim)
    n_tot = np.zeros(dim)
    nA = [np.zeros(dim) for _ in range(n_cells)]
    nB = [np.zeros(dim) for _ in range(n_cells)]
    for i in range(dim):
        v = 0.0
        for j in range(n_q):
            b = (i >> j) & 1
            v += 1.0 if b else 0.0
        n_tot[i] = v
        for c in range(n_cells):
            qA = 4 * c
            nA[c][i] = ((i >> qA) & 1) + ((i >> (qA + 1)) & 1)
            nB[c][i] = ((i >> (qA + 2)) & 1) + ((i >> (qA + 3)) & 1)
        for p in range(sites):
            up, dn = 2 * p, 2 * p + 1
            sz[i] += 0.5 * ((i >> up) & 1) - 0.5 * ((i >> dn) & 1)

    # ── S⁺：同格 (↓,空↑) → (↑,空↓)，系数 +1（无 JW 相） ──
    rows, cols, data = [], [], []
    for p in range(sites):
        up, dn = 2 * p, 2 * p + 1
        bit_up, bit_dn = 1 << up, 1 << dn
        # 只枚举 b_dn=1 且 b_up=0 的位形（约 dim/4 个）
        for i in range(dim):
            if (i & bit_dn) and not (i & bit_up):
                j = (i & ~bit_dn) | bit_up
                rows.append(i)
                cols.append(j)
                data.append(1.0)

    Splus = csc_matrix((data, (rows, cols)), shape=(dim, dim))
    return Splus, sz, n_tot, nA, nB


def spin_label(s2):
    """⟨S²⟩ → 'S=0/1/2/...'（含误差容限）"""
    for S in range(5):
        if abs(s2 - S * (S + 1)) < 1e-3:
            return f"S={S}"
    return f"S²={s2:.3f}"


def fingerprint(vec, Splus, sz, n_tot, nA, nB):
    """单个态向量 → (N, S², S_z, nA[], nB[])"""
    N = float(np.vdot(vec, n_tot * vec).real)
    Sz = float(np.vdot(vec, sz * vec).real)
    Spv = Splus @ vec
    S2 = Sz * Sz + Sz + float(np.vdot(Spv, Spv).real)
    densA = [float(np.vdot(vec, nA[c] * vec).real) for c in range(len(nA))]
    densB = [float(np.vdot(vec, nB[c] * vec).real) for c in range(len(nB))]
    return N, S2, Sz, densA, densB


# ============================================================================
# 主流程
# ============================================================================

def scan_U(v, w, boundary, U_grid):
    Splus, sz, n_tot, nA, nB = build_spin_ops(N_QUBITS)
    print(f"L={N_QUBITS // 4}  v={v}  w={w}  boundary={boundary}")
    print("  基矢 dim = 2^%d = %d\n" % (N_QUBITS, 1 << N_QUBITS))

    for U in U_grid:
        h = build_ssh_hubbard_hamiltonian(N_QUBITS // 4, v, w, U, boundary)
        eigvals, eigvecs = diagonalize_hamiltonian(h, k=K_ED)
        eigvals = np.asarray(eigvals, dtype=float).real
        order = np.argsort(eigvals)
        eigvals = eigvals[order]
        eigvecs = np.asarray(eigvecs, dtype=complex)[order]

        E0 = eigvals[0]
        n_degen = int(np.sum((eigvals - E0) < DEGEN_TOL))
        gap = eigvals[n_degen] - E0 if n_degen < len(eigvals) else np.nan

        print(f"── U={U:5.2f}   E0={E0:12.6f}   Δ(到第{n_degen+1}个)= {gap:10.3e}"
              f"   degen={n_degen}")
        for a in range(min(n_degen + 3, len(eigvals))):
            N, S2, Sz, dA, dB = fingerprint(eigvecs[a], Splus, sz, n_tot, nA, nB)
            mark = "GS" if a < n_degen else "↑"
            fmt_dens = "  nA=[" + ",".join(f"{x:.2f}" for x in dA) + \
                       "]  nB=[" + ",".join(f"{x:.2f}" for x in dB) + "]"
            print(f"   {mark} #{a}: N={N:.0f}  {spin_label(S2)} (⟨S²⟩={S2:5.3f})"
                  f"  ⟨S_z⟩={Sz:+5.3f}{fmt_dens}")
        print()


if __name__ == "__main__":
    # 标准点与 fidelity probe 一致：L=4, v=1, w=2, APBC
    scan_U(1.0, 2.0, 'APBC',
           [0.0, 1.0, 2.0, 3.0, 3.25, 3.5, 3.75, 4.0, 4.5, 5.0, 6.0])
