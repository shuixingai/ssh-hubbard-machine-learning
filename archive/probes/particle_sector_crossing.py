#!/usr/bin/env python3
"""
粒子数扇区交叉探针 — 验证 spin_sector_probe 的发现：U≥3.5 全空间基态漂到 N=6
============================================================================
背景（spin_sector_probe 2026-08-10）：
    全 2^16 空间对角化（不固定粒子数）下：
      U≤3.25  GS = N=8 唯一单态（Lieb 成立）
      U≥3.5   GS = N=6/N=7，N=8 单态变激发 → "U_c≈3.5 简并相" = 粒子数扇区漂移伪影
    因此 fidelity probe 的 Q=0.0000 是粒子数壁垒（DQAP 守恒 N=8 vs ED GS N=6），
    不是自旋扇区壁垒。

本探针把 ED 分别投影到 N=6/7/8 扇区对角化，回答：
  1. 粒子数交叉点在哪（E0(N=6) 何时压过 E0(N=8)）？
  2. N=8 扇区内：GS 是否全程唯一单态、能隙是否单调 → 半满模型到底有没有相变？
  3. 各扇区 GS 的自旋指纹。
"""
import os
import sys

try:
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
except Exception:
    pass

import numpy as np
from scipy.sparse.linalg import eigsh

from dqap_ssh_hubbard import build_ssh_hubbard_hamiltonian
from spin_sector_probe import build_spin_ops, spin_label, fingerprint

L, N_QUBITS = 4, 16
DIM = 1 << N_QUBITS
K = 6

# popcount 掩码：按 N 分扇区
POP = np.zeros(DIM, dtype=np.int32)
for i in range(DIM):
    POP[i] = bin(i).count('1')


def sector_e0(matrix, N, k=K):
    mask = POP == N
    idx = np.where(mask)[0]
    sub = matrix[mask][:, mask]
    evals, evecs = eigsh(sub, k=min(k, len(idx) - 1), which='SA')
    # 把子空间本征向量散布回全空间（fingerprint 按全空间位形算）
    full = np.zeros((evals.size, DIM), dtype=complex)
    full[:, idx] = evecs.T
    return evals, full, idx


def main(v, w, boundary, U_grid):
    Splus, sz, n_tot, nA, nB = build_spin_ops(N_QUBITS)
    print(f"L={L}  v={v}  w={w}  {boundary}\n  {'U':>5} | {'E0(N=6)':>10} "
          f"{'E0(N=7)':>10} {'E0(N=8)':>10} | 最低扇区 | N=8 GS 指纹")
    for U in U_grid:
        h = build_ssh_hubbard_hamiltonian(L, v, w, U, boundary)
        M = h.to_matrix(sparse=True).tocsr()
        E6, v6, _ = sector_e0(M, 6)
        E7, v7, _ = sector_e0(M, 7)
        E8, v8, _ = sector_e0(M, 8)
        E6, E7, E8 = E6[0], E7[0], E8[0]
        lo = min(E6, E7, E8)
        which = ('N=6' if abs(lo - E6) < 1e-9 else
                 'N=7' if abs(lo - E7) < 1e-9 else 'N=8')
        # N=8 扇区 GS 指纹（唯一单态则稳定）
        g8 = fingerprint(v8[0, :], Splus, sz, n_tot, nA, nB)
        N, S2, Sz, dA, dB = g8
        print(f"  {U:5.2f} | {E6:10.5f} {E7:10.5f} {E8:10.5f} | {which:>4}  "
              f"| N={N:.0f} {spin_label(S2)} ⟨S_z⟩={Sz:+.2f}")
    print("\n注：N=8 扇区 GS 若全程 'S=0' 且 E0(N=8)<E0(N=6) 之前唯一 → "
          "半满模型无自旋相变，'相变' = 粒子数交叉伪影。")


if __name__ == '__main__':
    main(1.0, 2.0, 'APBC',
         [0.0, 1.0, 2.0, 3.0, 3.25, 3.4, 3.5, 3.6, 3.75, 4.0, 4.5, 5.0, 6.0])
