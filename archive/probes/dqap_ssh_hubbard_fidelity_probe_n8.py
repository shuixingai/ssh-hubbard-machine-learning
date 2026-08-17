#!/usr/bin/env python3
"""
N=8 扇区保真度探针 — DQAP 对半满（固定 N=8）基态的"真实"态品质
========================================================================
背景（memory: dqap-polarization-hypotheses 2026-08-10 自旋探针收尾）：
    全空间 ED（μ=0，未固定粒子数）的 GS 随 U 漂到 N=6/7（罚双占据伪影），
    而 DQAP 电路严格守恒 N=8：
        初始态 = 每 cell (Aσ,Bσ) 成键 Bell 对 → 4 电子/cell × 4 = 8 电子
        H1/H2 层 = XX+YY（跳跃，粒子数守恒），HU 层 = RZ/RZZ（对角相）
    → DQAP 态永远在 N=8 扇区。因此 fidelity probe 的 Q=0.0000 是
      "DQAP(N=8) vs 全空间 GS(N=6)" 的粒子数正交，不是优化器失败。
      参考 ED 必须投影到 N=8 扇区，Q 才有物理意义。

本探针 = fidelity probe 的 N=8 修正版（唯一的改动：gs 流形取 N=8 扇区）：

    Part A：ED 投影 N=8 扇区沿 U 细扫 → Δ / n_degen / χ_F
            预期：能隙不塌陷、无简并、无 χ_F 峰值
                  → "U_c≈3.5 相变"消失，半满模型无相变（Lieb 一致）
    Part B：DQAP 态对 N=8 基态流形的投影 Q（复用 u4_probe.npz params）
            预期：U=0..2 保持 Q≈1；U=4 从 0.0000 变 ≈1
                  → DQAP 其实已到半满基态，此前被参考系选错冤枉成失败

判据：
    Q≈1 且 dE_N8≈0 → DQAP 状态制备成功（半满基态）
    Q≈0 且 dE_N8 大 → 半满扇区里也真失败 → 换优化器
    Δ_N8 恒非零     → 半满模型无粒子数伪影

用法：
    python dqap_ssh_hubbard_fidelity_probe_n8.py [u4_probe.npz 路径]
"""

import os
import sys

try:
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
except Exception:
    pass

import numpy as np
from scipy.sparse.linalg import eigsh
from qiskit.quantum_info import Statevector

from dqap_ssh_hubbard import (
    build_ssh_hubbard_hamiltonian,
    build_dqap_circuit_spinful,
)

DEGEN_TOL = 1e-4   # 与 fidelity probe / spin_sector_probe 一致
K_ED = 8
L, N_QUBITS = 4, 16
DIM = 1 << N_QUBITS
N_TARGET = 8        # 半满扇区

# popcount 掩码（LSB-first 基序，与 Qiskit Statevector 一致）
POP = np.zeros(DIM, dtype=np.int32)
for i in range(DIM):
    POP[i] = bin(i).count('1')


def sector_gs(M, N=N_TARGET, k=K_ED, tol=DEGEN_TOL):
    """N 扇区基态流形：sub-matrix 对角化 → 嵌回全空间（行向量）。

    NOTE: eigsh 返回的本征值不一定排序，必须先按能量排序再取流形。
    """
    mask = POP == N
    idx = np.where(mask)[0]
    sub = M[mask][:, mask]
    evals, evecs = eigsh(sub, k=min(k, len(idx) - 1), which='SA')
    full = np.zeros((evals.size, DIM), dtype=complex)
    full[:, idx] = evecs.T                     # 嵌回全空间基序
    evals = evals.real
    order = np.argsort(evals)
    evals = evals[order]
    full = full[order]
    E0 = evals[0]
    n_degen = int(np.sum((evals - E0) < tol))
    return E0, evals, full[:n_degen], n_degen


def subspace_fidelity(vecs_a, vecs_b):
    """F_sub = Σ|⟨a|b⟩|² / min(Da,Db) ∈ [0,1]（简并鲁棒）。"""
    if vecs_a.shape[0] == 0 or vecs_b.shape[0] == 0:
        return 0.0
    O = vecs_a @ vecs_b.conj().T
    D = min(vecs_a.shape[0], vecs_b.shape[0])
    return float((np.abs(O) ** 2).sum() / D)


# ============================================================================
# Part A: N=8 扇区 ED 沿 U 细扫
# ============================================================================

def fine_scan_n8(v, w, boundary, U_lo=2.0, U_hi=6.0, dU=0.25):
    print(f"[Part A] ED 投影 N={N_TARGET} 扇区，沿 U 细扫：Δ / degen / χ_F\n")
    U_grid = np.arange(U_lo, U_hi + 1e-9, dU)
    pts = []
    for U in U_grid:
        h = build_ssh_hubbard_hamiltonian(L, v, w, U, boundary)
        M = h.to_matrix(sparse=True).tocsr()
        E0, evals, vecs, n_degen = sector_gs(M)
        gap = evals[1] - E0 if len(evals) > 1 else np.nan
        pts.append({'U': float(U), 'E0': float(E0), 'gap': float(gap),
                    'n_degen': n_degen, 'vecs': vecs})
        print(f"  U={U:5.2f}  E0(N=8)={E0:12.6f}  Δ={gap:10.3e}  degen={n_degen}")

    for i in range(len(pts) - 1):
        F = subspace_fidelity(pts[i]['vecs'], pts[i + 1]['vecs'])
        pts[i]['chi_F'] = 1.0 - F
    if pts:
        pts[-1]['chi_F'] = np.nan

    print("\n  χ_F(U_i → U_{i+1}) = 1 - F_sub：")
    for i in range(len(pts) - 1):
        print(f"  [{pts[i]['U']:5.2f} → {pts[i+1]['U']:5.2f}]  χ_F = {pts[i]['chi_F']:.4f}")
    n_spike = sum(1 for p in pts[:-1] if p['chi_F'] > 0.1)
    print(f"\n  → N=8 扇区内 χ_F 峰值(>0.1) 数 = {n_spike}"
          f"  （原全空间探针在此区有 2 个 =1.0000 峰值）")
    return pts


# ============================================================================
# Part B: DQAP 态品质（对 N=8 基态流形）
# ============================================================================

def _parse_key(key):
    body = key[1:]
    if '_M' in body:
        u_str, m_str = body.split('_M')
        return float(u_str), int(m_str)
    return float(body), None


def dqap_vs_n8(npz_path, v, w, boundary):
    print(f"\n[Part B] DQAP 态对 N={N_TARGET} 基态流形的投影 Q\n")
    data = np.load(npz_path, allow_pickle=True)['results'].item()
    rows = []
    for key, rec in data.items():
        U, _ = _parse_key(key)
        params = np.asarray(rec['params'], dtype=float)
        M = len(params) // 3

        h = build_ssh_hubbard_hamiltonian(L, v, w, U, boundary)
        Mh = h.to_matrix(sparse=True).tocsr()
        E0_n8, _, vecs_n8, n_degen = sector_gs(Mh)

        # 全空间 GS 能量（对照：U≥3.5 时它是 N=6，不是 DQAP 能到达的目标）
        mask = POP == 6
        idx6 = np.where(mask)[0]
        E0_full = eigsh(Mh[mask][:, mask], k=1, which='SA')[0][0] if idx6.size > 1 else np.nan

        qc = build_dqap_circuit_spinful(L, params, v, w, U, boundary)
        psi = np.asarray(Statevector(qc).data, dtype=complex)

        ov = vecs_n8 @ psi.conj()
        Q = float((np.abs(ov) ** 2).sum())
        dE_n8 = float(rec['E']) - E0_n8
        dE_full = float(rec['E']) - float(E0_full)
        rows.append({'key': key, 'U': U, 'M': M, 'Q': Q, 'dE_N8': dE_n8,
                     'dE_full': dE_full, 'degen': n_degen,
                     'success': bool(rec['success'])})
        print(f"  {key:>7}  M={M}  Q(N=8流形)={Q:8.4f}  "
              f"ΔE vs N8-GS={dE_n8:9.3e}  ΔE vs 全空间GS={dE_full:9.3e}  "
              f"degen={n_degen}")
    return rows


# ============================================================================
# Main
# ============================================================================

def main(v=1.0, w=2.0, boundary='APBC', npz_path=None,
         save_path='fidelity_probe_n8.npz'):
    if npz_path is None:
        npz_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                'u4_probe.npz')
    if not os.path.exists(npz_path):
        print(f"[!] 找不到 {npz_path}，Part B 跳过；Part A 照跑。")

    print("=" * 74)
    print(f"N=8 扇区保真度探针 — L={L}, {boundary}, v={v}, w={w}（拓扑区）")
    print(f"npz = {os.path.basename(npz_path)}")
    print("改动：gs 流形 = N=8 扇区投影（DQAP 电路严格守恒 N=8）")
    print("=" * 74)

    pts = fine_scan_n8(v, w, boundary)
    out = {'partA': [{'U': p['U'], 'E0': p['E0'], 'gap': p['gap'],
                      'n_degen': p['n_degen'],
                      'chi_F': p.get('chi_F', np.nan)} for p in pts]}

    if os.path.exists(npz_path):
        rows = dqap_vs_n8(npz_path, v, w, boundary)
        out['partB'] = rows

    np.savez(save_path, **out)
    print(f"\n结果已存 {save_path}")

    print("\n" + "=" * 74)
    print("读法：")
    print("  Part A  若 Δ 恒非零、degen 恒 1、无 χ_F 峰值 → 半满模型无相变，")
    print("          'U_c≈3.5' 全是全空间粒子数漂移伪影。")
    print("  Part B  U=4 若 Q≈1 且 ΔE_N8≈0 → DQAP 已到半满基态，")
    print("          fidelity probe 的 Q=0.0000 = 粒子数正交，非优化器失败。")
    print("=" * 74)
    return out


if __name__ == '__main__':
    npz = sys.argv[1] if len(sys.argv) > 1 else None
    main(npz_path=npz)
