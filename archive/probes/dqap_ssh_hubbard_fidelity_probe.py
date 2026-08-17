#!/usr/bin/env python3
"""
保真度敏感率探针 — DQAP 态品质 vs ED 基态（DQAP vs QKM 对比的 ground truth）
==============================================================================
背景（memory: dqap-polarization-hypotheses 2026-08-10 结论闭合）：
    P 路线到此为止——U=4 基态简并（E1-E0=9.6e-14）→ 极化无定义，-0.490969
    非权威值，追它无意义。真实信号是 ΔE；推进方向 = 换序参量（能隙 /
    保真度敏感率 / Loschmidt echo）。本探针不包含任何 -0.490969 硬编码。

本探针把测量目标从"极化 P"换成"态 overlap"，产三样东西（都秒级，不重跑 VQE）：

  Part A（ED 精确，沿 U 细扫）：
      Δ(U)     = E1 - E0       主判据：小 → 拟简并/相变点
      n_degen  = 基态简并度    （E1-E0 < 1e-4）
      χ_F(U)   = 1 - F_sub     F_sub = 子空间保真度（简并鲁棒），相变处取峰
      → 回答："U=4 到底是物理相变点还是有限尺寸伪影？"

  Part B（复用 u4_probe.npz 存下的 params，重建 DQAP 电路）：
      Q(U) = Σ_a |⟨ψ_DQAP(U)|ψ_a(U)⟩|²  DQAP 态在 ED 基态流形上的投影
          = 态品质探针 ≡ QKM kernel 的 ground truth（fidelity kernel ≡
            Loschmidt echo ≡ 保真度敏感率，见 qkm-vs-classical-ml-comparison）
      对 U=4 同时看 M=3/4/5：增 M 到底改没改善态品质（"M 非杠杆"的态品质侧验证）

  Part C（QKM kernel 矩阵，7×7 双矩阵对比）：
      K_ED  ：K_ij = F_sub(U_i, U_j)，简并鲁棒的"理想 kernel"（ED 能精确算）
      K_DQAP：K_ij = |⟨ψ_DQAP(U_i)|ψ_DQAP(U_j)⟩|²，真量子硬件/QKM 会测到的量
      两矩阵之差 = 态制备误差对 QKM kernel 的污染 → DQAP vs QKM 对比的最小雏形

判据（修正版，替代 -0.490969）：
    Q≈1            → DQAP 态 ≈ ED 基态，态制备成功
    Q≈0 且 ΔE 大   → 优化器失败（态不在基态流形内）→ 换优化器（QNG/VarQITE/ADAPT）
    Q≈0 且 ΔE 小   → 拟简并/对称性硬限 → 换态制备方案
    χ_F 峰值       → 相变点信号（P 给不了的）

用法：
  python dqap_ssh_hubbard_fidelity_probe.py [npz_path]
       npz_path: u4_probe.npz 路径（默认与脚本同目录）
"""

import os
import sys

# Windows 控制台默认 GBK 编码，中文 / χ_F 等字符会显示成乱码。
# 转成 UTF-8（现代终端都支持；失败则忽略，不影响计算）。
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

DEGEN_TOL = 1e-4  # E1-E0 < 该值 → 基态视为简并流形（相对第二流形真实 gap ≥ 0.097）
K_ED = 8          # 稀疏 eigsh 取最低本征值个数（排序后够抓 3 重简并 + 余量）


# ============================================================================
# 共用工具
# ============================================================================

def gs_manifold(h, k=K_ED, tol=DEGEN_TOL):
    """返回 (E0, eigvals, manifold_vecs, n_degen)。

    manifold_vecs = 最低 n_degen 个本征向量（行向量，与 qiskit Statevector 同基序）。

    NOTE: scipy eigsh 在重度简并/近简并时返回的本征值**不一定排序**
    （本系统实测过 k=0,1,5 同能量而 2,3,4 更高）。必须先按能量排序，
    否则 eigvecs[:n] 会抓到错误的本征向量，污染 overlap / kernel 计算。
    """
    eigvals, eigvecs = diagonalize_hamiltonian(h, k=k)
    eigvals = np.asarray(eigvals, dtype=float).real
    order = np.argsort(eigvals)          # 按能量升序
    eigvals = eigvals[order]
    eigvecs = np.asarray(eigvecs, dtype=complex)[order]
    E0 = eigvals[0]
    n_degen = int(np.sum((eigvals - E0) < tol))
    return E0, eigvals, eigvecs[:n_degen], n_degen


def subspace_fidelity(vecs_a, vecs_b):
    """F_sub = Σ_{a,b}|⟨a|b⟩|² / min(Da,Db) ∈ [0,1]。子空间相同时 = 1，正交 = 0。

    简并鲁棒：不依赖求解器在简并子空间里返回哪个基向量。
    """
    if vecs_a.shape[0] == 0 or vecs_b.shape[0] == 0:
        return 0.0
    O = vecs_a @ vecs_b.conj().T          # (Da, Db) = ⟨a|b⟩
    D = min(vecs_a.shape[0], vecs_b.shape[0])
    return float((np.abs(O) ** 2).sum() / D)


# ============================================================================
# Part A: ED 沿 U 细扫 → 能隙 / 简并度 / 保真度敏感率
# ============================================================================

def fine_scan_U(L, v, w, boundary, U_lo=2.0, U_hi=6.0, dU=0.25):
    """ED 细扫 U，逐点出能隙与简并度，相邻点间出 χ_F。秒级。"""
    U_grid = np.arange(U_lo, U_hi + 1e-9, dU)
    if not np.any(np.abs(U_grid - 4.0) < dU / 2):          # 确保含判决点 U=4
        U_grid = np.sort(np.append(U_grid, 4.0))

    pts = []
    for U in U_grid:
        h = build_ssh_hubbard_hamiltonian(L, v, w, U, boundary)
        E0, eigvals, vecs, n_degen = gs_manifold(h)
        gap = eigvals[1] - E0 if len(eigvals) > 1 else np.nan
        pts.append({'U': float(U), 'E0': float(E0), 'gap': float(gap),
                    'n_degen': n_degen, 'vecs': vecs})
        print(f"  U={U:5.2f}  E0={E0:12.6f}  Δ={gap:10.3e}  "
              f"degen={n_degen}")

    # 相邻点保真度敏感率
    for i in range(len(pts) - 1):
        F = subspace_fidelity(pts[i]['vecs'], pts[i + 1]['vecs'])
        pts[i]['chi_F'] = 1.0 - F
    if pts:
        pts[-1]['chi_F'] = np.nan

    print("\n  保真度敏感率 χ_F(U_i → U_{i+1}) = 1 - F_sub：")
    for i in range(len(pts) - 1):
        marker = "  <<< 峰值" if pts[i]['chi_F'] == max(
            p['chi_F'] for p in pts[:-1]) else ""
        print(f"  [{pts[i]['U']:5.2f} → {pts[i+1]['U']:5.2f}]  "
              f"χ_F = {pts[i]['chi_F']:.4f}{marker}")
    return pts


# ============================================================================
# Part B: DQAP 态品质（复用 u4_probe.npz 的 params，不重跑 VQE）
# ============================================================================

def _parse_key(key):
    """'U0.5' → (0.5, None)；'U4_M3' → (4.0, 3)。"""
    body = key[1:]                       # 去掉 'U'
    if '_M' in body:
        u_str, m_str = body.split('_M')
        return float(u_str), int(m_str)
    return float(body), None


def dqap_state_quality(npz_path, L, v, w, boundary):
    """对 npz 里每个存下的 DQAP 解：重建电路 → 投影到 ED 基态流形。

    Q = Σ_a |⟨ψ_DQAP|ψ_a⟩|²（态品质，QKM kernel ground truth）
    ΔE = E_DQAP - E0（主判据）。U=0 处 ΔE≈0 → Q≈1，兼作基序一致性自检。
    """
    data = np.load(npz_path, allow_pickle=True)['results'].item()
    rows = []
    for key, rec in data.items():
        U, _ = _parse_key(key)
        params = np.asarray(rec['params'], dtype=float)
        M = len(params) // 3

        h = build_ssh_hubbard_hamiltonian(L, v, w, U, boundary)
        E0, _, vecs, n_degen = gs_manifold(h)

        qc = build_dqap_circuit_spinful(L, params, v, w, U, boundary)
        psi = np.asarray(Statevector(qc).data, dtype=complex)

        ov = vecs @ psi.conj()                       # ⟨ψ_a|ψ_DQAP⟩
        Q = float((np.abs(ov) ** 2).sum())           # 流形投影
        Q_max = float((np.abs(ov) ** 2).max())       # 最匹配单态
        dE = float(rec['E']) - E0
        rows.append({'key': key, 'U': U, 'M': M, 'dE': dE,
                     'Q': Q, 'Q_max': Q_max, 'n_degen': n_degen,
                     'success': bool(rec['success'])})
        print(f"  {key:>7}  M={M}  ΔE={dE:9.3e}  Q(流形)={Q:8.4f}  "
              f"Q_max={Q_max:8.4f}  degen={n_degen}  success={rec['success']}")
    return rows


# ============================================================================
# Part C: QKM kernel 双矩阵（ED 理想 vs DQAP 真实）
# ============================================================================

def qkm_kernel_matrices(npz_path, L, v, w, boundary):
    """两张 7×7 overlap 矩阵：

    K_ED[i,j]   = F_sub(流形(U_i), 流形(U_j))    理想 kernel（简并鲁棒）
    K_DQAP[i,j] = |⟨ψ_DQAP(U_i)|ψ_DQAP(U_j)⟩|²    真 QKM/硬件会测到的量

    差值 max|K_ED - K_DQAP| = 态制备误差对 kernel 的污染。
    """
    data = np.load(npz_path, allow_pickle=True)['results'].item()
    keys = list(data.keys())

    svs = {}
    mans = {}
    for key in keys:
        U, _ = _parse_key(key)
        params = np.asarray(data[key]['params'], dtype=float)
        qc = build_dqap_circuit_spinful(L, params, v, w, U, boundary)
        svs[key] = np.asarray(Statevector(qc).data, dtype=complex)

        h = build_ssh_hubbard_hamiltonian(L, v, w, U, boundary)
        _, _, vecs, _ = gs_manifold(h)
        mans[key] = vecs

    n = len(keys)
    K_ED, K_DQAP = np.zeros((n, n)), np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            K_ED[i, j] = subspace_fidelity(mans[keys[i]], mans[keys[j]])
            K_DQAP[i, j] = float(
                np.abs(np.vdot(svs[keys[i]], svs[keys[j]])) ** 2)

    print("\n  K_ED（理想，ED 精确，简并鲁棒）：")
    print("      " + " ".join(f"{k:>7}" for k in keys))
    for i in range(n):
        print(f"  {keys[i]:>5}  " + " ".join(f"{K_ED[i,j]:7.3f}" for j in range(n)))

    print("\n  K_DQAP（真实，QKM 会在硬件上测到的量）：")
    print("      " + " ".join(f"{k:>7}" for k in keys))
    for i in range(n):
        print(f"  {keys[i]:>5}  " + " ".join(f"{K_DQAP[i,j]:7.3f}" for j in range(n)))

    err = float(np.abs(K_ED - K_DQAP).max())
    print(f"\n  态制备污染 = max|K_ED - K_DQAP| = {err:.4f}")
    return K_ED, K_DQAP, keys


# ============================================================================
# Main
# ============================================================================

def main(L=4, v=1.0, w=2.0, boundary='APBC',
         npz_path=None, save_path='fidelity_probe.npz'):
    if npz_path is None:
        npz_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                'u4_probe.npz')
    if not os.path.exists(npz_path):
        print(f"[!] 找不到 {npz_path}，Part B/C 跳过；Part A（ED 细扫）照跑。")
        has_params = False
    else:
        has_params = True

    print("=" * 74)
    print(f"保真度敏感率探针 — L={L}, {boundary}, v={v}, w={w}（拓扑区）")
    print(f"npz = {os.path.basename(npz_path)}")
    print("测量目标已从极化 P 换成态 overlap（fidelity kernel / χ_F / Loschmidt echo）")
    print("=" * 74)

    # ── Part A: ED 细扫 ──
    print(f"\n{'─' * 74}\n[Part A] ED 沿 U 细扫：能隙 / 简并度 / 保真度敏感率\n{'─' * 74}")
    pts = fine_scan_U(L, v, w, boundary)

    out = {'partA': [{'U': p['U'], 'E0': p['E0'], 'gap': p['gap'],
                      'n_degen': p['n_degen'],
                      'chi_F': p.get('chi_F', np.nan)} for p in pts]}

    # ── Part B / C: 复用 DQAP params ──
    if has_params:
        print(f"\n{'─' * 74}\n[Part B] DQAP 态品质：Q(U) = 流形投影\n{'─' * 74}")
        rows = dqap_state_quality(npz_path, L, v, w, boundary)
        out['partB'] = rows

        print(f"\n{'─' * 74}\n[Part C] QKM kernel 双矩阵\n{'─' * 74}")
        K_ED, K_DQAP, keys = qkm_kernel_matrices(npz_path, L, v, w, boundary)
        out['partC'] = {'keys': keys, 'K_ED': K_ED, 'K_DQAP': K_DQAP}

    np.savez(save_path, **out)
    print(f"\n结果已存 {save_path}")

    # ── 读法（判据替代 -0.490969）──
    print("\n" + "=" * 74)
    print("读法：")
    print("  Part A  χ_F 峰值 / 能隙塌陷位置 → U=4 是物理相变点还是有限尺寸伪影")
    print("  Part B  Q≈1 → DQAP 态制备成功；Q≈0+ΔE大 → 优化器失败（非 M 问题）；")
    print("          Q≈0+ΔE小 → 拟简并硬限 → 换优化器/态制备方案")
    print("  Part C  K_ED vs K_DQAP 差异 = 态制备误差对 QKM kernel 的污染")
    print("          = DQAP vs QKM 对比的最小雏形（同一系统、同一 overlap 测量）")
    print("=" * 74)
    return out


if __name__ == '__main__':
    npz = sys.argv[1] if len(sys.argv) > 1 else None
    main(npz_path=npz)
