#!/usr/bin/env python3
"""
TBC 多体 Berry phase（Zak phase）— 自旋 SSH-Hubbard N=8 半满扇区
====================================================================
文献核对结论（2026-08-11，memory: tbc-berry-literature-reconciliation）：
    • Watanabe arXiv:2602.22578v3：多体 Zak = −Im ln Π_j ⟨Ψ(θ_j)|Ψ(θ_{j+1})⟩
      （离散 Wilson loop，Eq.18）；均匀规范 twist 经局域规范变换等价于边界
      规范（Eq.12-15）→ 边界相位只放边界跳跃，即脚本 1 的 boundary_phase。
    • 自旋ful 半满（N_e=2L）下 *总电荷* twist 的 γ ≡ 0 (mod 2π)（Watanabe
      Fig.5：两自旋通道各 π，加和 2π ≡ 0）→ 不能作 U≠0 拓扑标签。
    • Lin–Ke–Lee PRB 2023 (arXiv:2211.07494v3)：正确 U≠0 标签 = Z2 电荷
      Berry phase —— 只扭 ↑（boundary_phase_dn=0），U=0 时 = 单粒子 Zak
      （拓扑 π / 平凡 0）；边界/周期规范只差经典极化，不影响 0/π 分类。

本脚本（对应脚本 1 的 boundary_phase 参数）：
    spin='both'  → 总电荷 twist  → 验证 γ ≡ 0 (mod 2π) ∀U（Watanabe 一致性）
    spin='up'    → ↑-only twist → Z2 charge Berry phase（U≠0 拓扑标签）
    均在 N=8 半满扇区投影 ED（DQAP 电路严格守恒 N=8，粒子数正交伪影免疫）

内置交叉检验：
    --selftest  验证脚本 1 四项 Pauli 展开符号：
                (a) H(θ=0) == H('PBC')   矩阵相等
                (b) H(θ=π) == H('APBC')  矩阵相等
                (c) H(θ) 厄米 ∀θ         (d) H(θ+2π) == H(θ)
    --scan      U=0 时 γ_up 必须 = 单粒子 Zak phase（拓扑 π / 平凡 0），
                γ_both ≡ 0 (mod 2π) —— 任一不符 = 边界相位符号错误

用法：
    python dqap_ssh_hubbard_tbc_berry.py --selftest
    python dqap_ssh_hubbard_tbc_berry.py --scan            [拓扑 w>v，n_θ=36]
    python dqap_ssh_hubbard_tbc_berry.py --scan --quick    [快速烟测：n_θ=12, U=[0,2,4]]
    python dqap_ssh_hubbard_tbc_berry.py --scan --trivial  [平凡 w<v]
"""

import os
import sys
import argparse

try:
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
except Exception:
    pass

import numpy as np
from scipy.sparse.linalg import eigsh

from dqap_ssh_hubbard import build_ssh_hubbard_hamiltonian, build_tbc_hamiltonian

DEGEN_TOL = 1e-4   # 简并判据（与 fidelity_probe_n8 一致）
K_ED = 8
N_TARGET = 8       # 半满扇区（L=4 → 8 电子）
GAP_WARN = 1e-3    # min E1−E0 低于此值 → γ 不可靠警告
SNAP_TOL = 0.05    # γ 归 0/π 的容差（rad，有限 n_θ 离散误差）

# popcount 掩码（LSB-first，与 Qiskit Statevector 一致），按 n_qubits 缓存
_POP = {}


def _popcounts(n_qubits):
    if n_qubits not in _POP:
        dim = 1 << n_qubits
        _POP[n_qubits] = np.array([bin(i).count('1') for i in range(dim)],
                                  dtype=np.int32)
    return _POP[n_qubits]


def sector_gs(M, N=N_TARGET, k=K_ED, tol=DEGEN_TOL):
    """N 扇区基态流形：sub-matrix 对角化 → 嵌回全空间（行向量）。
    eigsh 本征值不保证排序，先按能量排序再取流形。"""
    n_qubits = int(round(np.log2(M.shape[0])))
    dim = M.shape[0]
    POP = _popcounts(n_qubits)
    mask = POP == N
    idx = np.where(mask)[0]
    sub = M[mask][:, mask]
    evals, evecs = eigsh(sub, k=min(k, len(idx) - 1), which='SA')
    full = np.zeros((evals.size, dim), dtype=complex)
    full[:, idx] = evecs.T                     # 嵌回全空间基序
    evals = evals.real
    order = np.argsort(evals)
    evals = evals[order]
    full = full[order]
    E0 = evals[0]
    n_degen = int(np.sum((evals - E0) < tol))
    return E0, evals, full[:n_degen], n_degen


# ============================================================================
# 单粒子 Zak phase（自旋无关 SSH，U=0 交叉检验参照）
# ============================================================================

def sp_ssh_hamiltonian(L, v, w, theta):
    """Spinless SSH 紧束缚，边界规范：边界跳跃 × e^{iθ}。
    站点序 0=A₀, 1=B₀, 2=A₁, ..., 2L-2=A_{L-1}, 2L-1=B_{L-1}。
    H[B,A] = −w e^{iθ}（c†_B c_A 项），H[A,B] = −w e^{−iθ} ——
    与 dqap_ssh_hubbard.boundary_phase / ssh_model.compute_sp_zak 同约定。"""
    H = np.zeros((2 * L, 2 * L), dtype=complex)
    for i in range(L):
        H[2 * i, 2 * i + 1] = -v
        H[2 * i + 1, 2 * i] = -v
    for i in range(L - 1):
        H[2 * i + 1, 2 * i + 2] = -w
        H[2 * i + 2, 2 * i + 1] = -w
    H[2 * L - 1, 0] = -w * np.exp(1j * theta)
    H[0, 2 * L - 1] = -w * np.exp(-1j * theta)
    return H


def sp_zak_phase(L, v, w, n_theta=61):
    """自旋无关 SSH 的 SP Zak phase（填满最低 L 个态 = 下带）。
    γ = −angle Π_j det(Ψ_j† Ψ_{j+1})。期望：w>v → π；w<v → 0。"""
    theta_j = 2.0 * np.pi * np.arange(n_theta) / n_theta
    states = []
    for th in theta_j:
        evals, evecs = np.linalg.eigh(sp_ssh_hamiltonian(L, v, w, th))
        states.append(evecs[:, :L].T)          # L×2L 行向量，填下带
    prod = 1.0 + 0j
    for j in range(n_theta):
        O = states[j] @ states[(j + 1) % n_theta].conj().T
        prod *= np.linalg.det(O)
    return -np.angle(prod)


# ============================================================================
# 多体 Berry phase（Wilson loop，N=8 扇区）
# ============================================================================

def mb_berry(L, v, w, U, spin, n_theta, N=N_TARGET):
    """多体 Zak phase，沿 twist θ_j = 2πj/n_θ 的 Wilson loop。

    γ = −angle Π_j ⟨Ψ₀(θ_j)|Ψ₀(θ_{j+1})⟩  （θ_{n_θ} ≡ θ_0 闭环）

    另报告：
        min_ov  — loop 上最小 |⟨Ψ_j|Ψ_{j+1}⟩|（≪1 → 相位定义质量差）
        min_gap — loop 上最小 E1−E0（< GAP_WARN → γ 不可靠/可能简并）
        max_deg — loop 上最大基态简并度
    """
    theta_j = 2.0 * np.pi * np.arange(n_theta) / n_theta
    gs_list, gaps, degens = [], [], []
    for th in theta_j:
        h = build_tbc_hamiltonian(L, v, w, U, float(th), spin=spin)
        M = h.to_matrix(sparse=True).tocsr()
        E0, evals, vecs, n_degen = sector_gs(M, N=N)
        gs_list.append(vecs[0])
        gaps.append(evals[1] - E0 if len(evals) > 1 else np.nan)
        degens.append(n_degen)
    prod = 1.0 + 0j
    ov = np.empty(n_theta)
    for j in range(n_theta):
        ov[j] = abs(np.vdot(gs_list[j], gs_list[(j + 1) % n_theta]))
        prod *= np.vdot(gs_list[j], gs_list[(j + 1) % n_theta])
    gamma = -np.angle(prod)
    return (gamma, float(ov.min()), float(np.min(gaps)), int(max(degens)))


def snap_pi(gamma, tol=SNAP_TOL):
    """γ 归约到 {0, π}（mod 2π 后 snap）。返回 (值, 是否成功归位)。"""
    g = gamma % (2.0 * np.pi)
    if g > np.pi:
        g -= 2.0 * np.pi
    ag = abs(g)
    if ag < tol:
        return 0.0, True
    if abs(ag - np.pi) < tol:
        return np.pi, True
    return g, False


# ============================================================================
# --selftest：验证脚本 1 的 boundary_phase 四项展开
# ============================================================================

def selftest(L=4, v=1.0, w=2.0, U=1.0):
    print("=" * 72)
    print("Selftest: boundary_phase 四项 Pauli 展开（脚本 1）")
    print("=" * 72)
    ok = True

    H_pbc = build_ssh_hubbard_hamiltonian(L, v, w, U, 'PBC')
    H_apbc = build_ssh_hubbard_hamiltonian(L, v, w, U, 'APBC')
    H0 = build_ssh_hubbard_hamiltonian(L, v, w, U, 'PBC', boundary_phase=0.0)
    Hpi = build_ssh_hubbard_hamiltonian(L, v, w, U, 'PBC', boundary_phase=np.pi)

    def mx(h):
        """稀疏矩阵。16-qubit 稠密化 (2^16 × 2^16 complex) = 64 GB → OOM，
        必须保持稀疏（比较用 Python 内建 abs → __abs__ → .abs().max()）。"""
        return h.to_matrix(sparse=True).tocsr()

    # (a)(b) θ=0⟺PBC, θ=π⟺APBC
    for name, A, B in [("H(θ=0)  == H(PBC)", mx(H0), mx(H_pbc)),
                       ("H(θ=π)  == H(APBC)", mx(Hpi), mx(H_apbc))]:
        diff = abs(A - B).max()
        status = "PASS" if diff < 1e-10 else "FAIL"
        ok &= (diff < 1e-10)
        print(f"  [{status}] {name}   max|Δ| = {diff:.2e}")

    # (c) 厄米性
    for th in [0.3, 1.0, np.pi / 2, 5.0]:
        H = build_ssh_hubbard_hamiltonian(L, v, w, U, 'PBC', boundary_phase=th)
        Mm = mx(H)
        diff = abs(Mm - Mm.conj().T).max()
        status = "PASS" if diff < 1e-10 else "FAIL"
        ok &= (diff < 1e-10)
        print(f"  [{status}] H(θ={th:.3f}) 厄米   max|H−H†| = {diff:.2e}")

    # (d) 2π 周期
    Ha = build_ssh_hubbard_hamiltonian(L, v, w, U, 'PBC', boundary_phase=0.7)
    Hb = build_ssh_hubbard_hamiltonian(L, v, w, U, 'PBC',
                                       boundary_phase=0.7 + 2 * np.pi)
    diff = abs(mx(Ha) - mx(Hb)).max()
    status = "PASS" if diff < 1e-10 else "FAIL"
    ok &= (diff < 1e-10)
    print(f"  [{status}] H(θ+2π) == H(θ)   max|Δ| = {diff:.2e}")

    print()
    print("  全部通过 → 四项展开符号正确，可进行 --scan。"
          if ok else "  存在 FAIL → 停止，检查边界相位符号。")
    return ok


# ============================================================================
# --scan：both/up 双 twist 沿 U 扫描
# ============================================================================

def scan(v, w, U_grid, n_theta, label, save_path):
    L = 4
    spz = sp_zak_phase(L, v, w, n_theta=61)
    exp_pi = np.pi if w > v else 0.0

    print("=" * 72)
    print(f"TBC 多体 Zak phase 扫描 — L={L}, v={v}, w={w}  [{label}]")
    print(f"SP Zak（自旋无关参照）= {spz/np.pi:+8.4f} π   "
          f"(期望 {exp_pi/np.pi:+.3f} π)   n_θ={n_theta}")
    print("=" * 72)

    rows = []
    for U in U_grid:
        row = {'U': float(U)}
        line = f"  U={U:5.2f}"
        for spin in ['both', 'up']:
            gamma, min_ov, min_gap, max_deg = mb_berry(L, v, w, U, spin,
                                                       n_theta)
            snapped, snapped_ok = snap_pi(gamma)
            row[f'gamma_{spin}'] = gamma
            row[f'snap_{spin}'] = snapped
            row[f'snap_ok_{spin}'] = snapped_ok
            row[f'min_ov_{spin}'] = min_ov
            row[f'min_gap_{spin}'] = min_gap
            row[f'max_deg_{spin}'] = max_deg

            mark = ' ✓' if snapped_ok else ' ?'
            line += (f"  {spin:>4}: γ={gamma:+8.4f} rad  "
                     f"(={snapped/np.pi:+5.2f}π{mark})  "
                     f"min|O|={min_ov:6.3f}  minΔ={min_gap:9.2e}")
            if max_deg > 1:
                line += "  ⚠degen"
            if min_gap < GAP_WARN:
                line += "  ⚠gap"
        print(line)
        rows.append(row)

    # ── U=0 解析交叉检验 ──
    print()
    print("-" * 72)
    print("U=0 解析交叉检验（四项展开 + Wilson loop 全链路的判据）")
    u0 = rows[0]
    g_up0 = u0['gamma_up']
    g_bo0 = u0['gamma_both']

    # γ_up(U=0) 应等于 SP Zak（mod 2π —— −π ≡ +π 是同一拓扑值，
    # 不做归约会把分支切割两侧的同一角误判为 FAIL）
    d_up = abs((g_up0 - spz) % (2 * np.pi))
    d_up = min(d_up, 2 * np.pi - d_up)
    ok_up = d_up < 0.2
    print(f"  γ_up (U=0) = {g_up0:+8.4f} rad  vs  SP Zak = {spz:+8.4f} rad"
          f"   max|Δ| = {d_up:.4f}   [{('PASS' if ok_up else 'FAIL')}]")

    # γ_both(U=0) 应 ≡ 0 (mod 2π)（Watanabe：两自旋各 π 相加）
    d_bo = abs((g_bo0 % (2 * np.pi)) - 0.0)
    d_bo = min(d_bo, 2 * np.pi - d_bo)
    ok_bo = d_bo < 0.2
    print(f"  γ_both(U=0) = {g_bo0:+8.4f} rad  (期望 0 mod 2π, Watanabe)"
          f"   偏离 = {d_bo:.4f}   [{('PASS' if ok_bo else 'FAIL')}]")

    print()
    print("  两项皆 PASS → boundary_phase 符号 + Wilson loop + N=8 投影全链路正确。"
          if (ok_up and ok_bo)
          else "  存在 FAIL → 停止，先查 --selftest 与边界相位符号。")

    # ── 汇总表（snap 后，π 单位）──
    print()
    print("  汇总（snap 后 γ 的 π 单位）：")
    print(f"  {'U':>5} | {'γ_both':>10} | {'γ_up':>10}")
    print("  " + "-" * 32)
    for r in rows:
        print(f"  {r['U']:5.1f} | {r['snap_both']/np.pi:10.3f} | "
              f"{r['snap_up']/np.pi:10.3f}")
    print("  " + "-" * 32)
    print("  读法：γ_up = 0/π 是否随 U 保持 → Z2 charge Berry phase 是否")
    print("        仍区分拓扑/平凡；γ_both ≡ 0 恒成立是 Watanabe 预言，正常。")

    np.savez(save_path, L=L, v=v, w=w, n_theta=n_theta,
             sp_zak=spz, rows=rows)
    print(f"\n结果已存 {save_path}")


# ============================================================================
# Main
# ============================================================================

def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--selftest', action='store_true', help='验证四项展开符号')
    ap.add_argument('--scan', action='store_true', help='U 扫描双 twist')
    ap.add_argument('--quick', action='store_true', help='快速烟测(n_θ=12, U=[0,2,4])')
    ap.add_argument('--trivial', action='store_true', help='平凡区 w<v (v=2,w=1)')
    ap.add_argument('--n-theta', type=int, default=None)
    ap.add_argument('--u-grid', type=str, default=None, help='逗号分隔 U 列表')
    ap.add_argument('--out', type=str, default='tbc_berry_scan.npz')
    args = ap.parse_args()

    if args.selftest:
        sys.exit(0 if selftest() else 1)

    if not args.scan and not args.quick and not args.trivial:
        ap.print_help()
        sys.exit(0)

    v, w = (2.0, 1.0) if args.trivial else (1.0, 2.0)
    label = "trivial (w<v)" if args.trivial else "topological (w>v)"

    if args.quick:
        n_theta = 12
        U_grid = [0.0, 2.0, 4.0]
    else:
        n_theta = args.n_theta or 36
        U_grid = ([0.0, 0.5, 1.0, 2.0, 4.0, 6.0]
                  if args.u_grid is None
                  else [float(x) for x in args.u_grid.split(',')])

    scan(v, w, U_grid, n_theta, label, args.out)


if __name__ == '__main__':
    main()
