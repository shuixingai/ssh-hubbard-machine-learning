#!/usr/bin/env python3
"""
DQAP SSH-Hubbard 回归验证脚本
==============================
回归验证此前关键判断是否仍然成立：
  1. U=0 谱完整性：E_sf = 2 × E_sl  (Z-string 谱正确性)
  2. L=4 M=3 各 U 值收敛情况
  3. M=6 U=4 更高深度能否收敛
  4. PBC/APBC 极化反转假设
  5. 初始态费米子相位 (CZ fix) 正确性
"""

import numpy as np
import time
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# ============================================================================
# 测试 1: U=0 谱完整性
# ============================================================================
def test_spectrum_integrity():
    """验证 spinful U=0 的谱 = 2 × spinless 谱"""
    print("\n" + "=" * 70)
    print("【测试 1】U=0 谱完整性：E_sf = 2 × E_sl")
    print("=" * 70)

    from dqap_ssh_reproduce import build_ssh_hamiltonian as build_h_sl
    from dqap_ssh_hubbard import build_ssh_hubbard_hamiltonian, compare_with_spinless
    from utils import diagonalize_hamiltonian

    results = {}
    for L in [2, 4]:
        print(f"\n── L={L} ──")
        # Spinless
        h_sl = build_h_sl(L, 1.0, 2.0, 'APBC')
        e_sl, _ = diagonalize_hamiltonian(h_sl)
        E0_sl = e_sl[0].real

        # Spinful U=0
        h_sf = build_ssh_hubbard_hamiltonian(L, 1.0, 2.0, 0.0, 'APBC')
        e_sf, _ = diagonalize_hamiltonian(h_sf)
        E0_sf = e_sf[0].real

        ratio = E0_sf / E0_sl
        print(f"  Spinless E₀ = {E0_sl:.10f}")
        print(f"  Spinful U=0 E₀ = {E0_sf:.10f}")
        print(f"  Ratio = {ratio:.8f}  (期望 2.0)")

        ok = abs(ratio - 2.0) < 1e-10
        status = "✅" if ok else "❌"
        print(f"  {status}  ratio-2 = {ratio - 2.0:.2e}")
        results[L] = {'ok': ok, 'ratio': ratio, 'E0_sl': E0_sl, 'E0_sf': E0_sf}

    return results

# ============================================================================
# 测试 2: 初始态验证（CZ fix + H₁ 基态确认）
# ============================================================================
def test_initial_state():
    """验证初始态确实是 H₁ 基态 (CZ fix 后能量正确)"""
    print("\n" + "=" * 70)
    print("【测试 2】初始态验证")
    print("=" * 70)

    from dqap_ssh_hubbard import (
        build_ssh_hubbard_hamiltonian, build_h1_hamiltonian,
        build_initial_state_spinful, compute_polarization_spinful
    )
    from utils import diagonalize_hamiltonian
    from qiskit.quantum_info import Statevector

    for L in [2, 4]:
        v, w, U = 1.0, 2.0, 0.0
        print(f"\n── L={L} ──")

        # 构建 Hamiltonian
        h_full = build_ssh_hubbard_hamiltonian(L, v, w, U, 'APBC')
        h1 = build_h1_hamiltonian(L, v)

        # H₁ 精确基态
        e_h1, v_h1 = diagonalize_hamiltonian(h1)
        E0_h1 = e_h1[0].real

        # 初始态 (双 |Ψ⁺⟩ Bell 态)
        qc_init = build_initial_state_spinful(L)
        sv_init = Statevector(qc_init)
        E_init_h1 = sv_init.expectation_value(h1).real

        # 粒子数检查（半填充：N_e = 2L）
        sv_data = sv_init.data
        probs = (sv_data.conjugate() * sv_data).real
        bits_list = [format(idx, f'0{4*L}b') for idx in range(len(sv_data))]
        n_electron = [sum(int(b) for b in bits) for bits in bits_list]
        avg_n = sum(p * n for p, n in zip(probs, n_electron))

        print(f"  H₁ 基态能量 E₀ = {E0_h1:.10f}")
        print(f"  初始态 ⟨H₁⟩    = {E_init_h1:.10f}")
        print(f"  ΔE(H₁)         = {E_init_h1 - E0_h1:.2e}")
        ok_h1 = abs(E_init_h1 - E0_h1) < 1e-8
        print(f"  {'✅' if ok_h1 else '❌'}  H₁ 基态验证")

        print(f"  平均粒子数      = {avg_n:.4f}  (期望 {2*L})")
        ok_n = abs(avg_n - 2*L) < 1e-10
        print(f"  {'✅' if ok_n else '❌'}  半填充验证")

        # U=0 全 Hamiltonian 的初始态能量
        E_init_full = sv_init.expectation_value(h_full).real
        print(f"  初始态 ⟨H_SSH⟩  = {E_init_full:.10f}")
        print(f"  H_SSH 基态 E₀  = {e_h1[0].real:.10f}")
        print(f"  ΔE(H_SSH)      = {E_init_full - e_h1[0].real:.2e}")

# ============================================================================
# 测试 3: L=4 M=3 U scan 复现（看之前的判断是否仍然成立）
# ============================================================================
def test_l4m3_scan():
    """复现 L=4 M=3 的 U scan，确认已知的收敛问题"""
    print("\n" + "=" * 70)
    print("【测试 3】L=4 M=3 U scan 复现")
    print("=" * 70)

    from dqap_ssh_hubbard import (
        build_ssh_hubbard_hamiltonian, build_initial_state_spinful,
        run_vqe, compute_polarization_spinful
    )
    from utils import diagonalize_hamiltonian
    from qiskit.quantum_info import Statevector

    L = 4
    M = 3
    v, w = 1.0, 2.0
    U_values = [0.0, 0.5, 1.0, 2.0, 4.0]
    boundary = 'APBC'

    results = {}
    for U in U_values:
        print(f"\n── U = {U} ──")
        h = build_ssh_hubbard_hamiltonian(L, v, w, U, boundary)

        # Exact
        eigs, vecs = diagonalize_hamiltonian(h)
        E0 = eigs[0].real
        sv_exact = Statevector(vecs[0])
        P_exact, _ = compute_polarization_spinful(sv_exact, L)

        # 初始态
        qc0 = build_initial_state_spinful(L)
        sv0 = Statevector(qc0)
        E0_init = sv0.expectation_value(h).real
        P_init, _ = compute_polarization_spinful(sv0, L)

        # VQE M=3
        t0 = time.time()
        result = run_vqe(L, v, w, U, boundary, M, h, verbose=False)
        t = time.time() - t0

        qc_opt = build_dqap_circuit_spinful(L, result.x, v, w, U, boundary)
        sv_opt = Statevector(qc_opt)
        P_opt, _ = compute_polarization_spinful(sv_opt, L)

        dE = result.fun - E0
        # 判断收敛状态
        if dE < 1e-6:
            conv_status = "🟢 完美"
        elif dE < 0.05:
            conv_status = "🟢 好"
        elif dE < 0.2:
            conv_status = "🟡 一般"
        elif dE < 1.0:
            conv_status = "🟠 差"
        else:
            conv_status = "🔴 失败"

        pol_ok = "✅" if abs(P_opt - P_exact) < 0.05 else "⚠️"
        print(f"  Exact:    E₀={E0:.8f}  P={P_exact:.4f}")
        print(f"  M=0:      E ={E0_init:.8f}  ΔE={E0_init - E0:.2e}")
        print(f"  M={M}: E ={result.fun:.8f}  ΔE={dE:.4e}  P={P_opt:.4f}  {conv_status}  {pol_ok}  ({t:.0f}s)")

        results[U] = {
            'E0': E0, 'P_exact': P_exact,
            'E_init': E0_init, 'P_init': P_init,
            'E_opt': result.fun, 'P_opt': P_opt,
            'dE': dE, 'conv_status': conv_status,
        }

    # 汇总表
    print("\n" + "─" * 70)
    print(f"{'U':>5} | {'E₀':>14} {'E₀_init':>14} {'E(M)':>14} {'ΔE':>10} | {'P_exact':>8} {'P(M)':>8} | {'状态':>8}")
    print("─" * 70)
    for U in U_values:
        r = results[U]
        print(f"{U:5.1f} | {r['E0']:14.8f} {r['E_init']:14.8f} {r['E_opt']:14.8f} {r['dE']:10.2e} | {r['P_exact']:8.4f} {r['P_opt']:8.4f} | {r['conv_status']:>8}")
    print("─" * 70)

    return results

# ============================================================================
# 测试 4: L=4 M=6 U=4 单点 — 更高深度能否收敛？
# ============================================================================
def test_l4m6_u4():
    """L=4 M=6 U=4：测试更高 M 能否解决强 U 收敛问题"""
    print("\n" + "=" * 70)
    print("【测试 4】L=4 M=6 U=4 单点 — 更高深度收敛性")
    print("=" * 70)

    from dqap_ssh_hubbard import (
        build_ssh_hubbard_hamiltonian, build_dqap_circuit_spinful,
        run_vqe, compute_polarization_spinful
    )
    from utils import diagonalize_hamiltonian
    from qiskit.quantum_info import Statevector

    L = 4
    v, w, U = 1.0, 2.0, 4.0
    boundary = 'APBC'
    h = build_ssh_hubbard_hamiltonian(L, v, w, U, boundary)

    # Exact
    eigs, vecs = diagonalize_hamiltonian(h)
    E0 = eigs[0].real
    sv_exact = Statevector(vecs[0])
    P_exact, _ = compute_polarization_spinful(sv_exact, L)
    print(f"Exact:    E₀={E0:.8f}  P={P_exact:.4f}")

    for M in [3, 4, 5, 6]:
        t0 = time.time()
        result = run_vqe(L, v, w, U, boundary, M, h, verbose=False)
        t = time.time() - t0

        qc_opt = build_dqap_circuit_spinful(L, result.x, v, w, U, boundary)
        sv_opt = Statevector(qc_opt)
        P_opt, _ = compute_polarization_spinful(sv_opt, L)
        dE = result.fun - E0

        pol_correct = abs(P_opt - P_exact) < 0.05
        print(f"  M={M}: E={result.fun:.8f}  ΔE={dE:.2e}  P={P_opt:.4f}  "
              f"{'✅' if pol_correct else '⚠️'}  ({t:.0f}s)")

# ============================================================================
# 测试 5: PBC vs APBC 极化对比
# ============================================================================
def test_pbc_vs_apbc():
    """验证 APBC π flux 翻转拓扑扇区的假设"""
    print("\n" + "=" * 70)
    print("【测试 5】PBC vs APBC 极化对比 — APBC π flux 假设验证")
    print("=" * 70)

    from utils import diagonalize_hamiltonian
    from qiskit.quantum_info import Statevector
    import importlib

    # 单独导入，避免跨模块 import 冲突
    rep = importlib.import_module('dqap_ssh_reproduce')
    hub = importlib.import_module('dqap_ssh_hubbard')

    L = 4
    v_triv, w_triv = 2.0, 1.0  # trivial (v > w)
    v_topo, w_topo = 1.0, 2.0  # topological (v < w)

    print(f"\nSpinless 参考:")
    for bc in ['PBC', 'APBC']:
        h_sl_triv = rep.build_ssh_hamiltonian(L, v_triv, w_triv, bc)
        h_sl_topo = rep.build_ssh_hamiltonian(L, v_topo, w_topo, bc)
        e_triv, vecs_triv = diagonalize_hamiltonian(h_sl_triv)
        e_topo, vecs_topo = diagonalize_hamiltonian(h_sl_topo)
        P_triv, _ = rep.compute_polarization(Statevector(vecs_triv[0]), L)
        P_topo, _ = rep.compute_polarization(Statevector(vecs_topo[0]), L)
        print(f"  {bc:5}: trivial P={P_triv:.4f}  topological P={P_topo:.4f}")

    print(f"\nSpinful U=0 & U=4（Hubbard 代码）：")
    for U in [0.0, 4.0]:
        print(f"  ── U={U} ──")
        for bc in ['PBC', 'APBC']:
            h_triv = hub.build_ssh_hubbard_hamiltonian(L, v_triv, w_triv, U, bc)
            h_topo = hub.build_ssh_hubbard_hamiltonian(L, v_topo, w_topo, U, bc)
            e_triv, vecs_triv = diagonalize_hamiltonian(h_triv)
            e_topo, vecs_topo = diagonalize_hamiltonian(h_topo)
            P_triv, _ = hub.compute_polarization_spinful(Statevector(vecs_triv[0]), L)
            P_topo, _ = hub.compute_polarization_spinful(Statevector(vecs_topo[0]), L)
            print(f"    {bc:5}: trivial P={P_triv:.4f}  topological P={P_topo:.4f}")


# ============================================================================
# Main
# ============================================================================
if __name__ == '__main__':
    print("=" * 70)
    print("DQAP SSH-Hubbard 回归验证脚本")
    print("时间:", time.strftime('%Y-%m-%d %H:%M:%S'))
    print("=" * 70)

    # 允许按编号运行特定测试
    tests = {
        1: test_spectrum_integrity,
        2: test_initial_state,
        3: test_l4m3_scan,
        4: test_l4m6_u4,
        5: test_pbc_vs_apbc,
    }

    if len(sys.argv) > 1:
        selected = [int(a) for a in sys.argv[1:] if a.isdigit()]
        for n in selected:
            if n in tests:
                tests[n]()
    else:
        for n in sorted(tests):
            tests[n]()

    print("\n" + "=" * 70)
    print("验证完成")
    print("=" * 70)
