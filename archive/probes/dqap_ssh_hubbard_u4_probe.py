#!/usr/bin/env python3
"""
U=4 判决探针 — L=4 grow-M 能否捕捉极化跳变
============================================
判据（见 memory dqap-hubbard-qubit-ordering 路线图）：
    U=4 P → −0.490969  = 表达力瓶颈解除（增 M 有效，值得继续）
    U=4 P 仍 = 0        = 优化或表达力硬限（增 M 无效 → 转 QNG/VarQITE → ADAPT/QITE）

策略（最便宜的重建，因为前一次 2 天扫描没存断点 save_path）：
  1. U 链 0→0.5→1→2 在 M=3（warm-start 延续，~小时级）重建 U=2 的 M=3 解
  2. U=4 M=3 从 U=2 解 warm-start（旧测 ΔE=1.29, P=0，重测确认）
  3. U=4 padding M=4 → M=5，每点设墙上时间预算，跑完立刻看 P
  4. 任一点 P 逼近 −0.49 即成功 → 自动停止；预算超时自动收尾继续下一步

用法：
  python dqap_ssh_hubbard_u4_probe.py [budget_hours] [maxiter]
      budget_hours: 每个 M 点的时间预算（默认 6 小时，按上次数据高 M 单点 10~12h 折半）
      maxiter:      L-BFGS-B 迭代上限（默认 2000）
  运行前先 Ctrl+C 杀掉正在跑的 grow 扫描（同目录，防止 CPU 争抢）。
"""

import numpy as np
import time
import sys
from types import SimpleNamespace
from scipy.optimize import minimize
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector

from dqap_ssh_hubbard import (
    build_ssh_hubbard_hamiltonian,
    build_dqap_circuit_spinful,
    compute_energy,
    pad_params,
    compute_polarization_spinful,
    diagonalize_hamiltonian,
)

P_EXACT_U4 = -0.490969  # 已精确对角化确认的 U=4 极化（L=4, APBC, v=1, w=2）


class _TimeBudgetExceeded(Exception):
    """Scipy callback 抛出以中止 minimize；携带当前最优参数。"""

    def __init__(self, xk):
        self.xk = np.asarray(xk, dtype=float)
        super().__init__("time budget exceeded")


def run_vqe_timeboxed(L, v, w, U, boundary, M, hamiltonian, x0,
                      budget_sec, maxiter=2000, verbose=True):
    """L-BFGS-B 但带墙上时间预算。超时用最后迭代点收尾（L-BFGS-B 单调降，末点≈最优）。

    Returns: SimpleNamespace(x, fun, success) — 兼容 run_vqe 的返回用法。
    """
    if x0 is None:
        x0 = np.random.default_rng(42).uniform(0.0, 0.5, 3 * M)
    start = time.time()

    def _cb(xk):
        if time.time() - start > budget_sec:
            raise _TimeBudgetExceeded(xk)

    try:
        result = minimize(
            compute_energy, x0,
            args=(L, v, w, U, boundary, hamiltonian),
            method='L-BFGS-B',
            callback=_cb,
            options={'maxiter': maxiter, 'ftol': 1e-12, 'gtol': 1e-08},
        )
    except _TimeBudgetExceeded as e:
        elapsed = time.time() - start
        f_last = compute_energy(e.xk, L, v, w, U, boundary, hamiltonian)
        result = SimpleNamespace(x=e.xk, fun=f_last, success=False,
                                 nfev=None, nit=None)
        if verbose:
            print(f"  [时间预算 {budget_sec:.0f}s 超时 ({elapsed:.0f}s)，"
                  f"用末点 f={f_last:.8f} 收尾]")
    return result


def probe_u4(L=4, boundary='APBC', M_chain=3, M_max=5,
             budget_hours=6.0, maxiter=2000, seed=42,
             save_path='u4_probe.npz'):
    v, w = 1.0, 2.0
    U_chain = [0.0, 0.5, 1.0, 2.0]
    budget_sec = budget_hours * 3600

    print("=" * 70)
    print(f"U=4 判决探针 — L={L}, {boundary}")
    print(f"U 链 M={M_chain} 重建 → U=4 M={M_chain}..{M_max} (padding)")
    print(f"v={v}, w={w} (拓扑区), 每点时间预算 {budget_hours:.0f}h, "
          f"P_exact(U=4)={P_EXACT_U4}")
    print("=" * 70)

    all_results = {}

    # ───────────────────────────── 步骤 1: U 链重建 ─────────────────────────────
    prev_x, prev_U = None, None
    for U in U_chain:
        print(f"\n{'─' * 70}\nU = {U} (M={M_chain})")
        h = build_ssh_hubbard_hamiltonian(L, v, w, U, boundary)
        eigvals, _ = diagonalize_hamiltonian(h)
        E0 = eigvals[0].real

        init_x = prev_x if prev_x is not None else None
        res = run_vqe_timeboxed(L, v, w, U, boundary, M_chain, h, init_x,
                                budget_sec, maxiter)
        sv = Statevector(build_dqap_circuit_spinful(L, res.x, v, w, U, boundary))
        P, _ = compute_polarization_spinful(sv, L)
        print(f"  M={M_chain}: E={res.fun:.8f} ΔE={res.fun - E0:.2e} "
              f"P={P:.6f}  init={'random' if prev_x is None else f'warm U={prev_U}'}")

        all_results[f'U{U}'] = {'M': M_chain, 'E': float(res.fun),
                                'dE': float(res.fun - E0), 'P': float(P),
                                'params': res.x, 'success': bool(res.success)}
        prev_x, prev_U = res.x, U
        if save_path:
            np.savez(save_path, results=all_results)
            print(f"  [checkpoint -> {save_path}]")

    # ───────────────────────────── 步骤 2: U=4 判决点 ─────────────────────────────
    U = 4.0
    print(f"\n{'=' * 70}\n★ 判决点 U = {U}\n{'=' * 70}")
    h = build_ssh_hubbard_hamiltonian(L, v, w, U, boundary)
    eigvals, eigvecs = diagonalize_hamiltonian(h)
    E0 = eigvals[0].real
    P_exact, _ = compute_polarization_spinful(Statevector(eigvecs[0]), L)
    print(f"  精确 GS: E₀ = {E0:.10f}, P_exact = {P_exact:.6f}")

    prev_M_x = prev_x  # U=2.0 的 M_chain 解
    jumped = False
    for M in range(M_chain, M_max + 1):
        if M == M_chain:
            init_x, desc = prev_M_x, f"warm U=2.0 (M={M_chain})"
        else:
            init_x = pad_params(prev_M_x, M, seed=seed)
            desc = f"pad M={M - 1}->{M}"
        print(f"\n  U=4 M={M}: init={desc}, 预算 {budget_hours:.0f}h")

        res = run_vqe_timeboxed(L, v, w, U, boundary, M, h, init_x,
                                budget_sec, maxiter)
        sv = Statevector(build_dqap_circuit_spinful(L, res.x, v, w, U, boundary))
        P, _ = compute_polarization_spinful(sv, L)
        dE = res.fun - E0
        print(f"  M={M}: E={res.fun:.8f} ΔE={dE:.2e} P={P:.6f} success={res.success}")

        all_results[f'U4_M{M}'] = {'E': float(res.fun), 'dE': float(dE),
                                   'P': float(P), 'params': res.x,
                                   'success': bool(res.success)}
        if save_path:
            np.savez(save_path, results=all_results)
            print(f"  [checkpoint -> {save_path}]")

        # ── 判决 ──
        if abs(P - P_exact) < 0.05:
            print(f"\n  ✓✓ P={P:.4f} ≈ P_exact={P_exact:.4f} — "
                  f"增 M 成功解除表达力瓶颈，值得继续。")
            jumped = True
            break
        if P < -0.05:
            print(f"\n  → P={P:.4f} 已偏离 0，方向对但未到 −0.49："
                  f"增 M 有效，建议继续 M={M + 1}。")
            prev_M_x = res.x
            continue
        # P ≈ 0，未移动
        if M == M_max:
            print(f"\n  ✗✗ 到 M={M_max} P 仍={P:.6f}（ΔE={dE:.2e}）")
            if dE < 1e-2:
                print("    ΔE 小但 P 钉在 0 → 准简并扇区/对称性硬限。")
                print("    增 M 到此为止，转 QNG/VarQITE 优化器判据（能=纯优化问题；仍 0=硬限）")
                print("    → 或 ADAPT/QITE 态制备 → 或探测侧 fidelity susceptibility（ED 直接算，绕开 VQE）")
            else:
                print("    ΔE 仍大 → 优化未收敛。加大 M 单点更贵且不解决优化面。")
                print("    同样转优化器路线，不要继续加 M。")
        else:
            print(f"    P 仍=0，继续 M={M + 1}。")
        prev_M_x = res.x

    print(f"\n{'=' * 70}\n结果已存 {save_path}\n{'=' * 70}")
    return all_results


if __name__ == '__main__':
    budget_hours = float(sys.argv[1]) if len(sys.argv) > 1 else 6.0
    maxiter = int(sys.argv[2]) if len(sys.argv) > 2 else 2000
    probe_u4(L=4, boundary='APBC', M_chain=3, M_max=5,
             budget_hours=budget_hours, maxiter=maxiter,
             save_path='u4_probe.npz')
