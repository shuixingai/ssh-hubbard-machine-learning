#!/usr/bin/env python3
"""
pilot_le.py — 闸门 1：L-e（Loschmidt echo）线路构造验证
====================================================================
U=0 pilot 三层闸门的第 1 层（脚本 1 / 3）。

目标：验证 fidelity kernel 的量子线路读出
    K(x_i, x_j) = |⟨ψ(x_i)|ψ(x_j)⟩|²
    K_le        = P(all-zero | U†_i U_j |0...0⟩)
与精确态矢量直接内积对账：max|K_le − K_direct| < 1e-8。

三部分：
  Part A — DQAP 随机参数态（真实引擎、零 VQE 成本，含 M=0 初态恒等冒烟）
  Part B — ED N=4 物理基态（U=0 spinless L=4 APBC，initialize 线路，
           Statevector 精确路径；物理态预存 npz 供闸门 2 复用）
  Part C — DQAP L-e 的 shot 读出链路（AerSimulator + transpile）：
           3σ 判据 + shots 预算预览（闸门 3 前先摸清 shot 成本）

关键实现约定（2026-08-12 实测确认）：
  • qubit 序：spinless SSH A_i@2i, B_i@2i+1（相邻无 JW string，仅边界项有）
  • L-e 构造：U†V|0⟩ 全零概率 = |⟨ψU|ψV⟩|²（全局相位不敏感）
  • Qiskit 2.1.2 的 qc.inverse() 对含 initialize 的线路崩溃（initialize_dg
    复数参数 bug）→ 用 Initialize.gates_to_uncompute() 兜底
  • gates_to_uncompute() 产生 multiplexer 指令：Statevector 支持（精确）、
    AerSimulator 不支持 → ED 态只走精确路径，shot 只对 DQAP 做
  • DQAP 线路逆：qc.inverse() 原生可用（PauliEvolutionGate/X/H/CX 全可逆）
  • Statevector LSB 序：sv.data[0] = 全零振幅

判据（精确模拟，无 shot 噪声）：
    max|K_le − K_direct| < 1e-8  ∧  diag(K) = 1  ∧  K 对称
    Part C（若跑）：|K_shot − K_direct| < 3σ + 0.005

用法：
    python pilot_le.py              # 全部（A + B + C）
    python pilot_le.py --quick      # 只跑 Part A（最快冒烟）
    python pilot_le.py --no-aer     # 跳过 Part C（无 qiskit-aer 时）
"""

import os
import sys
import time
import argparse

try:
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
except Exception:
    pass

import numpy as np
from qiskit import QuantumCircuit, ClassicalRegister
from qiskit.quantum_info import Statevector
from qiskit.circuit.library import Initialize

from dqap_ssh_reproduce import build_ssh_hamiltonian, build_dqap_circuit

# ============================================================================
# 常量
# ============================================================================

L = 4
BOUNDARY = 'APBC'
N_TARGET = 4           # spinless 半满扇区（L=4 → 4 电子/自旋通道）
GATE_TOL = 1e-8        # L-e 线路正确性容差（精确模拟）
DEGEN_TOL = 1e-4       # 基态简并判据（与 tbc_berry 一致）
SHOT_N = 20000         # Part C1 每 pair 的 shots
N_SHOT_PAIRS = 12      # Part C1 最多验证的 pair 数（选最难 = 最接近 K=0.5）
SLACK = 0.005          # Part C shot 额外容差（采样 + transpile 扰动）
SIGMA = 3.0

# 物理参数点（U=0 pilot，远离 t1=t2 对角线，无简并）：
#   trivial = v>w（拓扑数 0），topo = v<w（拓扑数 1）
PHYS_POINTS = [
    (2.0, 1.0, 'trivial'),   # v=2 w=1
    (3.0, 1.0, 'trivial'),   # v=3 w=1
    (1.0, 2.0, 'topo'),      # v=1 w=2
    (1.0, 3.0, 'topo'),      # v=1 w=3
]

# Part A 分组（DQAP 随机参数态）：(label, v, w, M, seed)
#   M=0 = 纯初始成键态 |t⟩^L（全组 6 态相同 → K≡1，测 U†U=I 恒等冒烟）
PART_A_GROUPS = [
    ('trivial_M0', 2.0, 1.0, 0, 11),
    ('trivial_M2', 2.0, 1.0, 2, 12),
    ('topo_M2',    1.0, 2.0, 2, 13),
    ('topo_M4',    1.0, 2.0, 4, 14),   # Part C 复用此组
    ('diag_M3',    1.0, 1.0, 3, 15),   # 对角线（随机态无简并问题，测一般性）
]

PART_C_GROUP = 'topo_M4'


# ============================================================================
# L-e 核心构造
# ============================================================================

def _inverse_circuit(qc):
    """线路的逆。DQAP 线路原生支持 qc.inverse()；含 initialize 的线路在
    Qiskit 2.1.2 崩溃（initialize_dg 复数参数 bug）→ 用 gates_to_uncompute()
    兜底（产生 multiplexer 指令，仅 Statevector 支持、Aer 不支持）。"""
    try:
        return qc.inverse()
    except Exception:
        inv_qc = QuantumCircuit(qc.num_qubits, name='U_dag')
        for inst in qc.data:
            op = inst.operation
            if op.name != 'initialize':
                raise RuntimeError(f'不支持的指令用于求逆: {op.name}')
            psi = np.asarray(op.params, dtype=complex)
            qidx = [qc.qubits.index(q) for q in inst.qubits]
            inv_qc.compose(Initialize(psi).gates_to_uncompute(), qidx, inplace=True)
        return inv_qc


def build_loschmidt_echo(qc_v, qc_u, with_measurement=False):
    """L-e 线路：U†V|0⟩。
    V、U 均为态制备线路（V|0⟩=|ψ_V⟩, U|0⟩=|ψ_U⟩）。
    全零振幅 = ⟨0|U†V|0⟩ = ⟨ψ_U|ψ_V⟩ → 全零概率 = |⟨ψ_U|ψ_V⟩|²。"""
    assert qc_v.num_qubits == qc_u.num_qubits, '两个态制备线路的 qubit 数必须一致'
    n = qc_v.num_qubits
    qc = QuantumCircuit(n, name='LoschmidtEcho')
    qc.compose(qc_v, inplace=True)                     # 制备 |ψ_V⟩
    qc.compose(_inverse_circuit(qc_u), inplace=True)   # 施加 U†
    if with_measurement:
        qc.add_register(ClassicalRegister(n, 'meas'))
        qc.measure(range(n), range(n))
    return qc


def allzero_prob_exact(qc_state):
    """精确全零概率：sv.data[0] = 全零振幅（LSB 序）。"""
    sv = Statevector(qc_state)
    return float(abs(sv.data[0]) ** 2)


def kernel_pair_le(qc_i, qc_j):
    """K_le(i,j) = P(all-zero | U†_i U_j |0⟩)：制备 j，逆 i。"""
    return allzero_prob_exact(build_loschmidt_echo(qc_j, qc_i))


def kernel_pair_direct(sv_i, sv_j):
    """K_direct(i,j) = |⟨ψ_i|ψ_j⟩|²（精确态矢量内积）。"""
    return float(abs(np.vdot(sv_i, sv_j)) ** 2)


def kernel_matrix_full(items, pair_func):
    """完整 K 矩阵（不强制对称，留待对称性检查）。"""
    m = len(items)
    K = np.zeros((m, m))
    for i in range(m):
        for j in range(m):
            K[i, j] = pair_func(items[i], items[j])
    return K


def _kernel_report(name, K_le, K_dir):
    err = float(np.max(np.abs(K_le - K_dir)))
    err_diag = float(np.max(np.abs(np.diag(K_le) - 1.0)))
    err_sym = float(np.max(np.abs(K_le - K_le.T)))
    print(f'  [{name}] max|K_le − K_direct| = {err:.3e} '
          f'| diag 偏差 = {err_diag:.3e} | 非对称 = {err_sym:.3e}')
    return err, err_diag, err_sym


# ============================================================================
# Part A：DQAP 随机参数态（真实引擎，无 VQE）
# ============================================================================

def build_part_a_states():
    """按 PART_A_GROUPS 生成 DQAP 随机态。返回 [(label, qcs, svs), ...]。
    每组 6 个随机参数态；M=0 组 6 态全为初态（测恒等冒烟）。"""
    groups = []
    for (label, v, w, M, seed) in PART_A_GROUPS:
        rng = np.random.default_rng(seed)
        qcs, svs = [], []
        for _ in range(6):
            params = rng.uniform(0.0, 2.0 * np.pi, (M, 2)) if M > 0 else np.zeros((0, 2))
            qc = build_dqap_circuit(L, params, v, w, boundary=BOUNDARY)
            sv = np.asarray(Statevector(qc).data, dtype=complex)
            qcs.append(qc)
            svs.append(sv)
        groups.append((label, qcs, svs))
    return groups


def part_a_random_dqap(groups):
    """对每组的 6 态算 6×6 完整 K_le / K_direct，返回 max 全局误差。"""
    print('\n[Part A] DQAP 随机参数态 — L-e(精确) vs 直接内积')
    max_err = 0.0
    report = {}
    for (label, qcs, svs) in groups:
        K_le = kernel_matrix_full(qcs, kernel_pair_le)
        K_dir = kernel_matrix_full(svs, kernel_pair_direct)
        err, err_diag, err_sym = _kernel_report(label, K_le, K_dir)
        report[label] = {'err': err, 'err_diag': err_diag, 'err_sym': err_sym}
        max_err = max(max_err, err)
    return max_err, report


# ============================================================================
# Part B：ED N=4 物理基态（initialize 线路，Statevector 精确路径）
# ============================================================================

def sector_gs_spinless(v, w, N=N_TARGET, boundary=BOUNDARY, tol=DEGEN_TOL):
    """U=0 spinless SSH 的 N 电子扇区基态流形（稠密，2^8=256 维，秒级）。
    与 tbc_berry.sector_gs 同构但用 np.linalg.eigh（dense，eigh 本征值升序）。
    返回 (E0, evals, manifold, degen)，manifold 为嵌回全空间的行向量。"""
    n_qubits = 2 * L
    ham = build_ssh_hamiltonian(L, v, w, boundary)
    M = ham.to_matrix()
    pop = np.array([bin(i).count('1') for i in range(1 << n_qubits)], dtype=np.int32)
    mask = pop == N
    idx = np.where(mask)[0]
    sub = M[mask][:, mask]
    evals, evecs = np.linalg.eigh(sub)
    degen = int(np.sum(evals - evals[0] < tol))
    manifold = np.zeros((degen, 1 << n_qubits), dtype=complex)
    manifold[:, idx] = evecs[:, :degen].T   # 嵌回全空间基序
    return float(evals[0].real), evals.real, manifold, degen


def part_b_ed_physical():
    """4 个物理参数点的 N=4 基态：4×4 完整 K_le / K_direct。
    物理态（含 E0/degen/标签）一并返回供 npz 预存（闸门 2 复用）。"""
    print('\n[Part B] ED N=4 物理基态 — initialize 线路 L-e vs 直接内积')
    n_qubits = 2 * L
    records = []   # (v, w, lab, sv, qc, E0, degen)
    for (v, w, lab) in PHYS_POINTS:
        E0, evals, manifold, degen = sector_gs_spinless(v, w)
        gs = manifold[0]
        qc = QuantumCircuit(n_qubits)
        qc.initialize(gs)
        records.append((v, w, lab, gs, qc, E0, degen))
        print(f'  ({v:4.1f}, {w:4.1f}) {lab:8s}  E0={E0:10.6f}  degen={degen}')

    svs = [r[3] for r in records]
    qcs = [r[4] for r in records]
    K_le = kernel_matrix_full(qcs, kernel_pair_le)
    K_dir = kernel_matrix_full(svs, kernel_pair_direct)
    err, err_diag, err_sym = _kernel_report('ED_N4', K_le, K_dir)

    return {
        'err': err, 'err_diag': err_diag, 'err_sym': err_sym,
        'K_ED_N4': K_le,
        'v': np.array([r[0] for r in records]),
        'w': np.array([r[1] for r in records]),
        'lab': np.array([r[2] for r in records]),
        'states': np.array(svs),                 # (4, 256) complex — 闸门 2 复用
        'E0': np.array([r[5] for r in records]),
        'degen': np.array([r[6] for r in records]),
    }


# ============================================================================
# Part C：DQAP L-e 的 shot 读出链路（AerSimulator + transpile）
# ============================================================================

def _transpile_for_aer(qc_le):
    from qiskit import transpile
    return transpile(qc_le, basis_gates=['cx', 'rz', 'sx', 'x', 'id'],
                     optimization_level=0)


def part_c_shot(groups):
    """C1：N_SHOT_PAIRS 对（选 K 最接近 0.5 的最难对）在 SHOT_N 下 3σ 判据。
    C2：预算预览——K≈0.5 那一对在 [100,1e3,1e4,1e5] shots 下的误差 vs 3σ。
    返回 (c1_ok, c1_dev, c1_sig, c2_rows)。"""
    print('\n[Part C] DQAP L-e shot 读出链路（AerSimulator + transpile）')
    try:
        from qiskit_aer import AerSimulator
    except ImportError:
        print('  [skip] qiskit-aer 未安装，跳过 Part C')
        return None, None, None, []

    sim = AerSimulator()
    n = 2 * L
    label, qcs, svs = next(g for g in groups if g[0] == PART_C_GROUP)
    pairs = []
    for i in range(len(svs)):
        for j in range(i + 1, len(svs)):
            K = abs(np.vdot(svs[i], svs[j])) ** 2
            pairs.append((i, j, K))
    pairs.sort(key=lambda t: abs(t[2] - 0.5))          # 最难对排前
    pairs = pairs[:N_SHOT_PAIRS]
    print(f'  {PART_C_GROUP} 组：{len(pairs)} 对（K 最接近 0.5，方差最大），'
          f'shots={SHOT_N}')

    # C1：3σ 判据
    c1_dev, c1_sig = 0.0, 0.0
    for (i, j, K) in pairs:
        qc_le = build_loschmidt_echo(qcs[j], qcs[i], with_measurement=True)
        qc_t = _transpile_for_aer(qc_le)
        counts = sim.run(qc_t, shots=SHOT_N).result().get_counts()
        K_shot = counts.get('0' * n, 0) / SHOT_N
        sig = SIGMA * np.sqrt(K * (1 - K) / SHOT_N)
        c1_dev = max(c1_dev, abs(K_shot - K))
        c1_sig = max(c1_sig, sig)
    c1_ok = c1_dev < c1_sig + SLACK
    print(f'  C1: max|K_shot − K_direct| = {c1_dev:.4f}  '
          f'(3σ={c1_sig:.4f} + slack={SLACK})  {"PASS" if c1_ok else "FAIL"}')

    # C2：预算预览
    i, j, K = pairs[0]
    rows = []
    print('  C2 预算预览（K≈%.3f 的 pair）：' % K)
    print('    %10s  %10s  %10s  %10s' % ('shots', 'K_shot', '|dev|', '3σ'))
    for shots in (100, 1000, 10000, 100000):
        qc_le = build_loschmidt_echo(qcs[j], qcs[i], with_measurement=True)
        qc_t = _transpile_for_aer(qc_le)
        counts = sim.run(qc_t, shots=shots).result().get_counts()
        K_shot = counts.get('0' * n, 0) / shots
        dev = abs(K_shot - K)
        sig = SIGMA * np.sqrt(K * (1 - K) / shots)
        rows.append([int(shots), K_shot, dev, sig])
        print('    %10d  %10.5f  %10.2e  %10.4f' % (shots, K_shot, dev, sig))
    return c1_ok, c1_dev, c1_sig, rows


# ============================================================================
# 主流程
# ============================================================================

def main(out='pilot_le.npz', quick=False, no_aer=False):
    print('=' * 68)
    print(f'闸门 1：L-e 线路构造验证  |  L={L} {BOUNDARY}  |  '
          f'判据 max|K_le−K_direct| < {GATE_TOL}')
    print('=' * 68)

    # ── Part A ──
    t0 = time.time()
    groups = build_part_a_states()
    err_a, partA_errors = part_a_random_dqap(groups)
    print(f'  Part A 用时 {time.time()-t0:.1f}s')

    # ── Part B ──
    partB = None
    if not quick:
        t0 = time.time()
        partB = part_b_ed_physical()
        print(f'  Part B 用时 {time.time()-t0:.1f}s')

    # ── Part C ──
    c1_ok, c1_dev, c1_sig, c2_rows = None, None, None, []
    if (not quick) and (not no_aer):
        t0 = time.time()
        c1_ok, c1_dev, c1_sig, c2_rows = part_c_shot(groups)
        print(f'  Part C 用时 {time.time()-t0:.1f}s')

    # ── 判据 ──
    err_b = partB['err'] if partB else None
    diag_ok = err_a < GATE_TOL and (err_b is None or partB['err_diag'] < GATE_TOL)
    sym_ok = err_a < GATE_TOL and (err_b is None or partB['err_sym'] < GATE_TOL)
    passed = (err_a < GATE_TOL
              and (err_b is None or err_b < GATE_TOL)
              and (c1_ok is None or c1_ok))

    print('\n' + '═' * 68)
    print('闸门 1 判定')
    print('═' * 68)
    print(f'  Part A  DQAP 随机态   max|K_le−K_direct| = {err_a:.3e}'
          f'  (判据 < {GATE_TOL})  {"PASS" if err_a < GATE_TOL else "FAIL"}')
    if err_b is not None:
        print(f'  Part B  ED 物理基态   max|K_le−K_direct| = {err_b:.3e}'
              f'  (判据 < {GATE_TOL})  {"PASS" if err_b < GATE_TOL else "FAIL"}')
        print(f'          diag 偏差 = {partB["err_diag"]:.3e}  '
              f'非对称 = {partB["err_sym"]:.3e}')
    if c1_ok is None:
        print('  Part C  shot 读出     跳过')
    else:
        print(f'  Part C  shot 读出     max|K_shot−K| = {c1_dev:.4f} < '
              f'3σ+slack = {c1_sig + SLACK:.4f}  '
              f'{"PASS" if c1_ok else "FAIL"}')
    print(f'\n  总判定：{"PASS ✅（闸门 1 通过，可进闸门 2）" if passed else "FAIL ❌（见上方哪一项未过）"}')

    # ── 输出 npz（含 Part B 物理态预存，供闸门 2 复用）──
    data = {
        'gate': 'pilot_le', 'passed': passed,
        'gate_tol': GATE_TOL, 'L': L, 'boundary': BOUNDARY, 'n_target': N_TARGET,
        'err_a': err_a,
        'err_b': err_b if err_b is not None else np.nan,
        'c1_ok': c1_ok if c1_ok is not None else np.nan,
        'c1_dev': c1_dev if c1_dev is not None else np.nan,
        'c1_sig': c1_sig if c1_sig is not None else np.nan,
        'c2_preview': np.asarray(c2_rows, dtype=float) if c2_rows else np.zeros((0, 4)),
    }
    # 逐项塞入（np.savez 不接受 None）
    for k, v in partA_errors.items():
        for sub, val in v.items():
            data[f'partA_{k}_{sub}'] = val
    if partB is not None:
        data['phys_v'] = partB['v']
        data['phys_w'] = partB['w']
        data['phys_lab'] = partB['lab']
        data['phys_states'] = partB['states']          # (4,256) complex — 闸门 2 复用
        data['phys_E0'] = partB['E0']
        data['phys_degen'] = partB['degen']
        data['K_ED_N4'] = partB['K_ED_N4']

    np.savez(out, **data)
    print(f'\n输出已保存: {os.path.abspath(out)}')

    return 0 if passed else 1


if __name__ == '__main__':
    p = argparse.ArgumentParser(description='闸门 1：L-e 线路构造验证')
    p.add_argument('--quick', action='store_true',
                   help='只跑 Part A（最快冒烟）')
    p.add_argument('--no-aer', action='store_true',
                   help='跳过 Part C（无 qiskit-aer 环境时）')
    p.add_argument('--out', default='pilot_le.npz', help='输出 npz 路径')
    args = p.parse_args()
    sys.exit(main(out=args.out, quick=args.quick, no_aer=args.no_aer))
