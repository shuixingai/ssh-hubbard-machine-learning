"""
ECDMRG 交叉验证：⟨(δn)²⟩、双占据、⟨m_s²⟩
==================================================
用 ED 跑指定测试点，输出关键序参量。
然后对比 DMRG 的对应值。

测试点 (t₁=t₂=0.01，均匀链极限):
  case A: U=4, V=0, L=6   — Mott/SDW
  case B: U=4, V=0, L=10  — Mott/SDW (大尺寸)
  case C: U=0, V=4, L=6   — CDW
  case D: U=0, V=4, L=10  — CDW (大尺寸)

用法:  python ED_DMRG_compare.py               (ED 部分)
       julia ED_DMRG_compare.jl                (DMRG 部分，需先跑完 ED)
"""

import sys, os
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ssh_model import SSHModel

import argparse
parser = argparse.ArgumentParser(description="ED-DMRG 交叉验证")
parser.add_argument('--L', type=int, nargs='+', default=[6, 10, 6, 10],
                    help="格点数列表")
parser.add_argument('--U', type=float, nargs='+', default=[4.0, 4.0, 0.0, 0.0],
                    help="U 列表")
parser.add_argument('--V', type=float, nargs='+', default=[0.0, 0.0, 4.0, 4.0],
                    help="V 列表")
parser.add_argument('--t1', type=float, default=0.01, help="t1 (奇数键)")
parser.add_argument('--t2', type=float, default=0.01, help="t2 (偶数键)")
parser.add_argument('--label', type=str, nargs='+',
                    default=['Mott L=6', 'Mott L=10', 'CDW L=6', 'CDW L=10'],
                    help="各测试点标签")
args = parser.parse_args()

test_cases = [
    {"L": L, "U": U, "V": V, "label": label}
    for L, U, V, label in zip(args.L, args.U, args.V, args.label)
]

t1 = args.t1
t2 = args.t2

print("=" * 72)
print(f"ED 计算: t₁ = {t1}, t₂ = {t2}")
print("=" * 72)

results = []
for case in test_cases:
    L = case["L"]
    U = case["U"]
    V = case["V"]
    label = case["label"]

    print(f"\n  --- {label} (L={L}, U={U}, V={V}) ---")

    model = SSHModel(L, t1, t2, U, V=V)
    energy, gs = model.get_ground_state()
    corr, _ = model.get_correlation_matrix(gs)

    # 传统指标
    dn  = model.get_staggered_charge_density(corr_matrix=corr)
    dB  = model.get_bond_order_alternation(corr_matrix=corr)
    docc = model.get_double_occupancy(gs)

    # 新指标：(δn)² 和 m_s²
    dn_sq = model.get_staggered_charge_squared(gs)
    ms_sq = model.get_staggered_magnetization_squared(gs)

    # 纠缠谱 gap4
    _, ent_spec = model.get_entanglement_spectrum(gs, n_max=10)
    if len(ent_spec) >= 5:
        gap4 = ent_spec[4] - ent_spec[3]
    else:
        gap4 = np.nan

    results.append({
        "label": label, "L": L, "U": U, "V": V,
        "energy": energy,
        "dn": dn, "dn_sq": dn_sq,
        "dB": dB, "docc_mean": docc.mean(),
        "ms_sq": ms_sq, "gap4": gap4,
    })

    occup_diag = np.diag(corr)
    print(f"    E0      = {energy:.10f}")
    print(f"    δn      = {dn:.6e}")
    print(f"    ⟨(δn)²⟩ = {dn_sq:.10f}")
    print(f"    δB      = {dB:.6f}")
    print(f"    ⟨docc⟩  = {docc.mean():.6f}")
    print(f"    ⟨m_s²⟩  = {ms_sq:.10f}")
    print(f"    gap4    = {gap4:.6f}")
    print(f"    occup   = {np.array2string(occup_diag, precision=4)}")

# ── 输出表格 ──
print("\n")
print("=" * 72)
print("ED 结果汇总")
print("=" * 72)
print(f"{'Case':<12} {'L':<3} {'U':<4} {'V':<4} {'E0':<14} {'⟨(δn)²⟩':<12} {'⟨docc⟩':<10} {'⟨m_s²⟩':<12} {'gap4':<8}")
print("-" * 72)
for r in results:
    print(f"{r['label']:<12} {r['L']:<3} {r['U']:<4} {r['V']:<4} "
          f"{r['energy']:<14.8f} {r['dn_sq']:<12.10f} "
          f"{r['docc_mean']:<10.6f} {r['ms_sq']:<12.10f} {r['gap4']:<8.4f}")

print("\nDMRG 部分待运行:  julia ED_DMRG_compare.jl")
print("=" * 72)
