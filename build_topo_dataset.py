#!/usr/bin/env python3
"""
简化版全扫描 — SSH-Hubbard 拓扑数据集构建 阶段1 (test)
=====================================================================
三维网格 (t1, t2, U) 上计算：
    γ_up    — ↑-only Z2 charge Berry phase（U≠0 拓扑标签，Lin–Ke–Lee 2211.07494）
    min_gap — Wilson loop 上最小 E1−E0（判有隙/临界，GAP_WARN 守卫）
    Δ_PBC   — θ=0（PBC）的 E1−E0（王远峰框架 Δ_n 口径）
    min_ov  — loop 上最小 |⟨Ψ_j|Ψ_{j+1}⟩|（相位定义质量）
    max_deg — loop 上最大基态简并度
    label   — 派生三类标签（拓扑/平凡/临界；阈值后处理可调，原始量已存）

标签链路（memory: tbc-berry-literature-reconciliation）：
    • 半满总电荷 twist γ_both ≡ 0 (mod 2π)（Watanabe）→ 不作标签，只验一致性
    • U≠0 正确标签 = per-spin Z2 charge Berry phase（只扭 ↑，U=0 时 = 单粒子 Zak）
    • γ_up snap 到 π → 拓扑；snap 到 0 → 平凡
    • max_deg>1 或 min_gap<GAP_WARN → 临界（能隙闭合，γ 不可靠）

数据格式借鉴 ssh_model.generate_dataset（已有数据生成规范）：
    网格形状数组 (n_t1, n_t2, n_U) + 轴数组 + np.savez_compressed
    _refined_linspace 加密 t1≈t2 附近 —— 注意：U≠0 时相互作用会偏移相边界，
    故保留完整 3D 网格，(t1/t2) 比值只是基准认知而非硬约束。

用法：
    python build_topo_dataset.py --quick                     # 烟测 5×5×3, n_θ=12
    python build_topo_dataset.py --n-t1 13 --n-t2 13 --n-u 7 --n-theta 24 \
        --out topo_dataset_full.npz                          # 全量 ≈1183 点
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

_HERE = os.path.dirname(os.path.abspath(__file__))
_QDIR = os.path.join(_HERE, 'qiskit_simulation')
if _QDIR not in sys.path:
    sys.path.insert(0, _QDIR)

from dqap_ssh_hubbard import build_tbc_hamiltonian
from dqap_ssh_hubbard_tbc_berry import (
    sector_gs, snap_pi, sp_zak_phase, GAP_WARN, N_TARGET)

L = 4                 # 4 格点 → 16 自旋轨道 → N=8 半满扇区
DEFAULT_N_THETA = 24

# 标签编码（王远峰框架三类，CDW 已砍）
LABEL_TRIVIAL, LABEL_TOPO, LABEL_CRITICAL, LABEL_UNRESOLVED = 0, 1, 2, 3
_LABEL_CH = {LABEL_TRIVIAL: '0', LABEL_TOPO: 'T', LABEL_CRITICAL: 'C',
             LABEL_UNRESOLVED: '?'}


def _refined_linspace(vmin, vmax, n):
    """arcsin 分布：加密 [vmin, vmax] 中心（t1≈t2 相变附近分辨率更高）。
    同 ssh_model.generate_dataset 的 _refined_linspace。"""
    x = np.linspace(0.0, 1.0, n)
    return vmin + (vmax - vmin) * (0.5 + np.arcsin(2.0 * x - 1.0) / np.pi)


def derive_label(snap, snap_ok, min_gap, max_deg, gap_warn=GAP_WARN):
    """派生三类标签。gap_warn 阈值后处理可调——脚本始终存原始量。"""
    if max_deg > 1 or min_gap < gap_warn:
        return LABEL_CRITICAL
    if not snap_ok:
        return LABEL_UNRESOLVED
    return LABEL_TOPO if abs(snap - np.pi) < 1e-9 else LABEL_TRIVIAL


def _mb_berry_full(L, t1, t2, U, n_theta, N=N_TARGET):
    """mb_berry（脚本2 同名逻辑）的本地扩展：额外返回 θ=0（PBC）的 E1−E0。

    θ_j = 2πj/n_θ 中 j=0 即 θ=0（PBC），故 gaps[0] 就是 Δ_PBC ——
    复用同一次 Wilson loop 对角化，避免为 Δ_n 单独再对角化 θ=0。"""
    theta_j = 2.0 * np.pi * np.arange(n_theta) / n_theta
    gs_list, gaps, degens = [], [], []
    for th in theta_j:
        h = build_tbc_hamiltonian(L, t1, t2, U, float(th), spin='up')
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
    return (gamma, float(ov.min()), float(np.min(gaps)),
            int(max(degens)), float(gaps[0]))


def scan_grid(t1_vals, t2_vals, U_vals, n_theta, verbose=True):
    """(t1,t2,U) 全网格扫描。每量返回一个 (n1,n2,nu) 网格形状 ndarray。

    另含 U=0 层解析交叉检验（γ_up vs 单粒子 Zak，mod-2π 折叠）。"""
    n1, n2, nu = len(t1_vals), len(t2_vals), len(U_vals)
    gamma = np.empty((n1, n2, nu))
    snap = np.empty((n1, n2, nu))
    snap_ok = np.zeros((n1, n2, nu), dtype=bool)
    min_ov = np.empty((n1, n2, nu))
    min_gap = np.empty((n1, n2, nu))
    gap_pbc = np.empty((n1, n2, nu))
    max_deg = np.zeros((n1, n2, nu), dtype=int)
    label = np.zeros((n1, n2, nu), dtype=np.int8)

    sp_zak = np.empty((n1, n2))            # U=0 单粒子参照，每 (t1,t2) 一次
    xchk_pass = np.zeros((n1, n2), dtype=bool)
    xchk_d = np.empty((n1, n2))

    n_pts = n1 * n2 * nu
    t0 = time.time()
    ip = 0
    for i, t1 in enumerate(t1_vals):
        for j, t2 in enumerate(t2_vals):
            sp_zak[i, j] = sp_zak_phase(L, t1, t2, n_theta=61)
            for k, U in enumerate(U_vals):
                g, mo, mg, md, gp = _mb_berry_full(L, t1, t2, U, n_theta)
                sn, sok = snap_pi(g)
                gamma[i, j, k] = g
                snap[i, j, k] = sn
                snap_ok[i, j, k] = sok
                min_ov[i, j, k] = mo
                min_gap[i, j, k] = mg
                gap_pbc[i, j, k] = gp
                max_deg[i, j, k] = md
                label[i, j, k] = derive_label(sn, sok, mg, md)

                if k == 0:    # U=0：γ_up 应 ≡ 单粒子 Zak（mod-2π 折叠，−π≡+π）
                    d = abs((g - sp_zak[i, j]) % (2 * np.pi))
                    d = min(d, 2 * np.pi - d)
                    xchk_pass[i, j] = d < 0.2
                    xchk_d[i, j] = d

                ip += 1
                if verbose and (ip % 10 == 0 or ip == n_pts):
                    el = time.time() - t0
                    eta = el / ip * (n_pts - ip)
                    print(f"  [{ip:4d}/{n_pts}] t1={t1:.2f} t2={t2:.2f} "
                          f"U={U:.2f}  elapsed={el:5.0f}s  eta≈{eta:5.0f}s",
                          flush=True)
    print(f"  完成：{n_pts} 点，耗时 {(time.time()-t0)/60:.1f} min")

    return dict(gamma=gamma, snap=snap, snap_ok=snap_ok, min_ov=min_ov,
                min_gap=min_gap, gap_pbc=gap_pbc, max_deg=max_deg,
                label=label, sp_zak=sp_zak, xchk_pass=xchk_pass,
                xchk_d=xchk_d)


def report(data, t1_vals, t2_vals, U_vals):
    """打印标签分布 + U=0 / U=U_max 两层的 (t1,t2) 标签图。"""
    label, snap_ok, min_gap, max_deg = (data['label'], data['snap_ok'],
                                        data['min_gap'], data['max_deg'])
    counts = np.bincount(label.flatten(), minlength=4)
    n_crit = counts[LABEL_CRITICAL]
    n_unres = counts[LABEL_UNRESOLVED]

    print("=" * 60)
    print("标签分布（网格全空间）")
    print(f"  平凡(0)={counts[LABEL_TRIVIAL]:5d}   拓扑(1)={counts[LABEL_TOPO]:5d}")
    print(f"  临界(2)={n_crit:5d}   未定(?)={n_unres:5d}")

    # 临界点应集中在 t1≈t2 附近 + 简并/U 相关位置；未定点应 ~0
    if n_crit == 0 and np.any(min_gap < GAP_WARN):
        print("  ⚠ min_gap<GAP_WARN 存在但 label 无临界 → 检查 derive_label 逻辑")

    # U=0 交叉检验通过率
    print("-" * 60)
    xp = data['xchk_pass']
    print(f"U=0 交叉检验（γ_up vs SP Zak, mod-2π）："
          f"{xp.mean()*100:.1f}% PASS  "
          f"max|Δ|={data['xchk_d'].max():.3f} rad")

    # 两层标签图
    for k, U in enumerate([0, len(U_vals) - 1]):
        if U not in (0, len(U_vals) - 1):
            continue
        Uv = U_vals[U]
        print("-" * 60)
        print(f"label 图  U={Uv:.2f}  （行=t1 从下↑上，列=t2 从左→右）")
        print("      t2 →  " + " ".join(f"{t:4.2f}" for t in t2_vals))
        for i in range(len(t1_vals) - 1, -1, -1):
            row = "".join(_LABEL_CH[label[i, j, U]] for j in range(len(t2_vals)))
            print(f"  t1={t1_vals[i]:4.2f}   {row}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--quick', action='store_true', help='烟测 5×5×3, n_θ=12')
    ap.add_argument('--n-t1', type=int, default=13)
    ap.add_argument('--n-t2', type=int, default=13)
    ap.add_argument('--n-u', type=int, default=7)
    ap.add_argument('--n-theta', type=int, default=DEFAULT_N_THETA)
    ap.add_argument('--t-max', type=float, default=4.0)
    ap.add_argument('--u-max', type=float, default=4.0)
    ap.add_argument('--out', type=str, default='topo_dataset_full.npz')
    args = ap.parse_args()

    if args.quick:
        n1, n2, nu, nth = 5, 5, 3, 12
        out = 'topo_dataset_smoke.npz'
    else:
        n1, n2, nu, nth = args.n_t1, args.n_t2, args.n_u, args.n_theta
        out = args.out

    t1 = _refined_linspace(0.25, args.t_max, n1)
    t2 = _refined_linspace(0.25, args.t_max, n2)
    U = np.linspace(0.0, args.u_max, nu)

    print("=" * 60)
    print(f"SSH-Hubbard 拓扑数据集扫描 — L={L}, N=8 半满扇区, n_θ={nth}")
    print(f"网格 {n1}×{n2}×{nu} = {n1*n2*nu} 点")
    print(f"  t1 ∈ [{t1[0]:.2f}, {t1[-1]:.2f}] (refined)  "
          f"t2 ∈ [{t2[0]:.2f}, {t2[-1]:.2f}] (refined)")
    print(f"  U ∈ [{U[0]:.2f}, {U[-1]:.2f}] (均匀)")
    print("=" * 60)

    data = scan_grid(t1, t2, U, nth)
    report(data, t1, t2, U)

    np.savez_compressed(
        out, L=L, t1_vals=t1, t2_vals=t2, U_vals=U, n_theta=nth,
        gamma_up=data['gamma'], snap_up=data['snap'],
        snap_ok_up=data['snap_ok'], min_ov=data['min_ov'],
        min_gap=data['min_gap'], gap_pbc=data['gap_pbc'],
        max_deg=data['max_deg'], label=data['label'],
        sp_zak=data['sp_zak'], xchk_pass=data['xchk_pass'],
        xchk_d=data['xchk_d'])
    print(f"\n结果已存 {out}")


if __name__ == '__main__':
    main()
