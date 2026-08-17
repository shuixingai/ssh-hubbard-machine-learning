#!/usr/bin/env python3
"""
qkm_ml.py — QKM 三栏对决（memory: gap4-classical-baseline §kernel 融合设计）
============================================================================
对同一批 (t1,t2,U) 点（qkm_grid_features.py 产出的 DQAP 态 fidelity kernel）
跑三个 SVM 栏（同点同协议 → 直接可比的 ablation）：

    ① classical-only   RBF-SVM on 5 个经典多体标量（与量子同点，经 idx 对齐）
    ② quantum-only     precomputed-SVM on K_DQAP（fidelity kernel）
    ③ hybrid           K_hyb(w) = w·K̂_Q + (1−w)·K̂_C
                       （各自 Frobenius 归一化，w 在数据上 CV 选）

诊断输出：PSD 数值检查 / 中心化 KTA（K_Q vs K_C 各分量）/ 三栏消融 + Wilson CI
        / 决策图数据（预测标签写回 (t1,t2) 平面逐 U 层）。

数据来源（无污染设计，勿破坏）：
    特征① = ssh_dataset_L8_labelgrid.npz（OBC ED 多体量）
    特征② = qkm_grid_M{M}.npz          （DQAP 电路态）
    标签   = topo_dataset_full.npz      （PBC/TBC γ_up，唯一标签来源）
三次计算完全独立，只共享 (t1,t2,U) 坐标。单粒子不变量（zak/winding）一律排除。

用法：
    python qkm_ml.py                     # 读 qkm_grid_M3.npz（默认 M=3）
    python qkm_ml.py --M 2 --grid qkm_grid_M2.npz
输出：控制台三栏表 + qkm_ml_M{M}.npz（含决策图数据）。
"""

import argparse
import os
import sys

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

_HERE = os.path.dirname(os.path.abspath(__file__))
_DATA = os.path.dirname(_HERE)
for p in (_HERE, _DATA):
    if p not in sys.path:
        sys.path.insert(0, p)

from baseline_ml import build_features_grid, FEATURE_NAMES_5
from kernel_ml_utils import (
    binom_ci, frob_normalize, hybrid_kernel, kta, psd_min_eig,
    rbf_matrix, run_cv_precomputed, run_cv_rbf, shuffle_null,
    tune_cv_precomputed, tune_cv_rbf, C_GRID, W_GRID,
)

LABEL_NPZ = os.path.join(_DATA, "topo_dataset_full.npz")
FEAT_NPZ = os.path.join(_DATA, "ssh_dataset_L8_labelgrid.npz")
OUT_TMPL = os.path.join(_HERE, "qkm_ml_M{M}.npz")


# ── 加载 + 对齐 ───────────────────────────────────────────────────────
def load_and_align(qkm_path):
    """读 qkm npz + 标签 npz + ssh npz；经典特征按 qkm idx（13×13×7 C 序
    扁平索引）坐标对齐。返回 (K_Q, X_C, y, coords)。"""
    q = np.load(qkm_path)
    topo = np.load(LABEL_NPZ)
    feat = np.load(FEAT_NPZ)

    # 轴对账（L 口径坑：ssh_model L=8 site vs topo L=4 cell=8 sites）
    assert np.allclose(q["t1_vals"], topo["t1_vals"]), "qkm 与标签 t1 轴不对齐"
    assert np.allclose(q["t2_vals"], topo["t2_vals"]), "qkm 与标签 t2 轴不对齐"
    assert np.allclose(q["U_vals"], topo["U_vals"]), "qkm 与标签 U 轴不对齐"

    K_Q = np.asarray(q["K_DQAP"], dtype=float)
    y = np.asarray(q["lab"], dtype=int)

    # 经典特征网格 → 按 qkm idx 对齐（同一坐标点，无重算、无新线路）
    X_grid, names = build_features_grid(feat)
    X_C = X_grid.reshape(-1, X_grid.shape[-1])[q["idx"]]

    coords = dict(t1=np.asarray(q["t1"], float),
                  t2=np.asarray(q["t2"], float),
                  U=np.asarray(q["U"], float))
    return K_Q, X_C, y, coords, names


# ── 决策图数据（画回 (t1,t2) 平面逐 U 层）─────────────────────────────
def decision_predict(K, y, C, seed=42):
    """在整核上训一个 precomputed-SVM（超参已由 CV 选定），预测所有点，
    返回 (pred, acc)。注：这是对已有 n 点的重构预测，供画决策边界用。"""
    clf = SVC(kernel="precomputed", C=C)
    clf.fit(K, y)
    pred = clf.predict(K)
    return pred, float(np.mean(pred == y))


# ── 主流程 ────────────────────────────────────────────────────────────
def main(M=3, qkm_path=None, seed=42):
    qkm_path = qkm_path or os.path.join(_HERE, f"qkm_grid_M{M}.npz")
    out_npz = OUT_TMPL.format(M=M)

    print("=" * 78)
    print(f"QKM 三栏对决 — qkm_grid_M{M}  |  特征：K_DQAP fidelity + "
          f"{len(FEATURE_NAMES_5)} 个经典多体标量")
    print("=" * 78)

    K_Q, X_C, y, coords, names = load_and_align(qkm_path)
    n = len(y)
    print(f"[1] 对齐：{n} 点（triv={int((y==0).sum())} "
          f"topo={int((y==1).sum())}）  经典特征按 idx 坐标对齐 ✓")

    # ── 诊断 1：PSD ──────────────────────────────────────────────────
    print(f"\n[2] PSD 检查（应 ≥ −1e-9）")
    for lbl, K in [("K_Q", K_Q)]:
        print(f"  {lbl:<6} min_eig = {psd_min_eig(K):+.3e}")

    # ── 栏 ① classical-only ─────────────────────────────────────────
    print(f"\n[3] 栏① 经典-only：RBF-SVM on {X_C.shape[1]} 标量")
    b1 = tune_cv_rbf(X_C, y, seed=seed)
    acc1, g1, C1, cor1, tot1 = b1
    lo1, hi1 = binom_ci(cor1, tot1)
    print(f"  acc={acc1*100:6.1f}%  CI=[{lo1*100:5.1f},{hi1*100:5.1f}]  "
          f"(γ={g1}, C={C1})")

    # 经典 K_C（全量标准化 + 栏①最佳 γ），供 KTA 与混合用
    from sklearn.preprocessing import StandardScaler
    Xc = StandardScaler().fit(X_C).transform(X_C)
    K_C = rbf_matrix(Xc, gamma=g1)

    # ── 栏 ② quantum-only ───────────────────────────────────────────
    print(f"\n[4] 栏② 量子-only：precomputed-SVM on K_DQAP")
    b2 = tune_cv_precomputed(K_Q, y, seed=seed)
    acc2, C2, cor2, tot2 = b2
    lo2, hi2 = binom_ci(cor2, tot2)
    print(f"  acc={acc2*100:6.1f}%  CI=[{lo2*100:5.1f},{hi2*100:5.1f}]  "
          f"(C={C2})")

    # ── 栏 ③ hybrid ─────────────────────────────────────────────────
    print(f"\n[5] 栏③ 混合：K_hyb(w)=w·K̂_Q+(1−w)·K̂_C，w∈{W_GRID}")
    best_w = None
    for w in W_GRID:
        K_h = hybrid_kernel(K_Q, K_C, w)
        acc_w, C_w, cor_w, tot_w = tune_cv_precomputed(K_h, y, seed=seed)
        print(f"  w={w:4.2f}  acc={acc_w*100:6.1f}%  CI=["
              f"{binom_ci(cor_w, tot_w)[0]*100:5.1f},"
              f"{binom_ci(cor_w, tot_w)[1]*100:5.1f}]  (C={C_w})"
              f"  min_eig={psd_min_eig(K_h):+.2e}")
        if best_w is None or acc_w > best_w[0]:
            best_w = (acc_w, w, C_w, cor_w, tot_w)
    acc3, w3, C3, cor3, tot3 = best_w
    lo3, hi3 = binom_ci(cor3, tot3)

    # ── 诊断 2：KTA ──────────────────────────────────────────────────
    print(f"\n[6] KTA（中心化，SVM 前先看哪个源真带相变信息）")
    for lbl, K in [("K_Q", K_Q), ("K_C(经典5)", K_C)]:
        print(f"  {lbl:<12} A = {kta(K, y):+.4f}")
    for i, name in enumerate(names):
        a = Xc[:, i].reshape(-1, 1)
        print(f"  K_C[{name:<6}]  A = {kta(rbf_matrix(a, gamma=g1), y):+.4f}")

    # ── 三栏消融表 ──────────────────────────────────────────────────
    print(f"\n[7] 三栏消融（同 {n} 点、同 5-fold 协议）")
    print("-" * 62)
    rows = [
        ("① 经典-only", acc1, lo1, hi1),
        ("② 量子-only", acc2, lo2, hi2),
        ("③ 混合(w=%.2f)" % w3, acc3, lo3, hi3),
    ]
    for name, a, lo, hi in rows:
        flag = ""
        if name.startswith("③"):
            better = max(acc1, acc2)
            flag = "  ✓ 赢过更好单侧" if a > better else "  ✗ 未赢更好单侧"
        print(f"  {name:<16} acc={a*100:6.1f}%  "
              f"CI=[{lo*100:5.1f},{hi*100:5.1f}]{flag}")
    print("-" * 62)
    print(f"  [判定] 混合须显著赢过更好单侧（CI 不交叠），否则这层复杂度不值")
    print(f"  [框架] gap4=69.5% 是 QKM 参照线；经典多特征=上界参照非栏")

    # ── 决策图数据 ──────────────────────────────────────────────────
    pred2, _ = decision_predict(K_Q, y, C2, seed=seed)
    pred3, _ = decision_predict(hybrid_kernel(K_Q, K_C, w3), y, C3, seed=seed)
    out = dict(M=M, n=int(n),
               t1=coords["t1"], t2=coords["t2"], U=coords["U"], lab=y,
               pred_quantum=pred2, pred_hybrid=pred3,
               results=dict(
                   classical=dict(acc=acc1, ci_lo=lo1, ci_hi=hi1, gamma=g1, C=C1),
                   quantum=dict(acc=acc2, ci_lo=lo2, ci_hi=hi2, C=C2),
                   hybrid=dict(acc=acc3, ci_lo=lo3, ci_hi=hi3, w=w3, C=C3),
                   kta_quantum=kta(K_Q, y), kta_classical=kta(K_C, y)))
    np.savez(out_npz, **out)
    print(f"\n决策图数据 + 结果已存 {os.path.abspath(out_npz)}"
          f"（t1/t2/U/lab/pred_quantum/pred_hybrid，画 (t1,t2) 逐 U 层）")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    p.add_argument("--M", type=int, default=3)
    p.add_argument("--grid", type=str, default=None,
                   help="显式指定 qkm npz 路径（默认 qkm_grid_M{M}.npz）")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    main(M=args.M, qkm_path=args.grid, seed=args.seed)
