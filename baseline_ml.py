#!/usr/bin/env python3
"""
SSH-Hubbard 经典 baseline —— QKM 的经典参照（memory: gap4-classical-baseline）
=================================================================================
给王远峰的两层对照：
  ① 随机 null（打乱标签）→ 应 ≈ 50%：证明"确实从特征学到了信息"，不是碰运气
  ② 经典 ED 多体量走 ML   → OBC 纠缠谱/关联矩阵标量判别 PBC/TBC ↑-only Z2
     相位拓扑标签（label）。这是 QKM（pilot_gate2 那套量子 kernel）必须跨的栏。

无污染设计（关键，勿破坏）：
  - 特征 ssh_dataset_L8_labelgrid.npz ← ssh_model，OBC ED 多体量（纠缠谱/关联/bond）
  - 标签 topo_dataset_full.npz        ← build_topo_dataset，PBC/TBC Wilson loop
    γ_up snap→π 拓扑 / snap→0 平凡
  两次计算完全独立，只共享 (t1,t2,U) 坐标。标签唯一来源 = topo 的 label 字段。
  禁止用 gap4 自标自；**禁止把 zak_phase/winding_number 等单粒子不变量当特征**
  （≈ γ_up 标签信息 → 泄漏）。

标签编码：0=trivial, 1=topological, 2=critical(能隙闭合,γ 不可靠), 3=unresolved。
二分类主分析剔除 2/3 —— 91 个临界点全在 t1=t2 对角线，属相边界而非相内点。

--feat 模式：
  gap4       单特征（ε5−ε4）——LogReg 复现 69.5% 栏 + RBF-SVM（QKM 参照线）
  ent8       前 8 个纠缠谱 level——完整信息上界对照（=100%）
  classical5 5 个多体标量（gap4/ee/dimer/s_occ/lam_half）→ RBF-SVM + KTA 诊断
             ——经典栏多特征定案（memory §三栏结构：经典侧给足特征，上界参照）

用法：
    python baseline_ml.py --feat gap4          # 复现 69.5% 栏
    python baseline_ml.py --feat classical5    # 经典多特征栏（默认）
输出：控制台表格 + baseline_results_{feat}.npz。
"""

import argparse
import os
import sys

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# 共享 kernel/SVM 层（同目录，两个入口文件共用）
from kernel_ml_utils import (
    binom_ci, kta, psd_min_eig, rbf_matrix,
    run_cv_rbf, shuffle_null, tune_cv_rbf,
)

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
FEAT_NPZ = os.path.join(THIS_DIR, "ssh_dataset_L8_labelgrid.npz")
LABEL_NPZ = os.path.join(THIS_DIR, "topo_dataset_full.npz")
OUT_NPZ_TMPL = os.path.join(THIS_DIR, "baseline_results_{feat}.npz")

N_SPLITS = 5
N_SHUFFLE = 5          # null 打乱次数，平均更稳

# classical5 的 5 个多体标量（全部来自 OBC ED，无单粒子泄漏）
FEATURE_NAMES_5 = ["gap4", "ee", "dimer", "s_occ", "lam_half"]
FEATURE_NOTES_5 = [
    "ES gap ε5−ε4（能隙，69.5% 栏）",
    "ES von Neumann 熵 S=−Σp ln p（最强判别，triv0.45/topo1.90）",
    "bond alternation δB（dimerization / 边缘态指示）",
    "corr 矩阵自然轨道占据熵（=EE via corr matrix，U 指纹）",
    "min|λ−0.5| corr 本征值贴近 1/2（边缘模分数占据，替代恒定的密度 IPR）",
]


# ── 数据加载 + 网格对齐门槛 ───────────────────────────────────────────
def load_aligned():
    feat = np.load(FEAT_NPZ)
    topo = np.load(LABEL_NPZ)
    # key 名不同（t1_arr vs t1_vals），逐轴对账——L 口径坑：
    # ssh_model 的 L 数 site（=8），build_topo 的 L 数元胞（=4 元胞=8 sites）
    assert np.allclose(feat["t1_arr"], topo["t1_vals"]), "t1 网格不对齐"
    assert np.allclose(feat["t2_arr"], topo["t2_vals"]), "t2 网格不对齐"
    assert np.allclose(feat["U_arr"], topo["U_vals"]), "U 网格不对齐"
    return feat, topo


# ── 特征构造 ──────────────────────────────────────────────────────────
def _ee_from_es(ent):
    """ES von Neumann 熵：ent_spectra 存 ε_α=−ln λ_α²，p_α=exp(−ε_α)
    （实测 20 层 Σp∈[0.9998,1.0]，截断可忽略）→ S=−Σ p ln p。"""
    p = np.exp(-ent)
    p = p / p.sum(-1, keepdims=True)
    return -(p * np.log(np.clip(p, 1e-15, None))).sum(-1)


def _corr_eigs(corr):
    """corr 矩阵 (…,8,8) 的本征值，降序。λ=自然轨道占据数。"""
    ev = np.linalg.eigvalsh(corr)            # 升序（实对称）
    return ev[..., ::-1]


def _s_occ_from_corr(corr):
    """占据熵 S_occ=−Σ[λ lnλ + (1−λ)ln(1−λ)]——单粒子密度矩阵的纠缠熵
    （王远峰"EE via corr matrix"；Slater 态=0，边缘模→分数占据→>0）。"""
    l = np.clip(_corr_eigs(corr), 1e-12, 1 - 1e-12)
    return -(l * np.log(l) + (1 - l) * np.log(1 - l)).sum(-1)


def build_features_grid(feat):
    """返回 (13,13,7,d) 特征网格 + 特征名。供 baseline 与 qkm_ml（坐标对齐）共用。"""
    ent = feat["ent_spectra"]
    corr = feat["corr_matrices"]
    gap4 = ent[..., 4] - ent[..., 3]
    ee = _ee_from_es(ent)
    dimer = feat["bond_alternation"]
    ev = _corr_eigs(corr)
    l = np.clip(ev, 1e-12, 1 - 1e-12)
    s_occ = -(l * np.log(l) + (1 - l) * np.log(1 - l)).sum(-1)
    lam_half = np.abs(ev - 0.5).min(-1)
    X_grid = np.stack([gap4, ee, dimer, s_occ, lam_half], axis=-1)
    return X_grid, list(FEATURE_NAMES_5)


def build_features(feat, topo, mode):
    label = topo["label"]            # (13,13,7) int8，唯一标签来源

    if mode == "gap4":
        X_all = (feat["ent_spectra"][..., 4]
                 - feat["ent_spectra"][..., 3]).reshape(-1, 1)
    elif mode == "ent8":
        X_all = feat["ent_spectra"][..., :8].reshape(-1, 8)
    elif mode == "classical5":
        X_grid, _ = build_features_grid(feat)
        X_all = X_grid.reshape(-1, 5)
    else:
        raise ValueError(mode)

    # 二分类：剔除 critical(2)/unresolved(3) → 只留 triv(0)/topo(1)
    mask = ((label == 0) | (label == 1)).reshape(-1)
    X, y = X_all[mask], label.reshape(-1)[mask].astype(int)

    n_triv = int((y == 0).sum())
    n_topo = int((y == 1).sum())
    n_crit = int((label == 2).sum())
    n_unres = int((label == 3).sum())
    print(f"  点计数：triv={n_triv}  topo={n_topo}  剔除 critical={n_crit}"
          f"  unresolved={n_unres}   → 二分类 {len(y)} 点（平衡={n_triv==n_topo}）")
    return X, y


# ── 逐特征诊断表 ──────────────────────────────────────────────────────
def print_feature_diagnostics(X, y, names, gamma_best):
    """逐特征：triv/topo 均值 + KTA + 单特征 RBF-SVM 判别力。"""
    print("\n  [诊断] 逐特征判别力（经典5栏，全量无监督标准化）")
    sc = StandardScaler().fit(X)
    Xs = sc.transform(X)
    print(f"  {'特征':<8}{'triv均值':>12}{'topo均值':>12}"
          f"{'KTA':>10}{'单特征acc':>12}")
    for i, name in enumerate(names):
        a = Xs[:, i]
        acc, correct, total, _ = run_cv_rbf(a.reshape(-1, 1), y,
                                            gamma_best, scale=False)
        triv_m = X[:, i][y == 0].mean()
        topo_m = X[:, i][y == 1].mean()
        print(f"  {name:<8}{triv_m:>12.4f}{topo_m:>12.4f}"
              f"{kta(rbf_matrix(a.reshape(-1,1), gamma=gamma_best), y):>10.4f}"
              f"{acc*100:>11.1f}%")


# ── 主流程 ────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("--feat", choices=["gap4", "ent8", "classical5"],
                    default="classical5")
    args = ap.parse_args()
    out_npz = OUT_NPZ_TMPL.format(feat=args.feat)

    print(f"[1] 加载 + 对齐门槛")
    feat, topo = load_aligned()
    print("  grid aligned ✓  (t1/t2/U 三轴 np.allclose 全过)")

    print(f"[2] 特征构造：mode={args.feat}")
    X, y = build_features(feat, topo, args.feat)

    print(f"[3] RBF-SVM（precomputed kernel，γ×C 在数据上 CV 选，"
          f"{N_SHUFFLE}× null）")
    results = {}
    print("-" * 78)

    if args.feat in ("gap4", "ent8"):
        # 线性参照（旧栏复现）：LogReg（gap4 应回 69.5%）
        for name, clf in [
            ("LogReg", make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))),
        ]:
            acc = correct = total = 0.0
            skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=42)
            c_accs = []
            for tr, te in skf.split(X, y):
                clf.fit(X[tr], y[tr])
                p = clf.predict(X[te])
                c_accs.append(accuracy_score(y[te], p))
                correct += int((p == y[te]).sum())
                total += len(te)
            acc = float(np.mean(c_accs))
            lo, hi = binom_ci(correct, total)
            results[name] = dict(acc=acc, ci_lo=lo, ci_hi=hi,
                                 correct=int(correct), total=int(total))
            print(f"  {name:<12} acc={acc*100:6.1f}%  "
                  f"CI=[{lo*100:5.1f},{hi*100:5.1f}]")

    # RBF-SVM（主模型）
    best = tune_cv_rbf(X, y)
    acc, gamma_best, C_best, correct, total = best
    lo, hi = binom_ci(correct, total)
    run_fn = lambda y2, seed: run_cv_rbf(X, y2, gamma_best, C_best,
                                         seed=seed)[0]
    null_m, null_lo, null_hi = shuffle_null(run_fn, y, N_SHUFFLE)
    delta = acc - null_m
    verdict = ("显著 > null" if lo > null_hi else
               ("弱于/≈ null" if hi < null_lo else "不显著(CI 交叠)"))
    print(f"  RBF-SVM     acc={acc*100:6.1f}%  CI=[{lo*100:5.1f},{hi*100:5.1f}]  "
          f"(γ={gamma_best}, C={C_best})")
    print(f"  null        {null_m*100:6.1f}%±2σ=[{null_lo*100:5.1f},"
          f"{null_hi*100:5.1f}]  Δ={delta*100:+.1f}%  → {verdict}")
    results["RBF-SVM"] = dict(acc=acc, ci_lo=lo, ci_hi=hi, gamma=gamma_best,
                              C=C_best, correct=int(correct), total=int(total),
                              null=null_m, null_lo=null_lo, null_hi=null_hi)

    if args.feat == "classical5":
        print_feature_diagnostics(X, y, FEATURE_NAMES_5, gamma_best)
        # PSD 检查（RBF kernel 结构保证 PSD，数值仍要查）
        sc = StandardScaler().fit(X)
        Kc = rbf_matrix(sc.transform(X), gamma=gamma_best)
        print(f"  [诊断] 整体5特征 KTA = {kta(Kc, y):+.4f}")

        print(f"\n  [诊断] K_C PSD min_eig = {psd_min_eig(Kc):+.3e}"
              f"（应 ≥ −1e-9）")
        print("  [框架警示] 经典多特征=完整信息上界参照（ent8 先例 100%），"
              "不是 QKM 要跨的栏；栏仍是 gap4=69.5%")

    print("-" * 78)
    np.savez(out_npz, feature_mode=args.feat,
             n_triv=int((y == 0).sum()), n_topo=int((y == 1).sum()),
             results=results)
    print(f"结果已存 {out_npz}")


if __name__ == "__main__":
    main()
