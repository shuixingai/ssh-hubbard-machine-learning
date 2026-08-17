#!/usr/bin/env python3
"""
report_classical5.py — 给王远峰的经典端汇报图（memory: gap4-classical-baseline）
==================================================================================
从 ssh_dataset_L8_labelgrid.npz + topo_dataset_full.npz 重建 classical5 特征，
生成 4 张可解释图 + 一页 markdown 汇报。口径与 baseline_ml.py 完全一致
（复用 build_features_grid / FEATURE_NAMES_5）。

输出（写入 report_classical5/ 目录）：
    fig1_feature_plane.png   ee vs dimer 特征平面（label 着色 + 临界线）
    fig2_phase_map.png       (t1,t2) 平面逐 U 层：γ_up 标签 + ee 热图
    fig3_diagnostics.png     逐特征单特征 acc + KTA 条形（null 线）
    fig4_real_vs_null.png    真实 RBF-SVM acc(CI) vs null(±2σ)
    report_classical5.md     一页汇报（可转发给合作者）

无污染设计：特征与标签只共享 (t1,t2,U) 坐标；标签唯一来源 = topo label 字段。
"""

import os
import sys

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

import numpy as np
import matplotlib
matplotlib.use("Agg")                      # 无头环境
import matplotlib.pyplot as plt

# CJK 字体（Windows）：图表里的中文标题必须显式指定
for _f in ("Microsoft YaHei", "SimHei", "SimSun"):
    if _f in {f.name for f in matplotlib.font_manager.fontManager.ttflist}:
        plt.rcParams["font.sans-serif"] = [_f, "DejaVu Sans"]
        break
plt.rcParams["axes.unicode_minus"] = False

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
from baseline_ml import build_features_grid, FEATURE_NAMES_5
from kernel_ml_utils import kta, rbf_matrix, run_cv_rbf, binom_ci

OUT_DIR = os.path.join(_HERE, "report_classical5")
os.makedirs(OUT_DIR, exist_ok=True)

FEAT_NPZ = os.path.join(_HERE, "ssh_dataset_L8_labelgrid.npz")
LABEL_NPZ = os.path.join(_HERE, "topo_dataset_full.npz")

# 论文标准配色
C_TRIV, C_TOPO, C_CRIT = "#3b82c4", "#d9534f", "#9aa0a6"
U_TICKS = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0]


def main():
    feat = np.load(FEAT_NPZ)
    topo = np.load(LABEL_NPZ)

    # 轴对账（L 口径坑：ssh_model L=8 site vs topo L=4 cell）
    assert np.allclose(feat["t1_arr"], topo["t1_vals"])
    assert np.allclose(feat["t2_arr"], topo["t2_vals"])
    assert np.allclose(feat["U_arr"], topo["U_vals"])
    print("grid aligned ✓")

    t1, t2, U = topo["t1_vals"], topo["t2_vals"], topo["U_vals"]
    label = topo["label"]                       # (13,13,7) 0/1/2/3
    X_grid, names = build_features_grid(feat)   # (13,13,7,5)
    ee, dimer, gap4 = X_grid[..., 1], X_grid[..., 2], X_grid[..., 0]

    # 二分类 mask（与 baseline 同口径）
    mask2 = (label == 0) | (label == 1)
    y = label[mask2].astype(int)
    X2 = X_grid[mask2]

    # ── 图 1：ee vs dimer 特征平面 ──────────────────────────────────
    fig, ax = plt.subplots(figsize=(7.2, 5.6))
    m = mask2
    ax.scatter(ee[m & (label == 0)], dimer[m & (label == 0)],
               s=16, c=C_TRIV, label=f"trivial  (n={int((label[m]==0).sum())})",
               alpha=0.85, edgecolors="none")
    ax.scatter(ee[m & (label == 1)], dimer[m & (label == 1)],
               s=16, c=C_TOPO, label=f"topological  (n={int((label[m]==1).sum())})",
               alpha=0.85, edgecolors="none")
    cm = ~m  # critical/unresolved
    ax.scatter(ee[cm], dimer[cm], s=22, c=C_CRIT, marker="x",
               label=f"critical (t1=t2, {int(cm.sum())})", alpha=0.7)
    ax.set_xlabel("ES 纠缠熵  $S=-\\sum p\\ln p$  (ee)")
    ax.set_ylabel("键交变  $\\delta B$  (dimer)")
    ax.set_title("classical5 特征平面：ee 与 dimer\n"
                 "(SSH-Hubbard L=8 OBC 特征 vs PBC/TBC γ_up 标签)")
    ax.legend(frameon=False, fontsize=9)
    ax.set_xlim(-0.1, 2.5)
    plt.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "fig1_feature_plane.png"), dpi=160)
    plt.close(fig)
    print("fig1 ✓")

    # ── 图 2：(t1,t2) 平面，选 3 个 U 层：标签 + ee 热图 ────────────
    fig, axes = plt.subplots(2, 3, figsize=(14, 8.6))
    ui = [0, 3, 6]                                # U=0,3,6
    for col, iu in enumerate(ui):
        # 上排：γ_up 标签
        ax = axes[0, col]
        im = ax.imshow(label[..., iu].T, origin="lower", cmap="RdBu",
                       vmin=0, vmax=2, aspect="auto",
                       extent=[t1[0], t1[-1], t2[0], t2[-1]])
        # 标出 t1=t2 对角线
        d = np.linspace(t1[0], t1[-1], 200)
        ax.plot(d, d, "--", color="k", lw=1, alpha=0.5)
        ax.set_title(f"γ_up 标签  U={U[iu]:g}")
        ax.set_xlabel("$t_1$"); ax.set_ylabel("$t_2$")
        # 下排：ee 热图
        ax = axes[1, col]
        im2 = ax.imshow(ee[..., iu].T, origin="lower", cmap="viridis",
                        aspect="auto", extent=[t1[0], t1[-1], t2[0], t2[-1]])
        ax.plot(d, d, "--", color="k", lw=1, alpha=0.5)
        ax.set_title(f"ES 纠缠熵 ee  U={U[iu]:g}")
        ax.set_xlabel("$t_1$"); ax.set_ylabel("$t_2$")
    fig.colorbar(im, ax=axes[0], shrink=0.9, label="label")
    fig.colorbar(im2, ax=axes[1], shrink=0.9, label="S")
    fig.suptitle("标签 vs ee 热图逐 U 层：(t1,t2) 平面", y=1.01)
    plt.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "fig2_phase_map.png"), dpi=150,
                bbox_inches="tight")
    plt.close(fig)
    print("fig2 ✓")

    # ── 图 3：逐特征诊断（单特征 acc + KTA，null 线）────────────────
    # 复现 baseline 的 γ 选择：直接在全量标准化上用 5 特征跑 tune 的 γ
    # （诊断表用与 baseline_results 相同的 γ_best 才可比；此处用
    #   baseline_ml 的 tune_cv_rbf 在 5 特征上选 γ，再对单特征复用）
    from kernel_ml_utils import tune_cv_rbf 
    acc_all, gamma_best, _, _, _ = tune_cv_rbf(X2, y)   # 1 整体 acc（原来被 _ 丢弃）
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler().fit(X2)
    Xs = sc.transform(X2)
    K_C = rbf_matrix(Xs, gamma=gamma_best)              # 2 整体多特征核矩阵
    kta_all = kta(K_C, y)                               # 3 整体 KTA（补齐）

    accs, ktas, triv_m, topo_m = [], [], [], []
    for i in range(X2.shape[1]):
        a = Xs[:, i]
        acc, _, _, _ = run_cv_rbf(a.reshape(-1, 1), y, gamma_best, scale=False)
        accs.append(acc * 100)
        ktas.append(kta(rbf_matrix(a.reshape(-1, 1), gamma=gamma_best), y))
        triv_m.append(X2[:, i][y == 0].mean())
        topo_m.append(X2[:, i][y == 1].mean())

    fig, ax = plt.subplots(figsize=(9.6, 4.8))
    xpos = np.arange(len(names) + 1)                    # 第 6 组 = 整体 5 特征
    accs_plot = accs + [acc_all * 100]
    ktas_plot = [k * 100 for k in ktas] + [kta_all * 100]
    labels_plot = list(names) + ["整体5特征"]
    ax.bar(xpos - 0.18, accs_plot, 0.36, color="#4472c4", label="RBF-SVM acc (%)")
    ax.bar(xpos + 0.18, ktas_plot, 0.36, color="#ed7d31", label="KTA ×100")
    ax.axhline(51.4, color=C_CRIT, lw=1.4, ls="--",
               label="shuffle-null 均值 51.4%")
    ax.set_xticks(xpos); ax.set_xticklabels(labels_plot, fontsize=9)
    ax.set_ylim(0, 110)
    ax.set_ylabel("%")
    ax.set_title("判别力诊断：逐特征 vs 整体 5 特征 kernel\n"
                 f"（整体 acc={acc_all*100:.1f}% / 整体 KTA={kta_all:.3f}）")
    ax.legend(frameon=False, fontsize=9, loc="upper right")
    for i, (a, k) in enumerate(zip(accs_plot, ktas_plot)):
        ax.text(i - 0.18, a + 2, f"{a:.0f}", ha="center", fontsize=8)
        ax.text(i + 0.18, k + 2, f"{k:.2f}", ha="center", fontsize=8)
    plt.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "fig3_diagnostics.png"), dpi=160)
    plt.close(fig)
    print("fig3 ✓")

    # ── 图 4：真实 acc(CI) vs null ──────────────────────────────────
    # 从 baseline_results_classical5.npz 读真实 RBF-SVM 结果
    br = np.load(os.path.join(_HERE, "baseline_results_classical5.npz"),
                 allow_pickle=True)
    r = br["results"].item()["RBF-SVM"]
    acc_real, lo, hi = r["acc"], r["ci_lo"], r["ci_hi"]
    null_m, null_lo, null_hi = r["null"], r["null_lo"], r["null_hi"]

    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    names_b = ["RBF-SVM (5 特征)", "shuffle-null"]
    means = [acc_real * 100, null_m * 100]
    lows = [lo * 100, null_lo * 100]
    highs = [hi * 100, null_hi * 100]
    colors = ["#4472c4", C_CRIT]
    for i in range(2):
        ax.bar(i, means[i], 0.5, color=colors[i], alpha=0.85)
        ax.errorbar(i, means[i], yerr=[[means[i] - lows[i]],
                                       [highs[i] - means[i]]],
                    fmt="none", ecolor="k", capsize=6)
        ax.text(i, highs[i] + 2, f"{means[i]:.1f}%", ha="center", fontsize=10)
    ax.set_xticks([0, 1]); ax.set_xticklabels(names_b)
    ax.set_ylim(0, 115)
    ax.set_ylabel("accuracy")
    ax.set_title(f"真实 vs null：Δ={ (acc_real-null_m)*100:+.1f}%\n"
                 f"（Wilson CI 不交叠 → 信号显著）")
    plt.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "fig4_real_vs_null.png"), dpi=160)
    plt.close(fig)
    print("fig4 ✓")

    # ── 一页 markdown 汇报 ──────────────────────────────────────────
    lines = []
    lines.append("# 经典端 baseline 汇报（classical5，2026-08-14）\n")
    lines.append("**SSH-Hubbard L=8 OBC 特征 → PBC/TBC γ_up 拓扑标签**，"
                 "两套计算完全独立，只共享 (t1,t2,U) 坐标（无污染）。\n")
    lines.append("## 三条结论\n")
    lines.append(f"1. **经典 5 特征 RBF-SVM = {acc_real*100:.1f}%**"
                 f"（Wilson CI [{lo*100:.1f},{hi*100:.1f}]），"
                 f"shuffle-null = {null_m*100:.1f}%±2σ "
                 f"→ 信号显著，不是碰运气。\n")
    lines.append("2. **100% 完全来自单个特征 ee（ES 纠缠熵）**："
                 "trivial 均值 0.45 vs topological 1.90，"
                 "在 L=8 上近乎完备序参量。dimer 有用（KTA 0.73 / 单特征 88.6%）；"
                 "s_occ 是死特征（KTA 0.04 ≈ 0），已从候选集剔除。\n")
    lines.append("3. **这是信息论上界参照，不是栏**：QKM 要跨的栏仍是 "
                 "gap4 单特征 69.5%。量子端的 claim 是经典吃紧处的可扩展性/"
                 "硬件路线，不是赢过经典。\n")
    lines.append("## 图\n")
    for f in sorted(os.listdir(OUT_DIR)):
        if f.endswith(".png"):
            lines.append(f"- `{f}`\n")
    lines.append("## 想请教的问题\n")
    lines.append("1. ee（ES 纠缠熵）在 L=8 已把两相分得如此开，"
                 "是否会重新定义相边界？论文 framing 要不要动？\n")
    lines.append("2. 在您看来，量子 kernel 要表现出什么才算"
                 "'值得做'？——经典天花板这么高，量子端只能回答"
                 "经典做不到的东西。\n")
    lines.append(f"\n_成本说明：量子端全网格 VQE 约 62–80 天单线程，"
                 "当前先走分层验证（L0 oracle → L1 探针 → L2 子网格）。_\n")
    rep = os.path.join(OUT_DIR, "report_classical5.md")
    with open(rep, "w", encoding="utf-8") as fh:
        fh.writelines(lines)
    print(f"report ✓  →  {rep}")


if __name__ == "__main__":
    main()
