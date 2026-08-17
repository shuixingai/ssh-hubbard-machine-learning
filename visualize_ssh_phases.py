"""
SSH Dataset — PCA & t-SNE Phase Visualization
===============================================
Loads ssh_dataset_L6.npz, performs dimensionality reduction on the
single-particle correlation matrix ρ_ij, and visualizes the topological
vs. trivial phase separation.

Features:  flattened 6×6 correlation matrix (36-dim)  OR  low-energy spectrum (20-dim)
Label:     Resta Z₂ polarisation P ≥ 0.25 → topological (blue),  P < 0.25 → trivial (red)

Figures saved to:  ./figures/  (adjacent to this script)
Interactive:       matplotlib Slider to sweep U
"""

import os

# ── OpenMP 运行时冲突修复 ─────────────────────────────────────
# numpy/scipy via Intel MKL 加载 libiomp，t-SNE/blas 加载 libomp，
# 两者共存会导致随机崩溃、死锁或 t-SNE 降质（点堆叠、颜色混在一起）。
# 必须在 numpy 导入前设置，让 Intel OpenMP 容忍另一个运行时存在。
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
# 可选：限制线程数以减少资源竞争（根据 CPU 核心数调整）
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")

import warnings
# threadpoolctl 的 OpenMP 冲突警告——KMP_DUPLICATE_LIB_OK=TRUE 已保证安全，
# 此警告只是告知性信息，过滤掉以免干扰输出。
warnings.filterwarnings("ignore", message=".*Intel OpenMP.*LLVM OpenMP.*")

import numpy as np
import matplotlib
matplotlib.use("TkAgg")   # interactive backend for local window
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

# ══════════════════════════════════════════════════════════════════════
#  Load dataset
# ══════════════════════════════════════════════════════════════════════

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
NPZ_PATH = os.path.join(THIS_DIR, "ssh_dataset_L6.npz")
FIG_DIR  = os.path.join(THIS_DIR, "figures")
os.makedirs(FIG_DIR, exist_ok=True)

data = np.load(NPZ_PATH)
t1_arr = data["t1_arr"]          # (11,)
t2_arr = data["t2_arr"]          # (11,)
U_arr  = data["U_arr"]           # (9,)
corr   = data["corr_matrices"]   # (11, 11, 9, 6, 6)  — full 3D grid
pol    = data["polarization"]    # (11, 11, 9)         — Resta Z₂ invariant
L      = int(data["L"])

print(f"Loaded:    {NPZ_PATH}")
print(f"Grid:      {len(t1_arr)}×{len(t2_arr)}×{len(U_arr)} = "
      f"{len(t1_arr)*len(t2_arr)*len(U_arr)} points")
print(f"Features:  flattened correlation matrix ({L}×{L} = {L*L} dim)")

# ══════════════════════════════════════════════════════════════════════
#  Flatten to 2D array  (n_points × n_features)
# ══════════════════════════════════════════════════════════════════════

N_t1, N_t2, N_U = corr.shape[:3]
n_total = N_t1 * N_t2 * N_U

# Reshape: (N_t1, N_t2, N_U, L, L) → (n_total, L*L)
X = corr.reshape(n_total, L * L)

# Corresponding parameter values for each row
t1_flat = np.broadcast_to(t1_arr[:, None, None], (N_t1, N_t2, N_U)).ravel()
t2_flat = np.broadcast_to(t2_arr[None, :, None], (N_t1, N_t2, N_U)).ravel()
U_flat  = np.broadcast_to(U_arr[None, None, :],  (N_t1, N_t2, N_U)).ravel()

# ══════════════════════════════════════════════════════════════════════
#  Phase labels via correlation matrix eigenvalues
# ══════════════════════════════════════════════════════════════════════
# Resta polarization with OBC + L=6 gives all-0.5 (finite-size effect).
# Instead we diagonalise ρ_ij at each point and use its eigenvalue
# structure to distinguish topological vs trivial phases.
#
# SSH topological phase  → eigenvalue(s) near 1 (edge states)
# SSH trivial phase      → eigenvalues cluster near 0 and 2 (bulk only)
# With Hubbard U the distinction blurs, but the metric varies continuously.
#
# Two methods:
#   "spread" : mean((n_α - 1)²)  — large = trivial, small = topological
#   "gap"    : n_{L/2} - n_{L/2-1}  — large = trivial, small = topological

def compute_corr_phase_labels(corr_matrices, method="spread", threshold=None):
    """Label phases by diagonalising ρ_ij and examining its eigenvalue structure."""
    N1, N2, N3, L, _ = corr_matrices.shape
    n_total = N1 * N2 * N3
    eig_all = np.linalg.eigvalsh(corr_matrices.reshape(n_total, L, L))  # (n_total, L)

    if method == "spread":
        # Trivial → eigenvalues near 0 or 2 → (n-1)² ≈ 1
        # Topological → eigenvalues spread around 1 → (n-1)² < 1 on average
        metric = np.mean((eig_all - 1.0) ** 2, axis=1)
        if threshold is None:
            threshold = 0.7
    elif method == "gap":
        # Mid-gap at the Fermi level: small → topological, large → trivial
        mid = L // 2
        metric = eig_all[:, mid] - eig_all[:, mid - 1]
        if threshold is None:
            threshold = 1.0
    else:
        raise ValueError(f"Unknown method: {method}")

    label_code_1d = np.where(metric < threshold, 1, 0).astype(int)
    return (label_code_1d.reshape(N1, N2, N3),
            metric.reshape(N1, N2, N3),
            eig_all.reshape(N1, N2, N3, L))


# ── Compute labels with default method ─────────────────────────────
label_code_grid, label_metric_grid, corr_eigvals_grid = compute_corr_phase_labels(
    corr, method="spread", threshold=None,
)
label_code = label_code_grid.ravel()
label_metric_flat = label_metric_grid.ravel()

label_colors = ["#e74c3c", "#3498db"]   # red = trivial, blue = topological
label_markers = ["o", "^"]

# ── Diagnostic ──────────────────────────────────────────────────────
n_topological = (label_code == 1).sum()
n_trivial = (label_code == 0).sum()
print(f"\nPhase labels via corr-eig method 'spread':")
print(f"  topological (label=1) : {n_topological} / {n_total}  ({100*n_topological/n_total:.1f}%)")
print(f"  trivial     (label=0) : {n_trivial} / {n_total}  ({100*n_trivial/n_total:.1f}%)")
print(f"  metric range : [{label_metric_flat.min():.6f}, {label_metric_flat.max():.6f}]")
print(f"  threshold = 0.7  (override via --label-threshold)")
# ══════════════════════════════════════════════════════════════════════
#  ══  STATIC FIGURES  ══
# ══════════════════════════════════════════════════════════════════════

def run_static(which_feature="corr", perplexity=30, random_state=42):
    """Generate and save static PCA + t-SNE scatter plots.

    Parameters
    ----------
    which_feature : "corr" or "spectrum"
        Which feature to use. "corr" uses flattened correlation matrix,
        "spectrum" uses the 20 lowest eigenvalues.
    """
    if which_feature == "corr":
        X_feat = X
        feat_label = "correlation matrix (36 dim)"
    else:
        spectra = data["spectra"].reshape(n_total, -1)   # (n_total, 20)
        X_feat = spectra
        feat_label = "low-energy spectrum (20 dim)"

    # ── PCA ──────────────────────────────────────────────────────────
    pca = PCA(n_components=2, random_state=random_state)
    X_pca = pca.fit_transform(X_feat)
    var_explained = pca.explained_variance_ratio_

    # ── t-SNE ────────────────────────────────────────────────────────
    tsne = TSNE(n_components=2, perplexity=perplexity,
                random_state=random_state, init="pca", learning_rate="auto")
    X_tsne = tsne.fit_transform(X_feat)

    # 检查 t-SNE / PCA 结果是否包含 NaN / inf 或全零
    for name, X_trans in [("PCA", X_pca), ("t-SNE", X_tsne)]:
        if np.any(np.isnan(X_trans)) or np.any(np.isinf(X_trans)):
            print(f"  ⚠ {name} 输出包含 NaN 或 inf，绘图可能异常！")
        ptp = np.ptp(X_trans, axis=0)  # max-min per component
        if np.any(ptp < 1e-10):
            print(f"  ⚠ {name} 各分量值几乎无变化 (range = {ptp})，所有点可能堆叠在一起")

    # ── Plot ─────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(f"SSH Phase Separation  (feature: {feat_label})",
                 fontsize=13, fontweight="bold")

    titles = [
        f"PCA  ({var_explained[0]:.1%} + {var_explained[1]:.1%} variance)",
        "t-SNE",
    ]
    for ax, X_trans, title in zip(axes, [X_pca, X_tsne], titles):
        for code in range(2):
            mask = label_code == code
            ax.scatter(
                X_trans[mask, 0], X_trans[mask, 1],
                c=label_colors[code], marker=label_markers[code],
                s=8, alpha=0.7, edgecolors="none",
                label=f"{['trivial','topological'][code]}",
            )
        ax.set_xlabel("Component 1")
        ax.set_ylabel("Component 2")
        ax.set_title(title)
        ax.legend(markerscale=3, fontsize=9)

    plt.tight_layout()
    fname = f"ssh_phases_{which_feature}.png"
    path = os.path.join(FIG_DIR, fname)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"  Saved → {path}")
    plt.close(fig)


# ══════════════════════════════════════════════════════════════════════
#  ══  INTERACTIVE FIGURE  ══   (matplotlib Slider for U)
# ══════════════════════════════════════════════════════════════════════

def run_interactive(which_feature="corr", perplexity=30):
    """Interactive window: slider sweeps U, PCA/t-SNE update in real time."""
    if which_feature == "corr":
        X_feat = X
    else:
        X_feat = data["spectra"].reshape(n_total, -1)

    # Pre-compute t-SNE for each U slice separately so we see per-U structure
    # Actually, to be interactively fluid, we run t-SNE once per U (9 times)
    # and cache. PCA can be re-fit on-the-fly, but let's also pre-compute.

    print("\nPre-computing PCA & t-SNE for each U slice ...")
    n_U_local = len(U_arr)
    slices_pca = []
    slices_tsne = []
    for ui in range(n_U_local):
        mask = np.abs(U_flat - U_arr[ui]) < 1e-12
        X_slice = X_feat[mask]          # (N_t1*N_t2, d)
        # PCA
        pca_s = PCA(n_components=2, random_state=42)
        pca_out = pca_s.fit_transform(X_slice)
        slices_pca.append(pca_out)
        # t-SNE
        tsne_s = TSNE(n_components=2, perplexity=perplexity,
                      random_state=42, init="pca", learning_rate="auto")
        tsne_out = tsne_s.fit_transform(X_slice)
        slices_tsne.append(tsne_out)

    # ── Build figure with slider ────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.subplots_adjust(bottom=0.18)   # space for slider

    # Initial slice (U = U_arr[0])
    ui0 = 0
    scatter_pca = []
    scatter_tsne = []
    for ax, slices in zip(axes, [slices_pca, slices_tsne]):
        for code in range(3):
            mask = label_code.reshape(N_t1, N_t2, n_U_local)[:, :, ui0].ravel() == code
            # mask is per-slice: we need to index into the slice results
            # The slice X_slice corresponds to fixed U, so we use label_code[ui_slice]
            pass

    # Rebuild: for each U slice, we have 1089/9 = 121 points
    # label_code for that slice is label_code.reshape(N_t1,N_t2,N_U)[:,:,ui].ravel()
    ui0 = 0
    lbl_slice = label_code.reshape(N_t1, N_t2, n_U_local)[:, :, ui0].ravel()
    lbl_slice_flat = lbl_slice  # (121,)

    scatters_pca = []
    scatters_tsne = []
    for code in range(2):
        mask = lbl_slice_flat == code
        s_p = axes[0].scatter(
            slices_pca[ui0][mask, 0], slices_pca[ui0][mask, 1],
            c=label_colors[code], marker=label_markers[code],
            s=20, alpha=0.8, edgecolors="none",
            label=f"{['trivial','topological'][code]}",
        )
        scatters_pca.append(s_p)
        s_t = axes[1].scatter(
            slices_tsne[ui0][mask, 0], slices_tsne[ui0][mask, 1],
            c=label_colors[code], marker=label_markers[code],
            s=20, alpha=0.8, edgecolors="none",
            label=f"{['trivial','topological'][code]}",
        )
        scatters_tsne.append(s_t)

    axes[0].set_title(f"PCA  (U = {U_arr[ui0]:.1f})")
    axes[1].set_title(f"t-SNE  (U = {U_arr[ui0]:.1f})")
    for ax in axes:
        ax.set_xlabel("Component 1")
        ax.set_ylabel("Component 2")
        ax.legend(markerscale=2, fontsize=9)

    # ── Slider ──────────────────────────────────────────────────────
    ax_slider = fig.add_axes([0.2, 0.05, 0.6, 0.04])
    slider = Slider(ax=ax_slider, label="U", valmin=0, valmax=n_U_local - 1,
                    valinit=ui0, valstep=1)

    def update(ui):
        ui = int(ui)
        lbl = label_code.reshape(N_t1, N_t2, n_U_local)[:, :, ui].ravel()
        for code in range(2):
            mask = lbl == code
            scatters_pca[code].set_offsets(
                np.column_stack([slices_pca[ui][mask, 0], slices_pca[ui][mask, 1]])
            )
            scatters_tsne[code].set_offsets(
                np.column_stack([slices_tsne[ui][mask, 0], slices_tsne[ui][mask, 1]])
            )
        axes[0].set_title(f"PCA  (U = {U_arr[ui]:.1f})")
        axes[1].set_title(f"t-SNE  (U = {U_arr[ui]:.1f})")
        fig.canvas.draw_idle()

    slider.on_changed(update)
    plt.show()
    return fig


# ══════════════════════════════════════════════════════════════════════
#  CLI
# ══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="SSH phase visualization")
    parser.add_argument("--feature", choices=["corr", "spectrum"],
                        default="corr",
                        help="Feature type (default: correlation matrix)")
    parser.add_argument("--perplexity", type=int, default=30,
                        help="t-SNE perplexity (default: 30)")
    parser.add_argument("--interactive", action="store_true",
                        help="Open interactive window with U slider")
    parser.add_argument("--label-method", choices=["spread", "gap"],
                        default="spread",
                        help="Phase labeling method via corr eigenvalues (default: spread)")
    parser.add_argument("--label-threshold", type=float, default=None,
                        help="Classification threshold (default: auto, 0.7 for spread, 1.0 for gap)")
    args = parser.parse_args()

    # Recompute phase labels if CLI overrides defaults
    if args.label_method != "spread" or args.label_threshold is not None:
        label_code_grid, label_metric_grid, corr_eigvals_grid = compute_corr_phase_labels(
            corr, method=args.label_method, threshold=args.label_threshold,
        )
        # Update module-level globals that run_static / run_interactive read
        import sys
        mod = sys.modules[__name__]
        mod.label_code_grid = label_code_grid
        mod.label_metric_grid = label_metric_grid
        mod.corr_eigvals_grid = corr_eigvals_grid
        mod.label_code = label_code_grid.ravel()
        mod.label_metric_flat = label_metric_grid.ravel()
        n_top = (label_code == 1).sum()
        n_triv = (label_code == 0).sum()
        print(f"\n[Recomputed labels: method='{args.label_method}', threshold={args.label_threshold}]")
        print(f"  topological (label=1) : {n_top} / {n_total}  ({100*n_top/n_total:.1f}%)")
        print(f"  trivial     (label=0) : {n_triv} / {n_total}  ({100*n_triv/n_total:.1f}%)")
        print(f"  metric range : [{label_metric_flat.min():.6f}, {label_metric_flat.max():.6f}]")
        del sys

    print("=" * 60)
    print("SSH Dataset — Phase Visualization")
    print("=" * 60)

    # Static figures
    run_static(which_feature=args.feature, perplexity=args.perplexity)
    print(f"  Figures saved to: {FIG_DIR}/")

    # Interactive (optional)
    if args.interactive:
        print("\nOpening interactive window ...")
        run_interactive(which_feature=args.feature, perplexity=args.perplexity)

    print("\nDone.")
