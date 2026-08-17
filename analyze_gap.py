"""
Phase 1: Charge Gap Analysis — from existing ssh_dataset_L6.npz

Extracts the charge gap Δ_c = E₁ − E₀ from the saved low-energy spectrum,
maps it across the t1-t2-U grid, and evaluates its suitability as a
topological phase label.

Key questions:
  1. Does Δ_c close along the t1 = t2 line at U = 0?  (sanity check)
  2. Does the gap-closing line shift with U?
  3. Can Δ_c (or a thresholded version of it) serve as a better label
     for ML-based phase recognition than Resta or corr spread?
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

# ── Paths ──────────────────────────────────────────────────────────────
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
NPZ_PATH = os.path.join(THIS_DIR, "ssh_dataset_L6.npz")
FIG_DIR  = os.path.join(THIS_DIR, "figures")
os.makedirs(FIG_DIR, exist_ok=True)

# ── Load ───────────────────────────────────────────────────────────────
data = np.load(NPZ_PATH)
t1_arr = data["t1_arr"]           # (11,)
t2_arr = data["t2_arr"]           # (11,)
U_arr  = data["U_arr"]            # (9,)
corr   = data["corr_matrices"]    # (11, 11, 9, 6, 6)
pol    = data["polarization"]     # (11, 11, 9)
spectra = data["spectra"]         # (11, 11, 9, 20)
energies = data["energies"]       # (11, 11, 9)
L = int(data["L"])

N1, N2, N3 = corr.shape[:3]
n_total = N1 * N2 * N3
print(f"Loaded {NPZ_PATH}")
print(f"Grid: {N1}×{N2}×{N3} = {n_total} points, L={L}")

# ══════════════════════════════════════════════════════════════════════
#  1. Charge gap from saved spectra
# ══════════════════════════════════════════════════════════════════════

# spectra[:, :, :, 0]  = ground-state energy (should match energies)
# spectra[:, :, :, 1]  = first excited state
# Δ_c = E₁ − E₀
E0 = spectra[..., 0]          # (11, 11, 9)
E1 = spectra[..., 1]          # (11, 11, 9)
gap = E1 - E0                 # (11, 11, 9)  charge gap

# Sanity: E0 from spectra should match energies array
max_dev = np.max(np.abs(E0 - energies))
print(f"\nSanity check: max |spectra[0] - energies| = {max_dev:.2e}")
gap_flat = gap.ravel()

# ══════════════════════════════════════════════════════════════════════
#  2. Phase labels from gap thresholding
# ══════════════════════════════════════════════════════════════════════
# Topological phase → gap closes near t1=t2 (small gap)
# Trivial phase → larger gap
# We'll determine a reasonable threshold

def label_from_gap(gap_grid, threshold=None):
    """Label: 1 = topological (small gap), 0 = trivial (large gap)."""
    if threshold is None:
        # Use median as automatic threshold
        threshold = np.median(gap_grid)
        print(f"  Auto threshold = median gap = {threshold:.6f}")
    labels = (gap_grid < threshold).astype(int)
    return labels, threshold

gap_labels_auto, thresh_auto = label_from_gap(gap)

# Also try: threshold at half the max gap at U=0 (physical: gap closing
# at t1=t2 should be well below bulk gap)
gap_U0 = gap[:, :, 0]
thresh_phys = 0.5 * gap_U0.max()
gap_labels_phys = (gap < thresh_phys).astype(int)
print(f"  Physical threshold (0.5 × max gap at U=0) = {thresh_phys:.6f}")

# ══════════════════════════════════════════════════════════════════════
#  3. Comparison with previous labeling schemes
# ══════════════════════════════════════════════════════════════════════

# Resta polarization label (original, but broken at L=6)
labels_pol = (pol.ravel() >= 0.25).astype(int).reshape(N1, N2, N3)

# Corr spread label (current, but actually measures correlation)
eig_all = np.linalg.eigvalsh(corr.reshape(n_total, L, L))
spread = np.mean((eig_all - 1.0) ** 2, axis=1)
labels_corr = (spread < 0.7).astype(int).reshape(N1, N2, N3)

# Print comparison stats
print("\n========== Label Comparison ==========")
methods = {
    "Resta P≥0.25": labels_pol,
    "corr spread<0.7": labels_corr,
    f"gap<{thresh_phys:.4f} (phys)": gap_labels_phys,
    f"gap<{thresh_auto:.4f} (median)": gap_labels_auto,
}
for name, labels in methods.items():
    n_top = (labels.ravel() == 1).sum()
    print(f"  {name:30s}:  top={n_top}/{n_total}  ({100*n_top/n_total:.1f}%)")

# Agreement matrices
print("\nAgreement rates between methods:")
method_items = list(methods.items())
for i in range(len(method_items)):
    for j in range(i+1, len(method_items)):
        n1, l1 = method_items[i]
        n2, l2 = method_items[j]
        agree = (l1.ravel() == l2.ravel()).sum()
        print(f"  {n1:25s} vs {n2:25s}:  {100*agree/n_total:.1f}%")

# ══════════════════════════════════════════════════════════════════════
#  4. Visualizations
# ══════════════════════════════════════════════════════════════════════

def plot_phase_diagrams():
    """t1-t2 phase diagram at selected U values for each label method."""
    nrows = 3
    ncols = 5
    U_plot_idx = [0, 2, 4, 6, 8]  # U = 0, 1, 2, 3, 4
    U_plot_vals = [U_arr[i] for i in U_plot_idx]

    fig, axes = plt.subplots(nrows, ncols, figsize=(22, 12))

    methods_plot = [
        ("Resta P ≥ 0.25", labels_pol, "RdYlBu"),
        ("Corr spread < 0.7", labels_corr, "RdYlBu"),
        (f"Gap Δ_c < {thresh_phys:.3f}", gap_labels_phys, "RdYlBu"),
    ]

    for row, (name, labels, cmap) in enumerate(methods_plot):
        for col, ui in enumerate(U_plot_idx):
            ax = axes[row][col]
            ax.pcolormesh(t1_arr, t2_arr, labels[:, :, ui].T,
                          shading="auto", cmap=cmap, vmin=0, vmax=1)
            ax.set_xlabel("t₁")
            ax.set_ylabel("t₂")
            ax.set_title(f"U = {U_arr[ui]:.1f}", fontsize=11)
            ax.set_aspect("equal")

        axes[row][0].set_ylabel(name, fontsize=11, fontweight="bold")

    plt.tight_layout()
    path = os.path.join(FIG_DIR, "gap_phase_diagrams.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"\nSaved → {path}")
    plt.close(fig)


def plot_gap_landscape():
    """Charge gap Δ_c as a function of t1, t2 at each U."""
    ncols = 3
    nrows = 3
    fig, axes = plt.subplots(nrows, ncols, figsize=(15, 12))

    vmin, vmax = gap.min(), gap.max()

    for idx in range(9):
        row, col = divmod(idx, ncols)
        ax = axes[row][col]
        im = ax.pcolormesh(t1_arr, t2_arr, gap[:, :, idx].T,
                           shading="auto", cmap="viridis",
                           vmin=vmin, vmax=vmax)
        ax.set_xlabel("t₁")
        ax.set_ylabel("t₂")
        ax.set_title(f"U = {U_arr[idx]:.1f}", fontsize=11)
        ax.set_aspect("equal")
        plt.colorbar(im, ax=ax, shrink=0.8)

    fig.suptitle("Charge Gap Δ_c = E₁ − E₀  across t₁-t₂-U space",
                 fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()
    path = os.path.join(FIG_DIR, "gap_landscape.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"Saved → {path}")
    plt.close(fig)


def plot_gap_vs_t1_div_t2():
    """Gap value along lines through parameter space:
       - Along t1 = t2 (should close at U=0)
       - Along t1 = const while varying t2
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # (a) Gap along t1 = t2 diagonal
    ax = axes[0]
    diag_mask = np.zeros((N1, N2), dtype=bool)
    for i in range(N1):
        for j in range(N2):
            if abs(t1_arr[i] - t2_arr[j]) < 1e-10:
                diag_mask[i, j] = True
    # Actually for our grid t1 = t2 only at index (5,5)
    # Better: sweep along slicing where t1 ≈ t2
    for ui, u_val in enumerate(U_arr):
        gap_diag = gap[:, :, ui][diag_mask]
        if len(gap_diag) > 0:
            ax.scatter(u_val, gap_diag[0], s=60, c="C0", zorder=5)
    # Also show gap along the diagonal for all t1=t2 points
    # The grid has 11×11×9 = same linspace, find where t1 == t2
    t1g, t2g = np.meshgrid(t1_arr, t2_arr, indexing="ij")
    on_diag = np.abs(t1g - t2g) < 1e-10
    for ui, u_val in enumerate(U_arr):
        vals_on_diag = gap[:, :, ui][on_diag]
        if len(vals_on_diag) > 0:
            ax.scatter([u_val]*len(vals_on_diag), vals_on_diag,
                       c="C0", alpha=0.5, s=20)
    ax.set_xlabel("U")
    ax.set_ylabel("Δ_c at t₁ = t₂")
    ax.set_title("Gap closing along the t₁ = t₂ line")
    ax.axhline(0, color="gray", linestyle="--", alpha=0.5)

    # (b) Gap as function of t2 - t1 for different U
    ax = axes[1]
    t2_minus_t1 = t2g - t1g
    # Pick slices at several U
    for ui in [0, 2, 4, 8]:
        ax.plot(t2_minus_t1.ravel(), gap[:, :, ui].ravel(),
                "o", markersize=2, alpha=0.5,
                label=f"U={U_arr[ui]:.1f}")
    ax.set_xlabel("t₂ − t₁")
    ax.set_ylabel("Δ_c")
    ax.set_title("Gap vs dimerization")
    ax.legend(fontsize=8)
    ax.axvline(0, color="gray", linestyle="--", alpha=0.5)

    # (c) Gap distribution (histogram) per U
    ax = axes[2]
    for ui in [0, 2, 4, 8]:
        ax.hist(gap[:, :, ui].ravel(), bins=20, alpha=0.4,
                label=f"U={U_arr[ui]:.1f}")
    ax.set_xlabel("Δ_c")
    ax.set_ylabel("count")
    ax.set_title("Gap distribution per U")
    ax.legend(fontsize=8)

    plt.tight_layout()
    path = os.path.join(FIG_DIR, "gap_statistics.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"Saved → {path}")
    plt.close(fig)


def plot_pca_tsne_gap():
    """PCA & t-SNE projected onto 2D, colored by gap value (continuous)."""
    # Flatten correlation matrix features (same as original visualize code)
    X = corr.reshape(n_total, L * L)

    # PCA
    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X)
    var_exp = pca.explained_variance_ratio_

    # t-SNE
    tsne = TSNE(n_components=2, perplexity=30, random_state=42,
                init="pca", learning_rate="auto")
    X_tsne = tsne.fit_transform(X)

    # Plot colored by continuous gap value
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for ax, X_trans, title in zip(
        axes, [X_pca, X_tsne],
        [f"PCA ({var_exp[0]:.1%}+{var_exp[1]:.1%})", "t-SNE"]
    ):
        sc = ax.scatter(X_trans[:, 0], X_trans[:, 1], c=gap_flat,
                        s=8, alpha=0.7, cmap="viridis", edgecolors="none")
        ax.set_xlabel("Component 1")
        ax.set_ylabel("Component 2")
        ax.set_title(title)
        plt.colorbar(sc, ax=ax, label="Δ_c", shrink=0.8)

    fig.suptitle("SSH-Hubbard: PCA & t-SNE colored by Charge Gap Δ_c",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    path = os.path.join(FIG_DIR, "pca_tsne_gap_color.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"Saved → {path}")

    # Also: scatter colored by binarized gap label
    fig2, axes2 = plt.subplots(1, 2, figsize=(14, 6))
    gap_labels_flat = gap_labels_phys.ravel()
    colors = ["#e74c3c", "#3498db"]
    markers = ["o", "^"]

    for ax, X_trans, title in zip(
        axes2, [X_pca, X_tsne],
        [f"PCA ({var_exp[0]:.1%}+{var_exp[1]:.1%})", "t-SNE"]
    ):
        for code in range(2):
            mask = gap_labels_flat == code
            ax.scatter(
                X_trans[mask, 0], X_trans[mask, 1],
                c=colors[code], marker=markers[code],
                s=8, alpha=0.7, edgecolors="none",
                label=f"{['trivial','topological'][code]}",
            )
        ax.set_xlabel("Component 1")
        ax.set_ylabel("Component 2")
        ax.set_title(title)
        ax.legend(markerscale=3, fontsize=9)

    fig2.suptitle("SSH-Hubbard: PCA & t-SNE colored by Gap-based Label",
                  fontsize=13, fontweight="bold")
    plt.tight_layout()
    path = os.path.join(FIG_DIR, "pca_tsne_gap_label.png")
    fig2.savefig(path, dpi=150, bbox_inches="tight")
    print(f"Saved → {path}")
    plt.close("all")


def gap_response_to_parameters():
    """Quantify how much gap varies with each parameter.
    This tells us whether Δ_c is responding to t1/t2 (good for topology)
    or mostly to U (bad — same problem as corr spread)."""
    gap_reshaped = gap.reshape(N1, N2, N3)

    print("\n========== Gap response analysis ==========")
    print("Variance of gap explained by each parameter:\n")

    # Gap variation with t1: fix t2, U → std across t1
    std_across_t1 = np.std(gap_reshaped, axis=0)  # (N2, N3)
    std_across_t2 = np.std(gap_reshaped, axis=1)  # (N1, N3)
    std_across_U  = np.std(gap_reshaped, axis=2)  # (N1, N2)

    print(f"  Mean std across t1 (fixed t2,U): {std_across_t1.mean():.6f}")
    print(f"  Mean std across t2 (fixed t1,U): {std_across_t2.mean():.6f}")
    print(f"  Mean std across U  (fixed t1,t2): {std_across_U.mean():.6f}")

    # Also: mean gap at each U
    print("\n  Mean gap at each U:")
    for ui, u_val in enumerate(U_arr):
        print(f"    U={u_val:.1f}:  mean gap = {gap_reshaped[:,:,ui].mean():.6f}  "
              f"(top half of t1-t2 plane: {gap_reshaped[N1//2:,:,ui].mean():.6f}, "
              f"bottom half: {gap_reshaped[:N1//2,:,ui].mean():.6f})")


# ══════════════════════════════════════════════════════════════════════
#  Run all analyses
# ══════════════════════════════════════════════════════════════════════

print("\n" + "="*60)
print("Phase 1: Charge Gap Analysis")
print("="*60)

plot_phase_diagrams()
plot_gap_landscape()
plot_gap_vs_t1_div_t2()
plot_pca_tsne_gap()
gap_response_to_parameters()

print("\nAll figures saved to:", FIG_DIR)
