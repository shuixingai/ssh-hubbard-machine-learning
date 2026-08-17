"""
Phase 4: Two-panel physical diagnostics
==========================================
Loads the ED dataset and produces 2 figures:
  1. gap4           — entanglement spectrum gap (topological marker)
  2. double occ.    — mean ⟨n_{i↑}n_{i↓}⟩ (Mott marker)

Charge gap moved to separate DMRG-based figure (grid mismatch).
Bond order postponed (L=6 boundary effect unclear).
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
NPZ_PATH = os.path.join(THIS_DIR, "ssh_dataset_L6_4picture.npz")

# ── Load ───────────────────────────────────────────────────────────────
data = np.load(NPZ_PATH)
t1_arr, t2_arr, U_arr = data["t1_arr"], data["t2_arr"], data["U_arr"]
corr = data["corr_matrices"]
ent = data["ent_spectra"]
docc = data["double_occupancy"]
L = int(data["L"])

N1, N2, N3 = corr.shape[:3]
n_total = N1 * N2 * N3
print(f"Loaded: {N1}x{N2}x{N3} = {n_total}, L={L}")

# ══════════════════════════════════════════════════════════════════════
#  Compute indicators
# ══════════════════════════════════════════════════════════════════════

gap4 = ent[:, :, :, 4] - ent[:, :, :, 3]
docc_mean = docc.mean(axis=3)

gap4_flat = gap4.ravel()
docc_flat = docc_mean.ravel()

# ══════════════════════════════════════════════════════════════════════
#  PCA + t-SNE
# ══════════════════════════════════════════════════════════════════════

X = corr.reshape(n_total, L * L)
pca_2d = PCA(n_components=2, random_state=42)
X_pca = pca_2d.fit_transform(X)
var_pca = pca_2d.explained_variance_ratio_

tsne = TSNE(n_components=2, perplexity=30, random_state=42,
            init="pca", learning_rate="auto")
X_tsne = tsne.fit_transform(X)

print(f"\nPCA: 2 dim ({var_pca[0]:.1%} + {var_pca[1]:.1%} = {var_pca.sum():.1%} variance)")

# ══════════════════════════════════════════════════════════════════════
#  Plot 2 figures
# ══════════════════════════════════════════════════════════════════════

configs = [
    (gap4_flat,  "gap4",  "gap4 = epsilon4 - epsilon3",  "viridis"),
    (docc_flat,  "double_occ",  "Double occupancy <n_up n_dn>", "magma"),
]

for arr, name, label, cmap in configs:
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for ax, Xd, xlabel in [
        (axes[0], X_pca, f"PCA ({var_pca[0]:.1%}+{var_pca[1]:.1%})"),
        (axes[1], X_tsne, "t-SNE"),
    ]:
        sc = ax.scatter(Xd[:, 0], Xd[:, 1], c=arr,
                        s=8, alpha=0.6, cmap=cmap, edgecolors="none")
        ax.set_xlabel("Component 1")
        ax.set_ylabel("Component 2")
        ax.set_title(xlabel, fontsize=11)
        plt.colorbar(sc, ax=ax, label=label, shrink=0.8)

    fig.suptitle(f"SSH-Hubbard: {label}", fontsize=13, fontweight="bold")
    plt.tight_layout()
    fpath = os.path.join(THIS_DIR, f"pca_{name}.png")
    fig.savefig(fpath, dpi=150, bbox_inches="tight")
    print(f"Saved -> {fpath}")
    plt.close(fig)

print(f"\nAll figures saved to: {THIS_DIR}")
