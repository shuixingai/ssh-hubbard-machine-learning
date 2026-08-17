"""
Phase 1b: Charge Gap Analysis WITH PCA dimensionality reduction
===============================================================
Pre-reduces the 36-dim correlation matrix to its intrinsic dimension (<= 6)
via PCA, then extracts charge gap labels and visualizes.

Purpose: see if removing the 30 dims of numerical noise improves
the phase separation in t-SNE / PCA space.
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
NPZ_PATH = os.path.join(THIS_DIR, "ssh_dataset_L6.npz")
FIG_DIR  = os.path.join(THIS_DIR, "figures")
os.makedirs(FIG_DIR, exist_ok=True)

data = np.load(NPZ_PATH)
t1_arr, t2_arr, U_arr = data["t1_arr"], data["t2_arr"], data["U_arr"]
corr = data["corr_matrices"]
spectra = data["spectra"]
L = int(data["L"])
N1, N2, N3 = corr.shape[:3]
n_total = N1 * N2 * N3
print(f"Loaded: {N1}x{N2}x{N3} = {n_total} points, L={L}")

# ── Gap labels (same as before) ──
gap = spectra[..., 1] - spectra[..., 0]
thresh_phys = 0.5 * gap[:, :, 0].max()
gap_labels = (gap < thresh_phys).astype(int)
gap_flat = gap.ravel()
gap_labels_flat = gap_labels.ravel()

# ── PCA reduction ──
X_raw = corr.reshape(n_total, L * L)           # (1089, 36)
pca_full = PCA(random_state=42).fit(X_raw)
cumvar = np.cumsum(pca_full.explained_variance_ratio_)
n_pc = np.searchsorted(cumvar, 0.999) + 1      # at least 99.9% var
n_pc = max(4, min(n_pc, L))                     # clamp to [4, L]
print(f"\nPCA: 36 dim -> {n_pc} dim (explains {cumvar[n_pc-1]:.2%} variance)")

pca = PCA(n_components=n_pc, random_state=42)
X_pca = pca.fit_transform(X_raw)

# ── kNN separability (compare raw vs reduced) ──
print("\nSeparability (kNN 5-fold CV, gap label):")
for name, Xf in [("Raw 36-dim", X_raw), (f"PCA {n_pc}-dim", X_pca)]:
    scores = cross_val_score(KNeighborsClassifier(5), Xf, gap_labels_flat, cv=5)
    print(f"  {name:20s}  accuracy = {scores.mean():.4f} +/- {scores.std():.4f}")

# ── t-SNE ──
print("\nComputing t-SNE on PCA-reduced features...")
tsne = TSNE(n_components=2, perplexity=30, random_state=42,
            init="pca", learning_rate="auto")
X_tsne = tsne.fit_transform(X_pca)

# Also PCA-2D for reference
pca_2d = PCA(n_components=2, random_state=42)
X_pca2 = pca_2d.fit_transform(X_raw)
var2d = pca_2d.explained_variance_ratio_

# ── Figure 1: PCA-2D and t-SNE on reduced features ──
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
colors = ["#e74c3c", "#3498db"]
markers = ["o", "^"]

for ax, Xd, title in [
    (axes[0], X_pca2, f"PCA-2D  ({var2d[0]:.1%}+{var2d[1]:.1%})"),
    (axes[1], X_tsne, f"t-SNE on PCA {n_pc}-dim"),
]:
    for code in range(2):
        mask = gap_labels_flat == code
        ax.scatter(Xd[mask, 0], Xd[mask, 1],
                   c=colors[code], marker=markers[code],
                   s=8, alpha=0.7, edgecolors="none",
                   label=f"{['trivial','topological'][code]}")
    ax.set_xlabel("Component 1")
    ax.set_ylabel("Component 2")
    ax.set_title(title, fontsize=11)
    ax.legend(markerscale=3, fontsize=9)

fig.suptitle("SSH-Hubbard: Gap Labels in PCA-Reduced Space",
             fontsize=13, fontweight="bold")
plt.tight_layout()
f1 = os.path.join(FIG_DIR, "pca_reduced_visualization.png")
fig.savefig(f1, dpi=150, bbox_inches="tight")
print(f"Saved -> {f1}")
plt.close(fig)

# ── Figure 2: phase diagrams across U ──
fig, axes = plt.subplots(2, 5, figsize=(20, 7))
U_plot = [0, 2, 4, 6, 8]
for row, (labels, title) in enumerate([
    (gap_labels, f"Gap label (thresh={thresh_phys:.3f})"),
]):
    for col, ui in enumerate(U_plot):
        ax = axes[row][col]
        ax.pcolormesh(t1_arr, t2_arr, labels[:, :, ui].T,
                      shading="auto", cmap="RdYlBu", vmin=0, vmax=1)
        ax.set_xlabel("t1"); ax.set_ylabel("t2")
        ax.set_title(f"U={U_arr[ui]:.1f}", fontsize=11)
        ax.set_aspect("equal")
    axes[row][0].set_ylabel(title, fontsize=10, fontweight="bold")

plt.tight_layout()
f2 = os.path.join(FIG_DIR, "pca_reduced_phases.png")
fig.savefig(f2, dpi=150, bbox_inches="tight")
print(f"Saved -> {f2}")
plt.close(fig)

# ── Figure 3: scree + cumulative var ──
fig, ax = plt.subplots(1, 1, figsize=(8, 4))
n_all = len(pca_full.explained_variance_ratio_)
ax.bar(range(1, n_all+1), pca_full.explained_variance_ratio_,
       alpha=0.6, color="steelblue", label="per component")
ax.plot(range(1, n_all+1), cumvar, "o-", color="darkorange",
        markersize=6, label="cumulative")
ax.axhline(0.999, color="gray", ls=":", alpha=0.7)
ax.axvline(n_pc, color="red", ls="--", label=f"n_PC = {n_pc}")
ax.set_xlabel("Component"); ax.set_ylabel("Variance ratio")
ax.set_title("PCA Explained Variance"); ax.legend(fontsize=9)
plt.tight_layout()
f3 = os.path.join(FIG_DIR, "pca_reduced_scree.png")
fig.savefig(f3, dpi=150, bbox_inches="tight")
print(f"Saved -> {f3}")
plt.close("all")

print("\nDone. Figures in:", FIG_DIR)
