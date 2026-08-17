"""
Phase 2: Entanglement Spectrum Analysis
=========================================
Loads the new dataset (with ent_spectra), analyzes the degeneracy pattern
of the lowest entanglement energies, and evaluates it as a topological
phase label for SSH-Hubbard.

Theory (Ye, Mu & Fan 2016, PRB 94):
  Topological phase -> lowest entanglement level 4-fold degenerate
  Trivial phase     -> lowest entanglement level non-degenerate

Figures saved to:  ./figure_spectrum/
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
FIG_DIR  = os.path.join(THIS_DIR, "figure_spectrum")
os.makedirs(FIG_DIR, exist_ok=True)

data = np.load(NPZ_PATH)
t1_arr, t2_arr, U_arr = data["t1_arr"], data["t2_arr"], data["U_arr"]
corr = data["corr_matrices"]       # (11, 11, 9, 6, 6)
spectra = data["spectra"]           # (11, 11, 9, 20)
ent     = data["ent_spectra"]       # (11, 11, 9, 20)
L = int(data["L"])
N1, N2, N3 = corr.shape[:3]
n_total = N1 * N2 * N3
print(f"Loaded: {N1}x{N2}x{N3} = {n_total} points, L={L}")

# ======================================================================
#  1. Entanglement spectrum structure
# ======================================================================

# Count valid levels (epsilon > 0 means non-zero Schmidt value)
n_valid = np.sum(ent > 1e-10, axis=3)
print(f"\nSchmidt rank: min={n_valid.min()}, max={n_valid.max()}, mean={n_valid.mean():.1f}")
print(f"(stored at most 20 levels -- if rank>20 all slots are used)")

# Gap between consecutive entanglement levels
# ent[:,:,:,k] is the k-th entanglement energy (sorted ascending)
# Small gap between levels = degenerate
gap_1 = ent[:, :, :, 1] - ent[:, :, :, 0]   # gap between level 1 and 0
gap_2 = ent[:, :, :, 2] - ent[:, :, :, 1]   # gap between level 2 and 1
gap_3 = ent[:, :, :, 3] - ent[:, :, :, 2]   # gap between level 3 and 2
gap_4 = ent[:, :, :, 4] - ent[:, :, :, 3]   # gap between level 4 and 3

# Metric: spread of first 4 levels (small = 4-fold degenerate)
spread_4 = ent[:,:,:,3] - ent[:,:,:,0]

# Metric: ratio of gap_4 (after 4th level) to gap_1 (after 1st level)
# Large ratio = 4 levels separated from the rest = 4-fold degenerate
ratio_41 = gap_4 / (gap_1 + 1e-15)

print(f"\nDegeneracy metrics:")
print(f"  gap1 (level 1-0): mean={gap_1.mean():.4f}, range=[{gap_1.min():.4f}, {gap_1.max():.4f}]")
print(f"  gap2 (level 2-1): mean={gap_2.mean():.4f}")
print(f"  gap3 (level 3-2): mean={gap_3.mean():.4f}")
print(f"  gap4 (level 4-3): mean={gap_4.mean():.4f}")
print(f"  spread4:          mean={spread_4.mean():.4f}")
print(f"  ratio_41:         mean={ratio_41.mean():.2f}")

# ======================================================================
#  2. Phase labels from entanglement spectrum
# ======================================================================

# Method A: spread of first 4 levels < threshold -> 4-fold degenerate -> topological
th_spr = np.median(spread_4)
label_spr = (spread_4 < th_spr).astype(int)
print(f"\nEnt spread label: threshold={th_spr:.4f}, top fraction={label_spr.mean():.3f}")

# Method B: gap4 < threshold -> edge states fill the gap -> topological
# (Both phases have 4-fold degenerate lowest level, but topological phase
#  has extra entanglement from edge states, reducing gap4)
th_g4 = np.median(gap_4)
label_g4 = (gap_4 < th_g4).astype(int)  # topological = SMALL gap after 4th level
print(f"Ent gap4 label:   threshold={th_g4:.4f}, top fraction={label_g4.mean():.3f}")

# Method C: gap1 < threshold -> first 2 levels close -> 4-fold degenerate
th_g1 = np.median(gap_1)
label_g1 = (gap_1 < th_g1).astype(int)  # topological = SMALL gap between first 2 levels
print(f"Ent gap1 label:   threshold={th_g1:.4f}, top fraction={label_g1.mean():.3f}")

# Compare: charge gap label
gap_c = spectra[:,:,:,1] - spectra[:,:,:,0]
th_gap = 0.5 * gap_c[:,:,0].max()
label_gap = (gap_c < th_gap).astype(int)
print(f"Gap charge label: threshold={th_gap:.4f}, top fraction={label_gap.mean():.3f}")

# ======================================================================
#  3. Numerical comparisons
# ======================================================================

methods = {
    "gap_charge":  label_gap.ravel(),
    "ent_spread4": label_spr.ravel(),
    "ent_gap4":    label_g4.ravel(),
    "ent_gap1":    label_g1.ravel(),
}
print(f"\n========== Label Comparison ==========")
for name, lbl in methods.items():
    print(f"  {name:20s}: top fraction = {100*lbl.mean():.1f}%")

print(f"\nAgreement rates:")
items = list(methods.items())
for i in range(len(items)):
    for j in range(i+1, len(items)):
        n1, l1 = items[i]
        n2, l2 = items[j]
        agree = (l1 == l2).sum()
        print(f"  {n1:15s} vs {n2:15s}:  {100*agree/n_total:.1f}%")

# PCA reduction
X_raw = corr.reshape(n_total, L*L)
pca_full = PCA(random_state=42).fit(X_raw)
cumvar = np.cumsum(pca_full.explained_variance_ratio_)
n_pc = max(4, min(np.searchsorted(cumvar, 0.999)+1, L))
pca = PCA(n_components=n_pc, random_state=42)
X_pca = pca.fit_transform(X_raw)
print(f"\nPCA: 36 -> {n_pc} dim ({cumvar[n_pc-1]:.2%} variance)")

print(f"\nkNN separability on PCA features:")
for name, lbl in methods.items():
    scores = cross_val_score(KNeighborsClassifier(5), X_pca, lbl, cv=5)
    print(f"  {name:20s}: accuracy={scores.mean():.4f}+/-{scores.std():.4f}")

# ======================================================================
#  4. Visualizations
# ======================================================================

# --- Figure 1: Phase diagrams at selected U ---
U_idx_plot = [0, 2, 4, 6, 8]
nc = len(U_idx_plot)
fig, axes = plt.subplots(4, nc, figsize=(24, 14))
plot_set = [
    (label_gap,  "Charge gap label"),
    (label_spr,  "Ent spread4 label"),
    (label_g4,   "Ent gap4 label"),
    (label_g1,   "Ent gap1 label"),
]
for row, (lbl, title) in enumerate(plot_set):
    for col, ui in enumerate(U_idx_plot):
        ax = axes[row][col]
        ax.pcolormesh(t1_arr, t2_arr, lbl[:,:,ui].T,
                      shading="auto", cmap="RdYlBu", vmin=0, vmax=1)
        ax.set_xlabel("t1"); ax.set_ylabel("t2")
        ax.set_title(f"U={U_arr[ui]:.1f}", fontsize=10)
        ax.set_aspect("equal")
    axes[row][0].set_ylabel(title, fontsize=9, fontweight="bold")
plt.tight_layout()
f1 = os.path.join(FIG_DIR, "phase_diagrams_all.png")
fig.savefig(f1, dpi=150, bbox_inches="tight")
print(f"Saved -> {f1}")
plt.close(fig)

# --- Figure 2: PCA / t-SNE visualization ---
pca2 = PCA(n_components=2, random_state=42)
X_pca2 = pca2.fit_transform(X_raw)
tsne = TSNE(n_components=2, perplexity=30, random_state=42,
            init="pca", learning_rate="auto")
X_tsne = tsne.fit_transform(X_pca)

colors = ["#e74c3c", "#3498db"]
markers = ["o", "^"]

for which_lbl, lbl, title_lbl in [
    ("gap4", label_g4, "Ent gap4 label"),
    ("gap1", label_g1, "Ent gap1 label"),
]:
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for ax, Xd, title in [
        (axes[0], X_pca2, "PCA-2D"),
        (axes[1], X_tsne, f"t-SNE on PCA {n_pc}-dim"),
    ]:
        for code in range(2):
            mask = lbl.ravel() == code
            ax.scatter(Xd[mask,0], Xd[mask,1],
                       c=colors[code], marker=markers[code],
                       s=8, alpha=0.7, edgecolors="none",
                       label=f"{['trivial','topological'][code]}")
        ax.set_xlabel("Comp 1"); ax.set_ylabel("Comp 2")
        ax.set_title(title, fontsize=11)
        ax.legend(markerscale=3, fontsize=9)
    fig.suptitle(f"SSH-Hubbard: {title_lbl}", fontsize=13, fontweight="bold")
    plt.tight_layout()
    f2 = os.path.join(FIG_DIR, f"visualization_{which_lbl}.png")
    fig.savefig(f2, dpi=150, bbox_inches="tight")
    print(f"Saved -> {f2}")
    plt.close(fig)

# --- Figure 3: Gap4 metric landscape ---
if True:
    fig, axes = plt.subplots(3, 3, figsize=(15, 12))
    for idx in range(9):
        row, col = divmod(idx, 3)
        ax = axes[row][col]
        im = ax.pcolormesh(t1_arr, t2_arr, gap_4[:,:,idx].T,
                           shading="auto", cmap="viridis")
        ax.set_xlabel("t1"); ax.set_ylabel("t2")
        ax.set_title(f"U={U_arr[idx]:.1f}", fontsize=11)
        ax.set_aspect("equal")
        plt.colorbar(im, ax=ax, shrink=0.8)
    fig.suptitle("Ent gap4 metric -- large = 4-fold degenerate = topological",
                 fontsize=13, fontweight="bold", y=1.01)
    plt.tight_layout()
    f3 = os.path.join(FIG_DIR, "gap4_landscape.png")
    fig.savefig(f3, dpi=150, bbox_inches="tight")
    print(f"Saved -> {f3}")
    plt.close(fig)

# --- Figure 4: Example entanglement spectra ---
i_list = [(0, -1, 0), (-1, 0, 0), (5, 5, 0)]  # (i1, i2, iU)
pt_labels = [
    "Topological deep (t1=0.5, t2=1.5)",
    "Trivial deep (t1=1.5, t2=0.5)",
    "Near boundary (t1=1.0, t2=1.0)",
]
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
for ax, (i1, i2, iU), lab in zip(axes, i_list, pt_labels):
    vals = ent[i1, i2, iU, :].copy()
    # Detect where padding starts (value = 0)
    valid_mask = vals > 1e-10
    nv = valid_mask.sum()
    if nv > 0:
        ax.plot(range(nv), vals[:nv], "o-", markersize=6)
    ax.axhline(0, color="gray", ls=":", alpha=0.5)
    ax.set_xlabel("Level index"); ax.set_ylabel("epsilon = -ln(lambda^2)")
    ax.set_title(f"{lab}\n(U={U_arr[iU]:.1f})", fontsize=10)
    ax.set_xticks(range(max(nv, 1)))
plt.tight_layout()
f4 = os.path.join(FIG_DIR, "ent_examples.png")
fig.savefig(f4, dpi=150, bbox_inches="tight")
print(f"Saved -> {f4}")
plt.close(fig)

# --- Figure 5: Topological fraction vs U ---
fig, ax = plt.subplots(1, 1, figsize=(8, 5))
for name, lbl in [
    ("Charge gap", label_gap),
    ("Ent spread4", label_spr),
    ("Ent gap4", label_g4),
    ("Ent gap1", label_g1),
]:
    frac = [lbl[:,:,ui].mean() for ui in range(N3)]
    ax.plot(U_arr, frac, "o-", label=name, markersize=6)
ax.set_xlabel("U"); ax.set_ylabel("Fraction topological")
ax.set_title("Topological fraction vs U"); ax.legend(fontsize=9)
plt.tight_layout()
f5 = os.path.join(FIG_DIR, "topological_fraction.png")
fig.savefig(f5, dpi=150, bbox_inches="tight")
print(f"Saved -> {f5}")
plt.close("all")

print(f"\nAll figures saved to: {FIG_DIR}")
