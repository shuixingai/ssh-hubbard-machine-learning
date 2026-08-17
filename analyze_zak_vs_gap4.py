"""
Phase 6: Zak Phase vs gap4 — Consistency Check
================================================
Loads ssh_dataset_L6_refined.npz, compares SP Zak phase (binary 0/π
topological label) against the entanglement spectrum gap4 (continuous
topological marker).



the final code is "python dqap_ssh_hubbard.py 4 3 grow 3 6"



Three-panel figure:
  1. PCA scatter colored by Zak phase
  2. PCA scatter colored by gap4
  3. t1-t2 phase diagram: gap4 heatmap + Zak boundary overlay
     (one row per U value, 3 selected U slices)
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
NPZ_PATH = os.path.join(THIS_DIR, "ssh_dataset_L6_refined.npz")
FIG_DIR  = os.path.join(THIS_DIR, "figures")
os.makedirs(FIG_DIR, exist_ok=True)

# ── Load ───────────────────────────────────────────────────────────────
data = np.load(NPZ_PATH)
t1_arr, t2_arr, U_arr = data["t1_arr"], data["t2_arr"], data["U_arr"]
corr    = data["corr_matrices"]   # (N1, N2, N3, 6, 6)
ent     = data["ent_spectra"]     # (N1, N2, N3, 20)
zak     = data["zak_phase"]       # (N1, N2, N3)
zak_ov  = data["zak_overlap_min"] # (N1, N2, N3)

N1, N2, N3 = corr.shape[:3]
n_total = N1 * N2 * N3
L = int(data["L"])
print(f"Loaded: {N1}×{N2}×{N3} = {n_total}, L={L}")

# ── Derived quantities ─────────────────────────────────────────────────
gap4 = ent[:, :, :, 4] - ent[:, :, :, 3]  # (N1, N2, N3)
# Zak phase: binary labels
zak_label = np.where(np.abs(zak - np.pi) < 1.0, 1.0, 0.0)  # 1=topo, 0=triv
# Unreliable region: near gap-closing where overlap is low
unreliable = zak_ov < 0.95

# Flatten for scatter
gap4_flat  = gap4.ravel()
zak_flat   = zak.ravel()
zak_label_flat = zak_label.ravel()
unreliable_flat = unreliable.ravel()

# PCA on correlation matrices
X = corr.reshape(n_total, L * L)
pca_2d = PCA(n_components=2, random_state=42)
X_pca = pca_2d.fit_transform(X)
var_pca = pca_2d.explained_variance_ratio_

tsne = TSNE(n_components=2, perplexity=40, random_state=42,
            init="pca", learning_rate="auto")
X_tsne = tsne.fit_transform(X)

print(f"PCA: {var_pca[0]:.1%}+{var_pca[1]:.1%} = {var_pca.sum():.1%}")
print(f"Zak: topo={zak_label.sum():.0f}  triv={(1-zak_label).sum():.0f}")

# ── Figure 1: PCA + t-SNE, side-by-side comparison ────────────────────
fig, axes = plt.subplots(2, 2, figsize=(14, 12))

titles = [("PCA", f"{var_pca[0]:.1%}+{var_pca[1]:.1%}"),
          ("t-SNE", "perplexity=40")]
projs = [X_pca, X_tsne]
cmaps = ["RdBu_r", "viridis"]  # Zak binary uses diverging, gap4 uses sequential

for col, (proj, (title, subt)) in enumerate(zip(projs, titles)):
    for row, (arr, cname, clabel) in enumerate([
        (zak_label_flat, "Zak", "Zak phase (topo=1, triv=0)"),
        (gap4_flat,      "gap4", "gap4 = ε₄−ε₃"),
    ]):
        ax = axes[row, col]
        sc = ax.scatter(proj[:, 0], proj[:, 1], c=arr,
                        s=6, alpha=0.5, cmap=cmaps[row],
                        edgecolors="none")
        # Mark unreliable points (overlap < 0.95) with open circles
        mask = unreliable_flat
        if mask.any():
            ax.scatter(proj[mask, 0], proj[mask, 1],
                      facecolors="none", edgecolors="red",
                      s=30, linewidths=0.5, alpha=0.6,
                      label=f"low overlap ({mask.sum()})")

        ax.set_xlabel("Component 1")
        ax.set_ylabel("Component 2")
        if col == 0:
            ax.set_ylabel(f"{cname}\nComponent 2", fontsize=10)
        cbar = plt.colorbar(sc, ax=ax, shrink=0.7)
        cbar.set_label(clabel, fontsize=9)
        if col == 0:
            ax.set_title(f"PCA ({var_pca[0]:.1%}+{var_pca[1]:.1%})", fontsize=11)
        else:
            ax.set_title("t-SNE", fontsize=11)

fig.suptitle("Zak Phase vs gap4 — Projection Comparison", fontsize=14, fontweight="bold")
plt.tight_layout()
fig.savefig(os.path.join(FIG_DIR, "zak_vs_gap4_pca.png"), dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"Saved -> {FIG_DIR}/zak_vs_gap4_pca.png")

# ── Figure 2: Phase diagrams — gap4 heatmap + Zak boundary overlay ─────
# Show 3 U slices: U=0, U=2, U=4
u_indices = [0, 4, 8]  # U = 0.0, 2.0, 4.0
u_labels = [f"U = {U_arr[i]:.1f}" for i in u_indices]

fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))

for idx, (ui, ax) in enumerate(zip(u_indices, axes)):
    t1_2d, t2_2d = np.meshgrid(t1_arr, t2_arr, indexing="ij")

    # gap4 heatmap
    g4 = gap4[:, :, ui]
    g4_min, g4_max = g4.min(), g4.max()
    im = ax.pcolormesh(t1_2d, t2_2d, g4,
                       cmap="viridis", shading="auto",
                       norm=Normalize(vmin=g4_min, vmax=g4_max))

    # Zak phase boundary: contour at 0.5 (between 0 and 1 in label)
    zl = zak_label[:, :, ui]
    # Only draw contour if both phases exist
    if zl.min() < 0.5 < zl.max():
        ax.contour(t1_2d, t2_2d, zl, levels=[0.5],
                   colors="red", linewidths=2.5, linestyles="--")

    # Unreliable region (overlap < 0.95) — hatch overlay
    ur = unreliable[:, :, ui]
    if ur.any():
        # Create a masked array for unreliable region
        ur_mask = np.ma.masked_where(~ur, np.ones_like(ur))
        ax.pcolormesh(t1_2d, t2_2d, ur_mask,
                      hatch="///", alpha=0.0,
                      cmap="Reds", shading="auto")

    # t₁ = t₂ diagonal (reference)
    diag = np.linspace(t1_arr[0], t1_arr[-1], 100)
    ax.plot(diag, diag, "w-", lw=0.8, alpha=0.4, label="t₁=t₂")

    ax.set_xlabel("t₁")
    ax.set_ylabel("t₂")
    ax.set_title(f"gap4 heatmap  (U={U_arr[ui]:.1f})", fontsize=11)
    ax.set_xlim(t1_arr[0], t1_arr[-1])
    ax.set_ylim(t2_arr[0], t2_arr[-1])
    ax.set_aspect("equal")

    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("gap4", fontsize=9)

    # legend handles
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    handles = []
    if zl.min() < 0.5 < zl.max():
        handles.append(Line2D([0], [0], color="red", ls="--", lw=2.5, label="Zak boundary"))
    if ur.any():
        handles.append(Patch(facecolor="none", edgecolor="red", hatch="///",
                             label="low overlap"))
    handles.append(Line2D([0], [0], color="white", lw=0.8, alpha=0.4, label="t₁=t₂"))
    ax.legend(handles=handles, fontsize=7, loc="lower right")

fig.suptitle("Zak Phase Boundary vs gap4 — Phase Diagram Comparison",
             fontsize=14, fontweight="bold")
plt.tight_layout()
fig.savefig(os.path.join(FIG_DIR, "zak_vs_gap4_phasediagram.png"),
            dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"Saved -> {FIG_DIR}/zak_vs_gap4_phasediagram.png")

# ── Figure 3: Disagreement map — where zak and gap4 tell different stories ──
# For each U slice, compute the disagreement metric:
# Normalize gap4 across the slice and binarize at the median gap
fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))

for idx, (ui, ax) in enumerate(zip(u_indices, axes)):
    t1_2d, t2_2d = np.meshgrid(t1_arr, t2_arr, indexing="ij")
    g4 = gap4[:, :, ui]
    zl = zak_label[:, :, ui]

    # Disagreement: points where gap4 binarisation (at median split)
    # disagrees with Zak label.  This catches misalignment.
    g4_binary = (g4 < np.median(g4)).astype(float)
    disagree = np.abs(g4_binary - zl)

    im = ax.pcolormesh(t1_2d, t2_2d, disagree,
                       cmap="RdBu_r", shading="auto",
                       vmin=0, vmax=1)

    # Superimpose Zak boundary as reference
    total = disagree.size
    ndis = disagree.sum()
    if zl.min() < 0.5 < zl.max():
        ax.contour(t1_2d, t2_2d, zl, levels=[0.5],
                   colors="k", linewidths=1.5, linestyles="--")

    ax.set_xlabel("t₁")
    ax.set_ylabel("t₂")
    ax.set_title(f"Disagreement at U={U_arr[ui]:.1f}", fontsize=11)
    ax.set_xlim(t1_arr[0], t1_arr[-1])
    ax.set_ylim(t2_arr[0], t2_arr[-1])
    ax.set_aspect("equal")

    cbar = plt.colorbar(im, ax=ax, shrink=0.8, ticks=[0, 0.5, 1])
    cbar.set_label("disagree", fontsize=9)

    ax.text(0.05, 0.95, f"disagree: {ndis:.0f}/{total} ({100*ndis/total:.1f}%)",
            transform=ax.transAxes, fontsize=9,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8))

fig.suptitle("Zak Phase vs gap4 — Disagreement Map (median-binarised gap4)",
             fontsize=14, fontweight="bold")
plt.tight_layout()
fig.savefig(os.path.join(FIG_DIR, "zak_vs_gap4_disagreement.png"),
            dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"Saved -> {FIG_DIR}/zak_vs_gap4_disagreement.png")

print("\nAll figures saved.  Summary (gap4 median-binarised vs Zak):")
for ui in u_indices:
    g4 = gap4[:, :, ui]
    zl = zak_label[:, :, ui]
    g4_binary = (g4 < np.median(g4)).astype(float)
    disagree_arr = np.abs(g4_binary - zl)
    total = disagree_arr.size
    ndis = disagree_arr.sum()
    print(f"  U={U_arr[ui]:.1f}:  disagree={ndis:.0f}/{total}  ({100*ndis/total:.1f}%)")
