"""
DMRG Visualization — read NPZ and produce PCA/t-SNE + phase diagrams
====================================================================
PCA is performed on the entanglement spectrum (20-dim) instead of corr matrix.

Usage:
  python visualize_dmrg.py
"""

import os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
NPZ_PATH = os.path.join(THIS_DIR, "dmrg_dataset.npz")

print(f"Loading: {NPZ_PATH}")
data = np.load(NPZ_PATH, allow_pickle=True)

# Handle NPZ dict-like access
if isinstance(data, np.lib.npyio.NpzFile):
    d = data
else:
    d = data.item()

t1_arr = np.asarray(d["t1_arr"])
t2_arr = np.asarray(d["t2_arr"])
U_arr  = np.asarray(d["U_arr"])
L = int(d["L"])
energies  = np.asarray(d["energies"])
ent_spec  = np.asarray(d["ent_spectra"])   # (N1, N2, N3, 20)
entropy   = np.asarray(d["ent_entropies"])
bdims     = np.asarray(d["bond_dims"])
has_gap = "charge_gap" in d
if has_gap:
    charge_gap = np.asarray(d["charge_gap"])

N1, N2, N3 = energies.shape
n_total = N1 * N2 * N3
print(f"Grid: {N1}x{N2}x{N3} = {n_total}, L={L}")
print(f"U: {U_arr}")

# ══════════════════════════════════════════════════════════════════════
#  Metrics
# ══════════════════════════════════════════════════════════════════════

gap_4 = ent_spec[:, :, :, 4] - ent_spec[:, :, :, 3]  # 0-based: index 4 = 5th level
th_g4 = np.median(gap_4)
label_g4 = (gap_4 < th_g4).astype(int)

if has_gap:
    th_cg = 0.5 * np.max(charge_gap[:, :, 0])
    label_cg = (charge_gap < th_cg).astype(int)

quality = bdims >= 10
n_bad = np.sum(~quality)
print(f"\nLabel fractions:")
print(f"  gap4 (th={th_g4:.4f}): topo = {100*np.mean(label_g4):.1f}%")
if has_gap:
    print(f"  charge gap (th={th_cg:.4f}): topo = {100*np.mean(label_cg):.1f}%")
    agree = (label_g4.ravel() == label_cg.ravel()).sum() / n_total
    print(f"  agreement: {100*agree:.1f}%")
print(f"  bond_dim < 10: {n_bad}/{n_total} = {100*n_bad/n_total:.1f}%")
print(f"  S_vN range: [{entropy.min():.4f}, {entropy.max():.4f}]")

# ══════════════════════════════════════════════════════════════════════
#  PCA on entanglement spectrum (20-dim)
# ══════════════════════════════════════════════════════════════════════

X = ent_spec.reshape(n_total, 20)

pca_full = PCA(random_state=42)
X_pca_full = pca_full.fit_transform(X)
cumvar = np.cumsum(pca_full.explained_variance_ratio_)
n_pc = np.searchsorted(cumvar, 0.95) + 1
print(f"\nPCA on ent-spec (20 dim):")
for i in range(min(6, 20)):
    print(f"  PC{i+1}: {pca_full.explained_variance_ratio_[i]:.2%}  (cum: {cumvar[i]:.2%})")
print(f"  95% variance at {n_pc} components")

pca_2d = PCA(n_components=2, random_state=42)
X_pca = pca_2d.fit_transform(X)
var_2d = pca_2d.explained_variance_ratio_

tsne = TSNE(n_components=2, perplexity=30, random_state=42,
            init="pca", learning_rate="auto")
X_tsne = tsne.fit_transform(X)

for name, lbl in [("gap4", label_g4.ravel()), ("charge gap", label_cg.ravel())]:
    scores = cross_val_score(KNeighborsClassifier(5), X_pca, lbl, cv=5)
    print(f"  kNN {name:15s}: accuracy={scores.mean():.4f}+/-{scores.std():.4f}")

# ══════════════════════════════════════════════════════════════════════
#  Figures
# ══════════════════════════════════════════════════════════════════════

colors = ["#e74c3c", "#3498db"]
markers = ["o", "^"]
bad = ~quality.ravel()

# ── Fig 1: PCA + t-SNE colored by gap4 label ──
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
for ax, Xd, title in [
    (axes[0], X_pca, f"PCA ({var_2d[0]:.1%}+{var_2d[1]:.1%})"),
    (axes[1], X_tsne, "t-SNE on ent-spec"),
]:
    for code in range(2):
        mask = label_g4.ravel() == code
        ax.scatter(Xd[mask, 0], Xd[mask, 1],
                   c=colors[code], marker=markers[code],
                   s=8, alpha=0.5, edgecolors="none",
                   label=f"{['trivial','topological'][code]}")
    ax.scatter(Xd[bad, 0], Xd[bad, 1],
               c="gray", marker="x", s=20, alpha=0.8, label="bond_dim<10")
    ax.set_xlabel("Comp 1"); ax.set_ylabel("Comp 2")
    ax.set_title(title, fontsize=11)
    ax.legend(markerscale=2, fontsize=8)
fig.suptitle(f"DMRG L={L}: PCA/t-SNE on entanglement spectrum (gap4 label)",
             fontsize=13, fontweight="bold")
plt.tight_layout()
fig.savefig(os.path.join(THIS_DIR, "dmrg_pca_tsne.png"), dpi=150)
print("Saved -> dmrg_pca_tsne.png")
plt.close(fig)

# ── Fig 1b: PCA colored by continuous gap4 ──
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
for ax, Xd, title in [
    (axes[0], X_pca, "PCA"),
    (axes[1], X_tsne, "t-SNE"),
]:
    sc = ax.scatter(Xd[:, 0], Xd[:, 1], c=gap_4.ravel(),
                    s=8, alpha=0.5, cmap="viridis", edgecolors="none")
    ax.scatter(Xd[bad, 0], Xd[bad, 1],
               c="gray", marker="x", s=20, alpha=0.8, label="bond_dim<10")
    ax.set_xlabel("Comp 1"); ax.set_ylabel("Comp 2")
    ax.set_title(title, fontsize=11)
    ax.legend(markerscale=2, fontsize=8)
    plt.colorbar(sc, ax=ax, label="gap4", shrink=0.8)
fig.suptitle("DMRG: PCA/t-SNE colored by continuous gap4",
             fontsize=13, fontweight="bold")
plt.tight_layout()
fig.savefig(os.path.join(THIS_DIR, "dmrg_pca_gap4_color.png"), dpi=150)
print("Saved -> dmrg_pca_gap4_color.png")
plt.close(fig)

# ── Fig 2: Phase diagrams ──
ncols = min(N3, 5)
U_idx = np.linspace(0, N3-1, ncols, dtype=int)
nrows = 2 if has_gap else 1
fig, axes = plt.subplots(nrows, ncols, figsize=(5*ncols, 4*nrows))
if nrows == 1: axes = np.array([axes]) if ncols == 1 else axes[np.newaxis, :]
for row, (lbl, title) in enumerate([
    (label_g4, "gap4 label"),
    (label_cg, "charge gap label") if has_gap else (None, ""),
]):
    if lbl is None: continue
    for col, ui in enumerate(U_idx):
        ax = axes[row, col]
        lbl_plot = lbl[:, :, ui].astype(float)
        lbl_plot[~quality[:, :, ui]] = np.nan
        ax.pcolormesh(t1_arr, t2_arr, lbl_plot.T,
                      shading="auto", cmap="RdYlBu", vmin=0, vmax=1)
        i_bad, j_bad = np.where(~quality[:, :, ui])
        ax.scatter(t1_arr[i_bad], t2_arr[j_bad],
                   c="gray", marker="x", s=15, alpha=0.7)
        ax.set_xlabel("t1"); ax.set_ylabel("t2")
        ax.set_title(f"U={U_arr[ui]:.1f}", fontsize=10)
        ax.set_aspect("equal")
    axes[row, 0].set_ylabel(title, fontsize=10, fontweight="bold")
plt.tight_layout()
fig.savefig(os.path.join(THIS_DIR, "dmrg_phase_diagrams.png"), dpi=150)
print("Saved -> dmrg_phase_diagrams.png")
plt.close(fig)

# ── Fig 3: Entropy landscape ──
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.flatten()
for idx, ui in enumerate([0, N3//3, 2*N3//3, N3-1][:4]):
    ax = axes[idx]
    im = ax.pcolormesh(t1_arr, t2_arr, entropy[:, :, ui].T,
                       shading="auto", cmap="viridis")
    ax.set_xlabel("t1"); ax.set_ylabel("t2")
    ax.set_title(f"SvN  U={U_arr[ui]:.1f}", fontsize=11)
    ax.set_aspect("equal")
    plt.colorbar(im, ax=ax, shrink=0.8)
ax = axes[4]
t1g, t2g = np.meshgrid(t1_arr, t2_arr, indexing="ij")
for ui, c in zip([0, N3//2, N3-1], ["blue", "green", "red"]):
    ax.scatter((t2g-t1g).ravel(), entropy[:, :, ui].ravel(),
               s=2, alpha=0.3, c=c, label=f"U={U_arr[ui]:.1f}")
ax.set_xlabel("t2 - t1"); ax.set_ylabel("SvN")
ax.set_title("Entanglement Entropy"); ax.legend(fontsize=8)
ax.axvline(0, color="gray", ls="--", alpha=0.5)
for idx in range(5, len(axes)): axes[idx].axis("off")
plt.tight_layout()
fig.savefig(os.path.join(THIS_DIR, "dmrg_entropy.png"), dpi=150)
print("Saved -> dmrg_entropy.png")
plt.close(fig)

# ── Fig 4: Charge gap landscape ──
if has_gap:
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    for idx, ui in enumerate([0, N3//3, 2*N3//3, N3-1][:4]):
        ax = axes[idx]
        im = ax.pcolormesh(t1_arr, t2_arr, charge_gap[:, :, ui].T,
                           shading="auto", cmap="plasma")
        ax.set_xlabel("t1"); ax.set_ylabel("t2")
        ax.set_title(f"Charge gap  U={U_arr[ui]:.1f}", fontsize=11)
        ax.set_aspect("equal")
        plt.colorbar(im, ax=ax, shrink=0.8)
    ax = axes[4]
    for ui, c in zip([0, N3//2, N3-1], ["blue", "green", "red"]):
        ax.scatter((t2g-t1g).ravel(), charge_gap[:, :, ui].ravel(),
                   s=2, alpha=0.3, c=c, label=f"U={U_arr[ui]:.1f}")
    ax.set_xlabel("t2 - t1"); ax.set_ylabel("Charge gap")
    ax.set_title("Charge Gap"); ax.legend(fontsize=8)
    ax.axvline(0, color="gray", ls="--", alpha=0.5)
    for idx in range(5, len(axes)): axes[idx].axis("off")
    plt.tight_layout()
    fig.savefig(os.path.join(THIS_DIR, "dmrg_charge_gap.png"), dpi=150)
    print("Saved -> dmrg_charge_gap.png")
    plt.close(fig)

# ── Fig 5: Summary ──
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
ax = axes[0, 0]
frac_g4 = [label_g4[:, :, ui].mean() for ui in range(N3)]
ax.plot(U_arr, frac_g4, "o-", label="gap4", color="blue")
if has_gap:
    frac_cg = [label_cg[:, :, ui].mean() for ui in range(N3)]
    ax.plot(U_arr, frac_cg, "s-", label="charge gap", color="red")
ax.set_xlabel("U"); ax.set_ylabel("Fraction topo"); ax.legend(); ax.set_title("Topological fraction")

ax = axes[0, 1]
mean_S = [entropy[:, :, ui].mean() for ui in range(N3)]
ax.plot(U_arr, mean_S, "o-", color="green")
ax.set_xlabel("U"); ax.set_ylabel("Mean SvN"); ax.set_title("Entropy vs U")

ax = axes[1, 0]
mean_g4 = [gap_4[:, :, ui].mean() for ui in range(N3)]
ax.plot(U_arr, mean_g4, "o-", color="purple")
ax.set_xlabel("U"); ax.set_ylabel("Mean gap4"); ax.set_title("gap4 vs U")

ax = axes[1, 1]
if has_gap:
    mean_cg = [charge_gap[:, :, ui].mean() for ui in range(N3)]
    ax.plot(U_arr, mean_cg, "o-", color="orange")
    ax.set_xlabel("U"); ax.set_ylabel("Mean charge gap"); ax.set_title("Charge gap vs U")

plt.tight_layout()
fig.savefig(os.path.join(THIS_DIR, "dmrg_summary.png"), dpi=150)
print("Saved -> dmrg_summary.png")
plt.close("all")

print(f"\nAll figures saved to: {THIS_DIR}")
