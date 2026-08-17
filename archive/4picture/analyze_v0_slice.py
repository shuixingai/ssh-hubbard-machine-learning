"""
V=0 子空间提取 + 多张单指标 PCA 可视化
============================================
从 4D (t₁,t₂,U,V) 数据中提取 V=0 slice，
做 PCA + t-SNE 降维，每张图用单种指示剂着色。

指示剂：
  - gap4         — 纠缠谱能隙，拓扑标记
  - double occ.  — 双占据平均，Mott 标记
  - δB           — 键序交替，BOW 标记
  - δn           — 交错电荷密度（V=0 应为零，作验证）

用法:  python analyze_v0_slice.py
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
NPZ_PATH = os.path.join(THIS_DIR, "ssh_dataset_L6_UVplane.npz")
OUT_V0 = os.path.join(THIS_DIR, "v0_slice.npz")

# ─── 1. Load 4D data ────────────────────────────────────────────────
print("Loading 4D data ...")
data = np.load(NPZ_PATH)
t1_arr = data["t1_arr"]    # (16,)
t2_arr = data["t2_arr"]    # (16,)
U_arr  = data["U_arr"]     # (17,)
V_arr  = data["V_arr"]     # (17,)
corr   = data["corr_matrices"]   # (16,16,17,17,6,6)
ent    = data["ent_spectra"]     # (16,16,17,17,20)
docc   = data["double_occupancy"]# (16,16,17,17,6)
bond_a = data["bond_alternation"]# (16,16,17,17)
stagg  = data["staggered_charge"]# (16,16,17,17)
L = int(data["L"])

print(f"  t1: {len(t1_arr)}  t2: {len(t2_arr)}  U: {len(U_arr)}  V: {len(V_arr)}")
print(f"  corr shape: {corr.shape}")

# ─── 2. Extract V=0 slice ───────────────────────────────────────────
iv0 = 0  # V_arr[0] == 0

t1_v0 = t1_arr
t2_v0 = t2_arr
U_v0  = U_arr
corr_v0  = corr[:, :, :, iv0, :, :]    # (16,16,17,6,6)
ent_v0   = ent[:, :, :, iv0, :]        # (16,16,17,20)
docc_v0  = docc[:, :, :, iv0, :]       # (16,16,17,6)
bond_v0  = bond_a[:, :, :, iv0]        # (16,16,17)
stagg_v0 = stagg[:, :, :, iv0]         # (16,16,17)

N1, N2, N3 = corr_v0.shape[:3]
n_total = N1 * N2 * N3
print(f"\nV=0 slice: {N1}x{N2}x{N3} = {n_total} points")

# ─── 3. Compute indicators ──────────────────────────────────────────
gap4_v0   = ent_v0[:, :, :, 4] - ent_v0[:, :, :, 3]   # ε₅ - ε₄  (0-indexed)
docc_mean = docc_v0.mean(axis=3)                        # avg over 6 sites
delta_B   = bond_v0
delta_n   = stagg_v0

gap4_flat   = gap4_v0.ravel()
docc_flat   = docc_mean.ravel()
deltaB_flat = delta_B.ravel()
deltaN_flat = delta_n.ravel()

print(f"  gap4:    [{gap4_flat.min():.3f}, {gap4_flat.max():.3f}]")
print(f"  docc:    [{docc_flat.min():.4f}, {docc_flat.max():.4f}]")
print(f"  δB:      [{deltaB_flat.min():.3f}, {deltaB_flat.max():.3f}]")
print(f"  δn:      [{deltaN_flat.min():.3e}, {deltaN_flat.max():.3e}]")

# ─── 4. Save V=0 slice for later use ────────────────────────────────
np.savez_compressed(OUT_V0,
    t1_arr=t1_v0, t2_arr=t2_v0, U_arr=U_v0,
    corr_matrices=corr_v0,
    ent_spectra=ent_v0,
    double_occupancy=docc_v0,
    bond_alternation=bond_v0,
    staggered_charge=stagg_v0,
    gap4=gap4_v0,
    docc_mean=docc_mean,
    L=L)
print(f"\nV=0 slice saved -> {OUT_V0}")

# ─── 5. PCA + t-SNE on correlation matrices ─────────────────────────
X = corr_v0.reshape(n_total, L * L)   # (4352, 36)

print("\nFitting PCA ...")
pca_2d = PCA(n_components=2, random_state=42)
X_pca  = pca_2d.fit_transform(X)
var_pca = pca_2d.explained_variance_ratio_

print(f"  PCA: 2 dim ({var_pca[0]:.1%} + {var_pca[1]:.1%} = {var_pca.sum():.1%} variance)")

print("Fitting t-SNE ...")
tsne = TSNE(n_components=2, perplexity=50, random_state=42,
            init="pca", learning_rate="auto")
X_tsne = tsne.fit_transform(X)

# ─── 6. Multi-panel: each indicator gets its own figure ─────────────
indicators = [
    (gap4_flat,   "gap4",        r"gap$_4$ = $\varepsilon_4 - \varepsilon_3$",   "viridis"),
    (docc_flat,   "double_occ",  r"$\langle n_\uparrow n_\downarrow \rangle$",   "magma"),
    (deltaB_flat, "delta_B",     r"$\delta B$ (bond-order alternation)",          "RdBu_r"),
    (deltaN_flat, "delta_n",     r"$\delta n$ (staggered charge)",               "coolwarm"),
]

for arr, name, label, cmap in indicators:
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

    fig.suptitle(f"SSH-Hubbard V=0: {label}", fontsize=13, fontweight="bold")
    plt.tight_layout()
    fpath = os.path.join(THIS_DIR, f"v0_pca_{name}.png")
    fig.savefig(fpath, dpi=150, bbox_inches="tight")
    print(f"  Saved -> {fpath}")
    plt.close(fig)

print(f"\nDone! All figures in: {THIS_DIR}")
