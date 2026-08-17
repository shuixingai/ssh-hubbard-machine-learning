#!/usr/bin/env python3
"""
kernel_ml_utils.py — 混合 kernel SVM 共享层（baseline_ml.py 与 qkm_ml.py 共用）
================================================================================
两类特征 → 两个 n×n PSD kernel → 加权融合（memory: gap4-classical-baseline
§"kernel 融合设计"）。把"理论标量特征做成 kernel 矩阵"的关键注意点都在这里。

核心原则 = 不要在特征空间合并，把两边都变成 kernel 再相加：
    K_hyb(w) = w·K̂_Q + (1−w)·K̂_C ,    K̂ = K / ‖K‖_F   (Frobenius 归一化)

★ 归一化用「正标量缩放」→ 不改变 PSD，SVM dual 恒可解。
★ 不要先双中心化再求和：中心化会去掉沿全 1 方向的贡献，可能造出负特征值，
  K_hyb 不再是合法 kernel，SVC 会不收敛。双中心化只用于 KTA 诊断
  （centered KTA = translation-invariant alignment，见 kta()）。

验证协议（怎么确认这些技巧合适）：
    1. PSD 数值检查    psd_min_eig() ≥ −1e-9
    2. KTA             对 K_Q / K_C 各分量单独算 → 低对齐=互补（混合有空间）、
                       高对齐=冗余（消融会暴露）
    3. 消融三栏        量子-only / 经典-only / 混合——混合须赢过更好单侧，
                       配 binomial CI（binom_ci）
    4. CV 纪律         缩放与超参只在训练折内（run_cv_rbf 内置 StandardScaler
                       训练折拟合）；混合栏全量无监督缩放 + 超参 CV 选，两栏同协议
    5. 决策图          SVM 决策画回 (t1,t2) 平面逐 U 层，边界应贴 t1=t2 线
"""

import numpy as np
from scipy.stats import binomtest
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


# ── kernel 构造 ───────────────────────────────────────────────────────
def rbf_matrix(X, Y=None, gamma=1.0):
    """RBF kernel：exp(−γ‖x−x'‖²)。Y=None 时做两两（X 自身），否则 X×Y。
    经典标量特征 xᵢ → K_C(i,j) = rbf_matrix(x, gamma)[i,j] 即得 PSD 矩阵。"""
    X = np.asarray(X, dtype=float)
    if Y is None:
        G = X @ X.T
        d = np.einsum("ii->i", G)
        D = d[:, None] + d[None, :] - 2 * G
    else:
        Y = np.asarray(Y, dtype=float)
        D = np.einsum("ii->i", X @ X.T)[:, None] \
            + np.einsum("ii->i", Y @ Y.T)[None, :] - 2 * (X @ Y.T)
    return np.exp(-gamma * np.maximum(D, 0.0))


def frob_normalize(K):
    """正标量缩放：K/‖K‖_F。保 PSD（缩放不改变符号），只对齐量级——
    MKL 加和前必做，否则量级大的 kernel 平凡主导。"""
    K = np.asarray(K, dtype=float)
    n = np.linalg.norm(K, "fro")
    return K / n if n > 0 else K


def center_kernel(K):
    """双中心化（仅诊断用，勿用于加和）。"""
    K = np.asarray(K, dtype=float)
    n = K.shape[0]
    J = np.ones((n, n)) / n
    return K - J @ K - K @ J + J @ K @ J


def hybrid_kernel(K_q, K_c, w):
    """K_hyb = w·K̂_Q + (1−w)·K̂_C（各自先 Frobenius 归一化）。"""
    return w * frob_normalize(K_q) + (1.0 - w) * frob_normalize(K_c)


# ── 诊断 ──────────────────────────────────────────────────────────────
def psd_min_eig(K):
    """PSD 数值检查：最小特征值（应 ≥ −1e-9）。"""
    K = (np.asarray(K, dtype=float) + np.asarray(K, dtype=float).T) / 2
    return float(np.linalg.eigvalsh(K).min())


def kta(K, y):
    """kernel-target alignment：A = ⟨K_c, yyᵀ⟩_F / (‖K_c‖_F·‖yyᵀ‖_F)。
    用双中心化 kernel → translation-invariant。SVM 跑之前就知道哪个源
    真带相变信息；低对齐 = 互补（混合有空间）、高对齐 = 冗余。"""
    y = np.asarray(y, dtype=float)
    Kc = center_kernel(np.asarray(K, dtype=float))
    yc = y - y.mean()
    yy = np.outer(yc, yc)
    num = float(np.sum(Kc * yy))
    den = np.linalg.norm(Kc, "fro") * np.linalg.norm(yy, "fro")
    return num / den if den > 0 else 0.0


def binom_ci(correct, total, level=0.95):
    """Wilson 区间（小样本 <100 测试点必须用）。"""
    lo, hi = binomtest(int(correct), int(total)).proportion_ci(
        confidence_level=level, method="wilson")
    return float(lo), float(hi)


# ── CV（precomputed kernel 版）────────────────────────────────────────
def _guard_splits(n_splits, y):
    k = int(min(np.bincount(y)))
    if k < 2:
        raise ValueError(f"每类样本不足（min={k}），无法分层 CV")
    return min(n_splits, k)


def run_cv_precomputed(K, y, C=1.0, n_splits=5, seed=42):
    """分层 5-fold on precomputed kernel。训练折用 K[tr][:, tr]，
    测试折用 K[te][:, tr]（行=测试、列=训练过的支持向量）。返回
    (mean_acc, correct, total, accs)。"""
    K = np.asarray(K, dtype=float)
    n_splits = _guard_splits(n_splits, y)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    accs, correct, total = [], 0, 0
    for tr, te in skf.split(K, y):
        clf = SVC(kernel="precomputed", C=C)
        clf.fit(K[np.ix_(tr, tr)], y[tr])
        pred = clf.predict(K[np.ix_(te, tr)])
        accs.append(accuracy_score(y[te], pred))
        correct += int((pred == y[te]).sum())
        total += len(te)
    return float(np.mean(accs)), correct, total, accs


def run_cv_rbf(X, y, gamma=1.0, C=1.0, n_splits=5, seed=42, scale=True):
    """分层 5-fold RBF-SVM。StandardScaler 只在训练折内拟合（CV 纪律）：
    训练折先 fit+transform 再建 K_tr；测试折用训练折的 scaler 变换后建
    K_te×tr。返回 (mean_acc, correct, total, accs)。"""
    X = np.asarray(X, dtype=float)
    n_splits = _guard_splits(n_splits, y)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    accs, correct, total = [], 0, 0
    for tr, te in skf.split(X, y):
        if scale:
            sc = StandardScaler().fit(X[tr])
            Xtr, Xte = sc.transform(X[tr]), sc.transform(X[te])
        else:
            Xtr, Xte = X[tr], X[te]
        clf = SVC(kernel="precomputed", C=C)
        clf.fit(rbf_matrix(Xtr, gamma=gamma), y[tr])
        pred = clf.predict(rbf_matrix(Xte, Xtr, gamma=gamma))
        accs.append(accuracy_score(y[te], pred))
        correct += int((pred == y[te]).sum())
        total += len(te)
    return float(np.mean(accs)), correct, total, accs


# ── 超参选择（在数据上 CV 选，两栏同协议可比）────────────────────────
GAMMA_GRID = [0.001, 0.01, 0.1, 0.5, 1.0, 5.0, 10.0, 50.0, 100.0]
C_GRID = [0.1, 1.0, 10.0]
W_GRID = [0.0, 0.25, 0.5, 0.75, 1.0]   # w=0 → 纯经典 kernel，w=1 → 纯量子 kernel


def tune_cv_precomputed(K, y, C_grid=C_GRID, seed=42):
    """在 C 网格上 CV 选最好的，返回 (best_acc, C, correct, total)。"""
    best = None
    for C in C_grid:
        acc, correct, total, _ = run_cv_precomputed(K, y, C=C, seed=seed)
        if best is None or acc > best[0]:
            best = (acc, C, correct, total)
    return best


def tune_cv_rbf(X, y, gamma_grid=GAMMA_GRID, C_grid=C_GRID, seed=42):
    """在 γ×C 网格上 CV 选最好的，返回 (best_acc, gamma, C, correct, total)。"""
    best = None
    for g in gamma_grid:
        for C in C_grid:
            acc, correct, total, _ = run_cv_rbf(X, y, g, C=C, seed=seed)
            if best is None or acc > best[0]:
                best = (acc, g, C, correct, total)
    return best


def shuffle_null(run_fn, y, n_shuffle=5, seed=0):
    """null：打乱标签 n_shuffle 次，各跑一次 run_fn(y_shuf, seed=s)。
    run_fn 是绑好数据/X 的闭包，签名 run_fn(y2, seed) → mean_acc。
    返回 (null_mean, null_lo, null_hi) = mean ± 2σ。"""
    rng = np.random.default_rng(seed)
    accs = []
    for s in range(n_shuffle):
        y_shuf = y.copy()
        rng.shuffle(y_shuf)
        accs.append(run_fn(y_shuf, seed=s))
    m, sd = float(np.mean(accs)), float(np.std(accs))
    return m, m - 2 * sd, m + 2 * sd


if __name__ == "__main__":
    # 自检：RBF / |G|² 都应 PSD；加和（Frobenius 归一化后）也应 PSD
    rng = np.random.default_rng(0)
    X = rng.normal(size=(40, 5))
    Kc = rbf_matrix(X, gamma=0.5)
    G = rng.normal(size=(40, 8)) + 1j * rng.normal(size=(40, 8))
    Kq = np.abs(G @ G.conj().T) ** 2
    Kh = hybrid_kernel(Kq, Kc, 0.5)
    print(f"min_eig K_C  = {psd_min_eig(Kc):+.3e}")
    print(f"min_eig K_Q  = {psd_min_eig(Kq):+.3e}")
    print(f"min_eig K_hyb= {psd_min_eig(Kh):+.3e}  (均应 ≥ −1e-9)")
