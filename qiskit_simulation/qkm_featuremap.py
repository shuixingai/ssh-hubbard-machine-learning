#!/usr/bin/env python3
"""
qkm_featuremap.py — ② 经典特征 → 量子特征图 → 量子核（QKM 三栏的"通用 QKM"栏）
============================================================================
QKM 三栏对决（同一批 (t1,t2,U) 点）：
    ① classical5 + RBF-SVM      ← baseline_ml.py（已完成，~100%）
    ② 同一批经典多体标量 → Havlíček ZZ 特征图 → 线路 fidelity 核 → SVM（本脚本）
    ③ DQAP 物理原生态 → fidelity 核（K_ED 精确参照 + K_DQAP 变分；qkm_u2_slice.py）

② 回答的问题（与 ③ 的对照）：
    把 ① 已经用到的经典特征"塞进通用电路再测 fidelity"，核几何是否
    比经典 RBF 核更好/不同？这是 QKM 文献里的"通用特征图"路线（Havlíček
    Nature 2019 的 ZZ 特征图）；③ 是"物理原生"路线（电路本身来自
    SSH-Hubbard Hamiltonian）。两条量子路线并排对照 = 论文方法完备性。
    诚实框架：n≤5 qubit 全部精确可经典模拟 → 不声称量子优势；
    ② 的价值 = 同特征下核几何的对照 + 管线自洽。

与 ① 同源同点（无污染设计，勿破坏）：
    - 特征 = ssh_dataset_L8_labelgrid.npz（OBC ED 多体量，build_features_grid）
    - 点集 = topo_dataset_full.npz（PBC/TBC γ_up 标签）同一 (t1,t2,U) 网格
    - 与 ③ 解耦：点集直接自标签 npz 重选（同 C 序 idx 数学），不依赖 ③ 的输出
      → Track B（本地②）与 Track A（服务器③）并行跑，互不干扰
    - 单粒子不变量（zak/winding）一律排除（≈ 标签泄漏）

管线：
    topo 选点 → 特征按 idx 对齐 → 标准化 → 选特征子集（--feat）
    → x_enc = (π/2)·tanh(x_std)（有界编码，避开周期核混叠）
    → Havlíček ZZ 特征图（n 特征 → n qubit，depth 层，linear/full 纠缠）
    → Statevector → K_Q(i,j)=|⟨φ_i|φ_j⟩|² → SVM 对决 + 诊断 → 存 drop-in npz

依赖标注（哪些调用原有代码/环境）：
    [原有] build_features_grid, FEATURE_NAMES_5 ← baseline_ml.py（经典特征，只读）
    [原有] load_grid, kernel_matrix            ← qkm_grid_features.py（网格/核）
    [原有] tune_cv_* / run_cv_* / shuffle_null ← kernel_ml_utils.py（SVM/CV/CI）
    [环境] qiskit QuantumCircuit, Statevector   —— 本脚本 feature map 自建，
           不用 qiskit.circuit.library → 版本无关
    [数据] topo_dataset_full.npz / ssh_dataset_L8_labelgrid.npz（唯一数据源）

用法：
    python qkm_featuremap.py                              # ee+dimer, U=2+U=0（与③同层）
    python qkm_featuremap.py --feat ee                    # 1-qubit 管线 pilot（闭式自检）
    python qkm_featuremap.py --feat gap4,ee,dimer,s_occ,lam_half   # 全 5 特征 → 5 qubit
    python qkm_featuremap.py --u-list all                 # 全 7 U 层（仍分钟级）
    python qkm_featuremap.py --depth 2 --entanglement full
输出：qkm_featuremap.npz（drop-in：含 K_DQAP 别名 key，qkm_ml.py --grid 可直接读）
"""

import argparse
import os
import sys
import time

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

import numpy as np
from sklearn.preprocessing import StandardScaler

_HERE = os.path.dirname(os.path.abspath(__file__))
_DATA = os.path.dirname(_HERE)
for p in (_HERE, _DATA):
    if p not in sys.path:
        sys.path.insert(0, p)

from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector

from baseline_ml import build_features_grid, FEATURE_NAMES_5
from qkm_grid_features import load_grid, kernel_matrix
from kernel_ml_utils import (
    binom_ci, kta, psd_min_eig, rbf_matrix,
    run_cv_precomputed, run_cv_rbf, shuffle_null,
    tune_cv_precomputed, tune_cv_rbf,
)

FEAT_NPZ = os.path.join(_DATA, "ssh_dataset_L8_labelgrid.npz")
OUT_DEFAULT = os.path.join(_HERE, "qkm_featuremap.npz")
N_SHUFFLE = 5           # null 打乱次数（同 baseline_ml.py）
ENCODING = "pi/2*tanh(std)"


# ── 选点（与 ③ 同 U 层、同 C 序 idx 数学）────────────────────────────
def select_u_layers(t1, t2, U, label, u_idxs, max_pts=None):
    """同 qkm_grid_features.select_binary / qkm_u2_slice.select_slice 的 C 序
    展平 idx，只取指定 U 层（默认 [3,0] = U=2, U=0，与 ③ 切片同层 → 同点可比）。
    返回 (idx, t1p, t2p, Up, lab)。"""
    n1, n2, nu = len(t1), len(t2), len(U)
    u_idxs = sorted(int(u) for u in u_idxs)
    idx_all = [i * n2 * nu + j * nu + k
               for i in range(n1) for j in range(n2) for k in u_idxs
               if label[i, j, k] in (0, 1)]
    if max_pts:
        rng = np.random.default_rng(42)
        pick = rng.choice(len(idx_all), size=max_pts, replace=False)
        pick.sort()
        idx_all = [idx_all[i] for i in pick]
    idx = np.array(idx_all, dtype=np.int64)
    t1p = np.array([t1[i // (n2 * nu)] for i in idx])
    t2p = np.array([t2[(i // nu) % n2] for i in idx])
    Up = np.array([U[i % nu] for i in idx])
    lab = np.array([int(label[i // (n2 * nu), (i // nu) % n2, i % nu])
                    for i in idx], dtype=np.int8)
    return idx, t1p, t2p, Up, lab


# ── 编码 + 特征图 ─────────────────────────────────────────────────────
def encode_features(X_std):
    """有界编码：x_enc = (π/2)·tanh(x_std) ∈ [−π/2, π/2]。
    ZZ map 本质是周期（三角）核，无界标准化特征会混叠——两个极值点可能正交，
    几何失去单调性。tanh 平滑压进一周期 → 编码单射、几何单调（记 ENCODING）。"""
    return np.pi / 2.0 * np.tanh(np.asarray(X_std, dtype=float))


def zz_pairs(n, entanglement):
    """纠缠对：linear = 最近邻链，full = 全连接。"""
    if entanglement == "linear":
        return [(i, i + 1) for i in range(n - 1)]
    if entanglement == "full":
        return [(i, j) for i in range(n) for j in range(i + 1, n)]
    raise ValueError(f"unknown entanglement: {entanglement}")


def zz_feature_map(x, depth=1, entanglement="linear"):
    """Havlíček ZZ 特征图（显式构造，不用 qiskit.circuit.library → 版本无关）。
    每层：H 全体 → RZ(2·x_i) 线性项 → CX-RZ-CX 实现 ZZ(x_i·x_j) 二次项。
    1 特征 depth=1 → |φ(x)⟩ = RZ(2x)|+⟩，K = cos²(x−y)（闭式可自检）。"""
    n = len(x)
    qc = QuantumCircuit(n)
    for _ in range(depth):
        qc.h(range(n))
        for i in range(n):
            qc.rz(2.0 * x[i], i)
        for i, j in zz_pairs(n, entanglement):
            qc.cx(i, j)
            qc.rz(2.0 * x[i] * x[j], j)
            qc.cx(i, j)
    return qc


def build_states(X_enc):
    """逐点 Statevector → S (n, 2^n_qubit)。n≤5 qubit 全核秒级（vs ③ 每点 ~82 min）。"""
    n, m = X_enc.shape
    S = np.zeros((n, 1 << m), dtype=complex)
    for k, x in enumerate(X_enc):
        S[k] = np.asarray(Statevector(zz_feature_map(x)).data, dtype=complex)
    return S


def closed_form_1q(x_enc):
    """1 特征 depth=1 的解析核：K(i,j) = cos²(x_i − x_j)（RZ 作用在 |+⟩ 上）。
    管线自检用——线路核必须逐元素吻合，抓 feature map 构造 bug。"""
    d = x_enc[:, None] - x_enc[None, :]
    return np.cos(d) ** 2


# ── 主流程 ────────────────────────────────────────────────────────────
def main(feat_names, u_list, depth=1, entanglement="linear",
         seed=0, max_pts=None, out=None):
    t0 = time.time()
    print("=" * 74)
    print(f"② 经典特征 → 量子特征图 → 量子核  |  特征={feat_names}  "
          f"{len(feat_names)} qubit × depth={depth} {entanglement}")
    print("=" * 74)

    # [1] 选点（同 ③ 的 U 层 → 同点可比；U 轴直接读自标签 npz，口径零漂移）
    t1, t2, U, label = load_grid()
    u_idxs = list(range(len(U))) if u_list == ["all"] else [int(u) for u in u_list]
    idx, t1p, t2p, Up, lab = select_u_layers(t1, t2, U, label, u_idxs, max_pts)
    y = lab.astype(int)
    n_triv, n_topo = int((y == 0).sum()), int((y == 1).sum())
    print(f"[1/6] 选点：{len(idx)} 点（U 层={u_idxs}，triv={n_triv} "
          f"topo={n_topo}，剔除临界/未定）")

    # [2] 特征按 idx 坐标对齐（与 ① 同源 → ② vs ① 是同信息消融）
    feat = np.load(FEAT_NPZ)
    assert np.allclose(feat["t1_arr"], t1) and np.allclose(feat["t2_arr"], t2) \
        and np.allclose(feat["U_arr"], U), "特征与标签网格不对齐（L 口径坑）"
    X_grid, names_all = build_features_grid(feat)
    X_slice = X_grid.reshape(-1, X_grid.shape[-1])[idx]
    col = [names_all.index(f) for f in feat_names]
    X_sub = X_slice[:, col]
    sc = StandardScaler().fit(X_sub)          # 全量无监督缩放（同 qkm_ml 对 K_Q 的协议）
    X_std = sc.transform(X_sub)
    X_enc = encode_features(X_std)
    print(f"[2/6] 特征对齐 ✓（{len(feat_names)} 个：{feat_names}）→ 编码 {ENCODING}")

    # [3] 管线自检：1 特征闭式解 vs 线路核
    if len(feat_names) == 1 and depth == 1:
        K_cf = closed_form_1q(X_enc[:, 0])
        K_ck = kernel_matrix(build_states(X_enc))
        err = float(np.abs(K_cf - K_ck).max())
        ok = err < 1e-10
        print(f"[3/6] 自检：线路核 vs cos²(x−y) 闭式  max|Δ|={err:.2e}  "
              f"{'PASS' if ok else 'FAIL'}")
        assert ok, "feature map 与解析式不符——先修线路再往下"
    else:
        print(f"[3/6] 自检：跳过（{len(feat_names)} 特征/非 depth=1，无闭式解）")

    # [4] 特征图 → K_Q
    print(f"[4/6] 逐点 Statevector → 核（{len(idx)} 点 …）", flush=True)
    tb = time.time()
    S = build_states(X_enc)
    K_Q = kernel_matrix(S)
    print(f"      线路构建 {time.time()-tb:.0f}s → K_Q ({K_Q.shape[0]}×"
          f"{K_Q.shape[1]})   min_eig={psd_min_eig(K_Q):+.3e}（应 ≥ −1e-9）")

    # [5] 诊断 + SVM 对决（同点同协议；① 同子集=同信息消融，① classical5=论文栏）
    b1 = tune_cv_rbf(X_sub, y, seed=seed)
    acc1, g1, C1, cor1, tot1 = b1
    b2 = tune_cv_precomputed(K_Q, y, seed=seed)
    acc2, C2, cor2, tot2 = b2
    b5 = tune_cv_rbf(X_slice, y, seed=seed)      # ① classical5 论文栏参照
    acc5, g5, C5, cor5, tot5 = b5

    K_C_same = rbf_matrix(X_std, gamma=g1)       # 同特征 RBF（KTA 对照）
    print(f"\n[5/6] KTA（中心化，SVM 前先看几何信息）")
    print(f"  K_Q(特征图)            A = {kta(K_Q, y):+.4f}")
    print(f"  K_C 同特征 RBF(γ={g1}) A = {kta(K_C_same, y):+.4f}")

    lo1, hi1 = binom_ci(cor1, tot1)
    lo2, hi2 = binom_ci(cor2, tot2)
    lo5, hi5 = binom_ci(cor5, tot5)
    print(f"\n[5/6] SVM 对决（同 {len(y)} 点、同 5-fold 协议）")
    print("-" * 62)
    print(f"  ① 同子集 {feat_names} RBF   acc={acc1*100:6.1f}%  "
          f"CI=[{lo1*100:5.1f},{hi1*100:5.1f}]  (γ={g1}, C={C1})")
    print(f"  ② 特征图核 precomputed    acc={acc2*100:6.1f}%  "
          f"CI=[{lo2*100:5.1f},{hi2*100:5.1f}]  (C={C2})")
    print(f"  ① classical5（论文栏参照）acc={acc5*100:6.1f}%  "
          f"CI=[{lo5*100:5.1f},{hi5*100:5.1f}]  (γ={g5}, C={C5})")
    run_fn = lambda y2, seed: run_cv_precomputed(K_Q, y2, C=C2, seed=seed)[0]
    null_m, null_lo, null_hi = shuffle_null(run_fn, y, N_SHUFFLE)
    print(f"  null（打乱标签）         {null_m*100:6.1f}%±2σ  "
          f"[{null_lo*100:5.1f},{null_hi*100:5.1f}]")
    print("-" * 62)
    print("  [框架] ② 同信息消融：② vs ①同子集 = 通用特征图核几何 vs RBF 核几何；")
    print("  [框架] 同点可跑 ③ 后并入终表：②(特征图) | ③(K_ED/K_DQAP) | ①(classical5)")

    # [6] 存 drop-in npz（K_DQAP 别名 key → qkm_ml.py --grid 可直接当"量子核"读）
    out_npz = out or OUT_DEFAULT
    np.savez(out_npz,
             t1_vals=t1, t2_vals=t2, U_vals=U, idx=idx,
             t1=t1p, t2=t2p, U=Up, lab=lab,
             K_Q=K_Q, K_DQAP=K_Q,                  # K_DQAP = drop-in 别名
             features=np.asarray(feat_names), depth=depth,
             entanglement=entanglement, encoding=ENCODING,
             u_idxs=np.asarray(u_idxs, dtype=int),
             n_triv=n_triv, n_topo=n_topo,
             acc_classical_subset=acc1, acc_quantum=acc2,
             acc_classical5=acc5,
             results=dict(
                 classical_subset=dict(acc=acc1, ci_lo=lo1, ci_hi=hi1,
                                       gamma=g1, C=C1),
                 quantum=dict(acc=acc2, ci_lo=lo2, ci_hi=hi2, C=C2,
                              null=null_m, null_lo=null_lo, null_hi=null_hi),
                 classical5=dict(acc=acc5, ci_lo=lo5, ci_hi=hi5,
                                 gamma=g5, C=C5),
                 kta_quantum=kta(K_Q, y), kta_classical_subset=kta(K_C_same, y)))
    print(f"\n[6/6] 已存 {os.path.abspath(out_npz)}"
          f"（K_Q + drop-in 别名 K_DQAP；总耗时 {time.time()-t0:.0f}s）")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    p.add_argument("--feat", default="ee,dimer",
                   help="逗号分隔特征子集（默认 ee,dimer=最小非平凡；ee=1q pilot）")
    p.add_argument("--u-list", default="3,0",
                   help="逗号分隔 U 层索引（默认 3,0 = U=2,U=0，与 ③ 同层）或 all")
    p.add_argument("--depth", type=int, default=1)
    p.add_argument("--entanglement", choices=["linear", "full"], default="linear")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max-pts", type=int, default=None,
                   help="冒烟：固定随机子集点数（seed=42 确定）")
    p.add_argument("--out", default=None, help="输出 npz 路径")
    a = p.parse_args()
    main([f.strip() for f in a.feat.split(",")],
         [s.strip() for s in a.u_list.split(",")],
         depth=a.depth, entanglement=a.entanglement, seed=a.seed,
         max_pts=a.max_pts, out=a.out)
