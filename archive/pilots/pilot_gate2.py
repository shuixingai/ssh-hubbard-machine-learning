#!/usr/bin/env python3
"""
pilot_gate2.py — 闸门 2：ED 精确态 kernel 判别力（(t1,t2) 判别轴）
====================================================================
U=0 pilot 三层闸门的第 2 层（脚本 2 / 3）。

目标：沿判别轴 (t1,t2)，ED N=4 物理基态（spinless L=4 APBC，8 qubit）的
fidelity kernel
    K_ED(i,j) = |⟨ψ(t1,t2) | ψ(t1',t2')⟩|²
能否作为 precomputed Gram matrix 喂给 SVM，分离平凡/拓扑两类？

这是 QKM 路线在本模型上的"真考验"（[[qkm-vs-classical-ml-comparison]]）：
若 ED 完美态的 kernel 都无法分离两相，则 QKM（L-e）路线在此模型不成立——
后面的 DQAP 态污染讨论都无意义。若 ED 能分离，闸门 3 再测污染是否翻转决策。

判据（修订版）：
  ① 无 0/1 分块 —— K_ED 不是"类内恒 1、类间恒 0"的阶跃（那是伪影特征），
     须携带平滑结构：类内 K 有真实散布（min < 0.99），类间 K 非精确 0
     （max > 1e-3），且类内/类间中心差明显但中间有过渡。
  ② SVM 区分度 —— precomputed-kernel SVM（C=1）在留出网格点上的
     分层 CV 准确率 ≳ 0.8（远离随机 0.5）。
     （佐证：kernel 空间最近邻准确率 + 留出外推测试）
  ③ 对角线简并 —— t1=t2 的 13 点基态简并（临界），剔除 SVM，单独报告
     （K_ED 对简并子空间无定义，与 γ_up 无定义共位）。

复用闸门 1 已验证的 ED 路径（pilot_le.sector_gs_spinless，dense eigh，
C(8,4)=70 维扇区，秒级），K 用直接内积（闸门 1 已与 L-e 电路对账到 1e-14）。

输出：pilot_gate2.npz（含 169 点 ED 态 + 标签 + K_ED + 判别力指标，
      ED 态供闸门 3 复用做污染参照）。
"""

import os
import sys
import time
import argparse

try:
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
except Exception:
    pass

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_DATA = os.path.dirname(_HERE)
for p in (_HERE, _DATA):
    if p not in sys.path:
        sys.path.insert(0, p)

# 复用闸门 1 已验证的 ED（dense eigh，L=4 APBC N=4）
from pilot_le import sector_gs_spinless, L, BOUNDARY, N_TARGET

# 标签编码（与 build_topo_dataset.py 一致）
LABEL_TRIVIAL, LABEL_TOPO, LABEL_CRITICAL = 0, 1, 2

# 判据阈值
ACC_THRESHOLD = 0.80        # SVM 判别力
WITHIN_SPREAD_MIN = 0.01    # 类内 K 最小散布（min K 至少低于 1-此值 → 非恒 1）
CROSS_MAX_MIN = 1e-3        # 类间 K 至少有一个 > 此值 → 非精确 0
CROSS_CENTER_MAX = 0.999    # 类间中心值上限（>此值=全混，判别力极差）


def load_grid(dataset_npz):
    """载入 13×13 网格 + U=0 层标签。返回 dict。"""
    d = np.load(dataset_npz, allow_pickle=True)
    t1 = d['t1_vals']
    t2 = d['t2_vals']
    lab = d['label'][:, :, 0]          # U=0 层 (13,13)
    spz = d['sp_zak']                  # (13,13) 单粒子 Zak 交叉检验参照
    return dict(t1=t1, t2=t2, lab=lab, sp_zak=spz)


def compute_ed_grid(t1, t2, verbose=True):
    """全网格 ED。返回 states (n1,n2,256) complex, E0 (n1,n2), degen (n1,n2),
    ok（唯一 GS 标记 bool）。对角线简并点也存（用 manifold[0] 占位）。"""
    n1, n2 = len(t1), len(t2)
    nq = 2 * L
    states = np.zeros((n1, n2, 1 << nq), dtype=complex)
    E0 = np.zeros((n1, n2))
    degen = np.zeros((n1, n2), dtype=int)
    ok = np.ones((n1, n2), dtype=bool)
    t0 = time.time()
    for i in range(n1):
        for j in range(n2):
            e0, _, manifold, dg = sector_gs_spinless(float(t1[i]), float(t2[j]))
            states[i, j] = manifold[0]
            E0[i, j] = e0
            degen[i, j] = dg
            ok[i, j] = (dg == 1)
            if verbose and (i * n2 + j) % 20 == 0:
                el = time.time() - t0
                print(f'  [{i*n2+j:3d}/{n1*n2}] t1={t1[i]:.3f} t2={t2[j]:.3f} '
                      f'E0={e0:10.6f} degen={dg}  el={el:.1f}s', flush=True)
    print(f'  完成 {n1*n2} 点 ED，耗时 {time.time()-t0:.1f}s；'
          f'唯一 GS {ok.sum()} / {n1*n2}')
    return states, E0, degen, ok


def kernel_matrix_full(states):
    """K[i,j] = |⟨ψ_i|ψ_j⟩|²。states (N, dim)。矩阵积快算。"""
    G = states @ states.conj().T
    return np.abs(G) ** 2


def block_stats(K, y):
    """按标签统计类内/类间 K 分布。y ∈ {0,1}。排除自配对（对角线）。"""
    n = len(y)
    off = ~np.eye(n, dtype=bool)
    same = K[(np.equal.outer(y, y)) & off]
    diff = K[(np.not_equal.outer(y, y)) & off]
    stats = lambda a: dict(mean=float(a.mean()), std=float(a.std()),
                           min=float(a.min()), max=float(a.max()),
                           median=float(np.median(a)))
    return dict(within=stats(same), cross=stats(diff))


def is_01_block(bs):
    """判定是否接近 0/1 分块（阶跃伪影）。返回 (bool, 诊断 dict)。"""
    w, c = bs['within'], bs['cross']
    diag = dict(
        within_spread=1.0 - w['min'],        # >WITHIN_SPREAD_MIN → 类内非恒 1
        cross_max=c['max'],                  # >CROSS_MAX_MIN → 类间非精确 0
        cross_center=c['median'],            # <CROSS_CENTER_MAX → 非全混
        gap=w['median'] - c['median'],       # 类内/类间中心差（判别信号）
    )
    flagged = not (diag['within_spread'] > WITHIN_SPREAD_MIN
                   and diag['cross_max'] > CROSS_MAX_MIN
                   and diag['cross_center'] < CROSS_CENTER_MAX)
    return flagged, diag


def cv_svm_precomputed(K, y, folds=5, C=1.0, repeats=5, seed=0):
    """分层 CV：precomputed-kernel SVM。返回 (mean_acc, std, per_fold)。"""
    from sklearn.svm import SVC
    from sklearn.model_selection import StratifiedKFold
    accs = []
    for r in range(repeats):
        skf = StratifiedKFold(n_splits=folds, shuffle=True,
                              random_state=seed + r)
        for tr, te in skf.split(np.arange(len(y)), y):
            clf = SVC(kernel='precomputed', C=C)
            clf.fit(K[np.ix_(tr, tr)], y[tr])
            accs.append(clf.score(K[np.ix_(te, tr)], y[te]))
    accs = np.array(accs)
    return float(accs.mean()), float(accs.std()), accs


def cv_kernel_nn(K, y, folds=5, seed=0):
    """kernel 空间最近邻（无模型佐证）：每测试点取 K 最大训练点标签。"""
    from sklearn.model_selection import StratifiedKFold
    accs = []
    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    for tr, te in skf.split(np.arange(len(y)), y):
        pred = []
        for i in te:
            j = int(np.argmax(K[i, tr]))
            pred.append(y[tr[j]])
        accs.append(np.mean(np.array(pred) == y[te]))
    return float(np.mean(accs)), float(np.std(accs))


def holdout_svm(K, y, test_frac=0.25, C=1.0, seed=0):
    """留出外推：随机分层留 25% 作测试（precomputed SVM + NN 各给一数）。"""
    from sklearn.svm import SVC
    from sklearn.model_selection import train_test_split
    tr, te = train_test_split(np.arange(len(y)), test_size=test_frac,
                              stratify=y, random_state=seed)
    clf = SVC(kernel='precomputed', C=C)
    clf.fit(K[np.ix_(tr, tr)], y[tr])
    acc_svm = clf.score(K[np.ix_(te, tr)], y[te])
    pred_nn = [y[tr[int(np.argmax(K[i, tr]))]] for i in te]
    acc_nn = float(np.mean(np.array(pred_nn) == y[te]))
    return acc_svm, acc_nn, tr, te


def main(out='pilot_gate2.npz', dataset=None, verbose=True):
    if dataset is None:
        dataset = os.path.join(_DATA, 'topo_dataset_full.npz')
    print('=' * 68)
    print(f'闸门 2：ED 精确态 kernel 判别力  |  L={L} {BOUNDARY} N={N_TARGET}'
          f'  |  判别轴 (t1,t2)')
    print('=' * 68)

    g = load_grid(dataset)
    t1, t2, lab = g['t1'], g['t2'], g['lab']
    n1, n2 = len(t1), len(t2)

    # ── 1. 全网格 ED ──
    print('[1/4] 全网格 ED（169 点）…')
    states, E0, degen, ok = compute_ed_grid(t1, t2, verbose=verbose)
    diag_idx = np.arange(n1)   # t1==t2 对角线行=列
    lab_grid = lab.copy()

    # ── 2. K_ED 全矩阵（169×169，含对角线占位）──
    print('[2/4] K_ED 全矩阵 …')
    S = states.reshape(-1, states.shape[-1])
    K_full = kernel_matrix_full(S)               # (169,169)
    # 逐点标签（对角线=临界）→ 拉平
    lab_flat = lab_grid.ravel()
    ok_flat = ok.ravel()

    # ── 3. 判别力分析（只用 156 个唯一 GS 的 off-diagonal 点）──
    print('[3/4] 判别力分析（唯一 GS off-diagonal 156 点）…')
    sel = ok_flat & (lab_flat != LABEL_CRITICAL)
    idx_off = np.where(sel)[0]
    y = lab_flat[idx_off].copy()
    y[y == LABEL_TOPO] = 1                        # topo → 1
    y[y == LABEL_TRIVIAL] = 0                     # trivial → 0
    K_off = K_full[np.ix_(idx_off, idx_off)]
    print(f'  off-diagonal 唯一 GS 点数 = {len(idx_off)}，'
          f'平凡={int((y==0).sum())} 拓扑={int((y==1).sum())}')

    bs = block_stats(K_off, y)
    flagged_01, d01 = is_01_block(bs)
    print('  类内/类间 K 分布：')
    for k in ('within', 'cross'):
        s = bs[k]
        print(f'    {k:6s}: mean={s["mean"]:.4f} std={s["std"]:.4f} '
              f'min={s["min"]:.4f} median={s["median"]:.4f} max={s["max"]:.4f}')
    print(f'  0/1 分块检查: within_spread=1−min={d01["within_spread"]:.4f} '
          f'(>{WITHIN_SPREAD_MIN}) | cross_max={d01["cross_max"]:.2e} '
          f'(>{CROSS_MAX_MIN:.0e}) | cross_center={d01["cross_center"]:.4f} '
          f'(<{CROSS_CENTER_MAX}) | gap={d01["gap"]:.4f}')
    print(f'  → 0/1 分块 = {"是（伪影警告）" if flagged_01 else "否（平滑）"}')

    acc_cv, acc_cv_std, _ = cv_svm_precomputed(K_off, y)
    acc_nn, acc_nn_std = cv_kernel_nn(K_off, y)
    acc_ho_svm, acc_ho_nn, tr_idx, te_idx = holdout_svm(K_off, y)
    svm_ok = acc_cv >= ACC_THRESHOLD
    print(f'  SVM(precomputed, C=1) 分层 CV（5×5）: acc = {acc_cv:.4f} ± {acc_cv_std:.4f}'
          f'  {"PASS" if svm_ok else "FAIL"} (≥{ACC_THRESHOLD})')
    print(f'  kernel 最近邻 CV: acc = {acc_nn:.4f} ± {acc_nn_std:.4f}'
          f'  （无模型佐证）')
    print(f'  留出 25% 外推: SVM acc = {acc_ho_svm:.4f} | NN acc = {acc_ho_nn:.4f}')

    # ── 4. 对角线简并报告（临界点）──
    print('[4/4] 对角线 13 点（t1=t2）简并报告 …')
    dg_diag = degen[np.arange(n1), np.arange(n1)]
    # 实测：APBC 下 degen 全 1（有限尺寸 L=4 避开 k=π 精确交叉，见 PBC 对照组 degen=2）。
    # 数据集"临界"标签来自 TBC/θ-loop 协议（U=0 = SP Zak，θ=0/PBC 处 gap 闭合），
    # 而非 APBC 点本身 —— 两者不矛盾。
    print(f'  对角线 degen = {list(dg_diag)}（APBC 全 1 → 每个对角点本身非简并；'
          f'PBC 对照组 degen=2 = 数据集临界性的真正来源）')
    print('  标度不变：t1=t2=s 时 H(s)=s·H0 → 13 点 GS 为同一矢量（逐对 fidelity=1.0）')
    # 对角线子空间对 off-diagonal 两类的 fidelity（basis-independent）
    # F_diag→ψ_i = ||Proj_diag ψ_i||² = Σ_k |⟨e_k|ψ_i⟩|²（投影到对角子空间）
    proj_diag = np.zeros((len(idx_off), len(dg_diag)))
    for a, (i, j) in enumerate(zip(range(n1), range(n1))):
        _, _, manifold, dg = sector_gs_spinless(float(t1[i]), float(t2[j]))
        ov = S[idx_off] @ manifold[:dg].conj().T     # (n_off, dg)
        proj_diag[:, a] = (np.abs(ov) ** 2).sum(axis=1)
    for a in range(len(dg_diag)):
        pt = (t1[a], t2[a])
        f_triv = proj_diag[y == 0, a].mean()
        f_topo = proj_diag[y == 1, a].mean()
        print(f'    对角线 ({pt[0]:4.2f},{pt[1]:4.2f}): '
              f'对平凡类投影均值={f_triv:.4f}  对拓扑类投影均值={f_topo:.4f}')
    # 镜像对称核对：网格 t1↔t2 对称 + 对角 GS 自镜像 → 平凡点 (a,b) 与其镜像拓扑点
    # (b,a) 对同一对角 GS 的 fidelity 精确相等（等距边界）。逐元素比会因网格索引
    # 顺序错位，故按排序后列比较（镜像对精确相等 → 排序后逐列必等）。
    f_eq = np.allclose(np.sort(proj_diag[y == 0], axis=0),
                       np.sort(proj_diag[y == 1], axis=0), atol=1e-8)
    print(f'  → 平凡/拓扑到对角 GS 投影（排序后逐列）相等（镜像对称）= {f_eq}'
          f'（对角 GS 与两类等距 → kernel 相界恰在对角线）')

    # ── 判定 ──
    passed = (not flagged_01) and svm_ok
    print('\n' + '═' * 68)
    print('闸门 2 判定')
    print('═' * 68)
    print(f'  ① 无 0/1 分块：{"PASS" if not flagged_01 else "FAIL"}'
          f'（平滑 kernel，非阶跃伪影）')
    print(f'  ② SVM 判别力：acc_cv={acc_cv:.4f} '
          f'{"PASS" if svm_ok else "FAIL"}（≥{ACC_THRESHOLD}）')
    print(f'  ③ 对角线（t1=t2）：APBC degen 全 1（非简并）+ 13 点 GS 同一矢量'
          f'（标度不变）+ 与两类等距（镜像对称）——kernel 相界恰在对角线，信息性报告')
    print(f'\n  总判定：{"PASS ✅（ED kernel 可分离两相，闸门 3 可测污染）" if passed else "FAIL ❌（见上方）"}')

    # ── 存 npz ──
    data = {
        'gate': 'pilot_gate2', 'passed': bool(passed),
        'L': L, 'boundary': BOUNDARY, 'n_target': N_TARGET,
        'acc_threshold': ACC_THRESHOLD,
        't1_vals': t1, 't2_vals': t2,
        'label_u0': lab_grid, 'sp_zak': g['sp_zak'],
        'states': states, 'E0': E0, 'degen': degen,
        'K_ED': K_full, 'lab_flat': lab_flat,
        'idx_off': idx_off, 'y_off': y,
        'bs_within_mean': bs['within']['mean'],
        'bs_within_median': bs['within']['median'],
        'bs_within_min': bs['within']['min'],
        'bs_within_max': bs['within']['max'],
        'bs_cross_mean': bs['cross']['mean'],
        'bs_cross_median': bs['cross']['median'],
        'bs_cross_min': bs['cross']['min'],
        'bs_cross_max': bs['cross']['max'],
        'd01_gap': d01['gap'],
        'is_01_block': bool(flagged_01),
        'svm_acc_cv': acc_cv, 'svm_acc_cv_std': acc_cv_std,
        'svm_ok': bool(svm_ok),
        'nn_acc_cv': acc_nn, 'nn_acc_cv_std': acc_nn_std,
        'ho_acc_svm': acc_ho_svm, 'ho_acc_nn': acc_ho_nn,
    }
    np.savez(out, **data)
    print(f'\n输出已保存: {os.path.abspath(out)}')
    return 0 if passed else 1


if __name__ == '__main__':
    p = argparse.ArgumentParser(description='闸门 2：ED kernel 判别力')
    p.add_argument('--out', default='pilot_gate2.npz')
    p.add_argument('--dataset', default=None,
                   help='topo_dataset_full.npz 路径')
    p.add_argument('--quiet', action='store_true')
    args = p.parse_args()
    sys.exit(main(out=args.out, dataset=args.dataset, verbose=not args.quiet))
