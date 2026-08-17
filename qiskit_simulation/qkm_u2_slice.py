#!/usr/bin/env python3
"""
qkm_u2_slice.py — ③ 量子原生栏：U=2 主切片 + U=0 参照切片（并行 + 可选 warm-start）
============================================================================
定位：qkm_grid_features.py 的"切片化 + 并行化"运行入口，产出 qkm_ml.py 直接
可用的 npz。不做任何新物理，只把原有全网格流程（逐点 DQAP-VQE）收敛到**少量
U 层**上跑，并顺手补一个精确参照核 K_ED。

与 qkm_grid_features.py 的关系：
    原有全网格版逐点**串行**（qkm_grid_features.py:159 的 for 循环，实测
    ~82 min/点，M=3, maxiter=2000）。本脚本 = 同一套原语 + 四个增量：
        A) --u-list 显式选 U 层（U=2 主 + U=0 参照 → 不用跑满 7 层 linspace(0,4)）
        B) --workers 并行：按 (U, t1) 分行，行内串行、行间 ProcessPool 并行
        C) --warm-start 行内 t2 链式热启动（x0 = 前一格点收敛参数；
           跨 t1=t2 对角线后 Q 是否塌 = --micro 微基准要回答的问题）
        D) K_ED：逐点存 ED GS 流形 → 流形间 fidelity 参照核（精确、~免费）

依赖标注（哪些调原有代码/环境，改动前请先读对应文件）：
    ── 原有代码 ──────────────────────────────────────────────────
    dqap_ssh_hubbard.py            build_ssh_hubbard_hamiltonian /
                                   build_dqap_circuit_spinful / run_vqe（x0 热启动）
    dqap_ssh_hubbard_tbc_berry.py  sector_gs / N_TARGET / DEGEN_TOL（ED N=8 扇区）
    qkm_grid_features.py           load_grid / kernel_matrix（仅复用，不修改）
    （本脚本**不碰** baseline_ml.py / kernel_ml_utils.py —— ① 的 feature 冻结期）
    ── 原有数据文件 ───────────────────────────────────────────────
    ../topo_dataset_full.npz       t1_vals/t2_vals/U_vals/label
                                   （唯一规格来源，轴直接读自它 → 口径零漂移）
    ── 环境 ──────────────────────────────────────────────────────
    numpy / qiskit.quantum_info.Statevector / scipy（optimize，经 run_vqe）
    concurrent.futures.ProcessPoolExecutor（Windows spawn → 必须 __main__ 保护；
    重模块 qiskit 在每 worker 里重新 import，属预期开销）

用法：
    # A0 微基准（先跑这个定样本量。3 次 VQE，两波并行，~2.7h 墙钟）
    python qkm_u2_slice.py --micro --workers 8

    # U=2 主 + U=0 参照切片（A0 定 max-pts 后；warm-start 待 A0 判定）
    python qkm_u2_slice.py --u-list 3,0 --max-pts 156 --workers 8
    python qkm_u2_slice.py --u-list 3,0 --max-pts 156 --workers 8 --warm-start --save-ed

    # 全网格（等于旧版行为，仅保底；实际应切片）
    python qkm_u2_slice.py --workers 8

输出：qkm_grid_M{M}.npz（默认路径 = qkm_ml.py 的默认读取路径，drop-in 兼容）
    = 原有 key（t1_vals/t2_vals/U_vals, idx, t1/t2/U/lab, K_DQAP, params,
                 E, E_ed, Q, M, maxiter, boundary, n_target, n_triv, n_topo）
    + K_ED      (n,n)  ED GS 流形 fidelity 核（精确参照，与 K_DQAP 同点同规）
    + ed_dims   (n,)   每点 ED GS 简并度（流形投影维度）
    --save-ed 时另存 ed_states (n, 2^(4L), dmax) 复数流形（~1MB/点，谨慎）
    --micro 时另出 qkm_a0_bench.npz（计时/Q 表）
"""

import argparse
import os
import sys
import time

try:
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
except Exception:
    pass

import numpy as np
from concurrent.futures import ProcessPoolExecutor
from qiskit.quantum_info import Statevector

_HERE = os.path.dirname(os.path.abspath(__file__))
_DATA = os.path.dirname(_HERE)
for p in (_HERE, _DATA):
    if p not in sys.path:
        sys.path.insert(0, p)

# ── 原有代码：DQAP 线路 / VQE 主循环 / ED N=8 扇区 ─────────────────────
from dqap_ssh_hubbard import (
    build_ssh_hubbard_hamiltonian,
    build_dqap_circuit_spinful,
    run_vqe,
)
from dqap_ssh_hubbard_tbc_berry import sector_gs, N_TARGET, DEGEN_TOL

# ── 原有代码：网格轴读取 + 单态 fidelity kernel（K_DQAP 复用）──────────
from qkm_grid_features import load_grid, kernel_matrix

L = 4
BOUNDARY = 'APBC'          # 与 pilot / 标签计算口径一致（勿改）
OUT_TMPL = os.path.join(_HERE, 'qkm_grid_M{M}.npz')
MICRO_OUT = os.path.join(_HERE, 'qkm_a0_bench.npz')


# ════════════════════════════════════════════════════════════════════════
# 切片选择
# ════════════════════════════════════════════════════════════════════════
def select_slice(t1, t2, U, label, u_idxs, max_pts=None):
    """C 序展平后取 U 层 ∈ u_idxs 且 label∈{0,1} 的点（剔除临界 2 / 未定 3，
    同 baseline_ml mask 口径）。返回 (idx, t1p, t2p, Up, lab)，idx 为全网格
    扁平索引 —— 与 qkm_ml.py 的 load_and_align 对齐口径完全一致。"""
    n1, n2, nu = len(t1), len(t2), len(U)
    u_idxs = sorted(u_idxs)                 # 升序 → idx 单调递增（与旧版同口径）
    idx_all = [i * n2 * nu + j * nu + k
               for i in range(n1) for j in range(n2)
               for k in u_idxs
               if 0 <= k < nu and label[i, j, k] in (0, 1)]
    if max_pts:
        rng = np.random.default_rng(42)      # 确定性子集（与旧 select_binary 同种子）
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


# ════════════════════════════════════════════════════════════════════════
# 并行 worker（顶层函数，Windows spawn 可 pickle；全在 qiskit/scipy 环境内）
# ════════════════════════════════════════════════════════════════════════
def _point_single(v, w, uu, M, maxiter, seed, x0=None):
    """单点：ED N=8 流形（Q 参照 + K_ED 素材）→ DQAP-VQE → ψ_VQE + 品质 Q。
    全为对原有代码的调用（见文件头依赖标注）。"""
    h = build_ssh_hubbard_hamiltonian(L, v, w, uu, BOUNDARY)      # 原有代码
    Mh = h.to_matrix(sparse=True).tocsr()
    E0_ed, _, vecs_ed, _ = sector_gs(Mh, N=N_TARGET, tol=DEGEN_TOL)  # 原有代码
    if x0 is not None:                        # run_vqe 内部也会验 3M 尺寸，这里
        x0 = np.asarray(x0, dtype=float).reshape(-1)   # 先兜底，错尺寸就当无热启动
        if x0.size != 3 * M:
            x0 = None
    res = run_vqe(L, v, w, uu, BOUNDARY, M, h,          # 原有代码（x0=热启动）
                  maxiter=maxiter, verbose=False, seed=seed, x0=x0)
    x = np.asarray(res.x, dtype=float)
    qc = build_dqap_circuit_spinful(L, x, v, w, uu, BOUNDARY)     # 原有代码
    sv = np.asarray(Statevector(qc).data, dtype=complex)          # qiskit 环境
    ov = vecs_ed @ sv.conj()                      # 向 ED GS 流形投影
    Q = float((np.abs(ov) ** 2).sum())
    return x, sv, float(res.fun), Q, float(E0_ed), np.asarray(vecs_ed, dtype=complex)


def _point_timed(v, w, uu, M, maxiter, seed, x0=None):
    """_point_single + 计时（A0 微基准用；顶层函数保证 ProcessPool 可 pickle）。"""
    t0 = time.perf_counter()
    x, sv, e, q, e0, v_ed = _point_single(v, w, uu, M, maxiter, seed, x0)
    return x, e, q, e0, time.perf_counter() - t0


def _row_worker(args):
    """一行 (U,t1) 的点：C 序内已按 t2 升序（idx 中 j 在中位），链式跑。
    warm=True 时 x0 = 前一格点收敛参数（参数延拓）；False 时全部随机起。
    返回 ((u_idx, t1_idx), [(flat_idx, x, sv, e, q, e0, v_ed), ...])。"""
    (u_idx, t1_idx), pts, M, maxiter, seed, warm = args
    out = []
    x0 = None
    for k, (fl, v, w, uu, _lab) in enumerate(pts):
        x, sv, e, q, e0, v_ed = _point_single(
            float(v), float(w), float(uu), M, maxiter, seed + int(fl), x0)
        out.append((int(fl), x, sv, e, q, e0, v_ed))
        if warm:
            x0 = x                                  # 链：下一格点热启动
    return (u_idx, t1_idx), out


def run_slice(idx, t1p, t2p, Up, lab, M, maxiter, seed, workers, warm,
              n2, nu, verbose=True):
    """按 (U,t1) 分行 → ProcessPool 并行；结果按原 idx 顺序重组。
    n2/nu 是网格尺寸（扁平索引反推 (i,j,k) 用），由 main 传入。"""
    i_of = lambda fl: fl // (n2 * nu)
    rows = {}
    for fl, v, w, uu, lb in zip(idx, t1p, t2p, Up, lab):
        i = i_of(int(fl))
        k = int(fl) % nu
        rows.setdefault((i, k), []).append((int(fl), v, w, uu, int(lb)))
    tasks = [((i, k), pts, M, maxiter, seed, warm)
             for (i, k), pts in rows.items()]
    results = {}
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for done, ((i, k), out) in enumerate(pool.map(_row_worker, tasks), 1):
            for fl, x, sv, e, q, e0, v_ed in out:
                results[fl] = (x, sv, e, q, e0, v_ed)
            if verbose:
                print(f'  行 {done}/{len(tasks)} (t1_idx={i}, U_idx={k}) '
                      f'完成  el={time.time()-t0:.0f}s', flush=True)
    # 按原 idx 顺序重组（输出顺序与 qkm_grid_features 一致）
    xs, svs, Es, Qs, E0s, Veds = [], [], [], [], [], []
    for fl in idx:
        x, sv, e, q, e0, v_ed = results[int(fl)]
        xs.append(x); svs.append(sv); Es.append(e); Qs.append(q)
        E0s.append(e0); Veds.append(v_ed)
    return (np.array(xs), np.array(svs), np.array(Es), np.array(Qs),
            np.array(E0s), Veds)


def kernel_manifold(Veds):
    """ED GS 流形 fidelity：K[i,j] = Tr(P_i P_j)/√(d_i d_j)。
    P_i = Σ|φ_a⟩⟨φ_a| 为 ED GS 流形投影（sector_gs 的列正交归一）。归一化使
    diag≡1、off-diag∈[0,1]，与 K_DQAP 同规可比；d=1 时退化为 |⟨φ|φ⟩|²。"""
    n = len(Veds)
    d = np.array([V.shape[1] for V in Veds], dtype=float)
    K = np.zeros((n, n))
    for i in range(n):
        Vi = Veds[i]
        for j in range(i, n):
            g = Vi.conj().T @ Veds[j]       # (d_i, d_j)
            K[i, j] = K[j, i] = float((np.abs(g) ** 2).sum()) / np.sqrt(d[i] * d[j])
    return K


# ════════════════════════════════════════════════════════════════════════
# A0 微基准：warm-start 值不值 + 跨对角线 Q 是否塌 → 定样本量
# ════════════════════════════════════════════════════════════════════════
def micro_bench(t1, t2, U, M, maxiter, seed, workers):
    """U=2 层、t1≈2.0 行、跨 t1=t2 对角线 3 点：
        p1 对角线下方（trivial 侧）random-init
        p2 对角线上方（topo 侧）  random-init
        p3 = p2 同点 warm-start（x0 = p1 参数）  ← 检验跨线热启动
    两波并行（wave1: p1；wave2: p2 random ∥ p2 warm）→ 表 + 结论。
    这是切片跑之前唯一要做的成本决策实验（~2.7h 墙钟, 8 核）。"""
    i_t1 = int(np.argmin(np.abs(t1 - 2.0)))
    i_u = int(np.argmin(np.abs(U - 2.0)))
    v0 = float(t1[i_t1]); uu = float(U[i_u])
    js = [j for j in range(len(t2)) if t2[j] < v0]
    j_b = max(js); j_a = j_b + 1                 # 紧邻对角线两侧的格点
    w1, w2 = float(t2[j_b]), float(t2[j_a])
    print('=' * 74)
    print(f'A0 微基准 — U={uu:.3f}（U 索引 {i_u}）, t1={v0:.3f}（索引 {i_t1}）')
    print(f'  跨 t1=t2 对角线：t2 下方 {w1:.3f}（trivial） 上方 {w2:.3f}（topo）')
    print('=' * 74)

    print(f'[wave 1] p1 (t2={w1:.3f}) random-init …', flush=True)
    x1, e1, q1, e01, t1s = _point_timed(v0, w1, uu, M, maxiter, seed)

    print(f'[wave 2] p2 (t2={w2:.3f}) random-init ∥ warm-start …', flush=True)
    # 依赖：warm 需要 x1 → 分两波；wave2 内部两任务可并行
    with ProcessPoolExecutor(max_workers=min(workers, 2)) as pool:
        f_r = pool.submit(_point_timed, v0, w2, uu, M, maxiter, seed, None)
        f_w = pool.submit(_point_timed, v0, w2, uu, M, maxiter, seed, x1)
        (x_r, e_r, q_r, e0_r, t_r) = f_r.result()
        (x_w, e_w, q_w, e0_w, t_w) = f_w.result()

    print('\n结果（t2 上方点，同一格点两种起法）：')
    print(f'  random-init :  Q={q_r:.4f}  ΔE={e_r-e0_r:9.3e}  耗时 {t_r:6.0f}s')
    print(f'  warm-start  :  Q={q_w:.4f}  ΔE={e_w-e0_w:9.3e}  耗时 {t_w:6.0f}s')
    speedup = t_r / t_w if t_w > 0 else float('nan')
    print(f'  提速 = {speedup:.2f}×   Q 差 = {q_w-q_r:+.4f}')
    print(f'\n参照（下方 trivial 点 random）：Q={q1:.4f}  ΔE={e1-e01:9.3e}'
          f'  耗时 {t1s:6.0f}s')

    ok_q = q_w >= max(q_r - 0.02, 0.90)          # warm 不塌 Q（宽松判据）
    ok_t = t_w < t_r                             # warm 至少不更慢
    print('\n' + '═' * 74)
    print('A0 判定')
    print('═' * 74)
    print(f'  ① warm 跨线后 Q 不塌（≥ max(Q_random−0.02, 0.90)）: '
          f'Q_warm={q_w:.4f}  {"PASS" if ok_q else "FAIL"}')
    print(f'  ② warm 不更慢（t_warm < t_random）: '
          f'{t_w:.0f}s vs {t_r:.0f}s  {"PASS" if ok_t else "FAIL"}')
    print('  含义：PASS → 切片可用 --warm-start（省的是墙钟，样本量可再想）；'
          'FAIL → 只用 ProcessPool 并行 + --max-pts 控样本')
    np.savez(MICRO_OUT, U=uu, t1=v0, t2_below=w1, t2_above=w2,
             q_random=q_r, q_warm=q_w, q_below=q1,
             dt_random=t_r, dt_warm=t_w, dt_below=t1s,
             de_random=e_r - e0_r, de_warm=e_w - e0_w,
             ok_q=bool(ok_q), ok_t=bool(ok_t), speedup=float(speedup))
    print(f'\n已存 {os.path.abspath(MICRO_OUT)}')
    return ok_q and ok_t


# ════════════════════════════════════════════════════════════════════════
# 主流程
# ════════════════════════════════════════════════════════════════════════
def main(M=3, maxiter=2000, seed=0, u_idxs=None, max_pts=None,
         workers=None, warm=False, save_ed=False, out=None, micro=False,
         verbose=True):
    t1, t2, U, label = load_grid()               # 原有代码：轴 = 唯一规格来源
    n1, n2, nu = len(t1), len(t2), len(U)

    if micro:
        return 0 if micro_bench(t1, t2, U, M, maxiter, seed,
                                workers or os.cpu_count() or 4) else 1

    if u_idxs is None:
        u_idxs = list(range(nu))                 # 默认全层（保底，实际应切片）
    else:
        bad = [k for k in u_idxs if not (0 <= k < nu)]
        if bad:
            print(f'[warn] U 索引越界被忽略：{bad}（有效 0..{nu-1}）')
            u_idxs = [k for k in u_idxs if 0 <= k < nu]
    u_str = '+'.join(f'U{U[iu]:.2f}' for iu in u_idxs)
    workers = workers or os.cpu_count() or 4

    print('=' * 78)
    print(f'③ 切片：DQAP-VQE fidelity kernel  |  L={L} {BOUNDARY} N={N_TARGET} 半满')
    print(f'  U 层={u_str}   M={M}  maxiter={maxiter}  workers={workers}'
          f'  warm={warm}')
    print('=' * 78)

    idx, t1p, t2p, Up, lab = select_slice(t1, t2, U, label, u_idxs,
                                          max_pts=max_pts)
    n = len(idx)
    if n == 0:
        print('所选 U 层无有效点（全为临界/未定？），退出。')
        return 1
    print(f'[1/4] 选点（剔除临界/未定，U 层={u_str}）: {n} 点'
          f'（triv={int((lab==0).sum())} topo={int((lab==1).sum())}）')

    print(f'[2/4] 逐点 DQAP-VQE（并行 {workers} worker'
          f'{"，" + "行内 t2 链式热启动" if warm else ""}）…', flush=True)
    xs, svs, Es, Qs, E0s, Veds = run_slice(
        idx, t1p, t2p, Up, lab, M, maxiter, seed, workers, warm, n2, nu,
        verbose)

    print(f'\n[3/4] 态品质：Q = |⟨ψ_DQAP|ψ_ED,N8⟩|² '
          f'mean={Qs.mean():.4f} min={Qs.min():.4f}  |  '
          f'ΔE mean={np.mean(Es-E0s):9.3e} max={np.max(np.abs(Es-E0s)):9.3e}')

    print('[4/4] kernel：K_DQAP（VQE 态）+ K_ED（ED 流形，精确参照）…')
    K_DQAP = kernel_matrix(svs)                    # 原有代码：|S S†|²
    K_ED = kernel_manifold(Veds)
    ed_dims = np.array([V.shape[1] for V in Veds])
    n_deg = int((ed_dims > 1).sum())
    print(f'  K_ED 简并点 {n_deg}/{n}（d>1）→ 投影流形 fidelity；'
          f'非简并处 = 单态 |⟨φ|φ⟩|²，同规可比')

    out_path = out or OUT_TMPL.format(M=M)
    data = dict(t1_vals=t1, t2_vals=t2, U_vals=U, idx=idx,
                t1=t1p, t2=t2p, U=Up, lab=lab,
                K_DQAP=K_DQAP, K_ED=K_ED, ed_dims=ed_dims,
                params=xs, E=Es, Q=Qs, E_ed=E0s,
                M=M, maxiter=maxiter, boundary=BOUNDARY, n_target=N_TARGET,
                n_triv=int((lab == 0).sum()), n_topo=int((lab == 1).sum()),
                u_idxs=np.array(u_idxs, dtype=int), warm=bool(warm))
    if save_ed:
        dmax = int(ed_dims.max())
        pad = np.zeros((n, svs[0].size, dmax), dtype=complex)
        for k, V in enumerate(Veds):
            pad[k, :, :V.shape[1]] = V          # 简并度不足的列补 0（投影核不用它们）
        data['ed_states'] = pad
    np.savez(out_path, **data)
    print(f'\n结果已存 {os.path.abspath(out_path)}'
          f'（qkm_ml.py 直接 --grid 读；含 K_ED 精确参照核）')
    return 0


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    p.add_argument('--M', type=int, default=3)
    p.add_argument('--maxiter', type=int, default=2000)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--u-list', type=str, default=None,
                   help='U 轴索引，逗号分隔（如 3,0 → U=2.00 主 + U=0.00 参照；'
                        '默认全部 0..6）')
    p.add_argument('--max-pts', type=int, default=None,
                   help='确定性随机子集上限（seed=42；warm-start 需整行，'
                        '子集时链的相邻格点可能跳远）')
    p.add_argument('--workers', type=int, default=None,
                   help='ProcessPool 并行数（默认 = CPU 数）')
    p.add_argument('--warm-start', action='store_true',
                   help='行内 t2 链式热启动（先跑 --micro 判定）')
    p.add_argument('--save-ed', action='store_true',
                   help='额外存 ED GS 流形（~1MB/点，默认只存 K_ED）')
    p.add_argument('--out', type=str, default=None,
                   help='输出 npz 路径（默认 qkm_grid_M{M}.npz，qkm_ml 直接读）')
    p.add_argument('--micro', action='store_true',
                   help='A0 微基准：3 点 warm vs random 定样本量后退出')
    a = p.parse_args()
    u_idxs = None if a.u_list is None else [
        int(x) for x in a.u_list.split(',') if x.strip()]
    sys.exit(main(M=a.M, maxiter=a.maxiter, seed=a.seed, u_idxs=u_idxs,
                  max_pts=a.max_pts, workers=a.workers, warm=a.warm_start,
                  save_ed=a.save_ed, out=a.out, micro=a.micro))
