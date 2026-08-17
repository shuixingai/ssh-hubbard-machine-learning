# 经典端 baseline 汇报（classical5，2026-08-14）
**SSH-Hubbard L=8 OBC 特征 → PBC/TBC γ_up 拓扑标签**，两套计算完全独立，只共享 (t1,t2,U) 坐标（无污染）。
## 三条结论
1. **经典 5 特征 RBF-SVM = 100.0%**（Wilson CI [99.6,100.0]），shuffle-null = 51.4%±2σ → 信号显著，不是碰运气。
2. **100% 完全来自单个特征 ee（ES 纠缠熵）**：trivial 均值 0.45 vs topological 1.90，在 L=8 上近乎完备序参量。dimer 有用（KTA 0.73 / 单特征 88.6%）；s_occ 是死特征（KTA 0.04 ≈ 0），已从候选集剔除。
3. **这是信息论上界参照，不是栏**：QKM 要跨的栏仍是 gap4 单特征 69.5%。量子端的 claim 是经典吃紧处的可扩展性/硬件路线，不是赢过经典。
## 图
- `fig1_feature_plane.png`
- `fig2_phase_map.png`
- `fig3_diagnostics.png`
- `fig4_real_vs_null.png`
## 想请教的问题
1. ee（ES 纠缠熵）在 L=8 已把两相分得如此开，是否会重新定义相边界？论文 framing 要不要动？
2. 在您看来，量子 kernel 要表现出什么才算'值得做'？——经典天花板这么高，量子端只能回答经典做不到的东西。

_成本说明：量子端全网格 VQE 约 62–80 天单线程，当前先走分层验证（L0 oracle → L1 探针 → L2 子网格）。_
