# SSH-Hubbard 模型拓扑标签：Zak Phase 原理、计算与局限

> 文档目的：从 Berry phase 基本概念出发，完整覆盖 Zak phase 在 SSH-Hubbard 模型中的定义、计算方法、实现局限以及 ML 流水线中的角色。

---

## 1. Berry Phase 与 Zak Phase

### 1.1 Berry Phase 基本定义

考虑一个含参 Hamiltonian $H(\lambda)$，其本征态 $|n(\lambda)\rangle$ 在参数空间沿闭合路径 $\Gamma$ 绝热演化时，基态获得额外的相位：

$$
\gamma_n = \oint_\Gamma \mathcal{A}_n(\lambda) \cdot d\lambda
$$

其中 $\mathcal{A}_n(\lambda) = i\langle n(\lambda) | \nabla_\lambda | n(\lambda)\rangle$ 为 Berry 联络（Berry connection），$\gamma_n$ 即 Berry 相位。这个相位是**规范依赖**的，但 Berry 相位本身是**规范不变**（模 $2\pi$）的量。

### 1.2 Zak Phase：一维晶格的 Berry Phase

对于一维周期晶格，晶格动量 $k$ 是天然的参数——它定义在布里渊区 $[-\pi/a, \pi/a)$ 这个紧致流形上。Zak Phase 是 Bloch 态的 Berry Phase 沿整个布里渊区的闭合路径积分：

$$
\gamma_{\text{Zak}} = i \oint_{\text{BZ}} \langle u(k) | \partial_k | u(k) \rangle \, dk
$$

其中 $|u(k)\rangle$ 是 Bloch 函数的周期部分。和普通 Berry Phase 不同的是：Zak Phase 的规范依赖问题严重——它和 Bloch 动量的相位约定直接相关。这个问题的解决是：Zak Phase 本身不是直接可观测量，但它的模 $2\pi$ 取值（即 $e^{i\gamma_{\text{Zak}}}$）以及 **0 和 $\pi$ 的量子化跳变**对应拓扑相变。

---

## 2. SSH 模型 U=0 的解析拓扑不变量

### 2.1 SSH 模型 Hamiltonian

SSH 模型描述一维 dimerized 晶格上的无自旋费米子（或等价地，自旋极化的电子）：

$$
H = -\sum_{i} (t_i c^\dagger_{i+1} c_i + \text{h.c.})
$$

其中交替跳跃 $t_{2j-1} = t_1$（cell 内跳跃），$t_{2j} = t_2$（cell 间跳跃），$t_1, t_2 > 0$。

### 2.2 动量空间 Bloch Hamiltonian

在 PBC（周期边界条件）下做 Fourier 变换，得到 $2\times2$ Bloch Hamiltonian：

$$
H(k) = -\begin{pmatrix}
0 & t_1 + t_2 e^{-ik} \\
t_1 + t_2 e^{ik} & 0
\end{pmatrix} = \boldsymbol{d}(k) \cdot \boldsymbol{\sigma}
$$

其中 $d_x(k) = -(t_1 + t_2 \cos k),\ d_y(k) = -t_2 \sin k,\ d_z(k) = 0$。

### 2.3 缠绕数 W

由于 $d_z(k) = 0$，$\boldsymbol{d}(k)$ 在二维平面（$d_x$-$d_y$）中运动。缠绕数（winding number）衡量 $\boldsymbol{d}(k)$ 绕原点的圈数：

$$
W = \frac{1}{2\pi} \oint_{\text{BZ}} \partial_k \arg[d_x(k) + i d_y(k)]\, dk
$$

解析计算结果为：

$$
W = \begin{cases}
0, & t_1 > t_2 \quad (\text{trivial}) \\
1, & t_2 > t_1 \quad (\text{topological})
\end{cases}
$$

### 2.4 Zak Phase 与缠绕数的关系

对于 SSH 模型，Zak Phase 等于缠绕数的 $\pi$ 倍：

$$
\gamma_{\text{Zak}} = \pi\, W \quad (\text{mod } 2\pi)
$$

即：
- $t_1 > t_2$（平庸相）：$\gamma = 0$
- $t_2 > t_1$（拓扑相）：$\gamma = \pi$

---

## 3. 单粒子 Zak Phase 的 TBC 数值计算

这是在**数值 ED（精确对角化）**框架下，对任意参数（包括 $U > 0$）计算 Zak Phase 的方法——但需要注意，它本质上不包含相互作用。

### 3.1 方法：Twisted Boundary Condition (TBC)

在 OBC（开始边界条件）系统中，无法直接遍历布里渊区 $k$。TBC 通过将相位因子 $\theta$ 引入边界项来模拟 Bloch 动量：

$$
H(\theta) = -\sum_{i=1}^{L-1} t_i (c^\dagger_{i+1}c_i + \text{h.c.})
- t_{L}(c^\dagger_{1}c_0 e^{i\theta} + \text{h.c.})
$$

其中 $t_L$ 是最后一个键的跳跃，$e^{\pm i\theta}$ 是 Peierls 相位。当 $\theta$ 从 $0$ 扫描到 $2\pi$ 时，系统遍历所有边界条件，等价于连续改变动量 $k$。

### 3.2 Slater 行列式的 Berry Phase

对**自旋简并的** spinful 费米子系统（双自旋），单粒子 Zak Phase 的流程是：

1. 构造 **L×L 单粒子跳跃矩阵** $H_{\text{SP}}(\theta)$（只含跳跃项，不含 Hubbard $U$）
2. 每个 $\theta_n$ 点对角化 $H_{\text{SP}}(\theta_n)$，获得 L 个单粒子本征态 $\{\phi_1(\theta_n), \ldots, \phi_L(\theta_n)\}$
3. 填充最低 $L/2$ 个态（半满，一个自旋物种）构成 Slater 行列式 $\Psi_n = \text{Det}[\phi_1, \ldots, \phi_{L/2}]$
4. 相邻 $\theta$ 点的 Slater 行列式重叠：

$$
\langle \Psi_n | \Psi_{n+1} \rangle = \det[ \Phi_n^\dagger \Phi_{n+1} ],
\quad \Phi_n = [\phi_1(\theta_n), \ldots, \phi_{L/2}(\theta_n)]
$$

5. Zak Phase（Berry Phase）为这些重叠的乘积的相位：

$$
\gamma = -\text{Im}\ln \prod_{n=0}^{N_\theta-1} \det[\Phi_n^\dagger \Phi_{n+1}]
$$

![](https://i.imgur.com/mZ4iEmG.png)
*图示：$\theta$ 扫描中相邻 Slater 行列式示意图*

### 3.3 数值结果示例

扫描 $\theta$ 约 41-61 个点就可以获得稳定的 0/π 量子化结果。在 SSH 模型的点 $(t_1=0.5, t_2=1.5)$ 处：

```
U = 0.0: γ = 3.141593 (topological)
U = 2.0: γ = 3.141593 (topological)
U = 4.0: γ = 3.141593 (topological)
```

数值检验标准：
- `overlap_min`——相邻 $\theta$ 的最小重叠，近相变点时重叠骤降至 < 0.99
- `min_gap`——HOMO-LUMO 最小能隙，gap→0 处 Zak Phase 不连续

---

## 4. 为什么 Zak Phase 不包含 Hubbard U

这是核心问题。代码中的 `compute_sp_zak_phase()` **完全不包含 Hubbard U**，原因如下。

### 4.1 单粒子 vs 多体：根本区别

SSH-Hubbard 模型的全 Hamiltonian：

$$
H = \underbrace{-\sum_{i,\sigma} t_i(c^\dagger_{i,\sigma}c_{i+1,\sigma} + \text{h.c.})}_{H_{\text{hop}}}
+ \underbrace{U\sum_i n_{i\uparrow}n_{i\downarrow}}_{H_U}
$$

- **$H_{\text{hop}}$** 是二次型（quadratic），对角化一个 L×L 矩阵即可
- **$H_U$** 是四次型（quartic），在 Fock 空间中 $\binom{L}{L/2}^2$ 维度的矩阵

如果要在 Zak Phase 计算中包含 $U$，必须用**完整多体 Hamiltonian $H(\theta)$** 在每个 $\theta$ 点做 **Fock 空间 ED**。这就是下一节的多体 Zak Phase。

### 4.2 计算代价对比

| 步骤 | 单粒子 SP（当前） | 多体 MB（完整） |
|------|------------------|----------------|
| 矩阵维度 | L×L（L=6 → 6×6） | $\binom{L}{L/2}^2$（L=6 → ~400×400） |
| 每 $\theta$ 点对角化 | 纳秒级 | 毫秒级（L=6），秒级（L=10） |
| 重叠计算 | $\det(\Phi_n^\dagger\Phi_{n+1})$ | $\langle \Psi_{\text{GS}}(\theta_n) | \Psi_{\text{GS}}(\theta_{n+1}) \rangle$ |
| L 可扩展性 | L ≤ 200 | L ≤ 10（ED），L ≤ 20（稀疏矩阵） |
| 需要的 $\theta$ 点数 | 41-61 | 同上 |

### 4.3 物理论证：为什么 SP 就够用了

从物理角度，**SSH-Hubbard 模型（V=0）的拓扑相是受多体能隙保护的**。只要 $U$ 不大到关闭多体能隙、触发自发对称性破缺或 Mott 转变，系统的拓扑结构就和 $U=0$ 的非相互作用参考系绝热连接（adiabatically connected）：

$$
\gamma_{\text{SP}}(U) = \gamma_{\text{SP}}(U=0) = \pi W
\quad \text{当多体能隙保持打开时}
$$

当前数值范围 $U = 0 \sim 4$ 对 SSH-Hubbard 模型来说是中等相互作用强度。L=6 的 ED 观测表明多体能隙在该参数范围内保持开，因此 SP Zak Phase 作为标签是**正确且高效的近似**。

> 风险：在大 $U$ 极限（$U \gg t_1, t_2$）靠近 Mott 转变时，或靠近 $t_1 = t_2$ 相边界时，SP Zak Phase 可能和真正的多体拓扑有偏差。目前数据集尚未触碰这些危险区域。

---

## 5. 多体 Zak Phase

### 5.1 多体 Berry Phase 的一般形式

在 Fock 空间中，多体基态 $|\Psi_0(\theta)\rangle$ 在 $\theta$ 参数空间的 Berry Phase 是：

$$
\gamma_{\text{MB}} = -\text{Im}\ln \prod_{n} \langle\Psi_0(\theta_n)|\Psi_0(\theta_{n+1})\rangle
$$

这个计算在原理上和单粒子版本完全一致——只是 $|\Psi_0(\theta)\rangle$ 现在是 Fock 空间的基态而非 Slater 行列式。

### 5.2 SU(2) 对称性的致命问题

对于**自旋简并的 spinful 费米子系统**（SSH-Hubbard 就是这种情况），一个微妙的问题出现了：

**总多体 Zak Phase 恒等于 0（mod 2π）。**

原因是每个自旋物种各自贡献一个 Zak Phase $\gamma_\sigma$。由于 SU(2) 对称性（自旋上/下完全等价），两个自旋通道的贡献相同：

- 拓扑相：$\gamma_\uparrow = \pi,\ \gamma_\downarrow = \pi \rightarrow \gamma_{\text{MB}} = \gamma_\uparrow + \gamma_\downarrow = 2\pi \equiv 0 \pmod{2\pi}$
- 平庸相：$\gamma_\uparrow = 0,\ \gamma_\downarrow = 0 \rightarrow \gamma_{\text{MB}} = 0$

因此，对 SU(2) 对称的 SSH-Hubbard 模型，**总多体 Zak Phase 无法区分拓扑相和平庸相**——无论系统处在什么相，结果都是 0。

这正是代码中 `compute_mb_zak_phase()` 的警告信息的物理根源。

### 5.3 正确的多体拓扑不变量

对于 SU(2) 对称系统，正确的拓扑不变量是 $\mathbb{Z}_2$ **电荷 Berry Phase**（charge Berry phase / many-body polarization）：

$$
\gamma_c = -\text{Im}\ln \prod_{n} \langle\Psi_0(\theta_n)| e^{i\pi \hat{S}^z} |\Psi_0(\theta_{n+1})\rangle
$$

其中引入 `$\pi \hat{S}^z$` 相位打破 SU(2) 对称性。事实上，单粒子 Zak Phase 自动等价于这个 $\mathbb{Z}_2$ 不变量——因为 SP 计算通过忽略一个自旋物种（等价于做自旋极化）绕过了 SU(2) 问题。

关系总结：

| 计算方法 | U 依赖性 | SU(2) 问题 | 拓扑标签 |
|----------|---------|-----------|---------|
| SP Zak Phase | 无（参考系） | 无（取单自旋） | ✓ 0/π |
| MB Zak Phase（总） | 有 | 恒为 0 | ✗ 无用 |
| $\mathbb{Z}_2$ Charge Berry Phase | 有 | 无（$e^{i\pi S^z}$ 投影） | ✓ 0/π |
| SP Zak Phase（TBC） | 无 | 无 | **✓ 实用选择** |

### 5.4 代码中多体 Zak Phase 的实现

代码已包含 `compute_mb_zak_phase()`（约 738-817 行），其在每个 $\theta$ 点：

1. 构建完整多体 Hamiltonian $H(\theta)$（含 Hubbard U）
2. 在 Fock 空间进行精确对角化获得基态 $|\Psi_0(\theta)\rangle$
3. 计算相邻 $\theta$ 点的基态重叠
4. 累积 Berry Phase

**验证实验**：对一个拓扑点运行这个函数，结果总为 0（mod 2π），和理论预测一致。

---

## 6. Resta 极化

### 6.1 定义

Resta 极化（Resta 1998）提供了一种直接在实空间中计算多体极化的方法，不需要扫描 $\theta$：

$$
P = \frac{1}{2\pi} \text{Im} \ln \langle \Psi_0 | e^{i 2\pi \hat{X} / L} | \Psi_0 \rangle
$$

其中 $\hat{X} = \sum_i x_i \hat{n}_i$ 是位置算符。对 SSH 模型，$P$ 在拓扑相中取 $1/2$（对应 $\gamma = \pi$），在平庸相中取 $0$（对应 $\gamma = 0$）。

### 6.2 优点与局限

**优点**：
- 只需要**一次** ED 计算（无需扫描 $\theta$），计算代价低
- 天然包含 Hubbard U
- 可扩展到更大的 L

**局限**：
- 在非常小的系统（L=6）中，由于 $\exp(i 2\pi \hat{X} / L)$ 的相位分辨率不足，极化值会偏离 0 和 1/2
- 实测 L=6 时本项目发现 Resta 极化失效（极化值在相变处的跳变被有限尺寸效应抹平）

### 6.3 SP Zak Phase vs Resta 极化

| | SP Zak Phase (TBC) | Resta 极化 |
|---|---|---|
| 计算代价 | $N_\theta$ 次 SP diag | 1 次 ED |
| 含 Hubbard U | ✗ | ✓ |
| L=6 可用 | ✓ | ✗（L 太小） |
| L≫1 可用 | ✓ | ✓ |
| 适用性 | 当前最优 | 未来 L 增大时可做交叉验证 |

---

## 7. 拓扑不变量全景图

```
Berry Phase (绝热, 含参闭合路径)
  │
  ├─ Bloch 态 Brillouin Zone 积分 → Zak Phase (1D 拓扑绝缘体)
  │     │
  │     ├─ U=0 解析：γ = π × W（缠绕数，严格解析）
  │     │
  │     ├─ U>0 数值-单粒子近似：γ = -Im ln ∏ det[Φⁿ†Φⁿ⁺¹]
  │     │     （SP Hamiltonian，L×L 对角线化，忽略 Hubbard U）
  │     │
  │     ├─ U>0 数值-多体：γ = -Im ln ∏ ⟨Ψⁿ₀|Ψⁿ⁺¹₀⟩
  │     │     （Fock space ED，含 U，但 SU(2) → ≡0 mod 2π）
  │     │
  │     └─ 正确的多体不变量：Z₂ Charge Berry Phase
  │           （引入 e^{iπS^z} 打破 SU(2) → 即 SP 的结果）
  │
  └─ Resta 极化 (实空间，一次 ED)
        P = (1/2π) Im ln ⟨Ψ₀|e^{i2πX/L}|Ψ₀⟩
        └─ L 大时等价于 Zak Phase，L=6 失效
```

### 实践结论

| 需求 | 推荐方法 |
|------|---------|
| U=0 快速标签 | `compute_winding_number()`（解析，零计算量） |
| U>0 数值标签（当前） | `compute_sp_zak_phase()`（SP TBC，高效可用） |
| 多体严格验证 | Z₂ Charge Berry Phase（需实现 $e^{i\pi\hat{S}^z}$ 投影） |
| 大 L 交叉验证 | Resta 极化（L≥10 可考虑） |

---

## 8. ML 流水线中的标签策略

基于以上所有讨论，SSH-Hubbard 拓扑相识别的 ML 标签策略为：

```
输入特征 X                          标签 y
────────────────────────────          ────────────
corr_matrices (36 维)                 SP Zak Phase
ent_spectra   (20 维)                 → 0 (trivial)
(备用: docc, δB 等标量)                 → π (topological)
```

**标签归属**：
- **y** = Zak Phase（三层标签体系：启发式 → 解析 W → SP Zak Phase）
- **X** = 可观测量（关联矩阵、纠缠谱等，由 ED 产生，**不能包含 Zak Phase 本身**）

**三个关键认知**：

1. **SP Zak Phase 是近似的**——它不包含 Hubbard U 的贡献，但在中等 U 下是绝热连接的正确标签
2. **ML 模型学会的是可观测量和拓扑之间的关联**——如果 ML 在 U>0 处精度下降，那可能标志着 SP 标签和真正多体拓扑的分歧，而非 ML 失败
3. **大 U 数据的标签需要升级**——当数据集扩展到 U > 6-8 时，应当实现 $\mathbb{Z}_2$ Charge Berry Phase 或使用 DMRG 的 Resta 极化做多体验证

---

> **参考文献**
>
> 1. Berry, M. V. "Quantal phase factors accompanying adiabatic changes." *Proc. R. Soc. Lond. A* 392, 45–57 (1984).
> 2. Zak, J. "Berry's phase for energy bands in solids." *Phys. Rev. Lett.* 62, 2747 (1989).
> 3. Resta, R. "Quantum-Mechanical Position Operator in Extended Systems." *Phys. Rev. Lett.* 80, 1800 (1998).
> 4. Su, W. P., Schrieffer, J. R. & Heeger, A. J. "Solitons in Polyacetylene." *Phys. Rev. Lett.* 42, 1698 (1979).
> 5. Manmana, S. R., et al. "Topological phases in the SSH-Hubbard model." *Phys. Rev. B* 85, 155118 (2012).
> 6. Niu, Q. & Thouless, D. J. "Quantised adiabatic charge transport in the presence of substrate disorder and many-body interactions." *J. Phys. A* 17, 2453 (1984).
