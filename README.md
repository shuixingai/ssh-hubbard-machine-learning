# SSH-Hubbard DQAP / QKM 机器学习

SSH-Hubbard 模型拓扑相识别（拓扑 ↔ 平庸）的机器学习项目。三条路线的同点对决：

| 轨道 | 方法 | 入口脚本 |
|------|------|----------|
| ① | 经典多体标量 → RBF-SVM | `baseline_ml.py --feat classical5` |
| ② | 经典特征 → Havlíček ZZ 特征图 → 线路 fidelity 核 → SVM | `qiskit_simulation/qkm_featuremap.py` |
| ③ | DQAP 物理原生态 → fidelity 核（K_ED 精确参照 + K_DQAP 变分） | `qiskit_simulation/qkm_u2_slice.py` + `qkm_ml.py` |

拓扑标签用 TBC Berry phase（γ_up，见 `build_topo_dataset.py`），DMRG 交叉验证在 `dmrg/`（Julia）。

## 环境（两个 Python + Julia，互不兼容）

| 环境 | 内容 | 安装 |
|------|------|------|
| conda `quspin_env`（py3.9） | 轨道① 经典 + `ssh_model.py` ED 数据生成 | `conda env create -f environment.yml` |
| pip（默认 env） | 轨道②③（qiskit/aer）+ ML 对决 | `pip install -r requirements-qiskit.txt` |
| Julia 1.11（`dmrg/`） | DMRG 交叉验证 | `cd dmrg && julia --project=.` |

依赖互斥原因：quspin 需要 numpy≤2.3，qiskit 轨道用 numpy 2.5。

## 运行

```bash
# 轨道① 经典（quspin_env）
python baseline_ml.py --feat classical5

# 轨道② 特征图 QKM（pip 环境）
python qiskit_simulation/qkm_featuremap.py --feat ee,dimer
python qiskit_simulation/qkm_featuremap.py --feat gap4,ee,dimer,s_occ,lam_half

# 轨道③ DQAP fidelity 核
python qiskit_simulation/qkm_u2_slice.py --M 3 --save-ed --warm-start
python qiskit_simulation/qkm_ml.py --M 3 --grid qiskit_simulation/qkm_featuremap.npz

# 数据重生成（可选，数据集已随仓库提交）
python build_topo_dataset.py --quick          # γ_up 标签烟测
python build_topo_dataset.py                  # 全量标签（13×13×7 网格）
```

`run.py` 会自检当前环境并打印以上命令。

## 目录结构

```
data_creation/
├── baseline_ml.py          ① 经典三栏（gap4/ent8/classical5）
├── kernel_ml_utils.py      共享 kernel / SVM / CV 层
├── ssh_model.py            quspin ED 数据生成（generate_dataset）
├── build_topo_dataset.py   γ_up 拓扑标签（TBC Berry phase）
├── report_classical5.py + report_classical5/   ① 论文报告
├── qiskit_simulation/      ②③ 量子路线（特征图 / DQAP / 三栏对决）
├── dmrg/                   Julia DMRG（ITensors，Project/Manifest 已锁）
├── archive/                使命已完成的验证基建，仅参考（见下）
├── environment.yml + requirements-qiskit.txt
└── *.npz                   数据集（见下）
```

## 数据文件

| 文件 | 内容 |
|------|------|
| `topo_dataset_full.npz` | (t1,t2,U) 网格 + γ_up 拓扑标签 |
| `ssh_dataset_L8_labelgrid.npz` | L=8 OBC 多体特征（gap4/ee/dimer/s_occ/lam_half） |
| `ssh_dataset_L6.npz` / `_refined` | L=6 早期口径数据 |
| `baseline_results*.npz` | ① 各特征集 SVM 结果 |
| `qiskit_simulation/qkm_featuremap*.npz` | ② 核矩阵 + drop-in `K_DQAP` 别名 |
| `qiskit_simulation/qkm_grid_M3.npz` | ③ DQAP 态 fidelity 核 |
| `dmrg/dmrg_dataset.npz` | DMRG 交叉验证 |

## ⚠️ 口径警示（勿破坏）

拓扑标签与特征使用**不同尺寸/边界条件**：

- **标签**：L=4 cell（8 sites），PBC/TBC，γ_up Z2 Berry phase（`tbc_berry_scan`）
- **特征**：L=8 sites，OBC（`ssh_dataset_L8_labelgrid.npz`）
- 对齐由 `load_grid()` / `build_features_grid()` 内部 `np.allclose` 校验 t1/t2/U 轴，
  选点一律按 C 序展平 idx 数学，**不要手动重排点集**（L 口径坑）。
- `ssh_dataset_L6*.npz` 为早期口径，仅作对比，不作标签来源。

## archive/ 说明

`archive/` 是使命已完成的验证基础设施，仅作参考、不参与主流程：

- `pilots/`：DQAP 管线闸门 1–5（自包含互导簇）
- `probes/`：探测脚本 —— **运行需把仓库根目录加回 sys.path**（它们导入
  `qiskit_simulation/dqap_ssh_hubbard`）
- `verification/`：Zak/winding 标签校验
- `4picture/`：一次性 PCA（其中 50MB 大 npz 已 gitignore，不入库）
