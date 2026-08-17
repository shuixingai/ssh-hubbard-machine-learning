#!/usr/bin/env python3
"""
run.py — 仓库薄入口：环境自检 + 各轨道运行命令
============================================
本仓库横跨两个不兼容的 Python 环境 + Julia，run.py 只做两件事：
  1) 检测当前 Python 环境装了哪些依赖，判断能跑哪个轨道
  2) 打印三个轨道的完整命令（复制粘贴即可）

用法：
    python run.py         # 环境自检 + 命令
    python run.py --cmds  # 只打印命令
"""

import importlib.util
import sys

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

DEPS = {
    "quspin  (轨道① 数据生成 / 经典)": "quspin",
    "qiskit  (轨道②③ 量子)": "qiskit",
    "qiskit_aer": "qiskit_aer",
    "numpy": "numpy",
    "scipy": "scipy",
    "sklearn (ML 对决)": "sklearn",
    "matplotlib": "matplotlib",
}

CMDS = """\
────────────────────────────────────────────────────────────────
 轨道1 经典 SVM —— 需 quspin_env（conda activate quspin_env）
   python baseline_ml.py --feat classical5

 轨道2 特征图 QKM（Havlíček ZZ）—— pip 环境
   python qiskit_simulation/qkm_featuremap.py --feat ee,dimer
   python qiskit_simulation/qkm_featuremap.py --feat gap4,ee,dimer,s_occ,lam_half

 轨道3 DQAP fidelity 核（K_ED 精确参照 + K_DQAP 变分）
   python qiskit_simulation/qkm_u2_slice.py --M 3 --save-ed --warm-start
   python qiskit_simulation/qkm_ml.py --M 3 --grid qiskit_simulation/qkm_featuremap.npz

 数据重生成（可选，数据集已随仓库提交）
   python build_topo_dataset.py --quick     # γ_up 标签烟测
   python build_topo_dataset.py             # 全量标签 13×13×7
   python ssh_model.py --L 6 --t1 1.0 --t2 0.8 --U 2.0   # 单点 ED
   # ED 网格数据集（ssh_dataset_L6/L8*.npz）由 ssh_model.generate_dataset()
   # 批量生成，见其 docstring；commit 里已含现成数据，非必须重跑。

 DMRG 交叉验证（Julia 1.11）
   cd dmrg && julia --project=. ssh_dmrg.jl

 报告
   python report_classical5.py              # ① 论文报告
────────────────────────────────────────────────────────────────
"""


def _works(mod):
    """真 import 测试（find_spec 只查文件存在，quspin 装了但 numpy 不兼容也算失败）。"""
    try:
        importlib.import_module(mod)
        return True
    except Exception:
        return False


def check_env():
    print(f"Python {sys.version.split()[0]}   {sys.executable}")
    print("-" * 60)
    for label, mod in DEPS.items():
        ok = _works(mod)
        print(f"  [{'OK' if ok else '--'}]  {label}")
    print()
    if _works("quspin"):
        print("  → 当前是 quspin 环境：可跑 轨道1 数据生成 + 经典 ML")
    elif _works("qiskit"):
        print("  → 当前是 qiskit 环境：可跑 轨道23 量子路线 + ML 对决")
    else:
        print("  → 缺依赖：按 README 装 quspin_env（conda）或 qiskit（pip）")
    print()


if __name__ == "__main__":
    if "--cmds" in sys.argv:
        print(CMDS)
    else:
        check_env()
        print(CMDS)
