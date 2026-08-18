# 1. 导入必要的库
import os
import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
HERE=os.path.dirname(os.path.abspath(__file__))
FEAT_NPZ=os.path.join(HERE,"data_creation","ssh_dataset_L8_labelgrid.npz")
LABEL_NPZ=os.path.join(HERE,"data_creation","topo_dataset_full.npz")

feat=np.load(FEAT_NPZ)
topo=np.load(LABEL_NPZ)
assert np.allclose(feat["t1_arr"],topo["t1_vals"]),"T1 WRONG"
assert np.allclose(feat["t2_arr"],topo["t2_vals"]),"T2 WRONG"
assert np.allclose(feat["U_arr"],topo["U_vals"]),"U WRONG"
ent=feat["ent_spectra"]
gap4=ent[...,4]-ent[...,3]
gap3=ent[...,3]-ent[...,2]
gap2=ent[...,2]-ent[...,1]
gap1=ent[...,1]-ent[...,0]
p=np.exp(-ent);p=p / p.sum(-1,keepdims=True)
ee=-(p*np.log(np.clip(p,1e-15,None))).sum(-1)
dimer=feat["bond_alternation"]
corr=feat["corr_matrices"]
ev=np.linalg.eigvalsh(corr)[...,::-1]
l=np.clip(ev,1e-12,1 - 1e-12)
s_occ=-(l*np.log(l)+(1-l)*np.log(1-l)).sum(-1)
lam_half=np.abs(ev-0.5).min(-1)

"""
X_grid=np.stack([gap4,gap3,gap2,s_occ,lam_half],axis=-1)
feature_names = ['gap4','gap3','gap2','s_occ','lam_half']

X_grid=np.stack([gap4,gap3,gap2,gap1,dimer,ee,s_occ,lam_half],axis=-1)
feature_names = ['gap4','gap3','gap2','gap1','dimer','ee','s_occ','lam_half']
"""
X_grid=np.stack([gap1,dimer,ee],axis=-1)
feature_names = ['gap1','dimer','ee']


label=topo["label"]
mask=((label==0)|(label==1)).reshape(-1)
X_all=X_grid.reshape(-1,len(feature_names))
X=X_all[mask]
y=label.reshape(-1)[mask].astype(int)
sc=StandardScaler()

def center_kernel(K):
    n = K.shape[0]
    J = np.ones((n, n)) / n
    return K - J @ K - K @ J + J @ K @ J

def kta(K, y):
    y = np.asarray(y, dtype=float)
    Kc = center_kernel(np.asarray(K, dtype=float))
    yc = y - y.mean()
    yy = np.outer(yc, yc)
    num = float(np.sum(Kc * yy))
    den = np.linalg.norm(Kc, "fro") * np.linalg.norm(yy, "fro")
    return num / den if den > 0 else 0.0

C_GRID = [10.0, 30.0, 100.0, 300.0, 1000.0]
GAMMA_GRID = [2.0, 3.0, 4.0, 5.0, 6.0, 8.0]
KERNEL_NAMES = ["rbf", "poly"]   # kernel 家族：RBF（无限维）+ 多项式（低阶交互）
DEGREE_GRID = [2, 3]             # poly 专用；rbf 无 degree，占位 0
COEF0 = 1.0                      # poly 截距，固定不扫


Xs_all = sc.fit_transform(X)
K_C = np.exp(-(1 / len(feature_names)) * (
    (Xs_all ** 2).sum(1)[:, None] + (Xs_all ** 2).sum(1)[None, :] - 2 * (Xs_all @ Xs_all.T)))
print(f"KTA(中心化) = {kta(K_C, y):+.4f}")
for d in DEGREE_GRID:
    Kp = (Xs_all @ Xs_all.T / len(feature_names) + COEF0) ** d
    print(f"KTA(poly deg={d}) = {kta(Kp, y):+.4f}")


skf=StratifiedKFold(n_splits=5,shuffle=True,random_state=42)
accs=[]
for tr, te in skf.split(X,y):
    Xs_tr=sc.fit_transform(X[tr])
    Xs_te=sc.transform(X[te])
    clf=SVC(kernel="rbf",C=1.0,gamma="scale",random_state=42)
    clf.fit(Xs_tr,y[tr])
    accs.append(accuracy_score(y[te],clf.predict(Xs_te)))
print(f"平均准确率：{np.mean(accs)*100:.2f}%")


print(f"\n{'kernel':>5} {'deg':>3} {'C':>5} {'gamma':>7} {'acc%':>7} {'SV占比':>8}")
best = None
for kn in KERNEL_NAMES:
    for d in (DEGREE_GRID if kn == "poly" else [0]):
        for Cv in C_GRID:
            for g in GAMMA_GRID:
                accs_cg, svfracs = [], []
                for tr, te in skf.split(X, y):
                    Xs_tr = sc.fit_transform(X[tr])
                    Xs_te = sc.transform(X[te])
                    if kn == "poly":
                        clf = SVC(kernel="poly", C=Cv, gamma=g, degree=d,
                                  coef0=COEF0, random_state=42)
                    else:
                        clf = SVC(kernel="rbf", C=Cv, gamma=g, random_state=42)
                    clf.fit(Xs_tr, y[tr])
                    accs_cg.append(accuracy_score(y[te], clf.predict(Xs_te)))
                    svfracs.append(len(clf.support_) / len(tr))
                m_acc = np.mean(accs_cg) * 100
                m_sv = np.mean(svfracs)
                print(f"{kn:>5} {d:>3} {Cv:>5.1f} {g:>7.3g} {m_acc:>7.1f} {m_sv:>8.1%}")
                if best is None or m_acc > best[0]:
                    best = (m_acc, kn, d, Cv, g, m_sv)
print(f"\n最优：kernel={best[1]}, degree={best[2]}, C={best[3]}, gamma={best[4]}, "
      f"acc={best[0]:.1f}%, SV占比={best[5]:.1%}")



"""
# ── 单特征诊断：各自 acc + KTA ──────────────────────────────────
# X 每列 = 一个特征在所有点上的取值，切片 X[:, i] → 单列即可。
# 单特征标准化后 var=1 → sklearn gamma="scale" = 1/(1·1) = 1.0
gamma_single = 1.0
print(f"\n{'feature':<10}{'acc%':>8}{'KTA':>10}")
for i, name in enumerate(feature_names):
    a = X[:, i].reshape(-1, 1)

    # 单特征 acc：同一 5-fold 协议
    accs_i = []
    for tr, te in skf.split(a, y):
        Xs_tr = sc.fit_transform(a[tr])
        Xs_te = sc.transform(a[te])
        clf = SVC(kernel="rbf", C=1.0, gamma="scale", random_state=42)
        clf.fit(Xs_tr, y[tr])
        accs_i.append(accuracy_score(y[te], clf.predict(Xs_te)))
    acc_i = np.mean(accs_i) * 100

    # 单特征 KTA：全量标准化单列 + RBF（gamma=1.0）
    a_sc = sc.fit_transform(a)
    K_i = np.exp(-gamma_single * ((a_sc**2).sum(1)[:, None]
                                  + (a_sc**2).sum(1)[None, :] - 2 * (a_sc @ a_sc.T)))
    kta_i = kta(K_i, y)

    print(f"{name:<10}{acc_i:8.1f}{kta_i:+10.4f}")
"""
# ── ee vs gap1 冗余检查 ──────────────────────────────────────────
from scipy.stats import spearmanr, pearsonr
from numpy.polynomial import polynomial as P
ee_f = ee.reshape(-1)[mask]
g1_f = gap1.reshape(-1)[mask]
t1_f = np.broadcast_to(feat["t1_arr"][:, None, None], ee.shape).reshape(-1)[mask]
t2_f = np.broadcast_to(feat["t2_arr"][None, :, None], ee.shape).reshape(-1)[mask]

# 探针 1：秩相关
r_p, _ = pearsonr(ee_f, g1_f)
r_s, _ = spearmanr(ee_f, g1_f)
print(f"Pearson={r_p:+.3f}  Spearman={r_s:+.3f}   # |秩相关|~1 → 单调等价")

# 探针 2：残差判别力（gap1 去掉 ee 的单调成分后还剩什么）
from scipy.stats import rankdata
r_ee, r_g1 = rankdata(ee_f), rankdata(g1_f)
coef = np.polyfit(r_ee, r_g1, 3)
res  = g1_f - np.polyval(coef, r_ee)            # ee 解释不掉的 gap1
a_res = sc.fit_transform(res.reshape(-1, 1))
K_res = np.exp(-((a_res**2).sum(1)[:, None] + (a_res**2).sum(1)[None, :]
                 - 2 * (a_res @ a_res.T)))
print(f"gap1 残差 KTA = {kta(K_res, y):+.4f}   # ≈0 → 冗余；有信号 → 不冗余")

# 探针 3：临界条带（|t1-t2| 最小的 10%）
d_diag = np.abs(t1_f - t2_f)
strip  = d_diag <= np.sort(d_diag)[int(0.10 * len(d_diag))]
def cohen_d(a):
    m0 = a[y[strip] == 0]; m1 = a[y[strip] == 1]
    return (m1.mean() - m0.mean()) / np.sqrt((m0.std()**2 + m1.std()**2) / 2)
print(f"条带内 Cohen d:  gap1={cohen_d(g1_f[strip]):+.2f}   ee={cohen_d(ee_f[strip]):+.2f}")
