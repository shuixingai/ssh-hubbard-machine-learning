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
X_grid=np.stack([gap4,gap3,gap2,gap1,dimer,ee,s_occ,lam_half],axis=-1)
feature_names = ['gap4','gap3','gap2','gap1','dimer','ee','s_occ','lam_half']
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

Xs_all = sc.fit_transform(X)
K_C = np.exp(-(1 / len(feature_names)) * (
    (Xs_all ** 2).sum(1)[:, None] + (Xs_all ** 2).sum(1)[None, :] - 2 * (Xs_all @ Xs_all.T)))
print(f"KTA(中心化) = {kta(K_C, y):+.4f}")



skf=StratifiedKFold(n_splits=5,shuffle=True,random_state=42)
accs=[]
for tr, te in skf.split(X,y):
    Xs_tr=sc.fit_transform(X[tr])
    Xs_te=sc.transform(X[te])
    clf=SVC(kernel="rbf",C=1.0,gamma="scale",random_state=42)
    clf.fit(Xs_tr,y[tr])
    accs.append(accuracy_score(y[te],clf.predict(Xs_te)))
print(f"平均准确率：{np.mean(accs)*100:.2f}%")

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
