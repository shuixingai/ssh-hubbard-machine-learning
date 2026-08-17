#!/usr/bin/env python3
"""
测试 5 独立版：PBC vs APBC 极化对比 — 验证 π flux 假设
"""
import numpy as np
from qiskit.quantum_info import SparsePauliOp, Statevector

# 手动内联 build_ssh_hamiltonian 和 compute_polarization
# 避免跨文件 import 副作用
from utils import _pauli_label, diagonalize_hamiltonian

def build_ssh_hamiltonian(L, v, w, boundary='APBC'):
    nq = 2 * L
    terms = []
    for i in range(L):
        ai = 2 * i
        bi = 2 * i + 1
        terms.append((_pauli_label(nq, [ai, bi], ['X', 'X']), float(-v / 2.0)))
        terms.append((_pauli_label(nq, [ai, bi], ['Y', 'Y']), float(-v / 2.0)))
    for i in range(L - 1):
        bi = 2 * i + 1
        aj = 2 * (i + 1)
        terms.append((_pauli_label(nq, [bi, aj], ['X', 'X']), float(-w / 2.0)))
        terms.append((_pauli_label(nq, [bi, aj], ['Y', 'Y']), float(-w / 2.0)))
    if boundary in ('PBC', 'APBC'):
        bl = 2 * L - 1
        a0 = 0
        filler = list(range(a0 + 1, bl))
        sign = 1.0 if boundary == 'PBC' else -1.0
        terms.append((_pauli_label(nq, [a0] + filler + [bl], ['X'] + ['Z']*len(filler) + ['X']), float(sign * -w / 2.0)))
        terms.append((_pauli_label(nq, [a0] + filler + [bl], ['Y'] + ['Z']*len(filler) + ['Y']), float(sign * -w / 2.0)))
    return SparsePauliOp.from_list([(l, complex(c)) for l, c in terms]).simplify()

def compute_polarization(sv, L):
    nq = 2 * L
    sv_data = sv.data
    expval = 0.0 + 0.0j
    for idx, amp in enumerate(sv_data):
        if abs(amp) < 1e-15:
            continue
        bits = format(idx, f'0{nq}b')
        phase = 0
        for j in range(L):
            n_j = int(bits[2*j]) + int(bits[2*j+1])
            phase += (j + 1) * n_j
        prob = (amp.conjugate() * amp).real
        expval += prob * np.exp(1j * 2.0 * np.pi / L * phase)
    return np.angle(expval) / (2.0 * np.pi), expval

# ── Spinless 参考 ──
L = 4
v_triv, w_triv = 2.0, 1.0
v_topo, w_topo = 1.0, 2.0

print("=" * 70)
print("【测试 5】PBC vs APBC 极化 — π flux 假设验证")
print("=" * 70)

print("\nSpinless SSH:")
for bc in ['PBC', 'APBC']:
    h_triv = build_ssh_hamiltonian(L, v_triv, w_triv, bc)
    h_topo = build_ssh_hamiltonian(L, v_topo, w_topo, bc)
    e_triv, vecs_triv = diagonalize_hamiltonian(h_triv)
    e_topo, vecs_topo = diagonalize_hamiltonian(h_topo)
    P_triv, _ = compute_polarization(Statevector(vecs_triv[0]), L)
    P_topo, _ = compute_polarization(Statevector(vecs_topo[0]), L)
    print(f"  {bc:5}: trivial(v=2,w=1) P={P_triv:.4f}  topological(v=1,w=2) P={P_topo:.4f}")

# ── Spinful 直接算（不引入 Hubbard 模块） ──
def build_ssh_hubbard_direct(L, v, w, U, boundary):
    """内联构建 spinful SSH-Hubbard (没有 Z-string 问题)"""
    nq = 4 * L
    terms = []
    for i in range(L):
        a_up, a_dn, b_up, b_dn = 4*i+0, 4*i+1, 4*i+2, 4*i+3
        # Spin ↑ hopping (A↔B): Z on a_dn
        terms.append((_pauli_label(nq, [a_up, a_dn, b_up], ['X','Z','X']), -v/2.0))
        terms.append((_pauli_label(nq, [a_up, a_dn, b_up], ['Y','Z','Y']), -v/2.0))
        # Spin ↓ hopping (A↔B): Z on b_up
        terms.append((_pauli_label(nq, [a_dn, b_up, b_dn], ['X','Z','X']), -v/2.0))
        terms.append((_pauli_label(nq, [a_dn, b_up, b_dn], ['Y','Z','Y']), -v/2.0))
        # Hubbard
        terms.append(('I'*nq, U/4.0))
        terms.append((_pauli_label(nq, [a_up], ['Z']), -U/4.0))
        terms.append((_pauli_label(nq, [a_dn], ['Z']), -U/4.0))
        terms.append((_pauli_label(nq, [a_up, a_dn], ['Z','Z']), U/4.0))
        terms.append(('I'*nq, U/4.0))
        terms.append((_pauli_label(nq, [b_up], ['Z']), -U/4.0))
        terms.append((_pauli_label(nq, [b_dn], ['Z']), -U/4.0))
        terms.append((_pauli_label(nq, [b_up, b_dn], ['Z','Z']), U/4.0))
    for i in range(L - 1):
        _, _, b_up, b_dn = 4*i+0, 4*i+1, 4*i+2, 4*i+3
        aj_up, aj_dn = 4*(i+1)+0, 4*(i+1)+1
        terms.append((_pauli_label(nq, [b_up, b_dn, aj_up], ['X','Z','X']), -w/2.0))
        terms.append((_pauli_label(nq, [b_up, b_dn, aj_up], ['Y','Z','Y']), -w/2.0))
        terms.append((_pauli_label(nq, [b_dn, aj_up, aj_dn], ['X','Z','X']), -w/2.0))
        terms.append((_pauli_label(nq, [b_dn, aj_up, aj_dn], ['Y','Z','Y']), -w/2.0))
    if boundary in ('PBC', 'APBC'):
        bl_up, bl_dn = 4*L-2, 4*L-1
        a0_up, a0_dn = 0, 1
        sign = 1.0 if boundary == 'PBC' else -1.0
        filler_up = list(range(a0_up+1, bl_up))
        terms.append((_pauli_label(nq, [a0_up]+filler_up+[bl_up], ['X']+['Z']*len(filler_up)+['X']), sign*-w/2.0))
        terms.append((_pauli_label(nq, [a0_up]+filler_up+[bl_up], ['Y']+['Z']*len(filler_up)+['Y']), sign*-w/2.0))
        filler_dn = list(range(a0_dn+1, bl_dn))
        terms.append((_pauli_label(nq, [a0_dn]+filler_dn+[bl_dn], ['X']+['Z']*len(filler_dn)+['X']), sign*-w/2.0))
        terms.append((_pauli_label(nq, [a0_dn]+filler_dn+[bl_dn], ['Y']+['Z']*len(filler_dn)+['Y']), sign*-w/2.0))
    return SparsePauliOp.from_list([(l, complex(c)) for l, c in terms]).simplify()

def compute_polarization_spinful(sv, L):
    nq = 4 * L
    sv_data = sv.data
    expval = 0.0 + 0.0j
    for idx, amp in enumerate(sv_data):
        if abs(amp) < 1e-15:
            continue
        bits = format(idx, f'0{nq}b')
        phase = 0
        for j in range(L):
            n_j = sum(int(bits[4*j + k]) for k in range(4))
            phase += (j + 1) * n_j
        prob = (amp.conjugate() * amp).real
        expval += prob * np.exp(1j * 2.0 * np.pi / L * phase)
    return np.angle(expval) / (2.0 * np.pi), expval

print("\nSpinful U=0:")
for bc in ['PBC', 'APBC']:
    h_triv = build_ssh_hubbard_direct(L, v_triv, w_triv, 0.0, bc)
    h_topo = build_ssh_hubbard_direct(L, v_topo, w_topo, 0.0, bc)
    e_triv, vecs_triv = diagonalize_hamiltonian(h_triv)
    e_topo, vecs_topo = diagonalize_hamiltonian(h_topo)
    P_triv, _ = compute_polarization_spinful(Statevector(vecs_triv[0]), L)
    P_topo, _ = compute_polarization_spinful(Statevector(vecs_topo[0]), L)
    print(f"  {bc:5}: trivial P={P_triv:.4f}  topological P={P_topo:.4f}")

print("\nSpinful U=4:")
for bc in ['PBC', 'APBC']:
    h_triv = build_ssh_hubbard_direct(L, v_triv, w_triv, 4.0, bc)
    h_topo = build_ssh_hubbard_direct(L, v_topo, w_topo, 4.0, bc)
    e_triv, vecs_triv = diagonalize_hamiltonian(h_triv)
    e_topo, vecs_topo = diagonalize_hamiltonian(h_topo)
    P_triv, _ = compute_polarization_spinful(Statevector(vecs_triv[0]), L)
    P_topo, _ = compute_polarization_spinful(Statevector(vecs_topo[0]), L)
    print(f"  {bc:5}: trivial P={P_triv:.4f}  topological P={P_topo:.4f}")
