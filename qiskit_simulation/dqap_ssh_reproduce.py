#!/usr/bin/env python3
"""
DQAP Quantum Simulation of SSH Model
====================================
Reproduce Xie et al. (2025): Digital quantum simulation of the SSH model
using a parameterized quantum circuit.

Extension framework: SSH-Hubbard (U>0)

Qubit ordering: A₁, B₁, A₂, B₂, ..., A_L, B_L  (2L qubits total)
Each unit cell = 2 qubits (sublattice A, B)

With this ordering, both intra-cell (A_i↔B_i) and inter-cell (B_i↔A_{i+1})
hopping are BETWEEN ADJACENT qubits → no Jordan-Wigner string for bulk terms.
Only the PBC/APBC boundary term (B_L↔A_1) needs a JW string.

Reference:
    Xie, Seki, Shirakawa & Yunoki (2025), arXiv:2504.08543
    Seki, Shirakawa & Yunoki (2022), PRB 105, 155106
"""

import numpy as np
from scipy.optimize import minimize
from qiskit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp, Statevector
from qiskit.circuit.library import PauliEvolutionGate
import time
import os
from utils import _pauli_label, diagonalize_hamiltonian


# ============================================================================
# Part 1: SSH Hamiltonian (Jordan-Wigner → Pauli strings)
# ============================================================================


def build_ssh_hamiltonian(L, v, w, boundary='APBC'):
    """
    Build the full SSH Hamiltonian as SparsePauliOp.

    H_SSH = -v · Σ_i (c†_{A,i}c_{B,i} + h.c.)
           -w · Σ_i (c†_{B,i}c_{A,i+1} + h.c.)

    Jordan-Wigner (adjacent qubits): c†_i c_j + h.c. → (X_i X_j + Y_i Y_j) / 2

    Parameters:
        L: Number of unit cells
        v: Intra-cell hopping (A_i ↔ B_i)
        w: Inter-cell hopping (B_i ↔ A_{i+1})
        boundary: 'PBC', 'APBC', or 'OBC'
    """
    n_qubits = 2 * L
    terms = []

    # ── H₁: Intra-cell hopping (A_i ↔ B_i) ──
    # A_i at qubit 2i, B_i at qubit 2i+1 (0-indexed)
    # Adjacent → no JW string
    for i in range(L):
        ai = 2 * i       # A_i
        bi = 2 * i + 1   # B_i

        label_xx = _pauli_label(n_qubits, [ai, bi], ['X', 'X'])
        label_yy = _pauli_label(n_qubits, [ai, bi], ['Y', 'Y'])
        terms.append((label_xx, -v / 2.0))
        terms.append((label_yy, -v / 2.0))

    # ── H₂: Inter-cell hopping (B_i ↔ A_{i+1}) ──
    # B_i at qubit 2i+1, A_{i+1} at qubit 2(i+1) = 2i+2
    # Adjacent → no JW string
    for i in range(L - 1):
        bi = 2 * i + 1      # B_i
        aj = 2 * (i + 1)    # A_{i+1}

        label_xx = _pauli_label(n_qubits, [bi, aj], ['X', 'X'])
        label_yy = _pauli_label(n_qubits, [bi, aj], ['Y', 'Y'])
        terms.append((label_xx, -w / 2.0))
        terms.append((label_yy, -w / 2.0))

    # ── Boundary term ──
    # B_{L-1} at qubit 2L-1 (last qubit), A₀ at qubit 0 (first qubit)
    # NOT adjacent → need JW string on qubits 1, 2, ..., 2L-2
    if boundary in ('PBC', 'APBC'):
        bl = 2 * L - 1  # B_{L-1} (last qubit, 0-indexed)
        a0 = 0           # A_0 (first qubit)

        # JW string: Z on all qubits between a0 and bl
        filler_qubits = list(range(a0 + 1, bl))  # qubits 1, 2, ..., 2L-2

        # Boundary Pauli string: X_a0 · Z_filler · X_bl  or  Y_a0 · Z_filler · Y_bl
        def boundary_label(p1, p2):
            return _pauli_label(n_qubits, [a0] + filler_qubits + [bl],
                                [p1] + ['Z'] * len(filler_qubits) + [p2])

        label_xx = boundary_label('X', 'X')
        label_yy = boundary_label('Y', 'Y')

        # PBC: same sign as bulk (-w/2), APBC: opposite sign (+w/2)
        sign = 1.0 if boundary == 'PBC' else -1.0
        terms.append((label_xx, sign * -w / 2.0))
        terms.append((label_yy, sign * -w / 2.0))

    # Build SparsePauliOp
    ham = SparsePauliOp.from_list(terms)
    return ham.simplify()


def build_h1_hamiltonian(L, v):
    """H₁ = -v Σ_i (c†_{A,i}c_{B,i} + h.c.) — intra-cell only."""
    return build_ssh_hamiltonian(L, v, 0.0, 'OBC')


def build_h2_hamiltonian(L, w, boundary='APBC'):
    """H₂ = -w Σ_i (c†_{B,i}c_{A,i+1} + h.c.) — inter-cell + boundary."""
    return build_ssh_hamiltonian(L, 0.0, w, boundary)



# ============================================================================

# Part 2: DQAP Circuit
# ============================================================================

def build_initial_state(L):
    """
    Prepare the ground state of H₁.

    For each unit cell, the bonding state: |t⟩ = (|01⟩ + |10⟩) / √2

    Circuit (per cell):
        |0⟩_A ─── H ─── ┤●├
        |0⟩_B ─── X ─── ┤X├

    Result: (|01⟩ + |10⟩) / √2 = |t⟩
    """
    qc = QuantumCircuit(2 * L, name='|Ψ₀⟩')

    for i in range(L):
        a_idx = 2 * i      # A_i
        b_idx = 2 * i + 1  # B_i

        qc.x(b_idx)          # |0⟩_B → |1⟩_B  :  |00⟩ → |01⟩
        qc.h(a_idx)           # |0⟩_A → (|0⟩+|1⟩)/√2 : |01⟩ → (|01⟩+|11⟩)/√2
        qc.cx(a_idx, b_idx)   # CNOT: |01⟩→|01⟩, |11⟩→|10⟩ : → (|01⟩+|10⟩)/√2

    return qc


def build_dqap_circuit(L, params, v, w, boundary='APBC', return_all=False):
    """
    Build the full DQAP circuit.

    U(θ) = Π_{m=1}^{M} [e^{-iθ_{m,1} H₁} · e^{-iθ_{m,2} H₂}]

    Applied after the initial state |Ψ₀⟩.

    Parameters:
        L: Number of unit cells
        params: Array of shape (M, 2) or (2M,) — θ₁, θ₂ for each layer
        v, w: SSH hopping parameters
        boundary: Boundary condition
        return_all: If True, also return intermediate circuits for each M
    """
    if params.ndim == 1:
        M = len(params) // 2
        params = params.reshape(M, 2)
    else:
        M = params.shape[0]

    n_qubits = 2 * L

    # Start with initial state
    qc = build_initial_state(L)

    # Add M layers
    for m in range(M):
        theta_1 = params[m, 0]
        theta_2 = params[m, 1]

        # Layer: e^{-iθ₁H₁} · e^{-iθ₂H₂}

        # ── e^{-iθ₁ H₁} ──
        # H₁ = -v Σ (XX + YY)/2 on each intra-cell bond
        # e^{-iθ₁ H₁} = Π_cell e^{iθ₁v (XX+YY)/2}
        for i in range(L):
            ai = 2 * i
            bi = 2 * i + 1

            # Qiskit 2.x compact Pauli format: no indices, all qubits specified
            op_xxyy = SparsePauliOp.from_list([
                ("XX", 1.0),
                ("YY", 1.0),
            ])
            # PauliEvolutionGate: exp(-i·t·op)
            # We want exp(i·θ₁·v·(XX+YY)/2) = exp(-i·(-θ₁·v/2)·(XX+YY))
            qc.append(PauliEvolutionGate(op_xxyy, time=-v * theta_1 / 2.0),
                      [ai, bi])

        # ── e^{-iθ₂ H₂} ──
        # H₂ = -w Σ (XX + YY)/2 on each inter-cell bond
        for i in range(L - 1):
            bi = 2 * i + 1
            aj = 2 * (i + 1)

            op_xxyy = SparsePauliOp.from_list([
                ("XX", 1.0),
                ("YY", 1.0),
            ])
            qc.append(PauliEvolutionGate(op_xxyy, time=-w * theta_2 / 2.0),
                      [bi, aj])

        # ── Boundary term ──
        if boundary in ('PBC', 'APBC'):
            bl = 2 * L - 1
            a0 = 0
            z_string = ''.join(f'Z{j}' for j in range(a0 + 1, bl))

            sign = 1.0 if boundary == 'PBC' else -1.0
            label_x = f"X{bl}{z_string}X{a0}"
            label_y = f"Y{bl}{z_string}Y{a0}"

            # Handle the ordering: Qiskit reads left-to-right as qubit 0→N-1
            # So "X5Z4Z3Z2Z1X0" means X₅·Z₄·Z₃·Z₂·Z₁·X₀
            # But for n_qubits, we need to pad or specify all qubits
            # Let me use _pauli_label instead
            nq = n_qubits
            filler_qubits = list(range(a0 + 1, bl))
            # Build labels properly
            chars_x = ['I'] * nq
            chars_x[a0] = 'X'
            chars_x[bl] = 'X'
            for j in filler_qubits:
                chars_x[j] = 'Z'

            chars_y = ['I'] * nq
            chars_y[a0] = 'Y'
            chars_y[bl] = 'Y'
            for j in filler_qubits:
                chars_y[j] = 'Z'

            op_xx = SparsePauliOp.from_list([(''.join(chars_x), 1.0)])
            op_yy = SparsePauliOp.from_list([(''.join(chars_y), 1.0)])

            qc.append(PauliEvolutionGate(op_xx, time=sign * -w * theta_2 / 2.0),
                      range(nq))
            qc.append(PauliEvolutionGate(op_yy, time=sign * -w * theta_2 / 2.0),
                      range(nq))

    return qc


# ============================================================================
# Part 3: Energy Evaluation and VQE
# ============================================================================

def compute_energy(params, L, v, w, boundary, hamiltonian):
    """
    Compute E(θ) = ⟨Ψ(θ)|H|Ψ(θ)⟩ using exact statevector simulation.

    Args:
        params: (2M,) array of θ parameters
        hamiltonian: SparsePauliOp of the full SSH Hamiltonian

    Returns:
        Energy expectation value (real)
    """
    qc = build_dqap_circuit(L, params, v, w, boundary)
    sv = Statevector(qc)
    energy = sv.expectation_value(hamiltonian).real
    return energy


def run_vqe(L, v, w, boundary, M, hamiltonian, maxiter=2000, verbose=True):
    """
    Run DQAP-VQE to find the ground state for M layers.

    Args:
        L, v, w, boundary: SSH model parameters
        M: Number of DQAP layers
        hamiltonian: SparsePauliOp of total SSH Hamiltonian
        maxiter: Max iterations for L-BFGS-B
        verbose: Print progress

    Returns:
        opt_result: scipy optimize result
    """
    n_params = 2 * M

    # Initial guess: random in [0, π/2]
    # Better: use small random values for stability
    x0 = np.random.uniform(0.0, 0.5, n_params)

    start = time.time()

    def callback(xk):
        if verbose:
            eng = compute_energy(xk, L, v, w, boundary, hamiltonian)
            print(f"  iter: E = {eng:.8f}")

    result = minimize(
        compute_energy,
        x0,
        args=(L, v, w, boundary, hamiltonian),
        method='L-BFGS-B',
        options={'maxiter': maxiter, 'ftol': 1e-12, 'gtol': 1e-08},
    )

    elapsed = time.time() - start

    if verbose:
        print(f"VQE M={M} done in {elapsed:.1f}s: "
              f"E = {result.fun:.8f}, success = {result.success}")

    return result


# ============================================================================
# Part 4: Resta Polarization
# ============================================================================

def compute_polarization(sv, L):
    """
    Compute Resta polarization P_R from a statevector.

    P_R = (1/2π) · Im ln ⟨Ψ|U_R|Ψ⟩

    where U_R = exp(i 2π/L · Σ_{j=1}^{L} j · n_j)
    and n_j = n_{A,j} + n_{B,j} (total density at cell j)

    In qubit basis: n_j = (I-Z_{2j-2})/2 + (I-Z_{2j-1})/2 = I - (Z_{2j-2}+Z_{2j-1})/2

    Since U_R is diagonal in the computational basis:
    U_R|b⟩ = exp(i 2π/L · Σ_j j·(b_{2j-2}+b_{2j-1}))|b⟩

    Returns:
        polarization: P_R ∈ (-0.5, 0.5]
        expval: ⟨U_R⟩ complex value
    """
    n_qubits = 2 * L
    sv_data = sv.data

    expval = 0.0 + 0.0j
    for idx, amp in enumerate(sv_data):
        if abs(amp) < 1e-15:
            continue
        bits = format(idx, f'0{n_qubits}b')

        # Compute density at each cell: n_j = b_{2j} + b_{2j+1} (0-indexed)
        phase = 0.0
        for j in range(L):
            a_bit = int(bits[2 * j])       # A_j
            b_bit = int(bits[2 * j + 1])   # B_j
            n_j = a_bit + b_bit            # total density at cell j
            phase += (j + 1) * n_j         # j starts from 1 in the formula

        prob = (amp.conjugate() * amp).real
        expval += prob * np.exp(1j * 2.0 * np.pi / L * phase)

    polarization = np.angle(expval) / (2.0 * np.pi)
    return polarization, expval


def polarization_from_circuit(qc, L):
    """Compute Resta polarization from a built circuit."""
    sv = Statevector(qc)
    return compute_polarization(sv, L)


# ============================================================================
# Part 5: Main Reproduction — Energy Convergence + Polarization
# ============================================================================

def reproduce_results(L=4, boundary='APBC', M_max=6, v_trivial=2.0, w_trivial=1.0,
                      v_topological=1.0, w_topological=2.0, verbose=True):
    """
    Reproduce the key results from Xie et al. (2025):

    1. Same topological phase → exponential energy convergence
    2. Different topological phase → polynomial energy convergence
    3. Polarization jump at critical depth M*

    Parameters:
        L: Number of unit cells (APBC: L ∈ 4N)
        boundary: 'APBC' for anti-periodic BC
        M_max: Maximum number of DQAP layers to test
        v/w_trivial: Parameters for trivial phase (v > w)
        v/w_topological: Parameters for topological phase (v < w)
    """
    print("=" * 70)
    print(f"DQAP SSH Reproduction — L={L}, {boundary}, M_max={M_max}")
    print("=" * 70)

    n_qubits = 2 * L
    print(f"\nSystem: {L} cells × 2 sites = {n_qubits} qubits")

    # ── Exact ground state energies ──
    print("\n[1/5] Computing exact ground states...")

    h_trivial = build_ssh_hamiltonian(L, v_trivial, w_trivial, boundary)
    h_topo = build_ssh_hamiltonian(L, v_topological, w_topological, boundary)

    eigs_triv, _ = diagonalize_hamiltonian(h_trivial)
    eigs_topo, _ = diagonalize_hamiltonian(h_topo)
    E0_triv = eigs_triv[0].real
    E0_topo = eigs_topo[0].real

    print(f"  Trivial case (v={v_trivial}, w={w_trivial}): "
          f"E₀ = {E0_triv:.8f}")
    print(f"  Topological case (v={v_topological}, w={w_topological}): "
          f"E₀ = {E0_topo:.8f}")

    # ── Initial state baseline ──
    print("\n[2/5] Initial state (M=0) energy...")
    qc_init = build_initial_state(L)
    sv_init = Statevector(qc_init)
    E_init_triv = sv_init.expectation_value(h_trivial).real
    E_init_topo = sv_init.expectation_value(h_topo).real
    P_init_triv, _ = compute_polarization(sv_init, L)
    P_init_topo, _ = compute_polarization(sv_init, L)

    print(f"  Trivial: E(M=0) = {E_init_triv:.8f}, "
          f"ΔE = {E_init_triv - E0_triv:.2e}, P = {P_init_triv:.4f}")
    print(f"  Topological: E(M=0) = {E_init_topo:.8f}, "
          f"ΔE = {E_init_topo - E0_topo:.2e}, P = {P_init_topo:.4f}")

    # ── Run VQE for M = 0..M_max ──
    print("\n[3/5] Running VQE for increasing M...")

    results = {
        'trivial': {'energies': [E_init_triv], 'polarizations': [P_init_triv],
                     'params': [None]},
        'topo': {'energies': [E_init_topo], 'polarizations': [P_init_topo],
                  'params': [None]},
    }

    for M in range(1, M_max + 1):
        print(f"\n--- M = {M} ---")

        for case_name, v, w, h_ref, E0 in [
            ('trivial', v_trivial, w_trivial, h_trivial, E0_triv),
            ('topo', v_topological, w_topological, h_topo, E0_topo),
        ]:
            case_key = 'trivial' if case_name == 'trivial' else 'topo'

            result = run_vqe(L, v, w, boundary, M, h_ref, verbose=verbose)

            E_opt = result.fun
            qc_opt = build_dqap_circuit(L, result.x, v, w, boundary)
            sv_opt = Statevector(qc_opt)
            P_opt, _ = compute_polarization(sv_opt, L)

            results[case_key]['energies'].append(E_opt)
            results[case_key]['polarizations'].append(P_opt)
            results[case_key]['params'].append(result.x)

            delta_E = E_opt - E0

            if verbose:
                print(f"  {case_key}: E = {E_opt:.8f}, ΔE = {delta_E:.2e}, "
                      f"P = {P_opt:.4f}, converged = {result.success}")

    # ── Exact polarization from exact ground state ──
    print("\n[4/5] Exact ground state polarization...")

    _, vecs_triv = diagonalize_hamiltonian(h_trivial)
    _, vecs_topo = diagonalize_hamiltonian(h_topo)

    sv_exact_triv = Statevector(vecs_triv[0])
    sv_exact_topo = Statevector(vecs_topo[0])

    P_exact_triv, UR_triv = compute_polarization(sv_exact_triv, L)
    P_exact_topo, UR_topo = compute_polarization(sv_exact_topo, L)

    print(f"  Trivial: P = {P_exact_triv:.6f}, ⟨U_R⟩ = {UR_triv:.6f}")
    print(f"  Topological: P = {P_exact_topo:.6f}, ⟨U_R⟩ = {UR_topo:.6f}")

    # ── Summary ──
    print("\n[5/5] Results Summary")
    print("-" * 70)
    print(f"{'M':>3} | {'E_triv':>14} {'ΔE_triv':>12} {'P_triv':>8} | "
          f"{'E_topo':>14} {'ΔE_topo':>12} {'P_topo':>8}")
    print("-" * 70)

    for m in range(M_max + 1):
        E_tr = results['trivial']['energies'][m]
        E_to = results['topo']['energies'][m]
        dE_tr = E_tr - E0_triv
        dE_to = E_to - E0_topo
        P_tr = results['trivial']['polarizations'][m]
        P_to = results['topo']['polarizations'][m]

        # Format
        dE_str_tr = f"{dE_tr:.2e}" if m > 0 else "—"
        dE_str_to = f"{dE_to:.2e}" if m > 0 else "—"

        print(f"{m:3d} | {E_tr:14.8f} {dE_str_tr:>12} {P_tr:8.4f} | "
              f"{E_to:14.8f} {dE_str_to:>12} {P_to:8.4f}")

    print("-" * 70)
    print(f"Exact | {E0_triv:14.8f} {'':>12} {P_exact_triv:8.4f} | "
          f"{E0_topo:14.8f} {'':>12} {P_exact_topo:8.4f}")

    return results, (E0_triv, E0_topo, P_exact_triv, P_exact_topo)


# ============================================================================
# Part 6: Extended Hubbard U (Framework)
# ============================================================================

def build_ssh_hubbard_hamiltonian(L, v, w, U, boundary='APBC'):
    """
    Build SSH-Hubbard Hamiltonian: H_SSH + H_U.

    H_U = U · Σ_i n_{i↑} n_{i↓}

    For spinless SSH (no Hubbard), we use spinless Jordan-Wigner.
    For SSH-Hubbard with spin, we need 2 qubits per site:
    - A_{i,↑}, A_{i,↓}, B_{i,↑}, B_{i,↓}

    Ordering: A₁↑, A₁↓, B₁↑, B₁↓, A₂↑, A₂↓, B₂↑, B₂↓, ...
    Total qubits: 4L

    But this is for the EXTENSION phase — for now, focus on spinless SSH.
    """
    # TODO: SSH-Hubbard extension — will need 4L qubits with spin-↑↓ per site
    return None


# ============================================================================
# Plotting
# ============================================================================

def plot_results(L, results, exact_energies, M_max, output_dir='.'):
    """Plot energy convergence and polarization."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available, skipping plots")
        return

    E0_triv, E0_topo, P_exact_triv, P_exact_topo = exact_energies

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    M_vals = np.arange(M_max + 1)

    # ── Energy convergence ──
    ax = axes[0]
    ax.axhline(E0_triv, color='C0', linestyle='--', alpha=0.5, label='Exact trivial')
    ax.axhline(E0_topo, color='C1', linestyle='--', alpha=0.5, label='Exact topological')
    ax.plot(M_vals, results['trivial']['energies'], 'o-', color='C0', label='Trivial (same phase)')
    ax.plot(M_vals, results['topo']['energies'], 's-', color='C1', label='Topological (cross phase)')
    ax.set_xlabel('M (layers)')
    ax.set_ylabel('Energy')
    ax.set_title('Energy convergence')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # ── Polarization ──
    ax = axes[1]
    ax.axhline(P_exact_triv, color='C0', linestyle='--', alpha=0.5,
               label=f'Exact trivial (P={P_exact_triv:.3f})')
    ax.axhline(P_exact_topo, color='C1', linestyle='--', alpha=0.5,
               label=f'Exact topological (P={P_exact_topo:.3f})')
    ax.plot(M_vals, results['trivial']['polarizations'], 'o-', color='C0',
            label='Trivial')
    ax.plot(M_vals, results['topo']['polarizations'], 's-', color='C1',
            label='Topological')
    ax.set_xlabel('M (layers)')
    ax.set_ylabel('Polarization P_R')
    ax.set_title('Resta polarization vs M')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    # Save
    filename = os.path.join(output_dir, f'dqap_ssh_L{L}_reproduce.png')
    plt.savefig(filename, dpi=150)
    print(f"\nPlot saved: {filename}")

    if output_dir == '.':
        plt.show()


# ============================================================================
# Main entry point
# ============================================================================

if __name__ == '__main__':
    import sys

    # Default parameters
    L = 4
    boundary = 'APBC'
    M_max = 6

    # Parse args: python dqap_ssh_reproduce.py [L] [M_max]
    if len(sys.argv) > 1:
        L = int(sys.argv[1])
    if len(sys.argv) > 2:
        M_max = int(sys.argv[2])

    # Reproduce
    results, exact = reproduce_results(
        L=L,
        boundary=boundary,
        M_max=M_max,
        v_trivial=2.0,
        w_trivial=1.0,
        v_topological=1.0,
        w_topological=2.0,
        verbose=True,
    )

    # Plot
    plot_results(L, results, exact, M_max, output_dir='.')
