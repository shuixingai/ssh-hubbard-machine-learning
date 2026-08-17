"""
SSH Model with Hubbard & Extended Hubbard — Exact Diagonalization (QuSpin)
============================================================================
Implements the SSH model with onsite Hubbard U and nearest-neighbor V:

    H = - Σ t_i (c†_{i,σ} c_{i+1,σ} + h.c.)
        + U Σ n_{i,↑} n_{i,↓}
        + V Σ n_i n_{i+1}

with alternating hopping t_i = t1 (i even) or t2 (i odd),
open boundary conditions (OBC), and half-filling (N_up = N_down = L/2).

Outputs:
    - Ground state energy
    - Eigenvalue spectrum (lowest n_eig eigenvalues)
    - Single-particle density matrix ⟨c†_i c_j⟩
    - Entanglement spectrum (Schmidt decomposition at middle cut)
    - Double occupancy ⟨n_{i↑} n_{i↓}⟩

Also contains a dataset generator that sweeps t1, t2, U (and optionally V).
"""

import os
import sys
import time
import numpy as np
from itertools import product

# ── QuSpin imports ──────────────────────────────────────────────────
os.environ["OMP_NUM_THREADS"] = "8"
os.environ["MKL_NUM_THREADS"] = "16"

try:
    from quspin.operators import hamiltonian
    from quspin.basis import spinful_fermion_basis_1d
    _HAS_QUSPIN = True
except ImportError:
    _HAS_QUSPIN = False
    msg = (
        "QuSpin is not installed. Install it via:\n"
        "  pip install quspin\n"
        "or use a fallback with dense NumPy diagonalization "
        "(small L only)."
    )
    raise ImportError(msg)


# ══════════════════════════════════════════════════════════════════════
#  SSH Model Class
# ══════════════════════════════════════════════════════════════════════

class SSHModel:
    """SSH-Hubbard model on a 1D chain with OBC at half-filling.

    Parameters
    ----------
    L : int
        Number of sites (must be even).
    t1 : float
        Hopping on odd bonds (0-1, 2-3, …).
    t2 : float
        Hopping on even bonds (1-2, 3-4, …).
    U : float
        On-site Hubbard interaction strength.
    V : float
        Nearest-neighbour Coulomb repulsion.
    """

    def __init__(self, L, t1, t2, U, V=0.0):
        if L % 2 != 0:
            raise ValueError("L must be even for half-filling.")
        self.L = L
        self.t1 = t1
        self.t2 = t2
        self.U = U
        self.V = V

        # caches (cleared on parameter change if needed — here immutable)
        self._basis = None
        self._H = None
        self._corr_ops = None  # list of (op_up, op_down) per (i, j)
        self._phase_factors = None  # for Resta polarization
        self._delta_n_sq_op = None  # (δn)² operator cache
        self._stag_mag_sq_op = None  # staggered m_s² operator cache

    # ── helper ───────────────────────────────────────────────────────

    def _hopping_amplitudes(self):
        """Return list of length L-1 with alternating t1 / t2."""
        return [self.t1 if i % 2 == 0 else self.t2 for i in range(self.L - 1)]

    def _build_basis(self):
        """Half-filled (N_up = N_down = L/2) spinful-fermion basis."""
        N_up = self.L // 2
        N_down = self.L // 2
        self._basis = spinful_fermion_basis_1d(self.L, Nf=(N_up, N_down))
        return self._basis

    # ── Hamiltonian ──────────────────────────────────────────────────

    def build_hamiltonian(self):
        """Construct the SSH-Hubbard Hamiltonian.

        Returns
        -------
        H : quspin.operators.hamiltonian
        """
        if self._basis is None:
            self._build_basis()

        t = self._hopping_amplitudes()

        # Hopping: right (i → i+1) and left (i+1 → i)
        hop_right = [[t[i], i, i + 1] for i in range(self.L - 1)]
        hop_left  = [[t[i], i + 1, i] for i in range(self.L - 1)]

        # Hubbard U on every site: U * n_{i,↑} n_{i,↓}
        U_term = [[self.U, i, i] for i in range(self.L)]

        static = [
            # spin-up hopping:  c†_{↑,i} c_{↑,j}
            ["+-|", hop_right],
            ["+-|", hop_left],
            # spin-down hopping:  c†_{↓,i} c_{↓,j}
            ["|+-", hop_right],
            ["|+-", hop_left],
            # Hubbard interaction:  n_{↑,i} n_{↓,i}
            ["n|n", U_term],
        ]

        # Nearest-neighbour Coulomb repulsion:  V * n_i * n_{i+1}
        #   n_i n_{i+1} = (n_{↑,i}+n_{↓,i})(n_{↑,i+1}+n_{↓,i+1})
        #   expands to 4 static-list entries per bond
        if self.V != 0.0:
            V_list = [[self.V, i, i + 1] for i in range(self.L - 1)]
            V_rev  = [[self.V, i + 1, i] for i in range(self.L - 1)]
            static.extend([
                ["nn|", V_list],   # n_{↑,i} n_{↑,i+1}
                ["n|n", V_list],   # n_{↑,i} n_{↓,i+1}
                ["n|n", V_rev],    # n_{↓,i} n_{↑,i+1}
                ["|nn", V_list],   # n_{↓,i} n_{↓,i+1}
            ])

        self._H = hamiltonian(
            static, [],
            basis=self._basis,
            dtype=np.float64,
            check_herm=False,
            check_symm=False,
            check_pcon=False,
        )
        return self._H

    # ── Diagonalisation ──────────────────────────────────────────────

    def diagonalise(self, k=None):
        """Diagonalise the Hamiltonian.

        Parameters
        ----------
        k : int or None
            Number of lowest eigenpairs to compute.
            If None, full diagonalisation (eigh).

        Returns
        -------
        eigvals : ndarray, shape (k,) or (Ns,)
        eigvecs : ndarray, shape (Ns, k) or (Ns, Ns)
        """
        if self._H is None:
            self.build_hamiltonian()

        if k is None:
            eigvals, eigvecs = self._H.eigh()
        else:
            k_use = min(k, self._basis.Ns)
            try:
                eigvals, eigvecs = self._H.eigsh(k=k_use, which="SA")
            except Exception:
                # Sparse solver may fail on highly degenerate Hamiltonians
                # (e.g. t1 ~ t2 ~ 0). Fall back to full diagonalization.
                eigvals_full, eigvecs_full = self._H.eigh()
                idx = np.argsort(eigvals_full)
                eigvals = eigvals_full[idx[:k_use]]
                eigvecs = eigvecs_full[:, idx[:k_use]]
            # eigsh does not guarantee sorted order
            idx = np.argsort(eigvals)
            eigvals = eigvals[idx]
            eigvecs = eigvecs[:, idx]

        return eigvals, eigvecs

    # ── Observables ──────────────────────────────────────────────────

    def get_ground_state(self):
        """Return (ground_state_energy, ground_state_vector)."""
        eigvals, eigvecs = self.diagonalise(k=1)
        return float(eigvals[0]), eigvecs[:, 0]

    def get_spectrum(self, n_eig=20):
        """Return lowest n_eig eigenvalues."""
        eigvals, _ = self.diagonalise(k=n_eig)
        return eigvals

    def _build_corr_ops(self):
        """Pre-build all correlation operators c†_{σ,i} c_{σ,j}.

        These are independent of t1, t2, U (basis only), so they are
        cached once and reused.
        """
        if self._corr_ops is not None:
            return self._corr_ops

        if self._basis is None:
            self._build_basis()

        ops = [[None for _ in range(self.L)] for _ in range(self.L)]
        for i in range(self.L):
            for j in range(self.L):
                # c†_{↑,i} c_{↑,j}
                op_up = hamiltonian(
                    [["+-|", [[1.0, i, j]]]], [],
                    basis=self._basis,
                    dtype=np.float64,
                    check_herm=False,
                    check_symm=False,
                    check_pcon=False,
                )
                # c†_{↓,i} c_{↓,j}
                op_down = hamiltonian(
                    [["|+-", [[1.0, i, j]]]], [],
                    basis=self._basis,
                    dtype=np.float64,
                    check_herm=False,
                    check_symm=False,
                    check_pcon=False,
                )
                ops[i][j] = (op_up, op_down)

        self._corr_ops = ops
        return ops

    def get_correlation_matrix(self, gs=None):
        """Compute single-particle density matrix ρ_{ij} = ⟨c†_i c_j⟩.

        Parameters
        ----------
        gs : ndarray or None
            Ground-state vector.  If None, it is computed.

        Returns
        -------
        corr : ndarray, shape (L, L)
            ρ_{ij} = ⟨c†_{↑,i} c_{↑,j}⟩ + ⟨c†_{↓,i} c_{↓,j}⟩.
        energy : float
            Ground-state energy (returned for convenience).
        """
        if gs is None:
            energy, gs = self.get_ground_state()
        else:
            energy = None

        ops = self._build_corr_ops()
        corr = np.zeros((self.L, self.L), dtype=np.float64)

        for i in range(self.L):
            for j in range(self.L):
                op_up, op_down = ops[i][j]
                val = op_up.expt_value(gs) + op_down.expt_value(gs)
                corr[i, j] = val.real  # imaginary part is numerical noise

        return corr, energy

    def get_staggered_charge_density(self, corr_matrix=None, gs=None):
        """Staggered charge density  δn = 1/L Σ_i (-1)^i ⟨n_i⟩.

        ⟨n_i⟩ = ⟨c†_{i↑}c_{i↑} + c†_{i↓}c_{i↓}⟩ = diag(ρ).

        Parameters
        ----------
        corr_matrix : ndarray (L, L) or None
            Single-particle density matrix.  Computed if not given.
        gs : ndarray or None
            Ground-state vector (used if corr_matrix is not given).

        Returns
        -------
        delta_n : float
        """
        if corr_matrix is None:
            corr_matrix, _ = self.get_correlation_matrix(gs)
        n_i = np.diag(corr_matrix).real
        sgn = np.array([(-1) ** i for i in range(self.L)])
        return float(np.sum(sgn * n_i)) / self.L

    def get_bond_order_alternation(self, corr_matrix=None, gs=None):
        """Bond-order alternation  δB = 1/(L-1) Σ_i (-1)^i B_i.

        B_i = ⟨c†_i c_{i+1} + c†_{i+1} c_i⟩ = 2 Re(ρ[i, i+1]).

        Positive δB → odd bonds stronger; negative → even bonds stronger.
        The magnitude |δB| is the alternation strength.

        Parameters
        ----------
        corr_matrix : ndarray (L, L) or None
        gs : ndarray or None

        Returns
        -------
        delta_B : float
        """
        if corr_matrix is None:
            corr_matrix, _ = self.get_correlation_matrix(gs)
        B = np.array([2.0 * corr_matrix[i, i + 1].real
                      for i in range(self.L - 1)])
        sgn = np.array([(-1) ** i for i in range(self.L - 1)])
        return float(np.sum(sgn * B)) / (self.L - 1)

    def get_double_occupancy(self, gs=None):
        """Compute double occupancy D_i = ⟨n_{i↑} n_{i↓}⟩ at each site.

        Parameters
        ----------
        gs : ndarray or None
            Ground-state vector.  If None, it is computed.

        Returns
        -------
        D : ndarray, shape (L,)
            Double occupancy per site.
        """
        if gs is None:
            _, gs = self.get_ground_state()

        if self._basis is None:
            self._build_basis()

        D = np.zeros(self.L)
        for i in range(self.L):
            op = hamiltonian(
                [["n|n", [[1.0, i, i]]]], [],
                basis=self._basis,
                dtype=np.float64,
                check_herm=False,
                check_symm=False,
                check_pcon=False,
            )
            D[i] = op.expt_value(gs).real

        return D

    # ── (δn)² operator ───────────────────────────────────────────────

    def _build_delta_n_sq_op(self):
        """Build and cache operator for (δn)² = (1/L²) Σᵢⱼ (-1)ⁱ⁺ʲ nᵢ nⱼ.

        nᵢ = nᵢ↑ + nᵢ↓, so the product expands to 4 spin terms.
        """
        if self._delta_n_sq_op is not None:
            return self._delta_n_sq_op
        if self._basis is None:
            self._build_basis()

        L = self.L
        nn_up = []    # (coeff, i, j) for n↑_i n↑_j
        n_up_dn = []  # (coeff, i, j) for n↑_i n↓_j
        n_dn_up = []  # (coeff, i, j) for n↓_i n↑_j  = n↑_j n↓_i
        nn_dn = []    # (coeff, i, j) for n↓_i n↓_j

        for i in range(L):
            for j in range(L):
                coeff = (-1.0) ** (i + j) / (L * L)
                if abs(coeff) < 1e-15:
                    continue
                entry = [coeff, i, j]
                nn_up.append(entry)
                n_up_dn.append(entry)
                # n↓_i n↑_j = n↑_j n↓_i  → same coeff with swapped sites
                n_dn_up.append([coeff, j, i])
                nn_dn.append(entry)

        static = [
            ["nn|", nn_up],      # Σ n↑_i n↑_j
            ["n|n", n_up_dn],    # Σ n↑_i n↓_j
            ["n|n", n_dn_up],    # Σ n↓_i n↑_j  (= n↑_j n↓_i)
            ["|nn", nn_dn],      # Σ n↓_i n↓_j
        ]

        self._delta_n_sq_op = hamiltonian(
            static, [],
            basis=self._basis,
            dtype=np.float64,
            check_herm=False,
            check_symm=False,
            check_pcon=False,
        )
        return self._delta_n_sq_op

    def get_staggered_charge_squared(self, gs=None):
        """⟨(δn)²⟩ = (1/L²) Σᵢⱼ (-1)ⁱ⁺ʲ ⟨nᵢ nⱼ⟩.

        Computed directly as the expectation of a product operator,
        so it's exact (no Wick factorisation).
        """
        if gs is None:
            _, gs = self.get_ground_state()
        op = self._build_delta_n_sq_op()
        return float(op.expt_value(gs).real)

    # ── Staggered magnetization squared  ⟨m_s²⟩ ──────────────────────

    def _build_staggered_mag_sq_op(self):
        """Build and cache operator for ⟨m_s²⟩.

        m_s = (1/L) Σᵢ (-1)ⁱ Sᶻᵢ
        Sᶻᵢ = (1/2)(nᵢ↑ - nᵢ↓)

        So m_s² = 1/(4L²) Σᵢⱼ (-1)ⁱ⁺ʲ (nᵢ↑ - nᵢ↓)(nⱼ↑ - nⱼ↓)
               = 1/(4L²) Σᵢⱼ (-1)ⁱ⁺ʲ (n↑_i n↑_j - n↑_i n↓_j - n↓_i n↑_j + n↓_i n↓_j)
        """
        if self._stag_mag_sq_op is not None:
            return self._stag_mag_sq_op
        if self._basis is None:
            self._build_basis()

        L = self.L
        pref = 1.0 / (4.0 * L * L)

        nn_up = []    # +n↑_i n↑_j
        n_up_dn = []  # -n↑_i n↓_j
        n_dn_up = []  # -n↓_i n↑_j  (= -n↑_j n↓_i)
        nn_dn = []    # +n↓_i n↓_j

        for i in range(L):
            for j in range(L):
                sgn = (-1.0) ** (i + j) * pref
                if abs(sgn) < 1e-15:
                    continue
                nn_up.append([+sgn, i, j])
                n_up_dn.append([-sgn, i, j])    # minus sign for cross term
                n_dn_up.append([-sgn, j, i])    # minus sign, swapped
                nn_dn.append([+sgn, i, j])

        static = [
            ["nn|", nn_up],
            ["n|n", n_up_dn],
            ["n|n", n_dn_up],
            ["|nn", nn_dn],
        ]

        self._stag_mag_sq_op = hamiltonian(
            static, [],
            basis=self._basis,
            dtype=np.float64,
            check_herm=False,
            check_symm=False,
            check_pcon=False,
        )
        return self._stag_mag_sq_op

    def get_staggered_magnetization_squared(self, gs=None):
        """⟨m_s²⟩ where m_s = (1/L) Σᵢ (-1)ⁱ Sᶻᵢ.

        SDW indicator: > 0 means AFM correlations (Mott/SDW phase),
        ≈ 0 in CDW phase (spin singlets).
        """
        if gs is None:
            _, gs = self.get_ground_state()
        op = self._build_staggered_mag_sq_op()
        return float(op.expt_value(gs).real)

    def get_entanglement_spectrum(self, gs, half=None, n_max=20):
        """Entanglement spectrum via Schmidt decomposition at the middle cut.

        Splits the chain into left (sites 0 … half-1) and right
        (sites half … L-1), builds the Schmidt matrix Ψ_{αβ} from the
        ground-state coefficients, and SVD-decomposes it.  Returns the
        eigenvalues of the left reduced density matrix (λ_α²) and the
        entanglement energies ε_α = -ln(λ_α²).

        Parameters
        ----------
        gs : ndarray, shape (Ns,)
            Ground-state vector.
        half : int or None
            Cut position.  Default L//2.
        n_max : int
            Max number of entanglement levels to return.

        Returns
        -------
        schmidt_sq : ndarray, shape (n_svd,)
            λ_α²  — eigenvalues of ρ_L (sorted descending).
        ent_energies : ndarray, shape (n_svd,)
            ε_α = -ln(λ_α²)  (sorted ascending).
        """
        L = self.L
        if half is None:
            half = L // 2

        # ── Encoding: QuSpin spinful_fermion_basis_1d packs
        #    bits  0 … L-1  = spin-up   sites 0 … L-1
        #    bits  L … 2L-1 = spin-down  sites 0 … L-1
        mask_left  = (1 << half) - 1
        mask_right = ((1 << (L - half)) - 1) << half

        dim_left  = 1 << (2 * half)        # 2^(2*half)  — left  index range
        dim_right = 1 << (2 * (L - half))  # 2^(2*(L-half)) — right index range

        psi_mat = np.zeros((dim_left, dim_right), dtype=np.float64)

        for i, s in enumerate(self._basis.states):
            s = int(s)
            # left half
            up_L  = (s       & mask_left)
            dn_L  = (s >> L & mask_left)
            idx_L = (up_L << half) | dn_L
            # right half
            up_R  = (s       & mask_right) >> half
            dn_R  = (s >> L & mask_right) >> half
            idx_R = (up_R << (L - half)) | dn_R

            psi_mat[idx_L, idx_R] = gs[i]

        # SVD
        _, S, _ = np.linalg.svd(psi_mat, full_matrices=False)

        # Drop numerical zeros
        tol = 1e-14
        S = S[S > tol]

        # λ²  and  ε = -ln(λ²)
        S2 = S ** 2
        ent = -np.log(S2)

        # Pad / truncate to n_max
        n = min(len(S2), n_max)
        out_s2 = np.zeros(n_max)
        out_ent = np.zeros(n_max)
        out_s2[:n] = S2[:n]
        out_ent[:n] = ent[:n]
        return out_s2, out_ent

    # ── Twisted boundary condition Hamiltonian ─────────────────────────

    def build_tbc_hamiltonian(self, theta):
        """Build Hamiltonian with twisted periodic boundary condition.

        The boundary condition c_{i+L} = e^{iθ} c_i is implemented as a
        Peierls phase on the bond connecting site L-1 back to site 0:

            H_boundary = -t_{L-1} ( e^{iθ} c†_{L-1,σ} c_{0,σ} + h.c. )

        Returns a NEW Hamiltonian object — does NOT modify self._H.

        Parameters
        ----------
        theta : float
            Twist angle (magnetic flux through the ring).

        Returns
        -------
        H : quspin.operators.hamiltonian
            Complex Hermitian Hamiltonian (dtype=complex128).
        """
        if self._basis is None:
            self._build_basis()

        L = self.L
        # Hopping amplitude for each bond (including the PBC boundary bond)
        t = [self.t1 if i % 2 == 0 else self.t2 for i in range(L)]

        # ── Hopping (bonds 0 … L-2: same as OBC) ─────────────────────
        hop_right = [[t[i], i, i + 1] for i in range(L - 1)]
        hop_left  = [[t[i], i + 1, i] for i in range(L - 1)]

        # ── Twisted boundary bond (L-1 → 0) ──────────────────────────
        hop_right.append(
            [t[L - 1] * np.exp(1j * theta), L - 1, 0]
        )
        hop_left.append(
            [t[L - 1] * np.exp(-1j * theta), 0, L - 1]
        )

        # ── Hubbard U ─────────────────────────────────────────────────
        U_term = [[self.U, i, i] for i in range(L)]

        static = [
            ["+-|", hop_right],      # c†_{↑,i} c_{↑,j}
            ["+-|", hop_left],
            ["|+-", hop_right],      # c†_{↓,i} c_{↓,j}
            ["|+-", hop_left],
            ["n|n", U_term],         # U n↑_i n↓_i
        ]

        # ── Nearest-neighbour V ──────────────────────────────────────
        if self.V != 0.0:
            V_bulk = [[self.V, i, i + 1] for i in range(L - 1)]
            V_bdry = [[self.V, L - 1, 0]]
            V_all = V_bulk + V_bdry

            V_bulk_rev = [[self.V, i + 1, i] for i in range(L - 1)]
            V_bdry_rev = [[self.V, 0, L - 1]]
            V_rev_all = V_bulk_rev + V_bdry_rev

            static.extend([
                ["nn|", V_all],        # n↑_i n↑_{i+1}
                ["n|n", V_all],        # n↑_i n↓_{i+1}
                ["n|n", V_rev_all],    # n↓_i n↑_{i+1}
                ["|nn", V_all],        # n↓_i n↓_{i+1}
            ])

        H = hamiltonian(
            static, [],
            basis=self._basis,
            dtype=np.complex128,
            check_herm=False,
            check_symm=False,
            check_pcon=False,
        )
        return H

    # ── Single-particle Zak phase (spinless, non-interacting) ────────

    def compute_sp_zak_phase(self, n_theta=61, verbose=False):
        """Single-particle Zak phase via twisted boundary conditions.

        This works with an effectively **spinless** fermion chain —
        the L×L single-particle hopping matrix with TBC is diagonalised,
        the lowest L/2 single-particle states are filled (Slater determinant),
        and the Berry phase of the Slater determinant is computed.

        For U=0 (non-interacting), this is EXACT and equals π·W (mod 2π):
            γ ≈ 0   → trivial phase
            γ ≈ π   → topological phase

        For U>0, this neglects interactions and is an approximation,
        but the result should remain quantised at 0/π as long as the
        many-body gap stays open.

        Why not the full many-body Zak phase (with spin)?
        — For SU(2)-symmetric spinful fermions at half-filling, the total
          many-body Berry phase is ALWAYS 0 (mod 2π) because both spin
          species contribute π (topological) + π (topological) = 2π ≡ 0.
          The total Zak phase ≠ topological invariant in this case.
          The correct many-body invariant is the Z₂ charge Berry phase,
          which equals the single-particle Zak phase computed here.

        Parameters
        ----------
        n_theta : int
            Number of θ points in [0, 2π).
            Recommended: 61+ for accuracy near critical point.
        verbose : bool
            Print progress.

        Returns
        -------
        gamma : float
            Zak phase in radians, wrapped to [0, 2π).
        info : dict
            'overlap_min', 'n_theta', 'mode'
        """
        L = self.L
        theta_vals = np.linspace(0, 2 * np.pi, n_theta, endpoint=False)

        # Hopping amplitudes for all bonds (incl. boundary bond)
        t = [self.t1 if i % 2 == 0 else self.t2 for i in range(L)]

        occ_list = []  # list of (L, L/2) occupied eigenvector sets
        gaps = []

        for i_th, theta in enumerate(theta_vals):
            if verbose and (i_th % max(1, n_theta // 5) == 0):
                print(f"  SP TBC θ [{i_th+1}/{n_theta}]  θ={theta:.4f}")

            # L×L single-particle hopping matrix with TBC
            H_sp = np.zeros((L, L), dtype=np.complex128)
            for i in range(L - 1):
                H_sp[i, i + 1] = -t[i]
                H_sp[i + 1, i] = -t[i]
            # PBC boundary bond with Peierls phase
            H_sp[L - 1, 0] = -t[L - 1] * np.exp(1j * theta)
            H_sp[0, L - 1] = -t[L - 1] * np.exp(-1j * theta)

            eigvals, eigvecs = np.linalg.eigh(H_sp)
            # Fill lowest L/2 states (half-filling, one spin species)
            occ = eigvecs[:, : L // 2]  # (L, N_occ)
            occ_list.append(occ)

            if L // 2 < L:
                gaps.append(eigvals[L // 2] - eigvals[L // 2 - 1])

        # ── Many-body Berry phase of the Slater determinant ──────────
        prod = 1.0 + 0.0j
        for n in range(n_theta):
            n_next = (n + 1) % n_theta
            # Overlap of two Slater determinants = det(Ψⁿ† Ψⁿ⁺¹)
            S = occ_list[n].T.conj() @ occ_list[n_next]
            ov = np.linalg.det(S)
            prod *= ov

        gamma = -np.angle(prod)  # result in [-π, π)
        if gamma < 0:
            gamma += 2 * np.pi
        # Snap to {0, π}: the SP Zak phase is quantised to these two
        # values when the single-particle gap is open.  Numerical noise
        # can land trivial at 2π instead of 0 — snap to whichever is
        # closer, with a generous tolerance.
        eps = 1e-5
        gamma_mod = gamma % (2 * np.pi)
        if abs(gamma_mod - np.pi) < eps:
            gamma = np.pi
        elif abs(gamma_mod) < eps or abs(gamma_mod - 2 * np.pi) < eps:
            gamma = 0.0

        min_gap = min(gaps) if gaps else 0.0
        overlap_min = min(abs(np.linalg.det(
            occ_list[n].T.conj() @ occ_list[(n + 1) % n_theta]
        )) for n in range(n_theta))

        if verbose:
            label = "topological" if abs(gamma - np.pi) < 0.5 else "trivial"
            print(f"  → SP Zak phase = {gamma:.6f}  ({label})")
            print(f"  → Min spectral gap = {min_gap:.6f}")
            print(f"  → Min overlap = {overlap_min:.6f}")

        return gamma, {
            'overlap_min': overlap_min,
            'min_gap': min_gap,
            'n_theta': n_theta,
            'mode': 'single_particle',
        }

    # ── Many-body Zak phase via TBC (reference / educational) ───────

    def compute_mb_zak_phase(self, n_theta=21, verbose=False,
                             return_all=False):
        """Many-body Zak phase — NOTE: 0 (mod 2π) for SU(2) systems!

        This computes the total many-body Berry phase of the spinful
        Fock-space ground state under TBC.

        !! WARNING !!
        For SU(2)-symmetric spinful fermions at half-filling, the TOTAL
        many-body Zak phase is ALWAYS 0 (mod 2π), regardless of topology,
        because the two spin copies contribute 2πW = 0 (mod 2π).

        Use compute_sp_zak_phase() (single-particle, spinless) instead
        for a proper 0 / π topological label.

        Kept for reference / educational / debugging purposes only.

        Returns
        -------
        zak_phase : float
            Always ≈ 0 (mod 2π) for SU(2) symmetric half-filled SSH.
        """
        theta_vals = np.linspace(0, 2 * np.pi, n_theta, endpoint=False)
        gs_list = []
        energies = np.empty(n_theta)

        for i, theta in enumerate(theta_vals):
            if verbose and (i % max(1, n_theta // 5) == 0):
                print(f"  TBC θ [{i+1}/{n_theta}]  θ={theta:.4f}")

            H = self.build_tbc_hamiltonian(theta)
            dim = self._basis.Ns

            if dim <= 5000:
                mat = H.tocsr().toarray()
                eigvals, eigvecs = np.linalg.eigh(mat)
                gs = eigvecs[:, 0]
                energies[i] = eigvals[0]
                if dim > 1 and (eigvals[1] - eigvals[0]) < 1e-10:
                    import warnings
                    warnings.warn(
                        f"Near-degenerate GS at θ={theta:.4f}, "
                        f"ΔE={eigvals[1]-eigvals[0]:.2e}. "
                        "Zak phase may be ill-defined."
                    )
            else:
                from scipy.sparse.linalg import eigsh
                eigvals, eigvecs = eigsh(H.tocsr(), k=1, which='SA')
                gs = eigvecs[:, 0].ravel()
                energies[i] = eigvals[0]

            gs_list.append(gs)

        prod = 1.0 + 0.0j
        overlaps = np.empty(n_theta, dtype=np.complex128)
        for n in range(n_theta):
            n_next = (n + 1) % n_theta
            ov = np.vdot(gs_list[n], gs_list[n_next])
            overlaps[n] = ov
            prod *= ov

        zak_phase = -np.angle(prod)
        if zak_phase < 0:
            zak_phase += 2 * np.pi

        phase_err = 1.0 - np.min(np.abs(overlaps))

        if verbose:
            print(f"  → Total MB Zak phase = {zak_phase:.6f}  (mod 2π)")
            print(f"  → Overlap drop = {phase_err:.2e}")
            print(f"  ⚠  For SU(2) SSH: total MB Zak phase ≡ 0 (mod 2π).")
            print(f"  → Use compute_sp_zak_phase() for correct 0/π label.")

        if return_all:
            return zak_phase, {
                'energies': energies,
                'overlaps': overlaps,
                'phase_err': phase_err,
            }
        return zak_phase

    # ── Analytic winding number (U=0 reference) ─────────────────────

    def compute_winding_number(self):
        """Winding number for *non-interacting* SSH model (U=0).

        W = 1/(2π) ∮ ∂_k arg[q(k)] dk,
        where q(k) = t₁ + t₂ e^{-ik}.

        This is an exact analytic result — no diagonalisation needed.

        Returns
        -------
        W : int
            Winding number (0 or 1).
        zak_phase : float
            Winding-number Zak phase = π·W (mod 2π).
        """
        if abs(self.t2) > abs(self.t1):
            W = 1
        else:
            W = 0
        return W, np.pi * W


# ══════════════════════════════════════════════════════════════════════
#  Dataset Generator
# ══════════════════════════════════════════════════════════════════════

def _refined_linspace(vmin, vmax, n, **_):
    """Non-uniform grid denser near the center (arcsine distribution).

    Transforms uniform spacing via arcsin to concentrate points toward
    the middle of [vmin, vmax].  Total point count remains n.
    """
    x = np.linspace(0, 1, n)
    y = 0.5 + np.arcsin(2 * x - 1) / np.pi
    return vmin + (vmax - vmin) * y


def generate_dataset(
    L=6,
    t1_range=(0.25, 4.0),
    t2_range=(0.25, 4.0),
    U_range=(0.0, 4.0),
    V_range=None,       # e.g. (0.0, 2.5) — None → no V sweep, V=0
    n_t1=16,
    n_t2=16,
    n_U=9,
    n_V=11,
    n_eig_save=20,
    n_theta=41,         # number of θ points for SP Zak phase
    refine_boundary=True,
    save_path=None,
    verbose=True,
):
    """Sweep t1, t2, U (and optionally V) and compute observables.

    Parameters
    ----------
    L : int
        System size (even).
    t1_range : (float, float)
        (min, max) for t1.
    t2_range : (float, float)
        (min, max) for t2.
    U_range : (float, float)
        (min, max) for U.
    V_range : (float, float) or None
        (min, max) for V.  If None, V is not swept (fixed at 0).
    n_t1, n_t2, n_U, n_V : int
        Number of grid points along each axis.
    n_eig_save : int
        Number of lowest eigenvalues to store in the dataset.
    n_theta : int
        Number of θ points for the single-particle Zak phase.
        31–41 is sufficient for L=6–12 away from the gap-closing point.
    refine_boundary : bool
        If True, use a non-uniform grid denser near the t₁ = t₂ diagonal
        (where the topological phase transition occurs).  Keeps the same
        total number of points, but concentrates them in the physically
        interesting region.
    save_path : str or None
        If given, save the dataset to this .npz file.
    verbose : bool
        Print progress.

    Returns
    -------
    data : dict
        Keys: t1_arr, t2_arr, U_arr, energies, spectra, corr_matrices,
              ent_spectra, double_occupancy, staggered_charge,
              bond_alternation, zak_phase, zak_overlap_min, zak_min_gap,
              winding_number, L, n_eig_save.
        Plus V_arr when V is swept.
    """
    # ── parameter grid ───────────────────────────────────────────────
    if refine_boundary:
        t1_vals = _refined_linspace(*t1_range, n_t1)
        t2_vals = _refined_linspace(*t2_range, n_t2)
    else:
        t1_vals = np.linspace(*t1_range, n_t1)
        t2_vals = np.linspace(*t2_range, n_t2)
    U_vals  = np.linspace(*U_range, n_U)

    sweep_V = V_range is not None
    if sweep_V:
        V_vals = np.linspace(*V_range, n_V)
        n_total = n_t1 * n_t2 * n_U * n_V
        grid_desc = f"{n_t1}×{n_t2}×{n_U}×{n_V} = {n_total}"
    else:
        n_total = n_t1 * n_t2 * n_U
        grid_desc = f"{n_t1}×{n_t2}×{n_U} = {n_total}"

    if verbose:
        print(f"Grid: {grid_desc} points")
        print(f"  t1 ∈ [{t1_range[0]}, {t1_range[1]}],  n = {n_t1}")
        print(f"  t2 ∈ [{t2_range[0]}, {t2_range[1]}],  n = {n_t2}")
        print(f"  U  ∈ [{U_range[0]}, {U_range[1]}],   n = {n_U}")
        if sweep_V:
            print(f"  V  ∈ [{V_range[0]}, {V_range[1]}],   n = {n_V}")
        print(f"  L  = {L}   (half-filling)\n")

    # ── pre-build shared basis & correlation operators ───────────────
    # (these are independent of t1, t2, U — same for all grid points)
    basis_proto = SSHModel(L, 1.0, 1.0, 0.0)   # dummy parameter — only for basis
    N_up = L // 2
    N_down = L // 2
    basis = spinful_fermion_basis_1d(L, Nf=(N_up, N_down))
    basis_proto._basis = basis
    basis_proto._build_corr_ops()
    ref_ops = basis_proto._corr_ops   # cached (list of lists)

    # Pre-build entanglement-spectrum mapping (basis-dependent only)
    # We pass the model's pre-computed basis to the method logic later.

    # ── allocate ─────────────────────────────────────────────────────
    energies = np.empty(n_total, dtype=np.float64)
    spectra  = np.empty((n_total, n_eig_save), dtype=np.float64)
    corr_mat = np.empty((n_total, L, L), dtype=np.float64)
    ent_spec_arr = np.empty((n_total, n_eig_save), dtype=np.float64)
    double_occ_arr = np.empty((n_total, L), dtype=np.float64)
    stag_chg_arr = np.empty(n_total, dtype=np.float64)
    stag_mag_sq_arr = np.empty(n_total, dtype=np.float64)
    bond_alt_arr = np.empty(n_total, dtype=np.float64)
    zak_phase_arr = np.empty(n_total, dtype=np.float64)
    zak_overlap_arr = np.empty(n_total, dtype=np.float64)
    zak_min_gap_arr = np.empty(n_total, dtype=np.float64)
    winding_arr = np.empty(n_total, dtype=np.int8)

    # ── loop ─────────────────────────────────────────────────────────
    t0 = time.perf_counter()
    idx = 0
    param_iter = product(
        t1_vals, t2_vals, U_vals,
        V_vals if sweep_V else [0.0],
    )
    for t1, t2, U_val, V_val in param_iter:
        # Build model and share cached basis + corr ops
        model = SSHModel(L, t1, t2, U_val, V=V_val)
        model._basis = basis
        model._corr_ops = ref_ops
        model.build_hamiltonian()

        # Ground state
        energy, gs = model.get_ground_state()
        energies[idx] = energy

        # Spectrum
        spec = model.get_spectrum(n_eig_save)
        spectra[idx] = spec

        # Correlation matrix (use pre-built corr ops + gs)
        corr, _ = model.get_correlation_matrix(gs)
        corr_mat[idx] = corr

        # Entanglement spectrum via Schmidt decomposition
        _, ent_spec = model.get_entanglement_spectrum(gs, n_max=n_eig_save)
        ent_spec_arr[idx] = ent_spec

        # Double occupancy
        D = model.get_double_occupancy(gs)
        double_occ_arr[idx] = D

        # Staggered charge density  δn  (from corr diagonal)
        stag_chg_arr[idx] = model.get_staggered_charge_density(corr_matrix=corr)

        # Staggered magnetization squared  ⟨m_s²⟩  (SDW indicator)
        stag_mag_sq_arr[idx] = model.get_staggered_magnetization_squared(gs)

        # Bond-order alternation  δB  (from corr off-diagonal)
        bond_alt_arr[idx] = model.get_bond_order_alternation(corr_matrix=corr)

        # ── Topological labels ──────────────────────────────────────────
        zak, zak_info = model.compute_sp_zak_phase(n_theta=n_theta, verbose=False)
        zak_phase_arr[idx] = zak
        zak_overlap_arr[idx] = zak_info['overlap_min']
        zak_min_gap_arr[idx] = zak_info['min_gap']

        W, _ = model.compute_winding_number()
        winding_arr[idx] = W

        idx += 1
        if verbose and (idx % 50 == 0 or idx == n_total):
            elapsed = time.perf_counter() - t0
            pct = idx / n_total * 100
            eta = elapsed / idx * (n_total - idx)
            print(
                f"  [{idx:5d}/{n_total}]  {pct:5.1f}%  "
                f"elapsed {elapsed:.1f}s  ETA {eta:.1f}s"
            )

    # ── reshape to grids ─────────────────────────────────────────────
    if sweep_V:
        shape = (n_t1, n_t2, n_U, n_V)
    else:
        shape = (n_t1, n_t2, n_U)
    energy_grid = energies.reshape(shape)
    spectra_grid = spectra.reshape((*shape, n_eig_save))
    corr_grid = corr_mat.reshape((*shape, L, L))
    ent_spec_grid = ent_spec_arr.reshape((*shape, n_eig_save))
    double_occ_grid = double_occ_arr.reshape((*shape, L))
    stag_chg_grid = stag_chg_arr.reshape(shape)
    stag_mag_sq_grid = stag_mag_sq_arr.reshape(shape)
    bond_alt_grid = bond_alt_arr.reshape(shape)
    zak_phase_grid = zak_phase_arr.reshape(shape)
    zak_overlap_grid = zak_overlap_arr.reshape(shape)
    zak_min_gap_grid = zak_min_gap_arr.reshape(shape)
    winding_grid = winding_arr.reshape(shape)

    data = {
        "t1_arr": t1_vals,
        "t2_arr": t2_vals,
        "U_arr": U_vals,
        "energies": energy_grid,
        "spectra": spectra_grid,
        "corr_matrices": corr_grid,
        "ent_spectra": ent_spec_grid,
        "double_occupancy": double_occ_grid,
        "staggered_charge": stag_chg_grid,
        "staggered_mag_sq": stag_mag_sq_grid,
        "bond_alternation": bond_alt_grid,
        "zak_phase": zak_phase_grid,
        "zak_overlap_min": zak_overlap_grid,
        "zak_min_gap": zak_min_gap_grid,
        "winding_number": winding_grid,
        "L": L,
        "n_eig_save": n_eig_save,
    }
    if sweep_V:
        data["V_arr"] = V_vals

    # ── save ─────────────────────────────────────────────────────────
    if save_path is not None:
        np.savez(save_path, **data)
        if verbose:
            print(f"\n✓ Dataset saved → {save_path}")

    elapsed = time.perf_counter() - t0
    if verbose:
        print(f"  Total time: {elapsed:.1f}s  ({elapsed/n_total:.3f}s/point)")

    return data


# ══════════════════════════════════════════════════════════════════════
#  Demo / CLI
# ══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="SSH-Hubbard exact diagonalisation")
    parser.add_argument("--L", type=int, default=6, help="System size (even)")
    parser.add_argument("--t1", type=float, default=1.0, help="Odd-bond hopping")
    parser.add_argument("--t2", type=float, default=0.8, help="Even-bond hopping")
    parser.add_argument("--U", type=float, default=2.0, help="Hubbard interaction")
    parser.add_argument("--V", type=float, default=0.0, help="Nearest-neighbour Coulomb repulsion")
    parser.add_argument("--spectrum", action="store_true", help="Print low-lying spectrum")
    parser.add_argument("--corr", action="store_true", help="Print correlation matrix")
    parser.add_argument("--dataset", action="store_true", help="Generate full dataset")
    parser.add_argument("--save", type=str, default=None, help="Save path for .npz")
    parser.add_argument("--n-t1", type=int, default=16)
    parser.add_argument("--n-t2", type=int, default=16)
    parser.add_argument("--n-U", type=int, default=9)
    parser.add_argument("--n-V", type=int, default=11,
                        help="Number of V grid points (only for --dataset)")
    parser.add_argument("--V-start", type=float, default=0.0,
                        help="V sweep start (default 0; set --V-end > 0 to enable sweep)")
    parser.add_argument("--V-end", type=float, default=0.0,
                        help="V sweep end (default 0 = no sweep)")
    parser.add_argument("--t1-start", type=float, default=0.25,
                        help="t1 sweep start (dataset mode)")
    parser.add_argument("--t1-end", type=float, default=4.0,
                        help="t1 sweep end (dataset mode)")
    parser.add_argument("--t2-start", type=float, default=0.25,
                        help="t2 sweep start (dataset mode)")
    parser.add_argument("--t2-end", type=float, default=4.0,
                        help="t2 sweep end (dataset mode)")
    parser.add_argument("--U-start", type=float, default=0.0,
                        help="U sweep start (dataset mode)")
    parser.add_argument("--U-end", type=float, default=4.0,
                        help="U sweep end (dataset mode)")
    args = parser.parse_args()

    if args.dataset:
        # ── generate full dataset ────────────────────────────────────
        save = args.save or os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            f"ssh_dataset_L{args.L}.npz",
        )
        V_range = (
            (args.V_start, args.V_end)
            if abs(args.V_end - args.V_start) > 1e-12
            else None
        )
        generate_dataset(
            L=args.L,
            t1_range=(args.t1_start, args.t1_end),
            t2_range=(args.t2_start, args.t2_end),
            U_range=(args.U_start, args.U_end),
            V_range=V_range,
            n_t1=args.n_t1,
            n_t2=args.n_t2,
            n_U=args.n_U,
            n_V=args.n_V,
            save_path=save,
            verbose=True,
        )
    else:
        # ── single-point demo ────────────────────────────────────────
        model = SSHModel(args.L, args.t1, args.t2, args.U, V=args.V)
        print(f"SSH model:  L={args.L}  t1={args.t1}  t2={args.t2}"
              f"  U={args.U}  V={args.V}")
        print(f"  Basis size = {model._build_basis().Ns}")
        E0, gs = model.get_ground_state()
        print(f"  Ground-state energy E₀ = {E0:.10f}")

        if args.spectrum:
            spec = model.get_spectrum(10)
            print(f"  Low-lying spectrum (first 10):")
            for n, e in enumerate(spec):
                print(f"    E_{n} = {e:.8f}")

        if args.corr:
            corr, _ = model.get_correlation_matrix(gs)
            print(f"  Correlation matrix ⟨c†_i c_j⟩:")
            np.set_printoptions(precision=6, suppress=True, linewidth=100)
            print(corr)
            # Also print the diagonal occupations
            occ = np.diag(corr).real
            print(f"  Site occupations n_i = diag(ρ): {occ}")
            print(f"  Total occupancy Σ n_i = {occ.sum():.4f}  "
                  f"(expected {args.L})")
