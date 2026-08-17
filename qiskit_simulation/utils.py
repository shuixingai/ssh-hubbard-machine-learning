#!/usr/bin/env python3
"""
DQAP SSH Utilities
==================
Shared tools for spinless and spinful DQAP codes.

Provides:
    _pauli_label:      Build Pauli label strings for SparsePauliOp
    diagonalize_hamiltonian: Automatic sparse/dense diagonalization

Design: lightweight common layer, consumed by dqap_ssh_reproduce.py
        and dqap_ssh_hubbard.py. No circular imports.
"""

import numpy as np


def estimate_hilbert_memory(n_qubits):
    """Estimate dense diagonalization memory for n_qubits (in bytes).

    Full dense matrix is 2^n_qubits × 2^n_qubits of complex128
    (16 bytes per element), so total = 16 * 4^n_qubits.
    """
    return 16 * (4 ** n_qubits)


def _pauli_label(n_qubits, positions, paulis):
    """Build a Pauli label string for SparsePauliOp.

    Args:
        n_qubits: Total number of qubits
        positions: List of qubit indices
        paulis: List of Pauli characters ('X', 'Y', 'Z') matching positions

    Returns:
        String like 'X0Z1Z2X3' for n_qubits=4, positions=[0,3], paulis=['X','X']
        with Z identity on the filler qubits.
    """
    chars = ['I'] * n_qubits
    for pos, p in zip(positions, paulis):
        chars[pos] = p
    return ''.join(chars)


def diagonalize_hamiltonian(hamiltonian, k=1, memory_limit_gb=None):
    """Diagonalize Hamiltonian, returning (eigenvalues, eigenvectors).

    For systems ≤ 12 qubits: full dense diagonalization via np.linalg.eigh.
    For larger systems: sparse eigensolver (eigsh) for k lowest eigenvalues.
    If memory_limit_gb is set, systems whose dense matrix would exceed
    the limit are rejected with MemoryError instead of being computed.

    This avoids O(2^N) memory blowup at large L that occurred with
    the original matrix.toarray() approach (e.g. 16 TiB at L=10).

    Args:
        hamiltonian: SparsePauliOp (or any sparse/compatible matrix)
        k: Number of lowest eigenvalues to compute (default 1)
        memory_limit_gb: Optional memory cap (GiB) for dense diagonalization;
            exceeding it raises MemoryError (default None = unlimited)

    Returns:
        eigenvalues: 1D array of length k (or all eigenvalues if ≤ 12 qubits)
        eigenvectors: 2D array, each row is an eigenvector
    """
    matrix_sparse = hamiltonian.to_matrix(sparse=True)
    n_qubits = int(np.log2(matrix_sparse.shape[0]))

    # Added memory-limit check: route to sparse solver when dense
    # diagonalization would exceed the user-set budget.
    dense_bytes = estimate_hilbert_memory(n_qubits)
    exceed_limit = (
        memory_limit_gb is not None
        and dense_bytes > memory_limit_gb * 1024**3
    )

    if n_qubits <= 12 and not exceed_limit:
        # Dense diagonalization (up to 4096×4096 = ~256 MiB)
        matrix = matrix_sparse.toarray()
        eigvals, eigvecs = np.linalg.eigh(matrix)
        return eigvals[:k], eigvecs[:, :k].T
    if exceed_limit:
        # Strict memory limit: terminate instead of falling back to sparse.
        raise MemoryError(
            f"memory_limit_gb={memory_limit_gb} exceeded by dense matrix "
            f"({dense_bytes / 1024**3:.2f} GiB); aborting to prevent memory "
            "blowup."
        )
    # Sparse eigensolver for large systems
    from scipy.sparse.linalg import eigsh

    n_v = min(k + 2, matrix_sparse.shape[0] - 1)  # safety margin
    if (matrix_sparse - matrix_sparse.getH()).max() < 1e-12:
        eigvals, eigvecs = eigsh(matrix_sparse, k=n_v, which='SA')
    else:
        # Fallback: use Hermitian part (shouldn't trigger for SSH)
        eigvals, eigvecs = eigsh(
            matrix_sparse + matrix_sparse.getH(), k=n_v, which='SA'
        )
        eigvals = eigvals / 2.0
    return eigvals[:k], eigvecs.T[:k]
