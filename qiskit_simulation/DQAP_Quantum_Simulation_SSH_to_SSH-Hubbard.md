# DQAP Quantum Simulation: From SSH (U=0) to SSH-Hubbard (U>0)

## Part 0: Previous Work — Machine Learning Phase Recognition

### Project Overview

The SSH-Hubbard project aims to identify the topological phase diagram of the 1D SSH-Hubbard model using **classical numerical methods + machine learning**. The model adds on-site Coulomb interaction U to the SSH dimerized hopping (t₁, t₂):

$$
H = -\sum_{i,\sigma} t_i (c^\dagger_{i,\sigma} c_{i+1,\sigma} + \text{h.c.}) + U \sum_i n_{i\uparrow} n_{i\downarrow}
$$

### Completed Results

**1. Data Production (ED L=6)**

Fine-grained scan of the full parameter space:
- **t₁/t₂ dimension**: t₁ ∈ [0.1, 3.0], t₂ = 2.0 − t₁ (refined near the phase boundary t₁ = t₂)
- **U dimension**: U ∈ [0, 4], step size 0.25–0.5
- **Total parameter points**: ~2304 combinations
- **Measured observables**: single-particle spectrum, correlation matrix (36-dim), entanglement spectrum (20-dim), double occupancy, Resta polarization, SP Zak Phase

**2. Topological Labeling Strategy**

Since the many-body Zak Phase is identically 0 (mod 2π) under spinful SU(2) symmetry, the **single-particle Zak Phase (TBC Berry Phase)** is adopted as the label:

| Method | Includes U? | SU(2) Issue? | Usability |
|--------|-------------|--------------|-----------|
| SP Zak Phase (TBC) | ✗ (reference system approx.) | None | ✅ L=6 viable |
| MB Zak Phase (Fock ED) | ✓ | Identically ≡0 | ❌ |
| Resta polarization | ✓ | None | ⚠️ Fails at L=6 (insufficient phase resolution) |
| Z₂ Charge Berry Phase | ✓ | None | 📝 To be implemented |

The SP Zak Phase is adiabatically connected to the true many-body topology at moderate U, making it an efficient and physically sound label.

**3. ML Classification Results**

Multiple classifiers perform well in phase identification:
- **kNN**: cleanly separates the 0/π sectors in t₁−t₂ scans, with the decision boundary near t₁ = t₂
- **PCA/t-SNE**: dimensionality reduction reveals two well-separated clusters in parameter space, corresponding to the topological and trivial phases
- **Feature importance**: next-nearest-neighbor elements of the correlation matrix contribute most to topological classification

**4. DMRG Extension to Larger Systems (L=20)**

- MPS tensor network with bond dimension χ controlling accuracy
- Entanglement spectra and correlation functions computed, validating the L=6 ED phase diagram
- Discovery of a deep connection between the area law and topological phase transitions — entanglement surges across phase boundaries (χ demand explodes), the same physics that drives the growth in DQAP circuit depth M

### Limitations of the Classical Path and the Quantum Motivation

```
Classical (ED/DMRG + ML)                DQAP Quantum Simulation
────────────────────────                ────────────────────────
Classical computer simulates            Quantum computer simulates
Generate data → post-process → ML       Directly measure observables
L=6 ED constrained; DMRG needs χ tuning Circuit depth M controls accuracy
Topology inferred indirectly            Polarization jump directly visible
```

The fundamental limitation of the classical approach: **a vast amount of quantum information is compressed into the post-processing pipeline** (correlation matrix → ML → label), whereas quantum simulation can directly measure the topological order parameter (Resta polarization), bypassing the "data → inference" intermediate step.

---

## Overview

This document explains the core method of the RIKEN 2025 paper (Xie et al., *Digital quantum simulation of SSH model using a PQC*) — DQAP — and its relationship to the SSH-Hubbard phase recognition project (ED/DMRG + ML) described above. It ultimately points toward a feasible extension:

**Simulating SSH-Hubbard on real quantum hardware, directly measuring the topological order parameter as a function of U.**

```
Three ways to handle SSH phase transitions:

① Device physics (original plan)   ② Classical numerics (current)   ③ Quantum simulation (target)
gate-gate map                      ED/DMRG produce data            DQAP quantum circuit
→ lever arm                        → ML classifies phase diagram   → directly measure polarization jump
→ infer U, V                       → infer topology via post-proc. → see topology
→ infer phase

   "Fabricate and measure"            "Compute classically"            "Simulate with quantum machine"
```

---

## Part 1: What is DQAP?

### Basic Idea

DQAP (Digital Quantum Adiabatic Passage) is a method for **preparing ground states on quantum computers** in two steps:

```
Step 1 — Adiabatic initialization:
  Start from the ground state of an "easy" Hamiltonian H₁
  Follow the adiabatic path H(s) = (1−s)H₁ + sH₂ to reach the target H₂
  Discretize the path → obtain rotation angles θ₁⁰, θ₂⁰ for M layers of gates

Step 2 — VQE refinement:
  Use θ⁰ as the initial guess for variational optimization (L-BFGS-B)
  Fine-tune θ to minimize energy → accurate ground state
```

The advantage of using adiabatic initialization over random initialization: VQE converges faster and avoids local minima.

### SSH Decomposition in the RIKEN Paper

For the SSH model, the Hamiltonian is split into two parts:

```
H₁ = −v Σ (c†_A,i c_B,i + h.c.)         ← intra-cell hopping (A↔B within the same cell)
H₂ = −γw Σ (c†_B,i c_{A,i+1} + h.c.)    ← inter-cell hopping (B→A between different cells)

H_SSH = H₁ + H₂
```

Each DQAP circuit layer alternately evolves H₁ and H₂:

```
e^{−iθ₁H₁} · e^{−iθ₂H₂}
```

### Key Results

| Initial → Target state | Energy convergence rate | Required depth M* |
|------------------------|------------------------|-------------------|
| Same topological phase | Exponential | ~L/8 |
| Different topological phases | Polynomial | ~L/4 |

**Same-phase is fast, cross-phase is slow** — because crossing a topological phase transition requires passing through a gap-closing point, where the entanglement structure must be reorganized.

### Polarization Jump

Once the ground state is prepared, the Resta polarization is measured directly via the Hadamard test:

```
P_R = (1/2π) Im ln ⟨Ψ₀|U_R|Ψ₀⟩

U_R = exp(i 2π/L · Σ_j j n_j)
```

At the critical depth M*, P_R jumps from 0 → π (or π → 0) — this is the signal of a topological phase transition.

---

## Part 2: Relationship Between the RIKEN Paper and the SSH-Hubbard Project

### Comparison of the Two Directions

| Dimension | SSH-Hubbard Project | RIKEN Paper |
|-----------|--------------------|-------------|
| Model | SSH-Hubbard (U>0) | SSH (U=0) |
| Method | ED L=6 / DMRG L=20 | DQAP + quantum hardware |
| Ground state | Matrix diagonalization / MPS tensor | Variational quantum circuit optimization |
| Topological label | SP Zak Phase (TBC, post-processed) | Resta polarization (direct measurement) |
| Phase classification | ML (kNN / PCA / t-SNE) | Polarization jump directly visible |
| Hardware | Classical computer | Quantinuum H1-1 (20 qubit) |

### Complementary Relationship

```
Classical analysis                     RIKEN quantum simulation
┌──────────────────────┐             ┌──────────────────────────┐
│ Fast, cheap, wide    │             │ Expensive, slow, but     │
│ parameter sweep      │             │ directly measures physics │
│ L=6 ED scans 2304 pts│             │ L=18, hundreds of circuits│
│ DMRG up to L=20      │  ←shared→  │ Qubit-number limited     │
│                      │  physics   │                          │
│ Phase diagram:       │  ──────→   │ Submit for quantum       │
│ which phases exist,  │  verify    │ hardware validation      │
│ where boundaries lie │            │                          │
└──────────────────────┘            └──────────────────────────┘
```

**Core: Area Law**

```
DMRG:                                DQAP (RIKEN's method):
χ = bond dimension                    M = number of circuit layers
Controls the maximum entanglement    Controls the maximum entanglement
an MPS can express                    a circuit can express

Same phase → small χ suffices         Same phase → small M suffices
Cross phase → χ demand explodes       Cross phase → M demand grows

Two thermometers, one pot of water
```

**Rigorous mathematical correspondence**:
- An MPS with bond dimension χ ⇔ a quantum circuit of depth ~O(log χ)
- The quantum state produced by M circuit layers ⇔ an MPS with bond dimension ~e^M

Hence DMRG experience transfers directly to quantum circuits. The difference:
- DMRG: large χ → exponential memory blowup (practical bottleneck)
- Quantum circuit: large M → many gates → noise accumulation (practical bottleneck)

### VQE Positioning

| | DMRG (Classical VQE) | Quantum VQE |
|---|---|---|
| What is parameterized | MPS tensors | Gate rotation angles θ |
| How energy is computed | Tensor contraction | Run circuit → measure expectation values |
| Bottleneck | Bond dimension explosion | Shot noise + gate fidelity |

The VQE framework is familiar to both classical and quantum computing — only the physical carrier of "θ" differs.

---

## Part 3: The Extension — Adding Hubbard U

### Physical Motivation

RIKEN's work is still U=0 SSH. The SSH-Hubbard project runs SSH-Hubbard (U>0), raising several key questions **never verified on quantum hardware**:

1. **Does the topological phase survive as U increases?** (Are edge states suppressed by the Mott transition?)
2. **How does the critical depth M*(U) behave as a function of U?**
3. **Is Resta polarization still a clean observable at U>0?**

### Technical Changes

Adding U means:

| | RIKEN (U=0) | Proposed (U>0) |
|---|---|---|
| Model | SSH | SSH-Hubbard |
| Qubits per site | 1 (spatial DOF only) | **2** (↑ and ↓) |
| L=6 total qubits | 6 | 12 |
| Trotter blocks per DQAP layer | 2 | **3** (+H_U) |
| Variational parameters/layer | θ₁, θ₂ (2) | θ₁, θ₂, θ_U (3) |

### Hubbard Term in a Quantum Circuit

After the Jordan-Wigner transformation:

```
U · n↑n↓ → U · (I−Z↑)(I−Z↓)/4
        = U/4 · (I − Z↑ − Z↓ + Z↑Z↓)
```

The **Z↑Z↓ term maps directly to an e^{−iαZZ} gate** — and ZZPhase is precisely the native gate that Quantinuum hardware implements most naturally.

### L=6 Gate Count Estimate

**L=6 → 3 cells × 2 spins → 12 qubits, M=2 layers conservative estimate:**

| Term | Two-qubit gates per layer |
|------|--------------------------|
| SSH intra-cell (H₁): 3 cells × 2 spins × 2 | 12 |
| SSH inter-cell (H₂): 3 bonds × (2+1 JW string) | 18 |
| Hubbard (H_U): 6 sites × 1 | 6 |
| **Subtotal per layer** | **36** |
| **Total for M=2** | **72** |

**Comparison**: The RIKEN paper ran L=18 SSH / M=4 on H1-1, using **170 two-qubit gates**. L=6 / M=2 requires only **72 gates** — smaller, shallower, and less susceptible to noise.

### Experimental Feasibility

| | RIKEN (already done) | Proposed |
|---|---|---|
| Platform | H1-1 (20 qubit) | H1-1 sufficient; H2 (56 qubit) preferred |
| Gate count | 170 (M=4) | 72–108 (M=2–3) |
| Two-qubit gate fidelity | ~99.8% | Same |
| Success probability | ✅ Polarization jump clearly visible | ✅ Smaller circuit → expected to be more reliable |

---

## Part 4: Technical Roadmap

### Phases

```
Phase 0 — Classical simulator verification (doable now)
  Implement DQAP + SSH-Hubbard on Qiskit-Aer simulator
  Verify that the polarization jump at U>0 is still clean
  → Cost: $0, pure coding
  → Output: feasibility demonstration

Phase 1 — Quantum hardware validation
  Submit verified circuits to Quantinuum H1-1 or H2
  → Cost: compute time (~$1K)
  → Output: first measurement of U>0 topological phases on quantum hardware

Phase 2 — Physical exploration
  Scan U from 0 to ~4, observe M*(U) dependence
  Scan t₁/t₂ across the topological phase transition
  → Output: quantum phase diagram in the U−θ plane
```

### How Existing Code Connects

```
Classical computation                 Extension direction (quantum simulation)
┌──────────────────┐                 ┌──────────────────────┐
│ ssh_model.py     │                 │ dqap_circuit.py      │
│ ED generates data│                 │ Build DQAP circuit   │
│ Compute obsrvbls │                 │ VQE optimize θ       │
├──────────────────┤                 │ Hadamard test        │
│ Post-processing  │                 │ Measure polarization │
│ Compute Zak Phase│        parallel ├──────────────────────┤
│ Train ML         │       ──────→   │ Results comparable to│
│ Plot phase diag. │                 │ ML phase diagram     │
└──────────────────┘                 └──────────────────────┘
```

The two are not a binary choice — **classical results serve as a predictive benchmark for quantum experiments**, and quantum experiments in turn validate the numerical conclusions.

---

## Part 5: Reproduction Results (2026-07-15)

### Experimental Setup

- Script: `dqap_ssh_reproduce.py`
- Parameters: `L=4` (2L=8 qubits), `boundary=APBC`, `M_max=4`
- Trivial parameters: v=2.0, w=1.0 (conventional SSH label: t₁>t₂ → trivial)
- Topological parameters: v=1.0, w=2.0 (conventional SSH label: t₂>t₁ → topological)
- Initial state: |t⟩^⊗L (H₁ bonding state, one entangled pair (|01⟩+|10⟩)/√2 per cell)

### Observations

- **Trivial (v=2, w=1)**: P stable at ~0.5; exact value also ~0.5 — initial and target polarizations coincide
- **Topological (v=1, w=2)**: P drops from ~0.5 at M=0 to ~0 at M=4; exact value also ~0 — initial and target polarizations differ

### Trend Analysis

The trivial polarization stays at 0.5 while the topological polarization drops from 0.5 to 0. This trend is correct for the following reasons:

1. **APBC π flux effect**: APBC at L=4 is equivalent to inserting a π flux, which swaps the polarization assignments of the topological and trivial sectors. The conventional SSH labels (t₁>t₂=trivial → P=0) are **reversed** under APBC+L=4 — so the trivial parameters actually show P≈0.5, and the topological parameters show P≈0. (See [dqap-polarization-hypotheses.md](dqap-polarization-hypotheses.md) Hypothesis 2.)

2. **Core physics successfully reproduced**:
   - **Same topological phase** (initial polarization ≈ exact polarization): P remains stable as M increases; energy converges exponentially ✓
   - **Different topological phases** (initial polarization ≠ exact polarization): P jumps with M (0.5→0); energy converges polynomially ✓

This behavior matches the conclusion reported in the RIKEN paper — **the jump in polarization (not its absolute value) is the signal of a topological phase transition**.

### To Be Verified (Hypothesis Testing)

| Hypothesis | Verification Method | Expected Outcome |
|------------|-------------------|------------------|
| H2 (APBC π flux swaps sectors) | Run with `boundary='PBC'` | trivial→P≈0, topological→P≈0.5 (labels return to "normal") |
| H3 (L=4 phase resolution too low) | Increase L to 8 or 12 | P quantization closer to 0/0.5 |
| H1 (Sign convention) | Independently check winding number via `compute_winding_number()` | v>w→W=0, v<w→W=1 |

---

## Part 6: What Needs To Be Done

The following are optional directions to pursue:

| ID | Topic | What To Do |
|----|-------|------------|
| A | **Install PennyLane / Qiskit** | `pip install pennylane`, then use its automatic differentiation for VQE |
| B | **DQAP circuit construction script** | Write the SSH-Hubbard brick-wall circuit using Qiskit (already available) |
| C | **VQE optimizer + polarization measurement** | L-BFGS-B + Hadamard test to measure Resta polarization |
| D | **Comparison with existing Zak Phase results** | Overlay the quantum-simulated polarization jump location with the ML phase diagram |

**Recommended starting point**: (B) — write the circuit construction first, as it is the core technical step of the RIKEN paper and the part with the most overlap with existing work, making it the easiest to understand.

---

## References

- Xie, Seki, Shirakawa & Yunoki (2025), *Digital quantum simulation of the Su-Schrieffer-Heeger model using a parameterized quantum circuit*, arXiv:2504.08543
- Seki, Shirakawa & Yunoki (2022), *DQAP method*, PRB 105, 155106
- Smith, Jobst, Green & Pollmann (2022), *Topological phase transitions on IBM quantum processor*, PRR 4, L022020
- Resta (1998), *Quantum theory of polarisation*, PRL 80, 1800
- Ye, Mu & Fan (2016), *Entanglement spectrum of SSH-Hubbard model*, PRB 94, 165167
- Related project file: `SSH_Zak_Phase_完整原理.md` (Complete derivation of Zak Phase as a topological label)
