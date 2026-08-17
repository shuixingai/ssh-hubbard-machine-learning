"""
Test TBC Zak phase computation for the SSH model.

Compare 3 levels of topological labels:
  1. t₁/t₂ heuristic          (coarse baseline)
  2. Analytic winding number   (U=0 exact, no ED)
  3. Single-particle Zak via TBC (spinless Slater det, works for any U)

Usage:  python test_zak_phase.py
"""
import sys, os
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ssh_model import SSHModel


def test_zak_phase(L=6, t1=1.0, t2=1.5, U=0.0, n_theta=61):
    """Test a single parameter point."""
    print("=" * 60)
    print(f"SSH Model  L={L}  t1={t1}  t2={t2}  U={U}")
    print("=" * 60)

    model = SSHModel(L, t1, t2, U)

    # ── Level 0: t₁/t₂ heuristic ──────────────────────────────────
    t_ratio = t1 / t2
    label_heur = "topological" if t2 > t1 else "trivial"
    print(f"\n[Level 0]  t₁/t₂ = {t_ratio:.4f}  →  {label_heur}")

    # ── Level 1: Analytic winding number ────────────────────────────
    W, zak_analytic = model.compute_winding_number()
    label_an = "topological" if W == 1 else "trivial"
    print(f"[Level 1] Analytic W = {W}  γ = {zak_analytic:.6f}  →  {label_an}")

    # ── Level 2: Single-particle Zak phase via TBC ─────────────────
    print(f"\n[Level 2] Single-particle Zak phase  (n_θ={n_theta})")
    zak_sp, info = model.compute_sp_zak_phase(
        n_theta=n_theta, verbose=True,
    )
    label_sp = "topological" if abs(zak_sp - np.pi) < 0.5 * np.pi else "trivial"
    print(f"          γ = {zak_sp:.6f}  →  {label_sp}")

    # ── Consistency (U=0 only) ─────────────────────────────────────
    print(f"\n── Consistency check ──")
    if U == 0:
        diff = min(abs(zak_sp - zak_analytic) % (2 * np.pi),
                   abs(zak_analytic - zak_sp) % (2 * np.pi))
        print(f"  |γ_TBC - γ_analytic| (mod 2π) = {diff:.6e}")
        if diff < 1e-3:
            print("  ✓  TBC matches analytic winding number.")
        else:
            print("  ⚠  Deviation — investigate!")
    else:
        print(f"  U={U} > 0: SP Zak phase is the single-particle reference.")
        print(f"  (May deviate from exact phase boundary at large U.)")

    return zak_sp, info


def scan_t1_vs_t2(L=6, U=0.0, n_theta=61):
    """Scan t1 at fixed t2 and check the transition."""
    t2_fixed = 1.0
    t1_vals = np.linspace(0.2, 1.8, 9)

    print("\n" + "=" * 64)
    print(f"Scan t₁ at fixed t₂={t2_fixed}  (U={U})")
    print("=" * 64)
    print(f"{'t1':>6}  {'t1/t2':>8}  {'heuristic':>10}  {'W_an':>6}  "
          f"{'γ_SP':>10}  {'label':>12}  {'gap':>8}")
    print("-" * 64)

    for t1 in t1_vals:
        model = SSHModel(L, t1, t2_fixed, U)
        W, _ = model.compute_winding_number()
        zak, info = model.compute_sp_zak_phase(n_theta=n_theta, verbose=False)
        label = "TOPOLOGICAL" if abs(zak - np.pi) < 0.5 * np.pi else "trivial"
        t_ratio = t1 / t2_fixed
        heur = "t1>t2" if t1 > t2_fixed else "t2>t1"
        gap_str = f"{info['min_gap']:.4f}" if info['min_gap'] > 1e-6 else "<1e-6"
        print(f"{t1:6.2f}  {t_ratio:8.4f}  {heur:>10}  {W:6d}  "
              f"{zak:10.6f}  {label:>12}  {gap_str:>8}")


def scan_U_effect(L=6, t1=0.5, t2=1.5, U_vals=None, n_theta=61):
    """Check how U affects topology (via SP Zak phase reference)."""
    if U_vals is None:
        U_vals = [0.0, 0.5, 1.0, 2.0, 4.0]

    print("\n" + "=" * 64)
    print(f"U dependence in topological regime  (t₁={t1}, t₂={t2})")
    print("=" * 64)
    print(f"{'U':>6}  {'γ_SP':>10}  {'label':>12}  {'min_gap':>10}  "
          f"{'overlap_min':>12}")
    print("-" * 64)

    for U in U_vals:
        model = SSHModel(L, t1, t2, U)
        zak, info = model.compute_sp_zak_phase(n_theta=n_theta, verbose=False)
        label = "topological" if abs(zak - np.pi) < 0.5 * np.pi else "trivial"
        print(f"{U:6.2f}  {zak:10.6f}  {label:>12}  {info['min_gap']:10.6f}  "
              f"{info['overlap_min']:12.6f}")


if __name__ == "__main__":
    print("═══  SSH Zak Phase (Single-Particle TBC) — Test Suite  ═══\n")

    # Test 1: topological point
    test_zak_phase(L=6, t1=0.5, t2=1.5, U=0.0, n_theta=61)

    # Test 2: trivial point
    test_zak_phase(L=6, t1=1.5, t2=0.5, U=0.0, n_theta=61)

    # Test 3: scan across transition
    scan_t1_vs_t2(L=6, U=0.0, n_theta=61)

    # Test 4: U dependence in topological regime
    scan_U_effect(L=6, t1=0.5, t2=1.5)

    print("\n═══  Done  ═══")
