#!/usr/bin/env python3
"""
Verify Wang Yuanfeng's ground-truth claims (2026-08-10) numerically.

Wang's conventions:
    H(k) = [[0, q(k)], [q*(k), 0]],   q(k) = t1 + t2 * e^{ik}
    W  = (1/2pi) ∮ ∂_k arg q(k) dk
    gamma_Zak = i ∮ <u_-(k)|∂_k u_-(k)> dk   (occupied band)
    Claim: |t1|>|t2| => gamma=0, W=0 ;  |t2|>|t1| => gamma=pi, W=+1
           gamma = pi * W  (mod 2pi)

Also checks the convention-dependence (his section 2.1): flipping e^{ik}->e^{-ik}
or H->-H must leave the *binary classification* invariant (mod 2pi).
"""
import numpy as np

def q_func(k, t1, t2, sign=+1):
    """q(k) = t1 + t2 * e^{i*sign*k}.  Wang uses sign=+1."""
    return t1 + t2 * np.exp(1j * sign * k)

def winding_number(t1, t2, n_k=2001, sign=+1):
    """W = (1/2pi) ∮ ∂_k arg q(k) dk, via unwrapped phase differences."""
    k = np.linspace(0.0, 2*np.pi, n_k, endpoint=False)
    q = q_func(k, t1, t2, sign)
    dphi = np.unwrap(np.diff(np.angle(q)))
    return np.round(dphi.sum() / (2*np.pi)).astype(int)

def zak_phase(t1, t2, n_k=401, sign=+1, sign_H=+1):
    """Occupied-band Zak phase via Wilson loop (no finite-difference of |u>).

    For E_minus band: u_-(k) ∝ (1, -q*/|q|)  (sign of H as in Wang's H(k)).
    sign_H = -1  -> H(k) = -[[0,q],[q*,0]], occupied band is E_plus.
    gamma = -Im ln prod <u(k_j)|u(k_{j+1})>  (mod 2pi), wrapped to [0, 2pi).
    """
    k = np.linspace(0.0, 2*np.pi, n_k, endpoint=False)
    q = q_func(k, t1, t2, sign)
    mag = np.abs(q)
    # u_(k) = (1, sH * (-q*/|q|)) / sqrt(2), sH=+1 for E_minus band.
    u = np.stack([np.ones_like(q), sign_H * (-np.conj(q) / mag)], axis=-1) / np.sqrt(2)
    # Wilson line
    prod = 1.0
    for j in range(n_k):
        prod *= np.vdot(u[j], u[(j + 1) % n_k])
    gamma = -np.angle(prod)            # in (-pi, pi]
    return gamma % (2*np.pi)

def _mod2pi(g, atol=1e-6):
    """Wrap gamma into [0, 2pi), snapping 2pi-eps back to 0 (pure FP artifact)."""
    w = g % (2*np.pi)
    if abs(w - 2*np.pi) < atol:
        w = 0.0
    return w

def report(t1, t2, **kw):
    W_p, W_m = winding_number(t1, t2, sign=+1), winding_number(t1, t2, sign=-1)
    g_p   = _mod2pi(zak_phase(t1, t2, sign=+1))
    g_m   = _mod2pi(zak_phase(t1, t2, sign=-1))
    g_neg = _mod2pi(zak_phase(t1, t2, sign=+1, sign_H=-1))
    pred = "trivial" if abs(t1) > abs(t2) else "topological"
    ok = np.isclose(g_p, np.pi*W_p % (2*np.pi), atol=1e-6) and (abs(g_p - np.pi) < 1e-6 or g_p < 1e-6)
    print(f" t1={t1:>4} t2={t2:>4}  ({pred:11s})"
          f"  | W(e+ik)={W_p:>2}  W(e-ik)={W_m:>2}"
          f" |  gamma(e+ik)={g_p/ np.pi:+.4f}*pi"
          f"   gamma(e-ik)={g_m/ np.pi:+.4f}*pi"
          f"   gamma(-H,e+ik)={g_neg/ np.pi:+.4f}*pi"
          f" |  gamma=pi*W (mod2pi)? {ok}")

if __name__ == "__main__":
    print("== Verify Wang's claims: q(k)=t1+t2*e^{ik}, H(k)=[[0,q],[q*,0]] ==")
    print("   (all gamma modulo 2*pi; W(e-ik)/gamma(e-ik) show convention dependence, section 2.1)\n")
    for t in [(2.0, 1.0), (1.0, 2.0), (3.0, 1.0), (1.0, 3.0), (0.5, 2.0), (2.0, 0.5)]:
        report(*t)
    print("\n-- critical point |t1|=|t2| : bulk gap closes -> gamma undefined --")
    try:
        # At t1=t2 the two bands touch at k=pi; check |q| has a zero.
        k = np.linspace(0, 2*np.pi, 4001)
        gap = 2*np.min(np.abs(q_func(k, 1.0, 1.0)))
        print(f" t1=t2=1.0 : min |q| = {gap:.3e}  -> 2*|q| gap ~ {2*gap:.3e} (gap-closing)")
    except Exception as e:
        print("  critical check failed:", e)
