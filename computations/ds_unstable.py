"""Dancer–Strachan Kähler–Einstein system (Castro thesis, App. A.3–A.4): numerical check of an exact identity.

Written 2026-10-15 15:57 CEST (corrected from a guessed 16:00).  [Lit] system (A.3.4) with Lambda = -1:
    a' = a(b^2 + c^2 - a^2)/2,   b' = b(c^2 + a^2 - b^2)/2,   c' = c(a^2 + b^2 - c^2 + 2 a^2 b^2)/2.
Family (II): the unstable curve of the critical point (q, 0, q), q > 0, which blows up at a finite time t*.
The thesis defines L(q) = lim a/b at t*, and the CR parameter at infinity is a_CR = L^(-2).
[Ours] Claim: with tau(t) = int_{-inf}^t b^2, the ratio r = a^2/b^2 obeys dr/dtau = -2 r (r - 1) exactly, so
    r(t) = 1 / (1 - exp(-2 tau(t)))   and   a_CR = L^(-2) = 1 - exp(-2 T),  T = int_{-inf}^{t*} b^2 dt.
This script integrates the unstable curve (in log variables, with time slowed near blow-up) and checks the
identity along the whole trajectory, then reports L(q) and T(q).
2026-10-16 22:12 CEST (corrected from a guessed 22:58): added certify(): stop at the level m = a^2 = M and bracket the remaining time by the
theorem's own bounds, T - tau_M in [ln(1 + 3/(2M)), ln(1 + 2/M)], so a finite run gives an interval for T.
2026-10-16 22:21 CEST (corrected from a guessed 22:40): added tube_radius(): R = int a b dt, which Dancer-Strachan 2002 (Sec. 4) identify with the
radius of the tube; checks L = coth R and T = ln cosh R, so the bounds on T are bounds on cosh R.
Run: .venv/bin/python computations/ds_unstable.py
"""
import numpy as np
from scipy.integrate import solve_ivp


def rhs(_s, Y):
    x, y, z, tau, t = Y  # x = ln a, y = ln b, z = ln c
    a2, b2, c2 = np.exp(2 * x), np.exp(2 * y), np.exp(2 * z)
    S = 1.0 + a2 + b2 + c2 + 2 * a2 * b2  # dt/ds = 1/S keeps the solution finite in s
    return [0.5 * (b2 + c2 - a2) / S, 0.5 * (c2 + a2 - b2) / S, 0.5 * (a2 + b2 - c2 + 2 * a2 * b2) / S,
            b2 / S, 1.0 / S]


def run(q, delta_rel=1e-7):
    d = delta_rel * q
    Y0 = [np.log(q), np.log(d), np.log(q), d * d / (2 * q * q), 0.0]  # tau(0) ~ int_{-inf}^0 d^2 e^{2q^2 t} dt
    stop = lambda s, Y: (Y[0] + Y[1]) - np.log(1e9)  # a*b = 1e9
    stop.terminal = True
    sol = solve_ivp(rhs, [0, 1e9], Y0, events=stop, rtol=1e-11, atol=1e-13, dense_output=False, max_step=np.inf)
    x, y, tau = sol.y[0], sol.y[1], sol.y[3]
    r = np.exp(2 * (x - y))
    pred = 1.0 / (-np.expm1(-2 * tau))
    err = np.max(np.abs(r / pred - 1)[len(r) // 20:])  # skip the first 5% (initial-condition transient)
    L_direct = np.exp(x[-1] - y[-1])
    T = tau[-1]
    return L_direct, T, (1.0 / (-np.expm1(-2 * T))) ** 0.5, err, sol.y[4][-1]


def apparent_L(q, M, delta_rel=1e-7):
    """a/b where a run stopped at m = a^2 = M would read L off (too early: a/b is still decreasing)."""
    d = delta_rel * q
    Y0 = [np.log(q), np.log(d), np.log(q), d * d / (2 * q * q), 0.0]
    stop = lambda s, Y: Y[0] - 0.5 * np.log(M)
    stop.terminal = True
    sol = solve_ivp(rhs, [0, 1e9], Y0, events=stop, rtol=1e-11, atol=1e-13)
    return np.exp(sol.y[0][-1] - sol.y[1][-1])


def tube_radius(q, delta_rel=1e-7):
    """Return (R, L, T): R = int a b dt over the whole solution, alongside L and T."""
    d = delta_rel * q
    rhs6 = lambda s, Y: rhs(s, Y[:5]) + [np.exp(Y[0] + Y[1]) / (1.0 + np.exp(2 * Y[0]) + np.exp(2 * Y[1])
                                                                     + np.exp(2 * Y[2]) + 2 * np.exp(2 * Y[0] + 2 * Y[1]))]
    Y0 = [np.log(q), np.log(d), np.log(q), d * d / (2 * q * q), 0.0, d / q]  # int_{-inf}^0 q d e^{q^2 t} dt = d/q
    stop = lambda s, Y: (Y[0] + Y[1]) - np.log(1e9)
    stop.terminal = True
    sol = solve_ivp(rhs6, [0, 1e9], Y0, events=stop, rtol=1e-11, atol=1e-13)
    x, y, tau, R = sol.y[0][-1], sol.y[1][-1], sol.y[3][-1], sol.y[5][-1]
    return R, np.exp(x - y), tau


def certify(q, M, delta_rel=1e-7):
    """Integrate only until m = a^2 = M; return the interval for T certified by the remaining-time bounds."""
    d = delta_rel * q
    Y0 = [np.log(q), np.log(d), np.log(q), d * d / (2 * q * q), 0.0]
    stop = lambda s, Y: Y[0] - 0.5 * np.log(M)  # a^2 = M
    stop.terminal = True
    sol = solve_ivp(rhs, [0, 1e9], Y0, events=stop, rtol=1e-11, atol=1e-13)
    tau_M = sol.y[3][-1]
    return tau_M + np.log1p(1.5 / M), tau_M + np.log1p(2.0 / M)


if __name__ == "__main__":
    print(f"{'q':>6} {'L = a/b at end':>15} {'T = int b^2':>12} {'L from T':>10} {'max rel err r vs formula':>25} {'t at end':>9}")
    for q in (0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 3.0, 5.0, 7.5, 10.0):
        L, T, LT, err, tend = run(q)
        print(f"{q:6.2f} {L:15.6f} {T:12.6f} {LT:10.6f} {err:25.2e} {tend:9.3f}")
    print("\nCertified intervals for T from a run stopped at m = M > q^2 (theorem's remaining-time bounds);")
    print("reference: the interval from M = 1e8 (width 5e-9); 'nested' allows 1e-9 for integration error.")
    print(f"{'q':>6} {'M':>7} {'lower':>10} {'upper':>10} {'width':>9} {'nested':>7}")
    for q in (1.0, 10.0):
        ref_lo, ref_hi = certify(q, 1e8)
        for M in (1e1, 1e2, 1e3, 1e4):
            if M <= q * q:
                continue
            lo, hi = certify(q, M)
            print(f"{q:6.2f} {M:7.0f} {lo:10.6f} {hi:10.6f} {hi - lo:9.2e} {lo - 1e-9 <= ref_lo and ref_hi <= hi + 1e-9!s:>7}")
        print(f"{q:6.2f} {'1e8':>7} {ref_lo:10.7f} {ref_hi:10.7f} {ref_hi - ref_lo:9.2e}")
    print("\nThe pitfall: a/b read off at m = M, before blow-up, for q = 10 (true L = 5.460412):")
    for M in (1.01e2, 1.1e2, 2e2, 1e3, 1e4):
        print(f"  M = {M:8.0f}:  a/b = {apparent_L(10.0, M):12.4f}")
    print("\nTube radius R = int a b dt: L = coth R and T = ln cosh R (Remark 'Tubes'):")
    print(f"{'q':>6} {'R':>10} {'coth R':>10} {'L':>10} {'ln cosh R':>10} {'T':>10} {'1+3/(2q^2)<=cosh R<=1+2/q^2':>28}")
    for q in (0.5, 1.0, 2.0, 5.0, 10.0):
        R, L, T = tube_radius(q)
        ok = 1 + 1.5 / q**2 <= np.cosh(R) <= 1 + 2 / q**2
        print(f"{q:6.2f} {R:10.6f} {1/np.tanh(R):10.6f} {L:10.6f} {np.log(np.cosh(R)):10.6f} {T:10.6f} {ok!s:>28}")
