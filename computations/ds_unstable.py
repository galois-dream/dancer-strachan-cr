"""Dancer–Strachan Kähler–Einstein system (Castro thesis, App. A.3–A.4): numerical check of an exact identity.

Written 2026-10-15 15:57 CEST (corrected from a guessed 16:00).  [Lit] system (A.3.4) with Lambda = -1:
    a' = a(b^2 + c^2 - a^2)/2,   b' = b(c^2 + a^2 - b^2)/2,   c' = c(a^2 + b^2 - c^2 + 2 a^2 b^2)/2.
Family (II): the unstable curve of the critical point (q, 0, q), q > 0, which blows up at a finite time t*.
The thesis defines L(q) = lim a/b at t*, and the CR parameter at infinity is a_CR = L^(-2).
[Ours] Claim: with tau(t) = int_{-inf}^t b^2, the ratio r = a^2/b^2 obeys dr/dtau = -2 r (r - 1) exactly, so
    r(t) = 1 / (1 - exp(-2 tau(t)))   and   a_CR = L^(-2) = 1 - exp(-2 T),  T = int_{-inf}^{t*} b^2 dt.
This script integrates the unstable curve (in log variables, with time slowed near blow-up) and checks the
identity along the whole trajectory, then reports L(q) and T(q).
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


if __name__ == "__main__":
    print(f"{'q':>6} {'L = a/b at end':>15} {'T = int b^2':>12} {'L from T':>10} {'max rel err r vs formula':>25} {'t at end':>9}")
    for q in (0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 3.0, 5.0, 7.5, 10.0):
        L, T, LT, err, tend = run(q)
        print(f"{q:6.2f} {L:15.6f} {T:12.6f} {LT:10.6f} {err:25.2e} {tend:9.3f}")
