"""Machine check of the elementary proof for thesis App. A.4 (Dancer–Strachan family II).

Written 2026-10-15 16:04 CEST (corrected from a guessed 16:08).  [Ours]
Variables: u=a^2, v=b^2, w=c^2; tau-time d(tau) = b^2 dt; r = a^2/b^2; m = a^2; s = (c^2 - a^2)/b^2.
Symbolic part checks:
  (1) dr/dtau = -2 r (r - 1)            (so r = 1/(1 - e^{-2 tau}))
  (2) dm/dtau = m (1 + s)
  (3) ds/dtau = 2 m (r + s) - 2 s (2 r - 1 + s)
  (4) sigma = 2s - m obeys  dsigma/dtau = m(1+s) - 2(2r-1+s) sigma
  (5) rho = m - s obeys     drho/dtau  = (2r-1+s) m - 2(2r-1+s) rho
  (6) d ln(ac)/dtau = 1 + a^2
Numeric part checks, along computed unstable curves:
  m/2 < s < m (i.e. 1 + b^2/2 < c^2/a^2 < 1 + b^2),  ln(1+3/(2q^2)) <= T(q) <= ln(1+2/q^2),  T decreasing in q.
Run: .venv/bin/python computations/ds_proof_check.py
"""
import numpy as np
import sympy as sp
from scipy.integrate import solve_ivp

from ds_unstable import rhs

a, b, c = sp.symbols("a b c", positive=True)
da = a * (b**2 + c**2 - a**2) / 2
db = b * (c**2 + a**2 - b**2) / 2
dc = c * (a**2 + b**2 - c**2 + 2 * a**2 * b**2) / 2


def ddtau(expr):  # d/dtau = (1/b^2) d/dt
    return sp.simplify((sp.diff(expr, a) * da + sp.diff(expr, b) * db + sp.diff(expr, c) * dc) / b**2)


r, m, s = a**2 / b**2, a**2, (c**2 - a**2) / b**2
checks = {
    "(1) dr/dtau + 2r(r-1)": ddtau(r) + 2 * r * (r - 1),
    "(2) dm/dtau - m(1+s)": ddtau(m) - m * (1 + s),
    "(3) ds/dtau - [2m(r+s) - 2s(2r-1+s)]": ddtau(s) - (2 * m * (r + s) - 2 * s * (2 * r - 1 + s)),
    "(4) dsigma/dtau - [m(1+s) - 2(2r-1+s)sigma]": ddtau(2 * s - m) - (m * (1 + s) - 2 * (2 * r - 1 + s) * (2 * s - m)),
    "(5) drho/dtau - [(2r-1+s)m - 2(2r-1+s)rho]": ddtau(m - s) - ((2 * r - 1 + s) * m - 2 * (2 * r - 1 + s) * (m - s)),
    "(6) dln(ac)/dtau - (1+a^2)": ddtau(sp.log(a * c)) - (1 + a**2),
}
# (7) at s = 2m/3 (i.e. c^2 = a^2 + 2a^2 b^2/3), d/dtau(2m - 3s) = 2m(r - 1) > 0, so s < 2m/3 persists
boundary7 = sp.simplify((ddtau(2 * m - 3 * s) - 2 * m * (r - 1)).subs(c, sp.sqrt(a**2 + 2 * a**2 * b**2 / 3)))
checks["(7) d(2m-3s)/dtau - 2m(r-1) on s=2m/3"] = boundary7

if __name__ == "__main__":
    print("Symbolic identities (each should simplify to 0):")
    for k, v in checks.items():
        print(f"  {k}: {sp.simplify(v)}")
    print("\nNumeric checks along unstable curves:")
    print(f"{'q':>6} {'min (s/m)':>10} {'max (s/m)':>10} {'ln(1+3/(2q^2))':>14} {'T':>10} {'ln(1+2/q^2)':>12}  bounds ok")
    Ts = []
    for q in (0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 3.0, 5.0, 7.5, 10.0):
        d = 1e-7 * q
        Y0 = [np.log(q), np.log(d), np.log(q), d * d / (2 * q * q), 0.0]
        ev = lambda _s, Y: (Y[0] + Y[1]) - np.log(1e9)
        ev.terminal = True
        sol = solve_ivp(rhs, [0, 1e9], Y0, events=ev, rtol=1e-11, atol=1e-13)
        A, B, Cc = np.exp(sol.y[0]), np.exp(sol.y[1]), np.exp(sol.y[2])
        ratio = ((Cc**2 - A**2) / B**2) / A**2  # s/m
        k0 = len(ratio) // 20  # skip the initial-condition transient
        T = sol.y[3][-1]
        lo, hi = np.log(1 + 1.5 / q**2), np.log(1 + 2 / q**2)
        Ts.append(T)
        print(f"{q:6.2f} {ratio[k0:].min():10.6f} {ratio[k0:].max():10.6f} {lo:14.6f} {T:10.6f} {hi:12.6f}  {lo <= T <= hi}")
    print("T strictly decreasing in q:", all(x > y for x, y in zip(Ts, Ts[1:])))
