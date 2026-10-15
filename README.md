# dancer-strachan-cr

Verification code for the note *Every CR structure is reached: an elementary proof* (draft, October 2026).

Written 2026-10-15 16:18 CEST.

## Context

Dancer and Strachan studied four-dimensional Kähler–Einstein metrics that are symmetric under SU(2).
The symmetry reduces the Einstein equations (with the Einstein constant normalized to −1) to three ordinary
differential equations:

    a' = a (b² + c² − a²) / 2
    b' = b (c² + a² − b²) / 2
    c' = c (a² + b² − c² + 2a²b²) / 2

For each q > 0, one solution leaves the equilibrium (q, 0, q) with b > 0 and runs until it blows up. These
solutions ("family II") give complete metrics. Far out, each metric looks like a squashed three-sphere. The
squashing is L(q) = lim a/b, and the boundary at infinity carries the CR structure with parameter L⁻².

A 2010 PhD thesis (A. L. Castro, *Chains and Monsters*, UC Santa Cruz, Appendix A.4) predicted from numerical
experiments that L takes every value in [1, ∞), so that every such CR structure occurs. The note proves it,
using only separable equations and the intermediate value theorem, plus two standard facts about ODEs:

    L(q)⁻² = 1 − exp(−2 T(q)),   T(q) = ∫ b² dt over the whole solution,
    ln(1 + 3/(2q²)) ≤ T(q) ≤ ln(1 + 2/q²),

and T is continuous and non-increasing in q.

## Scripts

| script | what it checks |
|---|---|
| `computations/ds_proof_check.py` | the seven identities used in the proof (symbolically, each must simplify to 0); along computed solutions, the inequalities 1/2 < s/m < 2/3, the bounds on T(q), and that T decreases in q |
| `computations/ds_unstable.py` | integrates the family II solutions for q from 0.05 to 10; checks r = 1/(1 − e^(−2τ)) along each one (to about 1e-8); reports L(q) and T(q) |

## Running

```
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python computations/ds_proof_check.py
.venv/bin/python computations/ds_unstable.py
```

Python 3.10 or later. Each script's docstring says what it checks and labels its sources: [Lit] for the
literature, [Ours] for new arguments.
