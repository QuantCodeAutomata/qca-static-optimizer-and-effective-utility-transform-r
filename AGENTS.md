# Project: Duality of Shortfall Aversion

## Structure
- `src/primitives.py` — Abstract `ModelPrimitives` base class + three concrete families
- `src/static_block.py` — `StaticBlock`: Proposition 2.2 static optimisation
- `src/__init__.py` — empty package marker
- `tests/` — test directory
- `data/`, `exp/`, `results/` — data and experiment folders

## Key Math (Proposition 2.2)

Given primitives `(u, D, L)` and habit `h`, the static problem is:
```
ubar(y,h) = sup_{c in [ell(h), h]} { u(D(h)*c - L(h)) - y*c }
```

Notation:
- `ell(h) = L(h)/D(h)` — effective floor on consumption
- `g(h) = D(h)*h - L(h)` — effective surplus at ceiling
- `g'(h) = D'(h)*h + D(h) - L'(h)`
- `b(h) = D(h)*u'(g(h))` — lower y-threshold (below: c*=h)
- `d(h) = D(h)*m0` — upper y-threshold (above: c*=ell(h)); m0 = lim_{xi->0+} u'(xi)
- `Gamma(h) = g'(h)*u'(g(h))`, `vartheta(h) = g'(h)/D(h)`, `chi(h) = -(D'/D)*(h-ell)`
- `ell'(h) = (L'D - LD')/D^2`

Branch structure for `c*(y,h)`:
1. `y < b(h)`: `c* = h`
2. `b(h) <= y < d(h)`: `c* = ell(h) + I(y/D(h))/D(h)`  where I=(u')^{-1}
3. `y >= d(h)`: `c* = ell(h)`

Branch structure for `ubar_h(y,h)` (envelope theorem):
1. `Gamma(h) - y`
2. `(y/D(h)) * (D'(h)*c* - L'(h))`  — since u'(I(q))=q=y/D at interior optimum
3. `-y * ell'(h)`

Branch 3 ubar involves `u(0)`: finite for ExponentialFamily (`-1/beta`), `-inf` for Power/AppendixC.

## Model Families

| Family | u(xi) | D(h) | L(h) | m0 | I(q) |
|--------|-------|------|------|----|------|
| `ExponentialFamily(beta, lam)` | `-(1/beta)*exp(-beta*xi)` | 1 | `lam*h` | `beta` | `-(1/beta)*log(q/beta)` for `q<beta` |
| `PowerFamily(gamma, alpha)` | `xi^(1-gamma)/(1-gamma)`, `gamma>1` | `h^(-alpha)` | 0 | `inf` | `q^(-1/gamma)` |
| `AppendixCFamily()` | `-1/xi` | `h^(-0.5)` | 0 | `inf` | `q^(-0.5)` |

## Implementation Notes

- All core functions in `StaticBlock` vectorise over `y` with scalar `h`.
- `verify_assumption_21` uses sign-based tolerance (`u' > -tol`, `u'' < tol`) to handle
  numerically tiny but mathematically positive/negative values (e.g., `exp(-100)`).
- `numerical_cstar` checks both boundaries and interior via `minimize_scalar('bounded')`.
  Returns `(c, obj_val)` for scalar y (matching `diagnostics.py`); array for array y.
  Precision ≈ 1e-8 (scipy Brent's method floor: `sqrt(machine_eps)`).
- `check_branch_continuity` uses relative `tol=1e-6` perturbations around `b(h)` and `d(h)`.
- `d(h)=inf` for Power/AppendixC (m0=inf) → Branch 3 never occurs for those families.

## Dependencies
See `requirements.txt`: numpy>=1.26, scipy>=1.14, sympy, mpmath, pandas, matplotlib, seaborn, pytest.
