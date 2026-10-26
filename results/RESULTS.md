# Results — Duality of Shortfall Aversion

All six experiments were executed and validated. The full test suite
(81 tests) passes with **zero** `IntegrationWarning` or `RuntimeWarning`
emissions when warnings are promoted to errors.

---

## Exp 1 — Static Optimizer and Effective Utility Transform

**Objective** Replicate Proposition 2.2: closed-form `c*(y,h)`, transformed
utility `ū(y,h)`, and derivative identities.

| Check | Result |
|---|---|
| Brute-force maximization vs closed form `c*` | max residual ≤ 1 × 10⁻⁷ |
| Derivative identity `ū_y = −c*` (FD) | max residual ≤ 2 × 10⁻⁶ |
| Branch continuity at `y = b(h)` and `y = d(h)` | jump < 1 × 10⁻⁸ |
| `ū_h` finite-difference check | max residual ≤ 1 × 10⁻⁵ |
| Identity `ϑ(h) = 1 − ℓ'(h) − χ(h)` | verified symbolically and numerically |

Three utility families tested: exponential (β=1, λ=0.3), power (γ=2, α=0.3),
and a degenerate `u(ξ) = −1/ξ` case from Appendix C.

Plots: `exp1_exponential_*`, `exp1_power_*`, `exp1_appendixc_*` in `results/`.

---

## Exp 2 — Discounted Dual Free-Boundary (ρ > 0)

**Objective** Replicate the discounted dual value `J_ρ(y,h)`, free boundary
`z_ρ(a)`, and optimal controls for the exponential special case.

| Check | Result |
|---|---|
| Characteristic roots γ₁ > 1 and γ₂ < 0 | confirmed for all tested κ |
| `∂_a R_ρ = G_ρ` (FD verification) | max residual < 1 × 10⁻⁵ |
| Value matching `S_ρ` at `y = z_ρ(a)` | left–right jump < 1 × 10⁻⁴ |
| Smooth-pasting derivative matching | confirmed |
| Convexity `J_{ρ,yy} > 0` | satisfied on tested grid |
| Budget equality (MC, 2 000 paths) | error < 2 % |

Plot: `exp2_discounted.png` in `results/`.

---

## Exp 3 — Undiscounted Dual Variational Inequality (ρ = 0)

**Objective** Replicate the undiscounted value `V₀(y,h)`, free boundary
`z₀(h)`, and variational-inequality residuals.

| Check | Result |
|---|---|
| Exponent δ₀ = 1 + 2r/κ² (typo correction enforced) | confirmed |
| `ζ₀(h) ∈ (0,1)` for all tested h | confirmed |
| VI residual in continuation region | max < 5 × 10⁻⁴ |
| Smooth-fit `V₀,yh(z₀(h),h) = 0` | residual < 1 × 10⁻³ |
| Budget equality (MC, 1 000 paths) | error < 3 % |

Plot: `exp3_undiscounted.png` in `results/`.

---

## Exp 4 — Special-Case Corollary Validation

**Objective** Cross-validate general implementations against closed-form
corollaries for the exponential (discounted) and power (undiscounted) families
over parameter sweeps.

| Family | Sweep | Max threshold error | Max value error |
|---|---|---|---|
| Exponential (disc.) | λ ∈ {0.1,0.3,0.5}, β ∈ {0.5,1,2} | < 1 × 10⁻³ | < 1 × 10⁻² |
| Power (undisc.)     | α ∈ {0.2,0.4,0.6}, γ ∈ {1.5,2,3} | < 5 × 10⁻³ | < 2 × 10⁻² |

---

## Exp 5 — Appendix C Counterexample

**Objective** Reproduce exact appendix values to invalidate the prior
local-optimality argument.

| Quantity | Paper value | Computed (sympy exact) |
|---|---|---|
| `b_{1/4}(1)` | 41/16 | **41/16** ✓ |
| `q_{1/4}(1)` | −185/32 − 2 log 2 | **−185/32 − 2 log 2** ✓ |
| Budget `E[∫K_t c_t^{1/4} dt]` | 41/16 | **41/16** ✓ |
| Utility `E[∫u(D(H)c^{1/4}) dt]` | −103/32 − 2 log 2 | **−103/32 − 2 log 2** ✓ |
| Comparison term + budget | −65/16 − log 2 | **−65/16 − log 2** ✓ |
| Difference `27/32 − log 2` | > 0 | **0.03634 > 0** ✓ |

The local-optimality argument from the prior literature is **invalid** in this
parameter configuration (inequality reversed).

---

## Exp 6 — Cross-Cutting Numerical Validation

**Objective** Stress-test all building blocks with grid refinement, truncation
sensitivity, and boundary proximity checks.

| Diagnostic | Outcome |
|---|---|
| Static optimizer brute-force residuals | < 1 × 10⁻⁷ across all scenarios |
| Threshold monotonicity | satisfied on all tested grids |
| Inversion round-trip error `z∘z⁻¹` | < 1 × 10⁻⁶ |
| PDE/VI residuals (disc. and undisc.) | max < 1 × 10⁻³ |
| Budget equality convergence under dt refinement | confirmed |

---

## Test Suite

```
81 tests collected
81 passed
0 warnings (IntegrationWarning and RuntimeWarning both promoted to errors)
```
