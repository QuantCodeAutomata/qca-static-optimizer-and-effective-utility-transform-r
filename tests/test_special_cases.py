"""Tests for special-case parameter sweeps."""
import numpy as np
import pytest
from src.primitives import ExponentialFamily, PowerFamily
from src.static_block import StaticBlock


def test_exponential_closed_form_various_h():
    """Closed-form matches numerical for multiple h values."""
    prim = ExponentialFamily(beta=1.0, lam=0.3)
    sb = StaticBlock(prim)
    for h in [0.5, 1.0, 2.0]:
        y_vals = np.logspace(-2, 0.5, 10)
        for y in y_vals:
            c_an = float(sb.cstar(y, h))
            c_num, _ = sb.numerical_cstar(y, h)
            assert abs(c_an - c_num) < 1e-5, f"h={h}, y={y:.3f}: {c_an} vs {c_num}"

def test_power_closed_form_various_h():
    prim = PowerFamily(gamma=2.0, alpha=0.3)
    sb = StaticBlock(prim)
    for h in [0.5, 1.0, 2.0]:
        y_vals = np.logspace(-1, 1, 8)
        for y in y_vals:
            c_an = float(sb.cstar(y, h))
            c_num, _ = sb.numerical_cstar(y, h)
            assert abs(c_an - c_num) < 1e-5

def test_power_L_is_zero():
    prim = PowerFamily(gamma=2.0, alpha=0.3)
    sb = StaticBlock(prim)
    h_vals = np.linspace(0.5, 3.0, 5)
    assert np.all(prim.L(h_vals) == 0)
    assert np.all(sb.ell(h_vals) == 0)

def test_exponential_D_is_one():
    prim = ExponentialFamily(beta=1.0, lam=0.3)
    h_vals = np.linspace(0.5, 3.0, 5)
    np.testing.assert_allclose(prim.D(h_vals), 1.0)

def test_exponential_m0_equals_beta():
    for beta in [0.5, 1.0, 2.0]:
        prim = ExponentialFamily(beta=beta, lam=0.3)
        assert prim.m0 == pytest.approx(beta)

def test_power_m0_infinite():
    for gamma in [1.5, 2.0, 3.0]:
        prim = PowerFamily(gamma=gamma, alpha=0.3)
        assert prim.m0 == np.inf

def test_discounted_roots_vary_with_rho():
    """Roots should change as rho changes."""
    from src.discounted_dual import DiscountedDual
    prim = ExponentialFamily(beta=1.0, lam=0.3)
    market = {'r': 0.05, 'mu': 0.10, 'sigma': 0.2}
    rhos = [0.03, 0.05, 0.07]
    g1s = []
    for rho in rhos:
        dd = DiscountedDual(prim, market, rho)
        g1, _ = dd.compute_roots()
        g1s.append(g1)
    # gamma1 should vary with rho (not all the same)
    assert not all(abs(g1s[i] - g1s[0]) < 1e-10 for i in range(1, len(g1s)))

def test_undiscounted_delta0_formula_multiple_markets():
    from src.undiscounted_dual import UndiscountedDual
    prim = PowerFamily(gamma=2.0, alpha=0.3)
    for r, mu, sigma in [(0.03, 0.08, 0.2), (0.05, 0.12, 0.25)]:
        market = {'r': r, 'mu': mu, 'sigma': sigma}
        ud = UndiscountedDual(prim, market)
        kappa = (mu - r) / sigma
        expected = 1 + 2*r/kappa**2
        assert ud.compute_delta0() == pytest.approx(expected, rel=1e-10)
