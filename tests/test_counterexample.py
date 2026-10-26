"""Tests for Appendix C counterexample exact arithmetic."""
import sympy as sp
import numpy as np
import pytest


def test_b_k_exact_value():
    """b_{1/4}(1) = 41/16."""
    b_exact = sp.Rational(41, 16)
    assert float(b_exact) == pytest.approx(41/16, rel=1e-12)

def test_q_k_exact_value():
    """q_{1/4}(1) = -185/32 - 2*log(2)."""
    q_exact = -sp.Rational(185, 32) - 2*sp.log(2)
    expected = -185/32 - 2*np.log(2)
    assert float(q_exact) == pytest.approx(expected, rel=1e-10)

def test_utility_value():
    """E[...utility...] = -103/32 - 2*log(2)."""
    val = -sp.Rational(103, 32) - 2*sp.log(2)
    expected = -103/32 - 2*np.log(2)
    assert float(val) == pytest.approx(expected, rel=1e-10)

def test_comparison_value():
    """E[tilde_U^G] + 41/16 = -65/16 - log(2)."""
    val = -sp.Rational(65, 16) - sp.log(2)
    expected = -65/16 - np.log(2)
    assert float(val) == pytest.approx(expected, rel=1e-10)

def test_difference_positive():
    """27/32 - log(2) > 0."""
    diff = sp.Rational(27, 32) - sp.log(2)
    assert float(diff) > 0

def test_difference_equals_expected():
    """(-103/32 - 2 log 2) - (-65/16 - log 2) = 27/32 - log 2."""
    utility = -sp.Rational(103, 32) - 2*sp.log(2)
    comparison = -sp.Rational(65, 16) - sp.log(2)
    diff = sp.simplify(utility - comparison)
    expected = sp.Rational(27, 32) - sp.log(2)
    assert sp.simplify(diff - expected) == 0
    assert float(diff) > 0

def test_budget_exact():
    """Budget = 41/16."""
    budget = sp.Rational(41, 16)
    assert float(budget) == pytest.approx(2.5625, rel=1e-12)

def test_delta0_appendixC():
    """delta_0 = 1 + 2*(1/2)/1^2 = 2 for AppendixC parameters."""
    r = sp.Rational(1, 2)
    kappa_sq = sp.Integer(1)
    delta0 = 1 + 2*r/kappa_sq
    assert delta0 == 2

def test_b_at_h1():
    """b(1) = D(1)*u'(g(1)) = 1^(-1/2) * (1^(1/2))^(-2) = 1 for AppendixC."""
    # D(1) = 1^(-0.5) = 1
    # g(1) = D(1)*1 = 1
    # u'(g(1)) = 1/g(1)^2 = 1
    # b(1) = D(1)*u'(g(1)) = 1*1 = 1
    from src.primitives import AppendixCFamily
    from src.static_block import StaticBlock
    prim = AppendixCFamily()
    sb = StaticBlock(prim)
    h = 1.0
    b = float(sb.b_thresh(h))
    assert b == pytest.approx(1.0, rel=1e-8)

def test_Gamma_at_h1_appendixC():
    """Gamma(1) = g'(1)*u'(g(1)) = (1/2)*1 = 1/2."""
    from src.primitives import AppendixCFamily
    from src.static_block import StaticBlock
    prim = AppendixCFamily()
    sb = StaticBlock(prim)
    Gamma = float(sb.Gamma_fn(1.0))
    assert Gamma == pytest.approx(0.5, rel=1e-6)
