"""Tests for src/diagnostics.py and cross-cutting validation."""
import numpy as np
import pytest
from src.primitives import ExponentialFamily, PowerFamily
from src.static_block import StaticBlock
from src.diagnostics import (
    check_static_vs_numerical, check_derivative_identity_ubar_y,
    check_branch_continuity, check_derivative_ubar_h,
    check_budget_equality, check_primal_dual_gap
)
from src.simulation import simulate_Y_exact


@pytest.fixture
def exp_sb():
    return StaticBlock(ExponentialFamily(beta=1.0, lam=0.3))

@pytest.fixture
def pow_sb():
    return StaticBlock(PowerFamily(gamma=2.0, alpha=0.3))


def test_static_optimizer_exponential_passes(exp_sb):
    y_grid = np.logspace(-2, 0.9, 20)
    res = check_static_vs_numerical(exp_sb, y_grid, h_vals=[1.0, 1.5], tol=1e-6)
    assert res['pass_flag'], f"Max error: {res['max_error']:.2e}"

def test_static_optimizer_power_passes(pow_sb):
    y_grid = np.logspace(-2, 1, 20)
    res = check_static_vs_numerical(pow_sb, y_grid, h_vals=[1.0], tol=1e-6)
    assert res['pass_flag'], f"Max error: {res['max_error']:.2e}"

def test_derivative_identity_exponential(exp_sb):
    y_grid = np.logspace(-2, 1, 30)
    res = check_derivative_identity_ubar_y(exp_sb, y_grid, h_val=1.0)
    assert res['max_residual'] < 1e-5

def test_branch_continuity_exponential(exp_sb):
    res = check_branch_continuity(exp_sb, h_val=1.0)
    # Accept either key name variant
    b_ok = res.get('b_continuity_ok', res.get('cstar_continuous_at_b', False))
    assert b_ok

def test_branch_continuity_power(pow_sb):
    res = check_branch_continuity(pow_sb, h_val=1.0)
    b_ok = res.get('b_continuity_ok', res.get('cstar_continuous_at_b', False))
    assert b_ok

def test_ubar_h_fd_exponential(exp_sb):
    """ū_h matches FD away from branch points."""
    h_val = 1.0
    b_h = float(exp_sb.b_thresh(h_val))
    d_h = float(exp_sb.d_thresh(h_val))
    y_safe = np.array([y for y in np.logspace(-2, 1, 30)
                       if abs(y - b_h) > 0.05 and (not np.isfinite(d_h) or abs(y - d_h) > 0.05)])
    if len(y_safe) > 0:
        res = check_derivative_ubar_h(exp_sb, y_safe, h_val, eps=1e-5)
        assert res['max_residual'] < 0.1  # relaxed tolerance for FD

def test_budget_equality_check_pass():
    res = check_budget_equality(1.02, 1.0, tol=0.05)
    assert res['pass_flag']

def test_budget_equality_check_fail():
    res = check_budget_equality(1.5, 1.0, tol=0.05)
    assert not res['pass_flag']

def test_primal_dual_gap_check():
    res = check_primal_dual_gap(utility_estimate=-1.5, dual_value=-2.0, x=1.0, ystar=0.5, tol=0.1)
    # gap = |-1.5 - (-2.0 + 1.0*0.5)| = |-1.5 - (-1.5)| = 0
    assert res['pass_flag']

def test_simulate_Y_exact_shape():
    times, Y = simulate_Y_exact(y0=1.0, kappa=0.25, r=0.05, rho=0.05,
                                T=5.0, n_steps=50, n_paths=10, seed=0)
    assert Y.shape == (10, 51)
    assert len(times) == 51
    assert Y.shape[1] == len(times)

def test_simulate_Y_initial_value():
    y0 = 0.75
    _, Y = simulate_Y_exact(y0=y0, kappa=0.25, r=0.05, rho=0.05,
                            T=5.0, n_steps=20, n_paths=5, seed=1)
    np.testing.assert_allclose(Y[:, 0], y0)

def test_simulate_Y_positive():
    _, Y = simulate_Y_exact(y0=1.0, kappa=0.25, r=0.05, rho=0.05,
                            T=5.0, n_steps=50, n_paths=20, seed=2)
    assert np.all(Y > 0)
