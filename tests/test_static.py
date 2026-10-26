"""Tests for src/primitives.py and src/static_block.py."""
import numpy as np
import pytest
from src.primitives import ExponentialFamily, PowerFamily, AppendixCFamily
from src.static_block import StaticBlock


@pytest.fixture
def exp_family():
    return ExponentialFamily(beta=1.0, lam=0.3)

@pytest.fixture
def pow_family():
    return PowerFamily(gamma=2.0, alpha=0.3)

@pytest.fixture
def exp_sb(exp_family):
    return StaticBlock(exp_family)

@pytest.fixture
def pow_sb(pow_family):
    return StaticBlock(pow_family)


# --- ModelPrimitives: ExponentialFamily ---

def test_exponential_u_increasing(exp_family):
    """u' > 0 everywhere."""
    xi = np.array([0.1, 0.5, 1.0, 2.0])
    assert np.all(exp_family.u_prime(xi) > 0)

def test_exponential_u_concave(exp_family):
    """u'' < 0 everywhere."""
    xi = np.array([0.1, 0.5, 1.0, 2.0])
    assert np.all(exp_family.u_double_prime(xi) < 0)

def test_exponential_m0_finite(exp_family):
    assert np.isfinite(exp_family.m0)
    assert exp_family.m0 == pytest.approx(1.0)

def test_exponential_I_inverse(exp_family):
    """u'(I(q)) = q for 0 < q < m0."""
    q_vals = np.array([0.1, 0.3, 0.5, 0.9])
    for q in q_vals:
        z = exp_family.I(q)
        assert exp_family.u_prime(z) == pytest.approx(q, rel=1e-5)

def test_exponential_I_zero_extension(exp_family):
    """I(q)=0 for q >= m0."""
    assert exp_family.I(1.0) == pytest.approx(0.0, abs=1e-10)
    assert exp_family.I(2.0) == pytest.approx(0.0, abs=1e-10)

# --- ModelPrimitives: PowerFamily ---

def test_power_u_increasing(pow_family):
    xi = np.array([0.1, 0.5, 1.0, 2.0])
    assert np.all(pow_family.u_prime(xi) > 0)

def test_power_u_concave(pow_family):
    xi = np.array([0.1, 0.5, 1.0, 2.0])
    assert np.all(pow_family.u_double_prime(xi) < 0)

def test_power_m0_infinite(pow_family):
    assert pow_family.m0 == np.inf

def test_power_I_inverse(pow_family):
    """u'(I(q)) = q."""
    q_vals = np.array([0.5, 1.0, 2.0, 5.0])
    for q in q_vals:
        z = pow_family.I(q)
        assert pow_family.u_prime(z) == pytest.approx(q, rel=1e-5)

# --- AppendixCFamily ---

def test_appendixC_u_formula():
    prim = AppendixCFamily()
    xi = np.array([0.5, 1.0, 2.0])
    expected_u = -1.0 / xi
    np.testing.assert_allclose(prim.u(xi), expected_u)

def test_appendixC_u_prime():
    prim = AppendixCFamily()
    xi = np.array([0.5, 1.0, 2.0])
    expected = 1.0 / xi**2
    np.testing.assert_allclose(prim.u_prime(xi), expected)

def test_appendixC_I_inverse():
    prim = AppendixCFamily()
    q_vals = np.array([0.25, 1.0, 4.0])
    for q in q_vals:
        z = prim.I(q)
        assert prim.u_prime(z) == pytest.approx(q, rel=1e-6)

# --- StaticBlock ---

def test_ell_nonneg(exp_sb):
    h = np.array([0.5, 1.0, 2.0])
    assert np.all(exp_sb.ell(h) >= 0)

def test_ell_less_than_h(exp_sb):
    h = np.array([0.5, 1.0, 2.0])
    assert np.all(exp_sb.ell(h) < h)

def test_g_positive(exp_sb):
    h = np.array([0.5, 1.0, 2.0])
    assert np.all(exp_sb.g(h) > 0)

def test_b_thresh_positive(exp_sb):
    h = np.array([0.5, 1.0, 2.0])
    assert np.all(exp_sb.b_thresh(h) > 0)

def test_cstar_branch1_equals_h(exp_sb):
    """For y < b(h), c* = h."""
    h = 1.0
    b = float(exp_sb.b_thresh(h))
    y = b * 0.5
    assert float(exp_sb.cstar(y, h)) == pytest.approx(h, rel=1e-8)

def test_cstar_branch3_equals_ell_exponential(exp_sb):
    """For y >= d(h), c* = ell(h). Exponential family has finite d(h)."""
    h = 1.0
    d = float(exp_sb.d_thresh(h))
    if np.isfinite(d):
        y = d * 2.0
        assert float(exp_sb.cstar(y, h)) == pytest.approx(float(exp_sb.ell(h)), rel=1e-7)

def test_cstar_in_bounds_exponential(exp_sb):
    """c*(y,h) in [ell(h), h] for all y."""
    h = 1.0
    y_grid = np.logspace(-3, 2, 100)
    c = exp_sb.cstar(y_grid, h)
    ell_h = float(exp_sb.ell(h))
    assert np.all(c >= ell_h - 1e-9)
    assert np.all(c <= h + 1e-9)

def test_cstar_in_bounds_power(pow_sb):
    h = 1.0
    y_grid = np.logspace(-2, 2, 50)
    c = pow_sb.cstar(y_grid, h)
    ell_h = float(pow_sb.ell(h))
    assert np.all(c >= ell_h - 1e-9)
    assert np.all(c <= h + 1e-9)

def test_ubar_y_equals_neg_cstar(exp_sb):
    """ū_y = -c* identity (by construction)."""
    h = 1.0
    y_grid = np.logspace(-2, 1.5, 50)
    np.testing.assert_allclose(exp_sb.ubar_y(y_grid, h), -exp_sb.cstar(y_grid, h), rtol=1e-10)

def test_ubar_decreasing_in_y(exp_sb):
    """ū decreasing in y (ū_y = -c* < 0)."""
    h = 1.0
    y_grid = np.logspace(-2, 1, 50)
    ub = exp_sb.ubar(y_grid, h)
    fin = np.isfinite(ub)
    if fin.sum() > 1:
        assert np.all(np.diff(ub[fin]) <= 1e-10)

def test_cstar_continuous_at_b(exp_sb):
    """c* is continuous at y = b(h)."""
    h = 1.0
    b = float(exp_sb.b_thresh(h))
    eps = 1e-8
    left = float(exp_sb.cstar(b - eps, h))
    right = float(exp_sb.cstar(b + eps, h))
    assert abs(left - right) < 1e-5

def test_cstar_continuous_at_d_exponential(exp_sb):
    """c* continuous at y=d(h) for exponential (d finite)."""
    h = 1.0
    d = float(exp_sb.d_thresh(h))
    if np.isfinite(d):
        eps = 1e-8
        left = float(exp_sb.cstar(d - eps, h))
        right = float(exp_sb.cstar(d + eps, h))
        assert abs(left - right) < 1e-5

def test_closed_form_vs_numerical_exponential(exp_sb):
    """Closed-form c* matches numerical max within 1e-6."""
    h = 1.0
    y_test = np.logspace(-2, 0.9, 15)
    for y in y_test:
        c_an = float(exp_sb.cstar(y, h))
        c_num, _ = exp_sb.numerical_cstar(y, h)
        assert abs(c_an - c_num) < 1e-6, f"Mismatch at y={y:.4f}: analytical={c_an:.8f}, numerical={c_num:.8f}"

def test_closed_form_vs_numerical_power(pow_sb):
    h = 1.0
    y_test = np.logspace(-2, 0.9, 10)
    for y in y_test:
        c_an = float(pow_sb.cstar(y, h))
        c_num, _ = pow_sb.numerical_cstar(y, h)
        assert abs(c_an - c_num) < 1e-6

def test_assumption_21_exponential(exp_sb):
    h_grid = np.linspace(0.5, 3.0, 10)
    for h in h_grid:
        checks = exp_sb.verify_assumption_21(h)
        assert isinstance(checks, dict)

def test_assumption_21_power(pow_sb):
    h_grid = np.linspace(0.5, 3.0, 10)
    for h in h_grid:
        checks = pow_sb.verify_assumption_21(h)
        assert isinstance(checks, dict)

def test_vartheta_identity_exponential(exp_sb):
    """vartheta(h) = 1 - ell'(h) - chi(h)."""
    for h in [0.5, 1.0, 1.5, 2.0]:
        vt = exp_sb.vartheta(h)
        el_p = exp_sb.ell_prime(h)
        ch = exp_sb.chi(h)
        assert abs(vt - (1 - el_p - ch)) < 1e-7, f"Identity fails at h={h}"

def test_vartheta_identity_power(pow_sb):
    for h in [0.5, 1.0, 1.5, 2.0]:
        vt = pow_sb.vartheta(h)
        el_p = pow_sb.ell_prime(h)
        ch = pow_sb.chi(h)
        assert abs(vt - (1 - el_p - ch)) < 1e-7, f"Identity fails at h={h}"

def test_numerical_cstar_returns_tuple(exp_sb):
    c, obj = exp_sb.numerical_cstar(0.5, 1.0)
    assert isinstance(c, float)
    assert isinstance(obj, float)

def test_branch_continuity_check_runs(exp_sb):
    result = exp_sb.check_branch_continuity(1.0)
    assert isinstance(result, dict)
    assert 'b_continuity_ok' in result or 'cstar_continuous_at_b' in result
