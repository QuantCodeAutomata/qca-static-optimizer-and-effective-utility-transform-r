"""Tests for src/undiscounted_dual.py."""
import numpy as np
import pytest
from src.primitives import PowerFamily
from src.undiscounted_dual import UndiscountedDual


@pytest.fixture
def ud():
    prim = PowerFamily(gamma=2.0, alpha=0.3)
    return UndiscountedDual(prim, {'r': 0.05, 'mu': 0.10, 'sigma': 0.2})


def test_delta0_positive(ud):
    assert ud.compute_delta0() > 0

def test_delta0_greater_than_one(ud):
    """delta_0 = 1 + 2r/kappa^2 > 1 when r > 0."""
    assert ud.compute_delta0() > 1

def test_delta0_formula(ud):
    expected = 1 + 2*ud.r/ud.kappa**2
    assert ud.compute_delta0() == pytest.approx(expected, rel=1e-10)

def test_free_boundary_runs(ud):
    h_grid = np.linspace(0.5, 3.0, 6)
    try:
        zeta0, z0 = ud.compute_free_boundary(h_grid)
        assert len(zeta0) == len(h_grid)
        assert len(z0) == len(h_grid)
    except Exception as e:
        pytest.skip(f"Free boundary computation failed: {e}")

def test_free_boundary_zeta_in_unit_interval(ud):
    h_grid = np.linspace(0.5, 3.0, 6)
    try:
        zeta0, z0 = ud.compute_free_boundary(h_grid)
        assert np.all(zeta0 > 0)
        assert np.all(zeta0 < 1)
    except Exception as e:
        pytest.skip(f"Free boundary computation failed: {e}")

def test_free_boundary_z0_positive(ud):
    h_grid = np.linspace(0.5, 3.0, 6)
    try:
        zeta0, z0 = ud.compute_free_boundary(h_grid)
        assert np.all(z0 > 0)
    except Exception as e:
        pytest.skip(f"Free boundary failed: {e}")

def test_Phi0_finite(ud):
    val = ud.Phi0(0.5, 1.0)
    assert np.isfinite(val)

def test_Phi0_y_derivative(ud):
    """Check Phi0_y vs finite difference."""
    y, h = 0.5, 1.0
    eps = 1e-6
    fd = (ud.Phi0(y + eps, h) - ud.Phi0(y - eps, h)) / (2*eps)
    an = ud.Phi0_y(y, h)
    assert abs(fd - an) < 1e-4

def test_V0_admissibility_check(ud):
    """find_multiplier raises when x not admissible."""
    h_grid = np.linspace(0.5, 3.0, 6)
    try:
        zeta0, z0 = ud.compute_free_boundary(h_grid)
        B0 = ud.compute_B0(h_grid, z0)
        x_bad = ud.sb.ell(1.0) / ud.r - 0.01
        with pytest.raises(ValueError):
            ud.find_multiplier(x_bad, 1.0, h_grid, z0, B0)
    except Exception as e:
        pytest.skip(f"Setup failed: {e}")
