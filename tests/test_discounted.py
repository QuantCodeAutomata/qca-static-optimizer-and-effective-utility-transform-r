"""Tests for src/discounted_dual.py."""
import numpy as np
import pytest
from src.primitives import ExponentialFamily
from src.discounted_dual import DiscountedDual


@pytest.fixture
def dd():
    prim = ExponentialFamily(beta=1.0, lam=0.3)
    market = {'r': 0.05, 'mu': 0.10, 'sigma': 0.2}
    return DiscountedDual(prim, market, rho=0.05)


def test_roots_signs(dd):
    g1, g2 = dd.gamma1, dd.gamma2
    assert g1 > 1, f"gamma1={g1} should be > 1"
    assert g2 < 0, f"gamma2={g2} should be < 0"

def test_roots_satisfy_quadratic(dd):
    """Both roots satisfy (kappa^2/2)*g*(g-1) + (rho-r)*g - rho = 0."""
    k2 = dd.kappa**2
    def quad(g): return (k2/2)*g*(g-1) + (dd.rho-dd.r)*g - dd.rho
    assert abs(quad(dd.gamma1)) < 1e-10
    assert abs(quad(dd.gamma2)) < 1e-10

def test_kappa_positive(dd):
    assert dd.kappa > 0

def test_free_boundary_computed(dd):
    a_grid = np.linspace(0.5, 3.0, 6)
    z_vals = dd.compute_free_boundary(a_grid)
    assert len(z_vals) == len(a_grid)
    assert np.all(z_vals > 0)

def test_free_boundary_less_than_Gamma(dd):
    a_grid = np.linspace(0.5, 3.0, 6)
    z_vals = dd.compute_free_boundary(a_grid)
    Gamma = np.array([dd.sb.Gamma_fn(a) for a in a_grid])
    assert np.all(z_vals < Gamma + 1e-8)

def test_A_rho_computes(dd):
    a_grid = np.linspace(0.5, 3.0, 4)
    z_vals = dd.compute_free_boundary(a_grid)
    for a, z in zip(a_grid[:2], z_vals[:2]):
        A = dd.A_rho(a, z)
        assert np.isfinite(A)

def test_S_rho_value_matching(dd):
    """S_rho is continuous at z_rho."""
    a_grid = np.linspace(0.5, 3.0, 4)
    z_vals = dd.compute_free_boundary(a_grid)
    A_vals = np.array([dd.A_rho(a, z) for a, z in zip(a_grid, z_vals)])
    for a, z, A in zip(a_grid[:3], z_vals[:3], A_vals[:3]):
        eps = 1e-7
        s_l = dd.S_rho(z - eps, a, z, A)
        s_r = dd.S_rho(z + eps, a, z, A)
        assert abs(s_l - s_r) < 1e-4, f"Value mismatch at a={a:.2f}: jump={abs(s_l-s_r):.2e}"

def test_J_rho_computes(dd):
    a_grid = np.linspace(0.5, 3.0, 6)
    z_vals = dd.compute_free_boundary(a_grid)
    A_vals = np.array([dd.A_rho(a, z) for a, z in zip(a_grid, z_vals)])
    J = dd.J_rho(0.5, 1.0, a_grid, z_vals, A_vals)
    assert np.isfinite(J)

def test_J_rho_convex_in_y(dd):
    """J_rho should be convex in y (J_yy > 0)."""
    a_grid = np.linspace(0.5, 3.0, 6)
    z_vals = dd.compute_free_boundary(a_grid)
    A_vals = np.array([dd.A_rho(a, z) for a, z in zip(a_grid, z_vals)])
    Jyy = dd.J_rho_yy(0.5, 1.0, a_grid, z_vals, A_vals)
    assert Jyy > 0, f"Not convex: J_yy={Jyy:.4f}"

def test_find_multiplier_inadmissible(dd):
    """Should raise ValueError if x <= ell(h)/r."""
    a_grid = np.linspace(0.5, 3.0, 4)
    z_vals = dd.compute_free_boundary(a_grid)
    A_vals = np.array([dd.A_rho(a, z) for a, z in zip(a_grid, z_vals)])
    h = 1.0
    x_inadmissible = dd.sb.ell(h) / dd.r - 0.01
    with pytest.raises(ValueError):
        dd.find_multiplier(x_inadmissible, h, a_grid, z_vals, A_vals)
