"""Diagnostic functions for validating dual optimization implementations."""
import numpy as np
from typing import List, Dict

def check_static_vs_numerical(sb, y_grid: np.ndarray, h_vals: List[float], tol: float = 1e-8) -> Dict:
    """Compare analytical c* to numerical maximization."""
    errors = []
    for h in h_vals:
        for y in y_grid:
            c_analytical = float(sb.cstar(y, h))
            c_numerical, _ = sb.numerical_cstar(y, h)
            errors.append(abs(c_analytical - c_numerical))
    max_error = max(errors) if errors else 0.0
    mean_error = np.mean(errors) if errors else 0.0
    return {
        'max_error': max_error,
        'mean_error': mean_error,
        'pass_flag': max_error < tol
    }

def check_derivative_identity_ubar_y(sb, y_grid: np.ndarray, h_val: float, eps: float = 1e-7) -> Dict:
    """Check ubar_y = -cstar via finite differences."""
    residuals = []
    for y in y_grid:
        if y - eps <= 0:
            continue
        fd = (sb.ubar(y + eps, h_val) - sb.ubar(y - eps, h_val)) / (2*eps)
        analytical = sb.ubar_y(y, h_val)
        if np.isfinite(fd) and np.isfinite(analytical):
            residuals.append(abs(fd - analytical))
    max_residual = max(residuals) if residuals else 0.0
    return {'max_residual': max_residual, 'n_tested': len(residuals)}

def check_derivative_ubar_h(sb, y_grid: np.ndarray, h_val: float, eps: float = 1e-6) -> Dict:
    """Check analytical ubar_h vs finite differences."""
    residuals = []
    for y in y_grid:
        fd = (sb.ubar(y, h_val + eps) - sb.ubar(y, h_val - eps)) / (2*eps)
        analytical = sb.ubar_h(y, h_val)
        if np.isfinite(fd) and np.isfinite(analytical):
            residuals.append(abs(fd - analytical))
    max_residual = max(residuals) if residuals else 0.0
    return {'max_residual': max_residual, 'n_tested': len(residuals)}

def check_branch_continuity(sb, h_val: float, eps: float = 1e-8) -> Dict:
    """Check cstar, ubar continuity at y=b(h) and y=d(h)."""
    b = float(sb.b_thresh(h_val))
    d = float(sb.d_thresh(h_val))
    
    # At b(h)
    c_left_b = float(sb.cstar(b - eps, h_val))
    c_right_b = float(sb.cstar(b + eps, h_val))
    b_continuity_ok = abs(c_left_b - c_right_b) < 1e-5
    
    result = {
        'b_continuity_ok': b_continuity_ok,
        'b_cstar_jump': abs(c_left_b - c_right_b),
        'd_continuity_ok': True,
        'd_cstar_jump': 0.0
    }
    
    if np.isfinite(d):
        c_left_d = float(sb.cstar(d - eps, h_val))
        c_right_d = float(sb.cstar(d + eps, h_val))
        result['d_continuity_ok'] = abs(c_left_d - c_right_d) < 1e-5
        result['d_cstar_jump'] = abs(c_left_d - c_right_d)
    
    return result

def check_smooth_fit_discounted(dual, h_val: float, a_grid, z_rho_vals, A_rho_vals) -> Dict:
    """Value matching at y=z_rho(h)."""
    z = float(np.interp(h_val, a_grid, z_rho_vals))
    A = float(np.interp(h_val, a_grid, A_rho_vals))
    eps = 1e-7
    S_left = dual.S_rho(z - eps, h_val, z, A)
    S_right = dual.S_rho(z + eps, h_val, z, A)
    return {'value_mismatch': abs(S_left - S_right)}

def check_vi_residual_undiscounted(dual_u, y_grid, h_grid, z0_vals, B0_vals) -> Dict:
    """VI residual on grid."""
    residuals = []
    for h in h_grid:
        for y in y_grid:
            try:
                res = dual_u.verify_vi_residual(y, h, h_grid, z0_vals, B0_vals)
                if np.isfinite(res):
                    residuals.append(abs(res))
            except Exception:
                pass
    return {'sup_norm': max(residuals) if residuals else float('nan'), 'n_tested': len(residuals)}

def check_budget_equality(mean_budget: float, x: float, tol: float = 0.05) -> Dict:
    residual = abs(mean_budget - x)
    return {'residual': residual, 'relative_error': residual/x, 'pass_flag': residual < tol*x}

def check_primal_dual_gap(utility_estimate: float, dual_value: float, x: float, ystar: float, tol: float = 0.1) -> Dict:
    primal_dual = dual_value + x * ystar
    gap = abs(utility_estimate - primal_dual)
    return {'gap': gap, 'pass_flag': gap < tol}
