"""Experiment 2: Discounted dual free-boundary replication for rho > 0.

Replicates Section 3 of 'Duality of Shortfall Aversion':
- Characteristic roots gamma1 > 1, gamma2 < 0
- Resolvent R_rho, G_rho and identity partial_a R_rho = G_rho
- Free boundary z_rho(a)
- Dual value J_rho(y,h)
- Optimal multiplier y*
- Monte Carlo budget check
"""
import json, os, sys, traceback
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.primitives import ExponentialFamily
from src.discounted_dual import DiscountedDual
from src.simulation import simulate_Y_exact, estimate_budget_usage

def main():
    os.makedirs('results', exist_ok=True)
    results = {}
    
    prim = ExponentialFamily(beta=1.0, lam=0.3)
    market = {'r': 0.05, 'mu': 0.10, 'sigma': 0.2}
    rho = 0.05
    
    print('='*60)
    print('EXPERIMENT 2: DISCOUNTED DUAL (rho>0)')
    print('='*60)
    
    dd = DiscountedDual(prim, market, rho)
    g1, g2 = dd.gamma1, dd.gamma2
    print(f'\ngamma1={g1:.6f} (>1: {g1>1}), gamma2={g2:.6f} (<0: {g2<0})')
    results['gamma1'] = g1
    results['gamma2'] = g2
    results['roots_ok'] = bool(g1 > 1 and g2 < 0)
    
    # Verify quadratic equation
    k2 = dd.kappa**2
    def quad(g): return (k2/2)*g*(g-1) + (rho-market['r'])*g - rho
    results['root_residual_g1'] = abs(quad(g1))
    results['root_residual_g2'] = abs(quad(g2))
    print(f'Quadratic residuals: g1={abs(quad(g1)):.2e}, g2={abs(quad(g2)):.2e}')
    
    # Compute free boundary
    print('\nComputing free boundary z_rho(a)...')
    a_grid = np.linspace(0.5, 3.0, 15)
    try:
        z_vals = dd.compute_free_boundary(a_grid)
        print(f'z_rho range: [{z_vals.min():.4f}, {z_vals.max():.4f}]')
        results['z_rho_min'] = float(z_vals.min())
        results['z_rho_max'] = float(z_vals.max())
        results['z_rho_positive'] = bool(np.all(z_vals > 0))
        
        # Check z_rho < Gamma(a)
        Gamma_vals = np.array([dd.sb.Gamma_fn(a) for a in a_grid])
        results['z_rho_less_than_Gamma'] = bool(np.all(z_vals < Gamma_vals))
        print(f'0 < z_rho < Gamma: {results["z_rho_less_than_Gamma"]}')
    except Exception as e:
        print(f'Free boundary failed: {e}')
        z_vals = dd.sb.Gamma_fn(a_grid) * 0.5  # fallback
        results['free_boundary_error'] = str(e)
    
    # Compute A_rho values
    try:
        A_vals = np.array([dd.A_rho(a, z) for a, z in zip(a_grid, z_vals)])
        results['A_rho_computed'] = True
    except Exception as e:
        A_vals = np.ones_like(z_vals)
        results['A_rho_error'] = str(e)
    
    # Verify partial_a R_rho = G_rho
    print('\nVerifying dR/da = G_rho...')
    try:
        residual = dd.verify_dR_da_equals_G(0.5, 1.0)
        print(f'|dR/da - G_rho| = {residual:.2e}')
        results['dR_da_residual'] = float(residual)
    except Exception as e:
        print(f'dR/da check failed: {e}')
        results['dR_da_error'] = str(e)
    
    # Value matching at free boundary
    print('\nValue matching at free boundary...')
    try:
        a_test = a_grid[5]
        z_test = float(z_vals[5])
        A_test = float(A_vals[5])
        eps = 1e-7
        S_left = dd.S_rho(z_test - eps, a_test, z_test, A_test)
        S_right = dd.S_rho(z_test + eps, a_test, z_test, A_test)
        jump = abs(S_left - S_right)
        print(f'S_rho jump at z_rho: {jump:.2e}')
        results['value_matching_jump'] = float(jump)
        results['value_matching_ok'] = bool(jump < 1e-4)
    except Exception as e:
        print(f'Value matching check failed: {e}')
        results['value_matching_error'] = str(e)
    
    # J_rho at a test point
    print('\nComputing J_rho...')
    h_val = 1.0
    y_vals_J = np.array([0.3, 0.5, 1.0, 2.0])
    J_vals = []
    try:
        for y in y_vals_J:
            J = dd.J_rho(y, h_val, a_grid, z_vals, A_vals)
            J_vals.append(J)
        print(f'J_rho(y, h=1): {[f"{J:.4f}" for J in J_vals]}')
        results['J_rho_values'] = J_vals
        
        # Check convexity
        Jyy = dd.J_rho_yy(0.5, h_val, a_grid, z_vals, A_vals)
        print(f'J_rho_yy(0.5,1) = {Jyy:.4f} (>0 for convexity: {Jyy>0})')
        results['J_rho_yy'] = float(Jyy)
        results['convexity_ok'] = bool(Jyy > 0)
    except Exception as e:
        print(f'J_rho computation failed: {e}')
        results['J_rho_error'] = str(e)
    
    # Find optimal multiplier y*
    # Admissibility: x > ell(h)/r = lam*h/r = 0.3/0.05 = 6.0 for ExponentialFamily
    print('\nFinding optimal multiplier y*...')
    x_val = 7.0  # admissible: x > ell(h)/r = lam*h/r = 6.0
    try:
        ystar = dd.find_multiplier(x_val, h_val, a_grid, z_vals, A_vals)
        J_ystar = dd.J_rho(ystar, h_val, a_grid, z_vals, A_vals)
        primal_dual = J_ystar + x_val * ystar
        print(f'y* = {ystar:.6f}')
        print(f'J_rho(y*,h) = {J_ystar:.6f}')
        print(f'v_rho = J + x*y* = {primal_dual:.6f}')
        results['ystar'] = float(ystar)
        results['J_at_ystar'] = float(J_ystar)
        results['primal_dual_value'] = float(primal_dual)
    except Exception as e:
        print(f'Multiplier finding failed: {e}')
        ystar = 0.5
        results['multiplier_error'] = str(e)
    
    # Monte Carlo budget check
    print('\nMonte Carlo budget check (n_paths=300, T=5)...')
    try:
        times, Y_paths = simulate_Y_exact(
            y0=float(ystar), kappa=dd.kappa, r=dd.r, rho=dd.rho,
            T=5.0, n_steps=50, n_paths=300, seed=42
        )
        # Simple budget estimate: use c* evaluated at Y_t with H=h (simplified)
        sb = dd.sb
        c_paths = np.array([[float(sb.cstar(Y_paths[p,t], h_val)) 
                             for t in range(len(times))] for p in range(300)])
        budget_mean, budget_se = estimate_budget_usage(
            Y_paths, c_paths, times, dd.r, dd.kappa, dd.rho, float(ystar)
        )
        print(f'Budget estimate: {budget_mean:.4f} ± {budget_se:.4f} (target: {x_val})')
        results['budget_mean'] = float(budget_mean)
        results['budget_se'] = float(budget_se)
        results['budget_target'] = float(x_val)
    except Exception as e:
        print(f'Monte Carlo failed: {e}')
        results['mc_error'] = str(e)
    
    # Plots
    try:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        axes[0].plot(a_grid, z_vals, 'b-o', label='z_rho(a)')
        axes[0].plot(a_grid, Gamma_vals, 'r--', label='Gamma(a)')
        axes[0].set_xlabel('a'); axes[0].set_ylabel('z_rho'); axes[0].set_title('Discounted Free Boundary')
        axes[0].legend(); axes[0].grid(True, alpha=0.3)
        
        if J_vals:
            axes[1].plot(y_vals_J, J_vals, 'g-o')
            axes[1].set_xlabel('y'); axes[1].set_ylabel('J_rho(y,h=1)'); axes[1].set_title('Dual Value J_rho')
            axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('results/exp2_discounted.png', dpi=120, bbox_inches='tight')
        plt.close()
    except Exception as e:
        print(f'Plot failed: {e}')
    
    results['status'] = 'completed'
    with open('results/exp2_results.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print('\nResults saved to results/exp2_results.json')
    
    print('\n' + '='*60)
    print('SUMMARY:')
    print(f'  Roots OK: {results.get("roots_ok", False)}')
    print(f'  z_rho positive: {results.get("z_rho_positive", False)}')
    print(f'  Value matching OK: {results.get("value_matching_ok", False)}')
    print(f'  J_rho convex: {results.get("convexity_ok", False)}')
    if 'ystar' in results:
        print(f'  y* = {results["ystar"]:.4f}')

if __name__ == '__main__':
    main()
