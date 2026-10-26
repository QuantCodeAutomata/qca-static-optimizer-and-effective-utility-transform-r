"""Experiment 3: Undiscounted dual variational-inequality replication (rho=0).

Replicates Section 4 of 'Duality of Shortfall Aversion':
- Exponent delta_0 = 1 + 2r/kappa^2
- Free boundary z_0(h)
- Phi_0, B_0, V_0
- Variational inequality residual
- Smooth-fit condition
- Optimal multiplier
"""
import json, os, sys, traceback
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.primitives import PowerFamily
from src.undiscounted_dual import UndiscountedDual
from src.simulation import simulate_Y_exact, estimate_budget_usage

def main():
    os.makedirs('results', exist_ok=True)
    results = {}
    
    prim = PowerFamily(gamma=2.0, alpha=0.3)
    market = {'r': 0.05, 'mu': 0.10, 'sigma': 0.2}
    
    print('='*60)
    print('EXPERIMENT 3: UNDISCOUNTED DUAL (rho=0)')
    print('='*60)
    
    ud = UndiscountedDual(prim, market)
    delta0 = ud.compute_delta0()
    expected_delta0 = 1 + 2*ud.r/ud.kappa**2
    print(f'\ndelta_0 = {delta0:.6f} (expected {expected_delta0:.6f})')
    print('Note: delta_0 = 1 + 2r/kappa^2 corrects typo in Section 4 (matches Appendix B)')
    results['delta0'] = float(delta0)
    results['delta0_formula_match'] = abs(delta0 - expected_delta0) < 1e-10
    
    # Compute free boundary
    print('\nComputing free boundary z_0(h)...')
    h_grid = np.linspace(0.5, 3.0, 12)
    try:
        zeta0_vals, z0_vals = ud.compute_free_boundary(h_grid)
        print(f'zeta_0 range: [{zeta0_vals.min():.4f}, {zeta0_vals.max():.4f}]')
        print(f'z_0 range: [{z0_vals.min():.4f}, {z0_vals.max():.4f}]')
        zeta_in_01 = bool(np.all(zeta0_vals > 0) and np.all(zeta0_vals < 1))
        print(f'zeta_0 in (0,1): {zeta_in_01}')
        results['zeta0_min'] = float(zeta0_vals.min())
        results['zeta0_max'] = float(zeta0_vals.max())
        results['zeta0_in_01'] = zeta_in_01
        results['z0_positive'] = bool(np.all(z0_vals > 0))
    except Exception as e:
        print(f'Free boundary failed: {e}')
        traceback.print_exc()
        zeta0_vals = np.full(len(h_grid), 0.5)
        z0_vals = np.array([ud.sb.Gamma_fn(h)*0.5 for h in h_grid])
        results['free_boundary_error'] = str(e)
    
    # Compute Phi0 and B0
    print('\nComputing Phi0 and B0...')
    try:
        B0_vals = ud.compute_B0(h_grid, z0_vals)
        print(f'B0 range: [{B0_vals.min():.4f}, {B0_vals.max():.4f}]')
        results['B0_computed'] = True
        results['B0_min'] = float(B0_vals.min())
    except Exception as e:
        print(f'B0 computation failed: {e}')
        B0_vals = np.zeros_like(h_grid)
        results['B0_error'] = str(e)
    
    # Test Phi0_y derivative
    print('\nVerifying Phi0_y via finite difference...')
    h_test = 1.0
    y_test = 0.5
    try:
        eps = 1e-6
        phi_up = ud.Phi0(y_test + eps, h_test)
        phi_dn = ud.Phi0(y_test - eps, h_test)
        fd = (phi_up - phi_dn) / (2*eps)
        analytical = ud.Phi0_y(y_test, h_test)
        residual = abs(fd - analytical)
        print(f'|Phi0_y - FD| = {residual:.2e}')
        results['Phi0_y_fd_residual'] = float(residual)
    except Exception as e:
        print(f'Phi0_y check failed: {e}')
        results['Phi0_y_error'] = str(e)
    
    # Evaluate V0
    print('\nComputing V0...')
    y_test_vals = np.array([0.1, 0.3, 0.5, 1.0])
    V0_vals_test = []
    try:
        for y in y_test_vals:
            V = ud.V0(y, h_test, h_grid, z0_vals, B0_vals)
            V0_vals_test.append(V)
        print(f'V0(y, h=1): {[f"{V:.4f}" for V in V0_vals_test]}')
        results['V0_values'] = V0_vals_test
    except Exception as e:
        print(f'V0 computation failed: {e}')
        results['V0_error'] = str(e)
    
    # Check convexity at a y in the CONTINUATION region y > z0(h)
    print('\nChecking V0 convexity in y...')
    try:
        # z0 at h=h_test (interpolated)
        z0_at_h = float(np.interp(h_test, h_grid, z0_vals))
        # Pick y well inside continuation region
        y_cont = z0_at_h * 2.0
        Vyy = ud.V0_yy(y_cont, h_test, h_grid, z0_vals, B0_vals)
        print(f'z0(h={h_test}) = {z0_at_h:.4f};  testing y={y_cont:.4f} (continuation region)')
        print(f'V0_yy({y_cont:.2f},{h_test}) = {Vyy:.6f} (>0: {Vyy>0})')
        results['V0_yy'] = float(Vyy)
        results['convexity_ok'] = bool(Vyy > 0)
    except Exception as e:
        print(f'Convexity check failed: {e}')
        results['V0_yy_error'] = str(e)

    # VI residual check — also in continuation region
    print('\nChecking variational inequality residual...')
    try:
        z0_at_h = float(np.interp(h_test, h_grid, z0_vals))
        y_vi = z0_at_h * 2.0   # continuation region
        h_vi = h_test
        vi_res = ud.verify_vi_residual(y_vi, h_vi, h_grid, z0_vals, B0_vals)
        print(f'VI residual at (y={y_vi:.3f},h={h_vi}): {vi_res:.4e}')
        results['vi_residual'] = float(vi_res)
    except Exception as e:
        print(f'VI residual failed: {e}')
        results['vi_error'] = str(e)
    
    # Find optimal multiplier
    print('\nFinding optimal multiplier y*...')
    x_val = 1.0
    h_val = 1.0
    try:
        ystar = ud.find_multiplier(x_val, h_val, h_grid, z0_vals, B0_vals)
        V_ystar = ud.V0(ystar, h_val, h_grid, z0_vals, B0_vals)
        primal_dual = V_ystar + x_val * ystar
        print(f'y* = {ystar:.6f}')
        print(f'v_0 = V0 + x*y* = {primal_dual:.6f}')
        results['ystar'] = float(ystar)
        results['V0_at_ystar'] = float(V_ystar)
        results['primal_dual_value'] = float(primal_dual)

        # Analytical budget verification (exact by FOC construction):
        # -Phi0_y(y*, h) = x by definition of y*
        analytical_budget = -ud.Phi0_y(float(ystar), h_val)
        budget_err = abs(analytical_budget - x_val)
        print(f'Analytical budget check: -Phi0_y(y*,h) = {analytical_budget:.6f} (target {x_val}, err={budget_err:.2e})')
        results['analytical_budget'] = float(analytical_budget)
        results['analytical_budget_err'] = float(budget_err)
    except Exception as e:
        print(f'Multiplier finding failed: {e}')
        import traceback as _tb; _tb.print_exc()
        ystar = 1.0
        results['multiplier_error'] = str(e)

    # Monte Carlo — undiscounted case needs long horizon; use T=100, dt=0.2
    # At this T, E[int_0^T K_t c_t dt] / x ≈ 1 - exp(-beta*T) where
    # beta = (1-1/gamma)*(r + kappa^2/(2*gamma)).  Budget converges to x=1.0 as T→∞.
    print('\nMonte Carlo (n_paths=200, T=100, n_steps=500)...')
    try:
        T_mc = 100.0
        times, Y_paths = simulate_Y_exact(
            y0=float(ystar), kappa=ud.kappa, r=ud.r, rho=0.0,
            T=T_mc, n_steps=500, n_paths=200, seed=42
        )
        sb = ud.sb
        c_paths = np.array([[float(sb.cstar(Y_paths[p, t], h_val))
                             for t in range(len(times))] for p in range(200)])
        budget_mean, budget_se = estimate_budget_usage(
            Y_paths, c_paths, times, ud.r, ud.kappa, 0.0, float(ystar)
        )
        print(f'Budget estimate (T={T_mc}): {budget_mean:.4f} ± {budget_se:.4f} (target: {x_val})')
        print(f'Note: undiscounted budget converges to x only as T→∞; T={T_mc} captures ~(1-e^(-β·T)) fraction.')
        results['budget_mean'] = float(budget_mean)
        results['budget_se'] = float(budget_se)
        results['budget_T'] = T_mc
    except Exception as e:
        print(f'Monte Carlo failed: {e}')
        results['mc_error'] = str(e)
    
    # Plots
    try:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        Gamma_h = np.array([ud.sb.Gamma_fn(h) for h in h_grid])
        axes[0].plot(h_grid, z0_vals, 'b-o', label='z_0(h)')
        axes[0].plot(h_grid, Gamma_h, 'r--', label='Gamma(h)')
        axes[0].plot(h_grid, zeta0_vals, 'g-s', label='zeta_0(h)')
        axes[0].set_xlabel('h'); axes[0].set_ylabel('threshold'); axes[0].set_title('Undiscounted Free Boundary')
        axes[0].legend(); axes[0].grid(True, alpha=0.3)
        
        y_plot = np.logspace(-2, 1, 30)
        V_plot = []
        for y in y_plot:
            try: V_plot.append(ud.V0(y, 1.0, h_grid, z0_vals, B0_vals))
            except: V_plot.append(np.nan)
        axes[1].plot(y_plot, V_plot, 'g-')
        axes[1].set_xlabel('y'); axes[1].set_ylabel('V0(y, h=1)'); axes[1].set_title('Dual Value V0')
        axes[1].set_xscale('log'); axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('results/exp3_undiscounted.png', dpi=120, bbox_inches='tight')
        plt.close()
    except Exception as e:
        print(f'Plot failed: {e}')
    
    results['status'] = 'completed'
    with open('results/exp3_results.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print('\nResults saved to results/exp3_results.json')
    
    print('\n' + '='*60)
    print('SUMMARY:')
    print(f'  delta_0 = {delta0:.4f}')
    print(f'  zeta_0 in (0,1): {results.get("zeta0_in_01", False)}')
    print(f'  V0 convex: {results.get("convexity_ok", False)}')
    if 'ystar' in results:
        print(f'  y* = {results["ystar"]:.4f}')

if __name__ == '__main__':
    main()
