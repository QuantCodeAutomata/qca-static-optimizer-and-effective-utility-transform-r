"""Experiment 6: Cross-cutting numerical validation and robustness diagnostics.

Stress-tests the numerical implementation across representative scenarios.
"""
import json, os, sys, traceback
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.primitives import ExponentialFamily, PowerFamily
from src.static_block import StaticBlock
from src.discounted_dual import DiscountedDual
from src.undiscounted_dual import UndiscountedDual
from src.diagnostics import (
    check_static_vs_numerical, check_derivative_identity_ubar_y,
    check_branch_continuity, check_derivative_ubar_h
)

SCENARIOS = [
    {'name': 'S1_ExpDisc', 'family': 'exp', 'beta': 1.0, 'lam': 0.3,
     'r': 0.05, 'mu': 0.10, 'sigma': 0.2, 'rho': 0.05, 'x': 1.0, 'h': 1.0},
    {'name': 'S2_PowUnd', 'family': 'pow', 'gamma': 2.0, 'alpha': 0.3,
     'r': 0.05, 'mu': 0.10, 'sigma': 0.2, 'rho': 0.0, 'x': 1.0, 'h': 1.0},
    {'name': 'S3_ExpDisc2', 'family': 'exp', 'beta': 2.0, 'lam': 0.5,
     'r': 0.03, 'mu': 0.08, 'sigma': 0.15, 'rho': 0.03, 'x': 0.5, 'h': 0.8},
    {'name': 'S4_PowUnd2', 'family': 'pow', 'gamma': 3.0, 'alpha': 0.2,
     'r': 0.04, 'mu': 0.09, 'sigma': 0.18, 'rho': 0.0, 'x': 0.8, 'h': 1.2},
]

def run_scenario(sc):
    name = sc['name']
    print(f'\n--- Scenario {name} ---')
    row = {'name': name}
    
    try:
        # Build primitives
        if sc['family'] == 'exp':
            prim = ExponentialFamily(sc['beta'], sc['lam'])
        else:
            prim = PowerFamily(sc['gamma'], sc['alpha'])
        
        sb = StaticBlock(prim)
        market = {'r': sc['r'], 'mu': sc['mu'], 'sigma': sc['sigma']}
        h = sc['h']
        y_grid = np.logspace(-2, 1.5, 40)
        
        # a. Static block check
        try:
            static_res = check_static_vs_numerical(sb, y_grid, h_vals=[h], tol=1e-6)
            row['static_max_error'] = float(static_res['max_error'])
            row['static_pass'] = bool(static_res['pass_flag'])
            print(f'  Static optimizer: max_err={static_res["max_error"]:.2e}, PASS={static_res["pass_flag"]}')
        except Exception as e:
            row['static_error'] = str(e)[:80]
            print(f'  Static optimizer: ERROR - {e}')
        
        # b. Branch continuity
        try:
            bc = check_branch_continuity(sb, h)
            row['b_continuity_ok'] = bool(bc.get('b_continuity_ok', False))
            row['d_continuity_ok'] = bool(bc.get('d_continuity_ok', True))
            print(f'  Branch continuity: b={bc.get("b_continuity_ok")}, d={bc.get("d_continuity_ok")}')
        except Exception as e:
            row['continuity_error'] = str(e)[:80]
        
        # c. Derivative identity
        try:
            der_res = check_derivative_identity_ubar_y(sb, y_grid, h)
            row['ubar_y_max_residual'] = float(der_res['max_residual'])
            row['ubar_y_pass'] = bool(der_res['max_residual'] < 1e-5)
            print(f'  ubar_y identity: max_res={der_res["max_residual"]:.2e}')
        except Exception as e:
            row['ubar_y_error'] = str(e)[:80]
        
        # d. ubar_h finite difference check
        try:
            b_h = float(sb.b_thresh(h))
            d_h = float(sb.d_thresh(h))
            y_safe = np.array([y for y in y_grid
                               if abs(y-b_h)>0.05 and (not np.isfinite(d_h) or abs(y-d_h)>0.05)])
            if len(y_safe) > 0:
                uh_res = check_derivative_ubar_h(sb, y_safe, h, eps=1e-5)
                row['ubar_h_max_residual'] = float(uh_res['max_residual'])
                print(f'  ubar_h FD: max_res={uh_res["max_residual"]:.2e}')
        except Exception as e:
            row['ubar_h_error'] = str(e)[:80]
        
        # e. Threshold diagnostics
        if sc['rho'] > 0:
            try:
                dd = DiscountedDual(prim, market, sc['rho'])
                a_grid = np.linspace(h, h + 2.0, 8)
                z_vals = dd.compute_free_boundary(a_grid)
                Gamma_vals = np.array([sb.Gamma_fn(a) for a in a_grid])
                row['z_rho_positive'] = bool(np.all(z_vals > 0))
                row['z_rho_lt_Gamma'] = bool(np.all(z_vals < Gamma_vals))
                print(f'  z_rho: positive={row["z_rho_positive"]}, <Gamma={row["z_rho_lt_Gamma"]}')
            except Exception as e:
                row['threshold_disc_error'] = str(e)[:80]
        else:
            try:
                ud = UndiscountedDual(prim, market)
                h_test = np.linspace(h, h + 2.0, 6)
                zeta0, z0 = ud.compute_free_boundary(h_test)
                row['zeta0_in_01'] = bool(np.all(zeta0 > 0) and np.all(zeta0 < 1))
                row['z0_positive'] = bool(np.all(z0 > 0))
                print(f'  zeta_0 in (0,1): {row["zeta0_in_01"]}')
            except Exception as e:
                row['threshold_und_error'] = str(e)[:80]
    except Exception as e:
        row['scenario_error'] = str(e)[:100]
        print(f'  Scenario failed: {e}')
    
    return row

def run_grid_refinement(name='S1_ExpDisc'):
    """Grid refinement study for S1."""
    print(f'\n--- Grid refinement study ({name}) ---')
    prim = ExponentialFamily(beta=1.0, lam=0.3)
    market = {'r': 0.05, 'mu': 0.10, 'sigma': 0.2}
    dd = DiscountedDual(prim, market, rho=0.05)
    
    results = {}
    prev_z = None
    for n in [10, 20, 40]:
        a_grid = np.linspace(0.5, 3.0, n)
        try:
            z_vals = dd.compute_free_boundary(a_grid)
            if prev_z is not None:
                # Compare at common points (every 2nd)
                a_common = np.linspace(0.5, 3.0, 10)
                z_curr = np.interp(a_common, a_grid, z_vals)
                z_prev_interp = np.interp(a_common, np.linspace(0.5, 3.0, len(prev_z)), prev_z)
                sup_diff = float(np.max(np.abs(z_curr - z_prev_interp)))
                print(f'  n={n}: sup|z_rho_n - z_rho_{n//2}| = {sup_diff:.4e}')
                results[f'refinement_n{n}'] = sup_diff
            prev_z = z_vals
        except Exception as e:
            print(f'  n={n}: failed - {e}')
    return results

def main():
    os.makedirs('results', exist_ok=True)
    
    print('='*60)
    print('EXPERIMENT 6: CROSS-CUTTING VALIDATION & ROBUSTNESS')
    print('='*60)
    
    all_results = []
    for sc in SCENARIOS:
        try:
            row = run_scenario(sc)
        except Exception as e:
            row = {'name': sc['name'], 'error': str(e)}
        all_results.append(row)
    
    df = pd.DataFrame(all_results)
    
    refinement = {}
    try:
        refinement = run_grid_refinement()
    except Exception as e:
        print(f'Refinement study failed: {e}')
    
    print('\n' + '='*60)
    print('VALIDATION SUMMARY')
    print('='*60)
    
    check_cols = ['static_pass', 'b_continuity_ok', 'ubar_y_pass']
    for col in check_cols:
        if col in df.columns:
            pass_rate = df[col].sum() if df[col].dtype == bool else df[col].astype(bool).sum()
            print(f'  {col}: {pass_rate}/{len(df)} scenarios passed')
    
    report = {
        'scenarios': all_results,
        'grid_refinement': refinement,
    }
    with open('results/exp6_validation_report.json', 'w') as f:
        json.dump(report, f, indent=2, default=str)
    print('\nReport saved to results/exp6_validation_report.json')

if __name__ == '__main__':
    main()
