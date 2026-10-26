"""Experiment 4: Special-case corollary validation and parameter sweeps.

Validates the general implementation against:
- Family A: Discounted exponential (rho=r, D=1, L=lambda*h)
- Family B: Undiscounted power (rho=0, D=h^{-alpha}, L=0)
Across parameter sweeps.
"""
import json, os, sys, traceback
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.primitives import ExponentialFamily, PowerFamily
from src.discounted_dual import DiscountedDual
from src.undiscounted_dual import UndiscountedDual

def run_family_A_sweep():
    """Sweep over beta and lam for discounted exponential case."""
    market = {'r': 0.05, 'mu': 0.10, 'sigma': 0.2}
    rho = 0.05
    a_grid = np.linspace(0.5, 3.0, 10)
    x_val, h_val = 1.0, 1.0
    
    records = []
    betas = [0.5, 1.0, 2.0]
    lams = [0.1, 0.3, 0.5]
    
    for beta in betas:
        for lam in lams:
            row = {'beta': beta, 'lam': lam}
            try:
                prim = ExponentialFamily(beta=beta, lam=lam)
                dd = DiscountedDual(prim, market, rho)
                g1, g2 = dd.gamma1, dd.gamma2
                row['gamma1'] = g1
                row['gamma2'] = g2
                row['roots_ok'] = bool(g1 > 1 and g2 < 0)
                
                z_vals = dd.compute_free_boundary(a_grid)
                A_vals = np.array([dd.A_rho(a, z) for a, z in zip(a_grid, z_vals)])
                row['z_rho_min'] = float(z_vals.min())
                row['z_rho_max'] = float(z_vals.max())
                row['z_rho_monotone'] = bool(np.all(np.diff(z_vals) <= 0.1*z_vals[:-1]))  # roughly monotone
                
                try:
                    J_test = dd.J_rho(0.5, h_val, a_grid, z_vals, A_vals)
                    row['J_rho_at_test'] = float(J_test)
                except Exception as e2:
                    row['J_rho_error'] = str(e2)[:50]
                
                try:
                    ystar = dd.find_multiplier(x_val, h_val, a_grid, z_vals, A_vals)
                    row['ystar'] = float(ystar)
                except Exception as e2:
                    row['ystar_error'] = str(e2)[:50]
            except Exception as e:
                row['error'] = str(e)[:100]
            
            records.append(row)
            print(f'  beta={beta}, lam={lam}: roots_ok={row.get("roots_ok", False)}, ystar={row.get("ystar", "N/A")}')
    
    return pd.DataFrame(records)

def run_family_B_sweep():
    """Sweep over gamma and alpha for undiscounted power case."""
    market = {'r': 0.05, 'mu': 0.10, 'sigma': 0.2}
    h_grid = np.linspace(0.5, 3.0, 8)
    x_val, h_val = 1.0, 1.0
    
    records = []
    gammas = [1.5, 2.0, 3.0]
    alphas = [0.1, 0.3, 0.5]
    
    for gamma in gammas:
        for alpha in alphas:
            row = {'gamma': gamma, 'alpha': alpha}
            try:
                prim = PowerFamily(gamma=gamma, alpha=alpha)
                ud = UndiscountedDual(prim, market)
                delta0 = ud.compute_delta0()
                row['delta0'] = float(delta0)
                
                zeta0_vals, z0_vals = ud.compute_free_boundary(h_grid)
                B0_vals = ud.compute_B0(h_grid, z0_vals)
                
                row['zeta0_min'] = float(zeta0_vals.min())
                row['zeta0_max'] = float(zeta0_vals.max())
                row['zeta0_in_01'] = bool(np.all(zeta0_vals > 0) and np.all(zeta0_vals < 1))
                
                try:
                    V_test = ud.V0(0.5, h_val, h_grid, z0_vals, B0_vals)
                    row['V0_at_test'] = float(V_test)
                except Exception as e2:
                    row['V0_error'] = str(e2)[:50]
                
                try:
                    ystar = ud.find_multiplier(x_val, h_val, h_grid, z0_vals, B0_vals)
                    row['ystar'] = float(ystar)
                except Exception as e2:
                    row['ystar_error'] = str(e2)[:50]
            except Exception as e:
                row['error'] = str(e)[:100]
            
            records.append(row)
            print(f'  gamma={gamma}, alpha={alpha}: delta0={row.get("delta0", "N/A"):.3f}, zeta0_in_01={row.get("zeta0_in_01", False)}')
    
    return pd.DataFrame(records)

def main():
    os.makedirs('results', exist_ok=True)
    
    print('='*60)
    print('EXPERIMENT 4: SPECIAL-CASE COROLLARY VALIDATION')
    print('='*60)
    
    print('\n--- Family A: Discounted Exponential ---')
    df_A = run_family_A_sweep()
    df_A.to_csv('results/exp4_family_A.csv', index=False)
    print(df_A[['beta','lam','gamma1','gamma2','roots_ok','ystar']].to_string())
    
    print('\n--- Family B: Undiscounted Power ---')
    df_B = run_family_B_sweep()
    df_B.to_csv('results/exp4_family_B.csv', index=False)
    print(df_B[['gamma','alpha','delta0','zeta0_in_01','ystar']].to_string())
    
    # Plots
    try:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        a_plot = np.linspace(0.5, 3.0, 10)
        for lam in [0.1, 0.3, 0.5]:
            try:
                prim = ExponentialFamily(beta=1.0, lam=lam)
                dd = DiscountedDual(prim, {'r':0.05,'mu':0.10,'sigma':0.2}, 0.05)
                z = dd.compute_free_boundary(a_plot)
                axes[0].plot(a_plot, z, label=f'lam={lam}')
            except Exception as e:
                print(f'Plot A lam={lam}: {e}')
        axes[0].set_xlabel('a'); axes[0].set_ylabel('z_rho(a)')
        axes[0].set_title('Family A: z_rho vs a (beta=1, varying lam)')
        axes[0].legend(); axes[0].grid(True, alpha=0.3)
        
        h_plot = np.linspace(0.5, 3.0, 8)
        for alpha in [0.1, 0.3, 0.5]:
            try:
                prim = PowerFamily(gamma=2.0, alpha=alpha)
                ud = UndiscountedDual(prim, {'r':0.05,'mu':0.10,'sigma':0.2})
                _, z0 = ud.compute_free_boundary(h_plot)
                axes[1].plot(h_plot, z0, label=f'alpha={alpha}')
            except Exception as e:
                print(f'Plot B alpha={alpha}: {e}')
        axes[1].set_xlabel('h'); axes[1].set_ylabel('z_0(h)')
        axes[1].set_title('Family B: z_0 vs h (gamma=2, varying alpha)')
        axes[1].legend(); axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('results/exp4_special_cases.png', dpi=120, bbox_inches='tight')
        plt.close()
    except Exception as e:
        print(f'Plot failed: {e}')
    
    results = {
        'family_A_all_roots_ok': bool(df_A.get('roots_ok', pd.Series([False])).all()) if 'roots_ok' in df_A else False,
        'family_B_all_zeta_in_01': bool(df_B.get('zeta0_in_01', pd.Series([False])).all()) if 'zeta0_in_01' in df_B else False,
        'n_A_cases': len(df_A),
        'n_B_cases': len(df_B),
    }
    with open('results/exp4_results.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print('\nResults saved to results/exp4_results.json')

if __name__ == '__main__':
    main()
