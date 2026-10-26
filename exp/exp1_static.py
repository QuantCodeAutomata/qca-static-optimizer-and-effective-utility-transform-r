"""Experiment 1: Static optimizer c*(y,h) and effective utility transform ū(y,h).

Replicates and numerically verifies Proposition 2.2 from 'Duality of Shortfall Aversion':
- Closed-form static optimizer c*(y,h)
- Transformed utility ū(y,h)
- Derivative identity ū_y = -c*
- Branch continuity at y=b(h) and y=d(h)
- Brute-force numerical verification
"""
import json
import sys
import os

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.primitives import ExponentialFamily, PowerFamily, AppendixCFamily
from src.static_block import StaticBlock
from src.diagnostics import (
    check_static_vs_numerical,
    check_derivative_identity_ubar_y,
    check_derivative_ubar_h,
    check_branch_continuity,
)


def run_experiment_family(family_name: str, prim, results: dict) -> None:
    """Run all static-block checks for a given model family."""
    print(f"\n{'='*60}")
    print(f"  Family: {family_name}")
    print(f"{'='*60}")

    sb = StaticBlock(prim)
    h_grid = np.linspace(0.5, 3.0, 30)
    y_grid = np.logspace(-3, 2, 200)

    # 1. Assumption checks
    print("\n1. Assumption 2.1 checks:")
    assumption_results = {}
    for h in h_grid[:5]:
        check = sb.verify_assumption_21(h)
        assumption_results[float(h)] = check.get('all_satisfied', False)
    all_ok = all(assumption_results.values())
    print(f"   All Assumption 2.1 checks passed: {all_ok}")

    # 2. Plot c*(y,h) for selected h values
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    h_plot_vals = [1.0, 1.5, 2.0]
    colors = ['blue', 'orange', 'green']

    for h_val, color in zip(h_plot_vals, colors):
        c_vals = sb.cstar(y_grid, h_val)
        ub_vals = sb.ubar(y_grid, h_val)
        b_val = sb.b_thresh(h_val)
        d_val = sb.d_thresh(h_val)

        axes[0].semilogx(y_grid, c_vals, color=color, label=f'h={h_val}')
        axes[0].axvline(b_val, color=color, linestyle='--', alpha=0.5)
        if np.isfinite(d_val):
            axes[0].axvline(d_val, color=color, linestyle=':', alpha=0.5)

        finite_mask = np.isfinite(ub_vals)
        axes[1].semilogx(y_grid[finite_mask], ub_vals[finite_mask], color=color, label=f'h={h_val}')

    axes[0].set_xlabel('y')
    axes[0].set_ylabel('c*(y,h)')
    axes[0].set_title(f'{family_name}: Static Optimizer c*(y,h)')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].set_xlabel('y')
    axes[1].set_ylabel('ū(y,h)')
    axes[1].set_title(f'{family_name}: Transformed Utility ū(y,h)')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    safe_name = "".join(c if c.isalnum() or c == '_' else '_' for c in family_name.lower().replace(" ", "_"))
    fname = f'results/exp1_{safe_name}_cstar.png'
    plt.savefig(fname, dpi=120, bbox_inches='tight')
    plt.close()
    print(f"   Plot saved: {fname}")

    # 3. Verify ū_y = -c* via identity (exact by definition)
    print("\n3. Derivative identity ū_y = -c*:")
    h_test = 1.0
    y_test = np.logspace(-2, 1.5, 50)
    ub_y = sb.ubar_y(y_test, h_test)
    neg_c = -sb.cstar(y_test, h_test)
    identity_residual = np.max(np.abs(ub_y - neg_c))
    print(f"   Max |ū_y + c*| = {identity_residual:.2e}  (should be ~0)")

    # 4. Verify ū_y via finite differences
    print("\n4. Finite-difference check of ū_y:")
    fd_check = check_derivative_identity_ubar_y(sb, np.logspace(-2, 1.5, 40), h_test)
    print(f"   Max FD residual for ū_y: {fd_check['max_residual']:.2e}")

    # 5. Verify ū_h via finite differences
    print("\n5. Finite-difference check of ū_h:")
    # Avoid near branch points where FD may be unstable
    b_h = sb.b_thresh(h_test)
    d_h = sb.d_thresh(h_test)
    y_safe = np.array([y for y in np.logspace(-2, 1.5, 40)
                       if abs(y - b_h) > 0.05 and (not np.isfinite(d_h) or abs(y - d_h) > 0.05)])
    if len(y_safe) > 0:
        h_check = check_derivative_ubar_h(sb, y_safe, h_test, eps=1e-5)
        print(f"   Max FD residual for ū_h: {h_check['max_residual']:.2e} ({h_check['n_tested']} points)")
    else:
        h_check = {'max_residual': float('nan'), 'n_tested': 0}

    # 6. Branch continuity
    print("\n6. Branch continuity checks:")
    branch_results = {}
    for h_val in [0.8, 1.0, 1.5, 2.0]:
        bc = check_branch_continuity(sb, h_val)
        branch_results[h_val] = bc
        status = "PASS" if bc.get('b_continuity_ok', True) and bc.get('d_continuity_ok', True) else "FAIL"
        print(f"   h={h_val}: {status}  (b-jump={bc.get('b_cstar_jump', 0):.2e}, d-jump={bc.get('d_cstar_jump', 0):.2e})")

    # 7. Closed-form vs numerical maximization
    print("\n7. Closed-form c* vs numerical maximization:")
    y_check = np.logspace(-2, 1, 50)
    num_result = check_static_vs_numerical(sb, y_check, h_vals=[1.0, 1.5, 2.0], tol=1e-6)
    print(f"   Max error: {num_result['max_error']:.2e}")
    print(f"   Mean error: {num_result['mean_error']:.2e}")
    print(f"   Pass: {num_result['pass_flag']}")

    # 8. vartheta identity: vartheta = 1 - ell' - chi
    print("\n8. vartheta identity check:")
    vartheta_residuals = []
    for h_val in h_grid:
        vt = sb.vartheta(h_val)
        el_p = sb.ell_prime(h_val)
        ch = sb.chi(h_val)
        vartheta_residuals.append(abs(vt - (1 - el_p - ch)))
    print(f"   Max |vartheta - (1 - ell' - chi)| = {max(vartheta_residuals):.2e}")

    # 9. Heatmap of ū(y,h)
    y_hm = np.logspace(-2, 1.5, 50)
    h_hm = np.linspace(0.5, 2.5, 30)
    UB = np.zeros((len(h_hm), len(y_hm)))
    for i, h_val in enumerate(h_hm):
        ub_row = sb.ubar(y_hm, h_val)
        UB[i, :] = np.where(np.isfinite(ub_row), ub_row, np.nan)

    fig, ax = plt.subplots(figsize=(8, 5))
    # Clip for visualization
    UB_clipped = np.clip(UB, np.nanpercentile(UB, 5), np.nanpercentile(UB, 95))
    im = ax.contourf(np.log10(y_hm), h_hm, UB_clipped, levels=20, cmap='viridis')
    plt.colorbar(im, ax=ax)
    ax.set_xlabel('log10(y)')
    ax.set_ylabel('h')
    ax.set_title(f'{family_name}: ū(y,h) heatmap')
    safe_name = "".join(c if c.isalnum() or c == '_' else '_' for c in family_name.lower().replace(" ", "_"))
    fname_hm = f'results/exp1_{safe_name}_ubar_heatmap.png'
    plt.savefig(fname_hm, dpi=120, bbox_inches='tight')
    plt.close()
    print(f"\n   Heatmap saved: {fname_hm}")

    # Store results
    results[family_name] = {
        'assumption_21_all_ok': bool(all_ok),
        'identity_ubar_y_residual': float(identity_residual),
        'fd_ubar_y_max_residual': float(fd_check['max_residual']),
        'fd_ubar_h_max_residual': float(h_check['max_residual']),
        'branch_b_continuity_ok': all(r.get('b_continuity_ok', True) for r in branch_results.values()),
        'branch_d_continuity_ok': all(r.get('d_continuity_ok', True) for r in branch_results.values()),
        'numerical_max_error': float(num_result['max_error']),
        'numerical_pass': bool(num_result['pass_flag']),
        'vartheta_identity_max_residual': float(max(vartheta_residuals)),
    }
    return results


def main():
    os.makedirs('results', exist_ok=True)
    results = {}

    # Family 1: Exponential
    try:
        prim_exp = ExponentialFamily(beta=1.0, lam=0.3)
        run_experiment_family("Exponential (beta=1,lam=0.3)", prim_exp, results)
    except Exception as e:
        print(f"Exponential family failed: {e}")
        results["Exponential"] = {"error": str(e)}

    # Family 2: Power
    try:
        prim_pow = PowerFamily(gamma=2.0, alpha=0.3)
        run_experiment_family("Power (gamma=2,alpha=0.3)", prim_pow, results)
    except Exception as e:
        print(f"Power family failed: {e}")
        results["Power"] = {"error": str(e)}

    # Family 3: AppendixC
    try:
        prim_app = AppendixCFamily()
        run_experiment_family("AppendixC (u=-1/xi)", prim_app, results)
    except Exception as e:
        print(f"AppendixC family failed: {e}")
        results["AppendixC"] = {"error": str(e)}

    # Summary
    print("\n" + "="*60)
    print("EXPERIMENT 1 SUMMARY")
    print("="*60)
    for fam, res in results.items():
        if 'error' in res:
            print(f"  {fam}: ERROR - {res['error']}")
        else:
            print(f"  {fam}:")
            print(f"    Assumption 2.1: {'PASS' if res['assumption_21_all_ok'] else 'FAIL'}")
            print(f"    ū_y identity: {res['identity_ubar_y_residual']:.2e}")
            print(f"    FD ū_y: {res['fd_ubar_y_max_residual']:.2e}")
            print(f"    FD ū_h: {res['fd_ubar_h_max_residual']:.2e}")
            print(f"    Numerical match: {'PASS' if res['numerical_pass'] else 'FAIL'} (max_err={res['numerical_max_error']:.2e})")
            print(f"    Branch continuity: b={'PASS' if res['branch_b_continuity_ok'] else 'FAIL'}, d={'PASS' if res['branch_d_continuity_ok'] else 'FAIL'}")

    with open('results/exp1_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    print("\nResults saved to results/exp1_results.json")


if __name__ == "__main__":
    main()
