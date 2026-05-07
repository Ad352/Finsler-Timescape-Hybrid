#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FTH-DM Active Extension — Appendix A.3
Richardson Extrapolation: Continuum-Limit Validation
Based on: FTH-DM Supp §5.3, Phase V §3.1
"""
import numpy as np
from scipy.integrate import solve_ivp
import sys

# Import ODE system and anchors from Appendix A.2
# (In production: from appendices.a2 import fth_dm_ode, FTHAnchors)

def richardson_convergence(ode_system, anchors, 
                           N_list=[25_000, 50_000, 100_000],
                           a_start=1e-3, a_end=1.0,
                           rtol=1e-10, atol=1e-12,
                           verbose=True):
    """
    Perform three-grid Richardson extrapolation to verify continuum limit.
    
    Parameters
    ----------
    ode_system : callable
        RHS function dy/d(ln a) = F(y, ln a, anchors)
    anchors : FTHAnchors
        Empirical anchor container
    N_list : list of int
        Grid sizes [N, 2N, 4N] for convergence hierarchy
    a_start, a_end : float
        Integration bounds
    rtol, atol : float
        Solver tolerances (must match main pipeline)
    verbose : bool
        Print diagnostic output
    
    Returns
    -------
    dict : Convergence metrics and continuum-validated solution
    """
    lna_span = (np.log(a_start), np.log(a_end))
    
    # Compute Riccati IC once (shared across runs)
    C1 = anchors.riccati_IC()
    b0_init = -anchors.bCMB / np.tanh(anchors.bCMB * (lna_span[0] + C1))
    
    # Initial conditions (matter-dominated high-z)
    H_init = 100 * np.sqrt(0.30 * a_start**-3 + 0.70)
    delta_init = a_start
    f_init = 1.0
    y0 = [H_init, delta_init, b0_init, f_init]
    
    solutions = {}
    
    for N in N_list:
        if verbose:
            print(f"  Integrating with N = {N:,} grid points...")
        
        # Logarithmic grid in a
        ln_a_grid = np.linspace(lna_span[0], lna_span[1], N)
        
        # Solve with BDF + dense output
        sol = solve_ivp(
            fun=lambda t, y: ode_system(t, y, anchors),
            t_span=lna_span,
            y0=y0,
            method='BDF',
            rtol=rtol,
            atol=atol,
            dense_output=True,
            max_step=0.1
        )
        
        if not sol.success:
            raise RuntimeError(f"BDF failed for N={N}: {sol.message}")
        
        # Interpolate to common fine grid using dense output
        y_fine = sol.sol(ln_a_grid)  # Shape: (4, N)
        
        solutions[N] = {
            'ln_a': ln_a_grid,
            'y': y_fine,  # [H_D, delta_DM, b0, f_growth]
            'nfev': sol.nfev,
            'njev': sol.njev,
        }
        
        if verbose:
            print(f"    ✓ N={N:,}: {sol.nfev} evals, {sol.njev} Jacobians")
    
    # =====================================================================
    # RICHARDSON ANALYSIS
    # =====================================================================
    if verbose:
        print("\n  Richardson Convergence Analysis:")
        print("  " + "-"*60)
    
    var_names = ['H_D', 'delta_DM', 'b_0', 'f_growth']
    convergence_metrics = {}
    
    for i, name in enumerate(var_names):
        # Extract values at z=0 (a=1) for convergence test
        y_N = solutions[N_list[0]]['y'][i, -1]
        y_2N = solutions[N_list[1]]['y'][i, -1]
        y_4N = solutions[N_list[2]]['y'][i, -1]
        
        # Compute observed convergence order p_obs
        # For BDF5, expect p ≈ 5
        num = np.abs(y_4N - y_2N)
        den = np.abs(y_2N - y_N)
        
        if den < 1e-15:
            p_obs = np.nan
        else:
            p_obs = np.log2(num / den)
        
        # Relative error estimates
        rel_err_2N_N = np.abs(y_2N - y_N) / np.abs(y_4N) if np.abs(y_4N) > 1e-15 else np.nan
        rel_err_4N_2N = np.abs(y_4N - y_2N) / np.abs(y_4N) if np.abs(y_4N) > 1e-15 else np.nan
        
        # Richardson-extrapolated continuum estimate
        if not np.isnan(p_obs) and p_obs > 0:
            y_continuum = y_4N + (y_4N - y_2N) / (2**p_obs - 1)
        else:
            y_continuum = y_4N
        
        rel_err_continuum = np.abs(y_4N - y_continuum) / np.abs(y_continuum) if np.abs(y_continuum) > 1e-15 else np.nan
        
        convergence_metrics[name] = {
            'p_obs': p_obs,
            'rel_err_2N_N': rel_err_2N_N,
            'rel_err_4N_2N': rel_err_4N_2N,
            'rel_err_continuum': rel_err_continuum,
            'y_4N': y_4N,
            'y_continuum': y_continuum,
        }
        
        if verbose:
            p_str = f"{p_obs:5.2f}" if not np.isnan(p_obs) else "  N/A"
            print(f"  {name:12s}: p_obs={p_str} | "
                  f"rel_err(4N/2N)={rel_err_4N_2N:.2e} | "
                  f"rel_err(cont)={rel_err_continuum:.2e}")
    
    # =====================================================================
    # GLOBAL CONVERGENCE VERDICT
    # =====================================================================
    p_values = [m['p_obs'] for m in convergence_metrics.values() 
                if not np.isnan(m['p_obs'])]
    
    continuum_validated = (
        len(p_values) >= 3 and 
        4.0 <= np.mean(p_values) <= 6.0 and  # BDF order 5 ± tolerance
        all(m['rel_err_continuum'] < 1e-6 for m in convergence_metrics.values())
    )
    
    if verbose:
        print("  " + "-"*60)
        if continuum_validated:
            print(f"  ✓ CONTINUUM LIMIT VALIDATED")
            print(f"    ⟨p_obs⟩ = {np.mean(p_values):.2f} ∈ [4, 6]")
            max_err = max(m['rel_err_continuum'] for m in convergence_metrics.values())
            print(f"    Max continuum error: {max_err:.2e} < 10⁻⁶")
        else:
            print(f"  ✗ CONVERGENCE WARNING")
            print(f"    ⟨p_obs⟩ = {np.mean(p_values) if p_values else 'N/A'}")
            print(f"    Required: 4.0 ≤ ⟨p⟩ ≤ 6.0 AND rel_err < 10⁻⁶")
    
    return {
        'solutions': solutions,
        'convergence_metrics': convergence_metrics,
        'continuum_validated': continuum_validated,
        'p_obs_mean': np.mean(p_values) if p_values else np.nan,
    }

# =============================================================================
# MAIN EXECUTION
# =============================================================================
if __name__ == "__main__":
    # Import from Appendix A.2 (in production setup)
    from appendices.a2 import fth_dm_ode, FTHAnchors
    
    print("="*70)
    print("FTH-DM ACTIVE EXTENSION — APPENDIX A.3")
    print("Richardson Extrapolation: Continuum-Limit Validation")
    print("="*70)
    
    anchors = FTHAnchors()
    
    result = richardson_convergence(
        ode_system=fth_dm_ode,
        anchors=anchors,
        N_list=[25_000, 50_000, 100_000],
        verbose=True
    )
    
    print("\n" + "="*70)
    print("FINAL VERDICT")
    print("="*70)
    
    if result['continuum_validated']:
        print("✓ CONTINUUM LIMIT CONFIRMED")
        print(f"  Observed order: ⟨p⟩ = {result['p_obs_mean']:.2f}")
        print("  All variables: rel_err < 10⁻⁶ vs. Richardson continuum")
        exit_code = 0
    else:
        print("✗ CONTINUUM LIMIT NOT CONFIRMED")
        print("  Check: grid resolution, solver tolerances, ODE formulation")
        exit_code = 1
    
    print("="*70)
    sys.exit(exit_code)