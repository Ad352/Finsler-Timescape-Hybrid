#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FTH+DM Step F.2.4: Richardson Convergence Validation
Implements three-grid Richardson extrapolation to verify continuum limit.
"""
import numpy as np
from scipy.integrate import solve_ivp

def richardson_convergence_test(ode_system, a_span, y0, anchors, N_list=[25000, 50000, 100000], verbose=True):
    """
    Perform Richardson convergence analysis for FTH+DM ODE system.
    
    Parameters
    ----------
    ode_system : callable
        RHS of dy/d(ln a) = f(ln a, y)
    a_span : tuple
        (a_start, a_end) integration bounds
    y0 : array
        Initial conditions at a_start
    anchors : object
        FTHAnchors instance with empirical parameters
    N_list : list
        Grid sizes for convergence hierarchy [N, 2N, 4N]
    verbose : bool
        Print diagnostic output
    
    Returns
    -------
    results : dict
        Solutions and convergence metrics
    """
    solutions = {}
    
    for N in N_list:
        if verbose:
            print(f"🔧 Integrating with N={N:,} grid points...")
        
        # Logarithmic grid in a
        ln_a_grid = np.linspace(np.log(a_span[0]), np.log(a_span[1]), N)
        
        # Solve with BDF + dense output
        sol = solve_ivp(
            fun=lambda t, y: ode_system(t, y, anchors),
            t_span=(np.log(a_span[0]), np.log(a_span[1])),
            y0=y0,
            method='BDF',
            rtol=1e-10,
            atol=1e-12,
            dense_output=True,
            max_step=0.1
        )
        
        if not sol.success:
            raise RuntimeError(f"BDF solver failed for N={N}: {sol.message}")
        
        # Interpolate to common fine grid using dense output
        y_fine = sol.sol(ln_a_grid)  # Shape: (4, N)
        solutions[N] = {
            'ln_a': ln_a_grid,
            'y': y_fine,  # [H, delta, b0, f]
            'nfev': sol.nfev,
            'njev': sol.njev
        }
        if verbose:
            print(f"✅ N={N:,}: {sol.nfev} evals, {sol.njev} Jacobians")
    
    # --- Richardson analysis ---
    if verbose:
        print("\n📊 Richardson Convergence Analysis:")
        print("-" * 60)
    
    # Extract values at fixed redshift (z=0) for convergence test
    z_test = 0.0
    a_test = 1.0 / (1.0 + z_test)
    ln_a_test = np.log(a_test)
    
    convergence_metrics = {}
    var_names = ['H_D', 'delta_DM', 'b_0', 'f_growth']
    
    for i, name in enumerate(var_names):
        y_N = solutions[N_list[0]]['y'][i, -1]
        y_2N = solutions[N_list[1]]['y'][i, -1]
        y_4N = solutions[N_list[2]]['y'][i, -1]
        
        # Compute observed order p_obs
        num = np.abs(y_4N - y_2N)
        den = np.abs(y_2N - y_N)
        if den < 1e-15:
            p_obs = np.nan
        else:
            p_obs = np.log2(num / den)
        
        # Relative error estimates
        rel_err_2N_N = np.abs(y_2N - y_N) / np.abs(y_4N) if np.abs(y_4N) > 1e-15 else np.nan
        rel_err_4N_2N = np.abs(y_4N - y_2N) / np.abs(y_4N) if np.abs(y_4N) > 1e-15 else np.nan
        
        convergence_metrics[name] = {
            'p_obs': p_obs,
            'rel_err_2N_N': rel_err_2N_N,
            'rel_err_4N_2N': rel_err_4N_2N,
            'y_4N': y_4N
        }
        
        if verbose:
            print(f"{name:12s}: p_obs = {p_obs:5.2f} | "
                  f"rel_err(2N/N) = {rel_err_2N_N:.2e} | "
                  f"rel_err(4N/2N) = {rel_err_4N_2N:.2e}")
    
    # Global convergence verdict
    p_values = [convergence_metrics[name]['p_obs'] for name in var_names 
                if not np.isnan(convergence_metrics[name]['p_obs'])]
    
    continuum_validated = (
        len(p_values) >= 3 and 
        4.0 <= np.mean(p_values) <= 6.0 and 
        all(convergence_metrics[name]['rel_err_4N_2N'] < 1e-6 for name in var_names)
    )
    
    if verbose:
        print("-" * 60)
        if continuum_validated:
            print(f"✅ CONTINUUM LIMIT VALIDATED: ⟨p_obs⟩ = {np.mean(p_values):.2f} ∈ [4,6]")
            print(f"   Max relative error (4N/2N): {max(convergence_metrics[name]['rel_err_4N_2N'] for name in var_names):.2e}")
        else:
            print(f"⚠️  CONVERGENCE WARNING: ⟨p_obs⟩ = {np.mean(p_values) if p_values else 'N/A'}")
            print("   Required: 4.0 ≤ ⟨p⟩ ≤ 6.0 AND rel_err < 1e-6")
    
    return solutions[N_list[-1]], continuum_validated, convergence_metrics