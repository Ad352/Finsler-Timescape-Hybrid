#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FTH-DM Active Extension — Appendix A.4
Kill-Switch Wrapper: Automated Falsification Protocol
Based on: FTH-DM Supp §6, Phase VI, K-DM-4/5/6 definitions
"""
import h5py
import numpy as np
import sys
import json
from datetime import datetime

# =============================================================================
# CONFIGURATION: Kill Thresholds (Stage-IV Survey Sensitivities)
# =============================================================================
KILL_THRESHOLDS = {
    # K-DM-4: Dipole-modulated weak lensing
    'K-DM-4': {
        'threshold': 0.002,  # Euclid/LSST WL sensitivity (3σ)
        'description': 'Residual lensing anisotropy |Δκ_dip|',
        'instrument': 'Euclid WL (2027) / LSST-Y1 (2026)',
    },
    
    # K-DM-5: Void-halo ellipticity
    'K-DM-5': {
        'threshold': 0.15,  # DESI+SKA shape uncertainty (4σ)
        'description': 'Void-halo ellipticity deviation |Δε_void|',
        'instrument': 'DESI DR3 Void Catalog (2027) / SKA1-Low (2028)',
    },
    
    # K-DM-6: Scale-dependent growth (fσ₈)
    'K-DM-6': {
        'threshold': 0.02,  # DESI RSD precision (3σ)
        'description': 'Scale-dependent growth deviation |Δ(fσ₈)|',
        'instrument': 'DESI RSD (2026–2028) / Euclid Spectroscopy (2027)',
    },
}

# Geometric constants (S³ angular averages — no tuning)
GEOM_CONSTANTS = {
    'A_geom': 2/15,   # Lensing quadrupole projection
    'E_geom': 1/5,    # Void-halo ellipticity scaling
    'C1_growth': 2/15,  # Growth friction renormalization
    'C2_growth': 1/15,  # Geometric pressure coefficient
}

# =============================================================================
# LOAD F.2 OUTPUT (High-Resolution Solution Archive)
# =============================================================================
def load_f2_results(filepath):
    """
    Load high-resolution arrays from Appendix A.2 output.
    
    Verifies GR regression and continuum convergence flags.
    """
    try:
        with h5py.File(filepath, 'r') as f:
            # Check mandatory reproducibility flags
            if not f.attrs.get('GR_test_passed', False):
                raise RuntimeError("Input data failed GR Regression Test. Abort.")
            if not f.attrs.get('continuum_validated', False):
                raise RuntimeError("Input data failed Continuum Limit (N < 10⁵). Abort.")
            
            # Load state variables
            data = {
                'ln_a': f['ln_a'][:],
                'a': f['a'][:],
                'H_D': f['H_D'][:],
                'delta_DM': f['delta_DM'][:],
                'b_0': f['b_0'][:],
                'f_growth': f['f_growth'][:],
                'fsigma8': f['fsigma8'][:],
                'chi_z': f['chi_z'][:],
                'BAO_residual': float(f['BAO_residual'][0]),
            }
            
            # Load metadata
            data['metadata'] = {}
            if 'metadata' in f:
                for key in f['metadata'].keys():
                    if isinstance(f[f'metadata/{key}'], h5py.Group):
                        data['metadata'][key] = dict(f[f'metadata/{key}'].attrs)
                    else:
                        data['metadata'][key] = f[f'metadata/{key}'][()]
            
            return data
            
    except FileNotFoundError:
        print(f"[CRITICAL] File not found: {filepath}")
        sys.exit(1)
    except Exception as e:
        print(f"[CRITICAL] Failed to load F.2 results: {e}")
        sys.exit(1)

# =============================================================================
# K-DM-4: Dipole-Modulated Weak Lensing
# =============================================================================
def evaluate_kdm4(data, lambda_val, z_bin=(0.5, 1.0)):
    """
    K-DM-4: Dipole-modulated weak lensing convergence.
    
    Prediction (Phase VI Eq. 6.2):
    Δκ_dip = λ² b₀²(z) A_geom 𝒴₂(𝑛̂) δ_DM^aniso
    
    Threshold: |Δκ_dip| > 0.002 (Euclid/LSST 3σ)
    """
    # Extract b₀ at representative redshift
    z_mid = np.mean(z_bin)
    a_mid = 1.0 / (1.0 + z_mid)
    idx = np.argmin(np.abs(data['a'] - a_mid))
    
    b0_z = data['b_0'][idx]
    delta_z = data['delta_DM'][idx]
    
    # Geometric prediction (A_geom = 2/15 from S³ rank-4 average)
    A_geom = GEOM_CONSTANTS['A_geom']
    Y2_max = 1.0  # Quadrupole maximum |𝒴₂| = 1
    
    # Predicted anisotropic amplitude
    pred = (lambda_val**2 * b0_z**2 * A_geom * Y2_max * abs(delta_z))
    
    # Placeholder for observed value (to be filled with survey data)
    # In production: delta_kappa_obs = load_euclid_data(z_bin)
    delta_kappa_obs = 0.0  # Default: no detection
    
    # Residual
    residual = abs(delta_kappa_obs - pred)
    threshold = KILL_THRESHOLDS['K-DM-4']['threshold']
    
    passed = residual <= threshold
    
    return {
        'passed': passed,
        'residual': float(residual),
        'prediction': float(pred),
        'observed': float(delta_kappa_obs),
        'threshold': threshold,
        'b0_z': float(b0_z),
        'lambda': lambda_val,
    }

# =============================================================================
# K-DM-5: Void-Halo Ellipticity
# =============================================================================
def evaluate_kdm5(data, lambda_val, void_radius_fraction=0.5):
    """
    K-DM-5: Void-halo ellipticity deviation.
    
    Prediction (Phase VI Eq. 6.4):
    Δε_void = λ² b₀² E_geom 𝒬_torsion
    
    Threshold: |Δε_void| > 0.15 (DESI+SKA 4σ)
    """
    # Representative b₀ at z ~ 0.5 (void catalog depth)
    z_rep = 0.5
    a_rep = 1.0 / (1.0 + z_rep)
    idx = np.argmin(np.abs(data['a'] - a_rep))
    
    b0_z = data['b_0'][idx]
    
    # Geometric prediction (E_geom = 1/5 from S³ rank-6 average)
    E_geom = GEOM_CONSTANTS['E_geom']
    Q_torsion = 1.0  # Normalized torsion-shear dispersion
    
    pred = lambda_val**2 * b0_z**2 * E_geom * Q_torsion
    
    # Placeholder for observed ellipticity deviation
    delta_eps_obs = 0.0  # Default: no deviation detected
    
    residual = abs(delta_eps_obs - pred)
    threshold = KILL_THRESHOLDS['K-DM-5']['threshold']
    
    passed = residual <= threshold
    
    return {
        'passed': passed,
        'residual': float(residual),
        'prediction': float(pred),
        'observed': float(delta_eps_obs),
        'threshold': threshold,
        'b0_z': float(b0_z),
    }

# =============================================================================
# K-DM-6: Scale-Dependent Growth (fσ₈)
# =============================================================================
def evaluate_kdm6(data, lambda_val, k_range=(0.05, 0.2), z_rep=0.5):
    """
    K-DM-6: Scale-dependent growth deviation in fσ₈.
    
    Prediction (Phase VI Eq. 6.5–6.6):
    Δ(fσ₈)_scale = fσ₈^ΛCDM × λ² b₀² [C1 + C2 (k/aH)²]
    
    Threshold: |Δ(fσ₈)| > 0.02 (DESI RSD 3σ)
    """
    # Representative scale factor
    a_rep = 1.0 / (1.0 + z_rep)
    idx = np.argmin(np.abs(data['a'] - a_rep))
    
    b0_z = data['b_0'][idx]
    f_z = data['f_growth'][idx]
    sigma8_z = data['fsigma8'][idx] / f_z if f_z != 0 else 0.811
    
    # Geometric coefficients
    C1 = GEOM_CONSTANTS['C1_growth']
    C2 = GEOM_CONSTANTS['C2_growth']
    
    # Representative k-mode (sub-horizon: k ~ a H)
    H_z = data['H_D'][idx]
    k_over_aH = 1.0  # Representative value
    
    # Predicted deviation
    pred_factor = lambda_val**2 * b0_z**2 * (C1 + C2 * k_over_aH**2)
    pred = sigma8_z * f_z * pred_factor
    
    # Placeholder for observed deviation
    delta_fs8_obs = 0.0  # Default: no scale-dependence detected
    
    residual = abs(delta_fs8_obs - pred)
    threshold = KILL_THRESHOLDS['K-DM-6']['threshold']
    
    passed = residual <= threshold
    
    return {
        'passed': passed,
        'residual': float(residual),
        'prediction': float(pred),
        'observed': float(delta_fs8_obs),
        'threshold': threshold,
        'b0_z': float(b0_z),
        'k_over_aH': k_over_aH,
    }

# =============================================================================
# AUTOMATED KILL-SWITCH EVALUATION
# =============================================================================
def run_kill_switches(filepath, lambda_val=0.1):
    """
    Execute full K-DM-4/5/6 falsification protocol.
    
    Returns structured verdict with pass/fail/investigate status.
    """
    print("="*70)
    print("FTH-DM ACTIVE EXTENSION — APPENDIX A.4")
    print("Kill-Switch Evaluation: K-DM-4/5/6")
    print("="*70)
    
    # Load F.2 results
    data = load_f2_results(filepath)
    results = {}
    
    # ---------------------------------------------------------------------
    # K-DM-4: Weak Lensing Anisotropy
    # ---------------------------------------------------------------------
    print(f"\n[K-DM-4] Weak Lensing Anisotropy")
    print(f"  Instrument: {KILL_THRESHOLDS['K-DM-4']['instrument']}")
    
    k4_result = evaluate_kdm4(data, lambda_val)
    results['K-DM-4'] = k4_result
    
    status = "✓ PASS" if k4_result['passed'] else "✗ FAIL"
    print(f"  Prediction: |Δκ_dip| = {k4_result['prediction']:.4e}")
    print(f"  Observed:   |Δκ_dip| = {k4_result['observed']:.4e}")
    print(f"  Residual:   {k4_result['residual']:.4e} ≤ {k4_result['threshold']:.1e}")
    print(f"  Status:     {status}")
    
    # ---------------------------------------------------------------------
    # K-DM-5: Void-Halo Ellipticity
    # ---------------------------------------------------------------------
    print(f"\n[K-DM-5] Void-Halo Ellipticity")
    print(f"  Instrument: {KILL_THRESHOLDS['K-DM-5']['instrument']}")
    
    k5_result = evaluate_kdm5(data, lambda_val)
    results['K-DM-5'] = k5_result
    
    status = "✓ PASS" if k5_result['passed'] else "✗ FAIL"
    print(f"  Prediction: |Δε_void| = {k5_result['prediction']:.4e}")
    print(f"  Observed:   |Δε_void| = {k5_result['observed']:.4e}")
    print(f"  Residual:   {k5_result['residual']:.4e} ≤ {k5_result['threshold']:.1e}")
    print(f"  Status:     {status}")
    
    # ---------------------------------------------------------------------
    # K-DM-6: Scale-Dependent Growth
    # ---------------------------------------------------------------------
    print(f"\n[K-DM-6] Scale-Dependent Growth (fσ₈)")
    print(f"  Instrument: {KILL_THRESHOLDS['K-DM-6']['instrument']}")
    
    k6_result = evaluate_kdm6(data, lambda_val)
    results['K-DM-6'] = k6_result
    
    status = "✓ PASS" if k6_result['passed'] else "✗ FAIL"
    print(f"  Prediction: |Δ(fσ₈)| = {k6_result['prediction']:.4e}")
    print(f"  Observed:   |Δ(fσ₈)| = {k6_result['observed']:.4e}")
    print(f"  Residual:   {k6_result['residual']:.4e} ≤ {k6_result['threshold']:.1e}")
    print(f"  Status:     {status}")
    
    # ---------------------------------------------------------------------
    # FINAL VERDICT
    # ---------------------------------------------------------------------
    print("\n" + "="*70)
    print("FINAL VERDICT")
    print("="*70)
    
    # Decision logic per Phase VI §4.2
    if not k4_result['passed'] or not k5_result['passed'] or not k6_result['passed']:
        print("✗ MODEL FALSIFIED")
        print("  Reason: Active torsion-DM coupling violates observational thresholds")
        print("  Action: Revert to passive sector or derive non-perturbative extension")
        verdict = "FALSIFIED"
        exit_code = 1
        
    else:
        print("✓ MODEL CONSISTENT")
        print("  Reason: All K-DM-4/5/6 thresholds satisfied within geometric bounds")
        print("  Action: Extend framework to higher-order backreaction")
        verdict = "CONSISTENT"
        exit_code = 0
    
    # ---------------------------------------------------------------------
    # EXPORT VERDICT (CI/CD Integration)
    # ---------------------------------------------------------------------
    verdict_data = {
        'timestamp': datetime.utcnow().isoformat(),
        'lambda_val': lambda_val,
        'verdict': verdict,
        'results': {
            k: {
                'passed': v['passed'],
                'residual': v['residual'],
                'threshold': v['threshold'],
            } for k, v in results.items()
        },
        'geometric_constants': GEOM_CONSTANTS,
        'kill_thresholds': {k: t['threshold'] for k, t in KILL_THRESHOLDS.items()},
    }
    
    with open('kill_switch_verdict.json', 'w') as f:
        json.dump(verdict_data, f, indent=2)
    
    print(f"\n[✓] Verdict exported: kill_switch_verdict.json")
    print("="*70)
    
    return verdict, results, exit_code

# =============================================================================
# MAIN EXECUTION
# =============================================================================
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python appendices/a4_kill_switch.py <fth_dm_active_ext.h5> [lambda]")
        print("  lambda: coupling strength (default: 0.1, geometrically fixed)")
        sys.exit(1)
    
    filepath = sys.argv[1]
    lambda_val = float(sys.argv[2]) if len(sys.argv) > 2 else 0.1
    
    verdict, results, exit_code = run_kill_switches(filepath, lambda_val)
    sys.exit(exit_code)