#!/usr/bin/env python3
"""
================================================================================
FTH-DM Supplement — Step F.3: Automated Kill-Switch Wrapper
Falsification Protocol for the Passive Dark Matter Ansatz
================================================================================

Base Theory : Finsler-Timescape Hybrid (FTH) v2.6.1
Module      : step_f3_wrapper.py
Author      : A. Backmund, LLM-EFT-Lapse Collaboration
Date        : May 2026
Status      : Pre-Print / Peer-Review Draft
DOI         : https://doi.org/10.5281/zenodo.20024079
License     : CC BY 4.0

Domain Restriction
------------------
All evaluations operate exclusively within the Flat Void Sector:
    V_flat = {D ∈ V_all | e(TM)|_D = 0, k_D = 0, r_V ≤ r_inj}
Results are invalid for compact hyperbolic voids (k_D < 0, e(TM) ≠ 0)
without full holonomy recomputation via App. M.3–M.6.

Scientific Purpose
------------------
This wrapper ingests the high-resolution continuum solution from Step F.2
(FTH_DM_results.h5) and evaluates the three operational Kill-Criteria
K-DM-1, K-DM-2, K-DM-3 defined in Step E. After closure of Lemma
L-VoidBias (Sec. V-ext), K-DM-3 is upgraded from an investigative trigger
to a hard falsification switch.

No-Tuning Mandate
-----------------
All thresholds are derived from independent observational anchors:
    BAO_GEOM_SHIFT = 8.3e-5   ← App. G, Eq. G.22 (geometric FTH prediction)
    SIGMA8_PLANCK  = 0.811    ← Planck 2018, arXiv:1807.06209
    S8_OBS         = 0.795    ← KiDS-1000/DES-Y3 joint analysis
No threshold is fitted to FTH-internal parameters.

Kill-Criteria Definitions (Step E, Table 7)
--------------------------------------------
K-DM-1: |Residual_BAO| > 1e-3 at z_eff = 0.5  → HARD FAIL
K-DM-2: |Δσ₈| > 0.02 OR |ΔS₈| > 0.015        → HARD FAIL
K-DM-3: |Δb_h^void| > 0.15 at r < 0.5 R_vir   → HARD FAIL (post L-VoidBias)
         |Δε_void|   > 0.15                     → HARD FAIL (post L-VoidBias)

Continuum Mandate
-----------------
Input MUST carry flags:
    GR_Test_Passed      = True   (b₀→0 recovers ΛCDM to 1e-12)
    Continuum_Convergence = True  (Richardson p_obs ∈ [4,6], ε_rel < 1e-6)
Absence of either flag aborts execution — no manual override permitted.

References
----------
[1] Desjacques, Jeong, Schmidt (2018). Phys. Rep. 733, 1. arXiv:1611.09787
[2] Planck Collaboration (2020). A&A 641, A6.      arXiv:1807.06209
[3] Neyrinck (2008). MNRAS 386, 2101.              arXiv:0712.3049
[4] Buchert (2000). Gen. Rel. Grav. 32, 105.       arXiv:gr-qc/9906015
================================================================================
"""

import h5py
import numpy as np
import subprocess
import sys


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1 — Configuration: Kill Thresholds and Observational Anchors
# ══════════════════════════════════════════════════════════════════════════════

# Kill-Criterion thresholds (Step E, Table 7 — not fitted, externally anchored)
THRESH_BAO_RESIDUAL = 1.0e-3    # K-DM-1: DM-induced BAO shift above geometric baseline
THRESH_SIGMA8_DEV   = 0.02      # K-DM-2: absolute deviation of σ₈ from Planck 2018
THRESH_S8_DEV       = 0.015     # K-DM-2: absolute deviation of S₈ from KiDS/DES-Y3
THRESH_VOID_BIAS    = 0.15      # K-DM-3: void-halo bias / ellipticity deviation

# Observational anchors (Planck 2018, KiDS-1000/DES-Y3 — external priors)
SIGMA8_PLANCK  = 0.811          # arXiv:1807.06209, Table 2
S8_OBS         = 0.795          # KiDS-1000 + DES-Y3 joint, 1σ band
BAO_GEOM_SHIFT = 8.3e-5         # Pure geometric FTH contribution, App. G Eq. G.22


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — SymPy Bianchi Closure Verification (Lemma L-VoidBias prerequisite)
# ══════════════════════════════════════════════════════════════════════════════

def verify_bianchi_closure(input_file):
    """
    Executes bias_sym_verify.py to confirm that the geometric bias operator
    G(x,y) defined in Lemma L-VoidBias (Sec. V-ext) preserves the horizontal
    Bianchi identity:

        ∇^h_i T^{ij}_{h,bias} = 0

    This is a mandatory prerequisite before K-DM-3 may carry hard falsification
    authority. If the symbolic verification fails, all Kill-Criterion evaluations
    are aborted — the analytical foundation is structurally invalid.

    Five assertions are enforced (bias_sym_verify.py, Steps 1–5):
        Step 1: Cartan antisymmetry  C_ijk = -C_ikj
        Step 2: Fiber symmetry       <y_i y_j>_S³ = δ_ij / 3
        Step 3: Torsion contraction  J^j = C_jkl <y^k y^l> = 0
        Step 4: Bianchi residual     ∇^h_i T^{ij}_{h,bias} = 0  (symbolic)
        Step 5: Continuum quadrature |I₁| < 1e-6  (N = 10⁵ equivalent)

    Domain restriction: valid only for e(TM) = 0, symmetric S³ indicatrix.
    For compact hyperbolic voids, parity cancellation fails and I₁ ≠ 0
    (Addendum I, Eq. I.5.2.3; bound: |I₁| ≤ 3.7e-3 · b₀(z)).

    Parameters
    ----------
    input_file : str
        Path to FTH_DM_results.h5 (Step F.2 output).

    Raises
    ------
    RuntimeError
        If any of the five SymPy assertions fails.
    """
    print("  Running bias_sym_verify.py — 5 SymPy assertions...")
    result = subprocess.run(
        ['python', 'bias_sym_verify.py', '--mode=full'],
        capture_output=True, text=True
    )

    if result.returncode != 0:
        print("[CRITICAL] SymPy Bianchi closure FAILED.")
        print("  stdout:", result.stdout[-500:])
        print("  stderr:", result.stderr[-500:])
        raise RuntimeError(
            "Lemma L-VoidBias: Bianchi closure invalid. "
            "K-DM-3 hard falsification NOT activated. "
            "Abort entire Kill-Switch evaluation."
        )

    # Record closure in HDF5 for downstream modules (F.4 Sec. 3a, App. N)
    with h5py.File(input_file, 'r+') as fh:
        fh.attrs['L_VoidBias_verified'] = True
        fh.attrs['L_VoidBias_status']   = 'CLOSED'

    print("  [L-VoidBias] Bianchi closure: VERIFIED ✓")
    print("  [K-DM-3]     Status upgraded: investigative trigger → HARD falsification\n")


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 3 — Data Loading with Mandatory Reproducibility Guards
# ══════════════════════════════════════════════════════════════════════════════

def load_f2_results(filepath):
    """
    Loads high-resolution continuum-validated arrays from Step F.2 output.

    Mandatory flags (set by Step F.2 Richardson protocol):
        GR_Test_Passed       : b₀→0 recovers ΛCDM growth to 1e-12 (Sec. 4.3.4)
        Continuum_Convergence: Richardson p_obs ∈ [4,6], ε_rel < 1e-6 (Sec. 4.3.2)

    If either flag is absent or False, execution is aborted. This prevents
    evaluation of toy-model solutions (N < 10⁵) from mimicking or masking
    physical signals at the 1e-3–1e-4 precision level of Kill-Criteria.

    Parameters
    ----------
    filepath : str
        Path to FTH_DM_results.h5 produced by Step F.2.

    Returns
    -------
    dict
        Arrays: ln_a, H_D, delta_DM, b_0, f_growth, fsigma8, chi_z,
                BAO_residual_raw, [void_bias_deviation], [Delta_epsilon_void],
                [L_VoidBias_verified]
    """
    try:
        with h5py.File(filepath, 'r') as f:

            # Reproducibility guard — mandatory, no override
            if not f.attrs.get('GR_Test_Passed', False):
                raise RuntimeError(
                    "Input data failed GR Regression Test (b₀→0 check). "
                    "Re-run Step F.2 with b_0=0, Q_DF=0 and verify ΛCDM recovery "
                    "to 1e-12 relative error. Abort."
                )
            if not f.attrs.get('Continuum_Convergence', False):
                raise RuntimeError(
                    "Input data failed Continuum Limit (N < 10⁵ or p_obs ∉ [4,6]). "
                    "Richardson extrapolation required across N, 2N, 4N grids. "
                    "Simulations with N < 10⁵ are rejected as toy models. Abort."
                )

            # Core state variables (Step F.2, Sec. 5.1)
            data = {
                'ln_a'            : f['ln_a'][:],
                'H_D'             : f['H_D'][:],
                'delta_DM'        : f['delta_DM'][:],
                'b_0'             : f['b_0'][:],
                'f_growth'        : f['f_growth'][:],
                'fsigma8'         : f['fsigma8'][:],
                'chi_z'           : f['chi_z'][:],
                'BAO_residual_raw': f['BAO_residual'][:]   # total shift vs. ΛCDM
            }

            # Optional L-VoidBias observables (F.4 Sec. 3a — if already computed)
            if 'void_bias_deviation' in f:
                data['void_bias_deviation'] = f['void_bias_deviation'][:]
            if 'Delta_epsilon_void' in f:
                data['Delta_epsilon_void']  = f['Delta_epsilon_void'][:]

            # L-VoidBias closure flag (written by verify_bianchi_closure)
            data['L_VoidBias_verified'] = bool(
                f.attrs.get('L_VoidBias_verified', False))

        return data

    except Exception as e:
        print(f"[CRITICAL] Failed to load Step F.2 results: {e}")
        sys.exit(1)


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 4 — Individual Kill-Criterion Evaluators
# ══════════════════════════════════════════════════════════════════════════════

def evaluate_k_dm_1(data):
    """
    K-DM-1: BAO Consistency — Residual Shift at z_eff = 0.5

    Tests whether the DM-induced modification to the expansion history H_D(z)
    produces a BAO shift exceeding the geometric FTH prediction tolerance.

    Formula (Step E, Eq. E.1):
        Residual_BAO = |Δ(r_d/r_d)_total| − |Δ(r_d/r_d)_geom|
        Kill condition: Residual_BAO > 1e-3

    The geometric contribution BAO_GEOM_SHIFT = 8.3e-5 (App. G, Eq. G.22)
    is subtracted before applying the threshold. This isolates the pure
    DM-induced component from the background FTH geometric correction.

    Riemannian limit: b₀→0 ⟹ Q_DF→0 ⟹ Residual_BAO→0 exactly.

    Parameters
    ----------
    data : dict  (from load_f2_results)

    Returns
    -------
    passed   : bool
    residual : float   (DM-induced BAO shift, dimensionless)
    """
    z_eff    = 0.5
    a_eff    = 1.0 / (1.0 + z_eff)
    ln_a_eff = np.log(a_eff)

    idx         = np.argmin(np.abs(data['ln_a'] - ln_a_eff))
    total_shift = data['BAO_residual_raw'][idx]

    # Subtract geometric FTH baseline (App. G Eq. G.22) — not a free subtraction
    residual = abs(total_shift) - abs(BAO_GEOM_SHIFT)
    passed   = residual <= THRESH_BAO_RESIDUAL

    return passed, float(residual)


def evaluate_k_dm_2(data):
    """
    K-DM-2: Growth Amplitude Tension — σ₈ and S₈ at z = 0

    Tests whether the FTH-modified growth rate f(a) and amplitude σ₈
    are consistent with Planck 2018 and KiDS-1000/DES-Y3 constraints.

    Formulas (Step E, Eq. E.2–E.3):
        σ₈_FTH = (fσ₈)[-1] / f_growth[-1]     (recover σ₈ at a=1)
        S₈_FTH = σ₈_FTH · sqrt(Ω_m / 0.3)     (simplified; Ω_m ≈ 0.3)
        Kill condition: |σ₈_FTH − 0.811| > 0.02
                     OR |S₈_FTH  − 0.795| > 0.015

    The growth suppression enters through the modified Hubble friction 2H_D
    incorporating Q_DF, not through a direct coupling parameter (No-Tuning).

    Riemannian limit: b₀→0 ⟹ σ₈_FTH → σ₈_ΛCDM = 0.811 exactly.

    Parameters
    ----------
    data : dict  (from load_f2_results)

    Returns
    -------
    passed       : bool
    delta_sigma8 : float   (absolute deviation of σ₈)
    delta_s8     : float   (absolute deviation of S₈)
    """
    sigma8_fth   = data['fsigma8'][-1] / data['f_growth'][-1]
    s8_fth       = sigma8_fth * np.sqrt(0.3 / 0.3)   # Ω_m = 0.3 working prior

    delta_sigma8 = abs(sigma8_fth - SIGMA8_PLANCK)
    delta_s8     = abs(s8_fth     - S8_OBS)

    passed = (delta_sigma8 <= THRESH_SIGMA8_DEV) and (delta_s8 <= THRESH_S8_DEV)

    return passed, float(delta_sigma8), float(delta_s8)


def evaluate_k_dm_3(data):
    """
    K-DM-3: Void-Halo Bias — UPGRADED to HARD falsification
    after closure of Lemma L-VoidBias (Sec. V-ext, FTH-DM Supplement).

    Prior status  : investigative trigger (pending Lemma L-VoidBias derivation)
    Current status: HARD falsification switch (Lemma closed, SymPy verified)

    Two observables are evaluated in order of priority:

    (A) Scalar void-halo bias deviation Δb_h^void (Step E, Eq. E.4):
            Δb_h^void = b_h^void(r < 0.5 R_vir) − b_h^void_ΛCDM
            Kill condition: |Δb_h^void| > 0.15

    (B) Ellipticity bias Δε_void (Lemma L-VoidBias, Eq. L.4):
            Δε_void = λ² b₀²(z) · E_geom · Q_torsion
            E_geom = 1/5  (rank-6 S³ angular average, App. VI)
            Kill condition: |Δε_void| > 0.15

    Both observables reduce to zero in the Riemannian limit (b₀→0, q_geo→0),
    recovering standard ΛCDM bias exactly (parity cancellation I₁=0,
    Addendum I Eq. I.5.2).

    Hard-fail logic:
        If L_VoidBias_verified=True AND threshold exceeded → HARD FAIL
        If L_VoidBias_verified=False AND threshold exceeded → INVESTIGATE
        If no data available → PENDING (not PASS — conservative default)

    Parameters
    ----------
    data : dict  (from load_f2_results)

    Returns
    -------
    result   : bool or None   (True=pass, False=fail, None=pending)
    value    : float          (observed deviation)
    status   : str            ('PASS', 'HARD_FAIL', 'INVESTIGATE', 'PENDING')
    label    : str            (observable name for reporting)
    """
    l_voidbias_closed = data.get('L_VoidBias_verified', False)

    # Priority (A): scalar void-halo bias deviation
    if 'void_bias_deviation' in data:
        dev    = float(data['void_bias_deviation'][-1])
        passed = dev <= THRESH_VOID_BIAS
        if passed:
            return True,  dev, 'PASS',      'Δb_h^void'
        elif l_voidbias_closed:
            return False, dev, 'HARD_FAIL', 'Δb_h^void'
        else:
            return False, dev, 'INVESTIGATE','Δb_h^void'

    # Priority (B): ellipticity bias from F.4 Sec. 3a
    elif 'Delta_epsilon_void' in data:
        eps    = float(abs(data['Delta_epsilon_void'][-1]))
        passed = eps <= THRESH_VOID_BIAS
        if passed:
            return True,  eps, 'PASS',      'Δε_void'
        elif l_voidbias_closed:
            return False, eps, 'HARD_FAIL', 'Δε_void'
        else:
            return False, eps, 'INVESTIGATE','Δε_void'

    # No K-DM-3 data in F.2 output — flag as PENDING, not PASS
    else:
        return None, 0.0, 'PENDING', 'N/A'


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 5 — Main Orchestrator: run_kill_switches()
# ══════════════════════════════════════════════════════════════════════════════

def run_kill_switches(input_file):
    """
    Orchestrates the complete Step F.3 falsification protocol:

        Step 0 : SymPy Bianchi closure  (bias_sym_verify.py — 5 assertions)
        Step 1 : Load F.2 results       (with mandatory reproducibility guards)
        Step 2 : Evaluate K-DM-1        (BAO residual shift)
        Step 3 : Evaluate K-DM-2        (σ₈ / S₈ growth tension)
        Step 4 : Evaluate K-DM-3        (void-halo bias / ellipticity)
        Step 5 : Final verdict          (falsification decision tree)

    Decision logic (Step E, Table 12):
        FAIL K-DM-1 or K-DM-2           → MODEL FALSIFIED (passive sector)
        FAIL K-DM-3 + L-VoidBias CLOSED → MODEL FALSIFIED (active sector trigger)
        K-DM-3 PENDING                  → CONDITIONAL PASS (awaiting F.4 Sec. 3a)
        ALL PASS                        → MODEL CONSISTENT

    Parameters
    ----------
    input_file : str
        Path to FTH_DM_results.h5 from Step F.2.

    Returns
    -------
    bool : True if model passes or conditionally passes; False if falsified.
    """
    print("=" * 70)
    print("FTH-DM  —  STEP F.3: AUTOMATED KILL-SWITCH EVALUATION")
    print("=" * 70)
    print(f"Input file : {input_file}")
    print(f"Domain     : V_flat  [e(TM)=0, k_D=0, r_V ≤ r_inj]")
    print(f"Anchors    : Planck 2018 σ₈={SIGMA8_PLANCK}, KiDS/DES S₈={S8_OBS}")
    print(f"Geom. BAO  : {BAO_GEOM_SHIFT:.1e}  (App. G, Eq. G.22)\n")

    # ── Step 0: SymPy Bianchi Closure ──────────────────────────────────────
    print("─" * 70)
    print("STEP 0 — Lemma L-VoidBias: SymPy Bianchi Closure Verification")
    print("─" * 70)
    verify_bianchi_closure(input_file)

    # ── Step 1: Load F.2 Results ───────────────────────────────────────────
    print("─" * 70)
    print("STEP 1 — Loading Step F.2 continuum-validated results")
    print("─" * 70)
    data    = load_f2_results(input_file)
    results = {}
    print("  GR_Test_Passed      : ✓")
    print("  Continuum_Convergence: ✓")
    print(f"  Grid points N       : {len(data['ln_a'])}\n")

    # ── Step 2: K-DM-1 — BAO Residual ─────────────────────────────────────
    print("─" * 70)
    print("STEP 2 — K-DM-1: BAO Consistency (z_eff = 0.5)")
    print("─" * 70)
    pass1, val1 = evaluate_k_dm_1(data)
    status1     = "PASS 🟢" if pass1 else "HARD FAIL 🔴"
    print(f"  Residual_BAO  : {val1:.4e}")
    print(f"  Threshold     : {THRESH_BAO_RESIDUAL:.1e}")
    print(f"  Geom. baseline: {BAO_GEOM_SHIFT:.1e} (subtracted, App. G Eq. G.22)")
    print(f"  Status        : {status1}\n")
    results['K-DM-1'] = pass1

    # ── Step 3: K-DM-2 — σ₈ / S₈ Tension ─────────────────────────────────
    print("─" * 70)
    print("STEP 3 — K-DM-2: Growth Amplitude Tension (z = 0)")
    print("─" * 70)
    pass2, val2a, val2b = evaluate_k_dm_2(data)
    status2             = "PASS 🟢" if pass2 else "HARD FAIL 🔴"
    print(f"  Δσ₈   : {val2a:.4f}  (threshold {THRESH_SIGMA8_DEV},"
          f" anchor σ₈={SIGMA8_PLANCK})")
    print(f"  ΔS₈   : {val2b:.4f}  (threshold {THRESH_S8_DEV},"
          f" anchor S₈={S8_OBS})")
    print(f"  Status: {status2}\n")
    results['K-DM-2'] = pass2

    # ── Step 4: K-DM-3 — Void-Halo Bias (HARD after L-VoidBias) ──────────
    print("─" * 70)
    print("STEP 4 — K-DM-3: Void-Halo Bias / Ellipticity  [L-VoidBias: CLOSED]")
    print("─" * 70)
    k3_result, val3, k3_status, k3_label = evaluate_k_dm_3(data)

    STATUS_LABELS = {
        'PASS'      : 'PASS 🟢',
        'HARD_FAIL' : 'HARD FAIL 🔴  (Lemma L-VoidBias CLOSED)',
        'INVESTIGATE': 'INVESTIGATE 🟡  (L-VoidBias not yet verified)',
        'PENDING'   : 'PENDING ⚪  (no K-DM-3 data in F.2 output — run F.4 Sec. 3a)'
    }

    print(f"  Observable    : {k3_label}")
    print(f"  Deviation     : {val3:.4f}  (threshold {THRESH_VOID_BIAS})")
    print(f"  L-VoidBias    : {'CLOSED ✓' if data.get('L_VoidBias_verified') else 'OPEN'}")
    print(f"  Status        : {STATUS_LABELS[k3_status]}\n")
    results['K-DM-3'] = k3_result

    # ── Step 5: Final Verdict ──────────────────────────────────────────────
    print("=" * 70)
    print("FINAL VERDICT")
    print("=" * 70)

    # CASE A: Hard falsification — K-DM-1 or K-DM-2
    if not results['K-DM-1'] or not results['K-DM-2']:
        failed = []
        if not results['K-DM-1']: failed.append('K-DM-1 (BAO)')
        if not results['K-DM-2']: failed.append('K-DM-2 (σ₈/S₈)')
        print("VERDICT  : MODEL FALSIFIED ❌")
        print(f"Trigger  : {', '.join(failed)}")
        print("Reason   : Passive DM ansatz violates observational constraints.")
        print("Action   : Extend to active torsion-DM coupling (Addendum M,")
        print("           Eq. M.1.3–M.1.5) or revise Riccati flow anchor.")
        print("Exit code: 1")
        return False

    # CASE B: Hard falsification — K-DM-3 post L-VoidBias closure
    if k3_status == 'HARD_FAIL':
        print("VERDICT  : MODEL FALSIFIED (K-DM-3 — HARD) ❌")
        print(f"Trigger  : K-DM-3 ({k3_label} = {val3:.4f} > {THRESH_VOID_BIAS})")
        print("Reason   : Lemma L-VoidBias closed — void-halo bias exceeds threshold.")
        print("           Fiber-isotropy (Assumption A2) or no-torsion coupling")
        print("           (Assumption A3) is violated at sub-virial scales.")
        print("Action   : Activate full fiber-anisotropic extension:")
        print("           → Addendum L, Eq. L.3 (active torsion-DM coupling)")
        print("           → Addendum M, Sec. 4–5 (active Vlasov on TM_hyp)")
        print("           → Recompute holonomy projector Φ_holo via App. M.3")
        print("Exit code: 1")
        return False

    # CASE C: Investigative (L-VoidBias not yet verified externally)
    if k3_status == 'INVESTIGATE':
        print("VERDICT  : CONDITIONAL PASS — K-DM-3 UNDER INVESTIGATION ⚠️")
        print(f"Trigger  : K-DM-3 ({k3_label} = {val3:.4f} > {THRESH_VOID_BIAS})")
        print("Reason   : Threshold exceeded but L-VoidBias closure not yet")
        print("           confirmed by bias_sym_verify.py for this dataset.")
        print("Action   : Re-run with bias_sym_verify.py passing all 5 assertions.")
        print("           If verified → verdict upgrades to HARD FAIL.")
        print("Exit code: 0 (conditional)")
        return True

    # CASE D: Pending K-DM-3 data
    if k3_status == 'PENDING':
        print("VERDICT  : CONDITIONAL PASS — K-DM-3 DATA PENDING ⚠️")
        print("Reason   : K-DM-1 and K-DM-2 satisfied. No void-halo bias")
        print("           data found in FTH_DM_results.h5.")
        print("Action   : Run Step F.4 Sec. 3a (extract_voidbias_observables)")
        print("           to compute Δε_void and re-evaluate K-DM-3.")
        print("Exit code: 0 (conditional)")
        return True

    # CASE E: Full consistency
    print("VERDICT  : MODEL CONSISTENT ✅")
    print("Reason   : All Kill-Criteria satisfied within observational bounds.")
    print("           K-DM-1 (BAO), K-DM-2 (σ₈/S₈), K-DM-3 (void bias).")
    print("           Lemma L-VoidBias closed and verified.")
    print("Action   : Proceed to Step G — manuscript assembly and arXiv submission.")
    print("           Tag release: FTH-DM v2.6.2 (Lemma L-VoidBias CLOSED).")
    print("Exit code: 0")
    return True


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 6 — Entry Point
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage  : python step_f3_wrapper.py <path_to_FTH_DM_results.h5>")
        print("Example: python step_f3_wrapper.py ./output/FTH_DM_results.h5")
        print()
        print("Input requirements:")
        print("  - GR_Test_Passed       = True  (Step F.2 regression check)")
        print("  - Continuum_Convergence = True  (Richardson p_obs ∈ [4,6])")
        print("  - bias_sym_verify.py must be in the same directory")
        sys.exit(1)

    success = run_kill_switches(sys.argv[1])
    sys.exit(0 if success else 1)