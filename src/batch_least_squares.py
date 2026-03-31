"""
01_batch_least_squares.py
==========================
Iterative Batch Least Squares (BLS) orbit determination algorithm.

Author : Giuliano Pennacchio
Version: 1.0

Theory
------
The BLS algorithm minimises the weighted sum of squared residuals:

    J(x0) = sum_i [ w_rng * (rho_i - rho_hat_i(x0))^2
                  + w_rr  * (rho_dot_i - rho_dot_hat_i(x0))^2 ]

where rho_hat and rho_dot_hat are the computed (predicted) range and
range-rate obtained by propagating x0 to the observation time t_i.

Because the measurement model is nonlinear, the solution is found
iteratively (Differential Correction):

    (1)  Propagate state x_k and STM Phi(t_i, t_0) forward.
    (2)  Evaluate measurement partials:
             H_0 = H_local(t_i) @ Phi(t_i, t_0)
         where H_local are the analytic partials w.r.t. the state at t_i.
    (3)  Build the normal equations:
             Lambda = sum_i H_0^T W H_0        (information matrix)
             N      = sum_i H_0^T W (y_i - y_hat_i)
    (4)  Solve for the correction:
             dx = Lambda^{-1} N
    (5)  Update:  x_{k+1} = x_k + dx
    (6)  Repeat until convergence.

The correction is normalised (each H row divided by its sigma) to
improve conditioning before the lstsq solve. The covariance matrix is:
    P = (H_norm^T H_norm)^{-1}   [physical units: km^2, km^2/s^2]

Initial perturbation
--------------------
A realistic a priori uncertainty of 100 m / 0.1 m/s is added to the
true initial state, representing a typical TLE-derived initial guess.

Outputs
-------
data/bls_result.npz
    Keys: x_true, x_initial, x_estimated, P_covariance, iterations,
          residuals_prefit_range, residuals_prefit_rr,
          residuals_postfit_range, residuals_postfit_rr,
          obs_times, obs_stations, sigmas

data/bls_summary.txt
    Convergence table, final errors, covariance diagonal.

Run
---
    python 01_batch_least_squares.py
"""

import os
import time
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d

from od_utils import (
    SIGMA_RANGE, SIGMA_RANGERATE,
    f_j2, eom_scipy, propagate_truth, propagate_with_stm,
    lla_to_ecef, compute_observation, measurement_partials,
)

# ---------------------------------------------------------------------------
# BLS configuration
# ---------------------------------------------------------------------------
MAX_ITERATIONS  = 10
CONV_THRESHOLD  = 1e-5   # km — convergence criterion on |dx_position|

# Initial state perturbation (simulates a priori uncertainty)
# 100 m in position, 0.1 m/s in velocity — typical TLE-level uncertainty
np.random.seed(0)   # fixed for reproducibility of the perturbation
PERTURBATION = np.array([0.100, -0.050,  0.080,       # km
                          0.0001, -0.0001, 0.00005])   # km/s

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")


# ===========================================================================
# Core BLS functions
# ===========================================================================

def build_normal_equations(
    x_est:     np.ndarray,
    obs_df:    pd.DataFrame,
    gs_lookup: dict,
    t_eval:    np.ndarray,
) -> tuple:
    """
    Propagate x_est, compute H @ Phi for all observations, and
    accumulate the normalised normal equations.

    Parameters
    ----------
    x_est    : ndarray (6,)         Current state estimate [km, km/s].
    obs_df   : DataFrame            Observations (time_s, station, range_km, rangerate_kms).
    gs_lookup: dict {str: ndarray}  Station ECEF positions [km].
    t_eval   : ndarray              Epoch grid [s].

    Returns
    -------
    H_norm   : ndarray (2*N, 6)    Sigma-normalised design matrix.
    y_norm   : ndarray (2*N,)      Sigma-normalised residuals.
    res_range: ndarray (N,)        Raw range residuals [km].
    res_rr   : ndarray (N,)        Raw range-rate residuals [km/s].
    obs_t    : ndarray (N,)        Observation times [s].
    obs_gs   : list                Observation station names.
    sol      : ODE solution        Propagated trajectory (with STM).
    """
    # Propagate state + STM over the full arc
    sol = propagate_with_stm(x_est, t_eval)

    if sol.status != 0:
        raise RuntimeError(f"STM propagation failed: {sol.message}")

    tp = sol.t   # time points actually used by the integrator

    # Build interpolators for state (6) and STM (36 = 6x6 flattened)
    # Using linear interpolation for STM (smoother than cubic for a matrix)
    i_state = [
        interp1d(tp, sol.y[i],   kind="cubic", bounds_error=False, fill_value="extrapolate")
        for i in range(6)
    ]
    i_stm = [
        [
            interp1d(tp, sol.y[6 + i * 6 + j], kind="linear",
                     bounds_error=False, fill_value="extrapolate")
            for j in range(6)
        ]
        for i in range(6)
    ]

    H_rows_range = []
    H_rows_rr    = []
    y_range_raw  = []
    y_rr_raw     = []
    obs_t        = []
    obs_gs_list  = []

    for _, row in obs_df.iterrows():
        t_o  = row["time_s"]
        name = row["station"]
        if t_o > tp[-1]:
            continue

        ge = gs_lookup[name]

        # Interpolate state at observation epoch
        r_c = np.array([i_state[i](t_o) for i in range(3)])
        v_c = np.array([i_state[i](t_o) for i in range(3, 6)])

        # Interpolate STM at observation epoch
        phi = np.array([[i_stm[i][j](t_o) for j in range(6)] for i in range(6)])

        # Analytic H mapped to epoch 0 via STM, plus computed observations
        H_r, H_rr, rng_c, rr_c = measurement_partials(r_c, v_c, ge, t_o, phi)

        # Residuals: observed minus computed
        yr  = row["range_km"]      - rng_c
        yrr = row["rangerate_kms"] - rr_c

        # Normalise rows by sigma (makes the normal equations well-conditioned)
        H_rows_range.append(H_r  / SIGMA_RANGE)
        H_rows_rr.append(   H_rr / SIGMA_RANGERATE)
        y_range_raw.append(yr)
        y_rr_raw.append(yrr)
        obs_t.append(t_o)
        obs_gs_list.append(name)

    H_norm = np.vstack([
        np.array(H_rows_range),
        np.array(H_rows_rr),
    ])                                             # shape (2*N, 6)
    y_norm = np.concatenate([
        np.array(y_range_raw) / SIGMA_RANGE,
        np.array(y_rr_raw)    / SIGMA_RANGERATE,
    ])                                             # shape (2*N,)

    return (H_norm, y_norm,
            np.array(y_range_raw), np.array(y_rr_raw),
            np.array(obs_t), obs_gs_list, sol)


def compute_postfit_residuals(
    x_est:     np.ndarray,
    obs_df:    pd.DataFrame,
    gs_lookup: dict,
    t_eval:    np.ndarray,
) -> tuple:
    """
    Compute post-fit residuals using the final estimated state.

    This function propagates x_est WITHOUT the STM (faster) and
    evaluates the measurement model at each observation epoch.

    Returns
    -------
    res_range : ndarray (N,)   Post-fit range residuals [km].
    res_rr    : ndarray (N,)   Post-fit range-rate residuals [km/s].
    """
    from scipy.integrate import solve_ivp
    sol = propagate_truth(x_est, t_eval)
    i_state = [
        interp1d(sol.t, sol.y[i], kind="cubic",
                 bounds_error=False, fill_value="extrapolate")
        for i in range(6)
    ]

    res_range = []
    res_rr    = []
    for _, row in obs_df.iterrows():
        t_o  = row["time_s"]
        name = row["station"]
        if t_o > sol.t[-1]:
            continue
        r_c = np.array([i_state[i](t_o) for i in range(3)])
        v_c = np.array([i_state[i](t_o) for i in range(3, 6)])
        rng_c, rr_c, _ = compute_observation(r_c, v_c, gs_lookup[name], t_o)
        res_range.append(row["range_km"]      - rng_c)
        res_rr.append(   row["rangerate_kms"] - rr_c)

    return np.array(res_range), np.array(res_rr)


# ===========================================================================
# Main
# ===========================================================================

def main() -> None:
    os.makedirs(DATA_DIR, exist_ok=True)

    # -----------------------------------------------------------------------
    # Load observations and ground truth
    # -----------------------------------------------------------------------
    obs_path = os.path.join(DATA_DIR, "observations.csv")
    gt_path  = os.path.join(DATA_DIR, "ground_truth.csv")

    if not os.path.exists(obs_path):
        raise FileNotFoundError(
            f"Observations file not found: {obs_path}\n"
            "  -> Run 00_parse_gmd.py first."
        )

    print("=" * 60)
    print("Project 2 — Batch OD: Iterative Least Squares")
    print("=" * 60)

    obs_df = pd.read_csv(obs_path)
    gt_df  = pd.read_csv(gt_path)

    # True initial state (read from ground truth at t=0)
    x_true = gt_df[["x_km","y_km","z_km","vx_kms","vy_kms","vz_kms"]].iloc[0].values

    # Build epoch grid matching the ground truth
    t_eval = gt_df["time_s"].values

    # Ground station lookup
    gs_lookup = {
        "Kiruna":   lla_to_ecef(67.86,  20.23, 0.04),
        "Svalbard": lla_to_ecef(78.23,  15.40, 0.50),
    }

    print(f"\n  Observations loaded : {len(obs_df)} total")
    print(f"  Arc                 : {t_eval[-1]/86400:.1f} days")
    print(f"  True initial state  : {x_true[:3]} km")

    # -----------------------------------------------------------------------
    # Set initial estimate (true + perturbation)
    # -----------------------------------------------------------------------
    x_initial = x_true + PERTURBATION
    x_est     = x_initial.copy()

    pos_err_init = np.linalg.norm(PERTURBATION[:3]) * 1000   # m
    vel_err_init = np.linalg.norm(PERTURBATION[3:]) * 1000   # m/s

    print(f"\n  Initial perturbation:")
    print(f"    Position error : {pos_err_init:.1f} m")
    print(f"    Velocity error : {vel_err_init:.4f} m/s")

    # -----------------------------------------------------------------------
    # Pre-fit residuals (before any correction)
    # -----------------------------------------------------------------------
    print("\n  Computing pre-fit residuals...")
    res_range_pre, res_rr_pre = compute_postfit_residuals(
        x_initial, obs_df, gs_lookup, t_eval
    )
    obs_times_arr   = obs_df["time_s"].values[:len(res_range_pre)]
    obs_station_arr = obs_df["station"].values[:len(res_range_pre)]

    print(f"    RMS range     (pre-fit): {np.std(res_range_pre)*1000:.2f} m")
    print(f"    RMS range-rate(pre-fit): {np.std(res_rr_pre)*1000:.4f} m/s")

    # -----------------------------------------------------------------------
    # BLS iterations
    # -----------------------------------------------------------------------
    print(f"\n{'─'*60}")
    print(f"  {'Iter':>4}  {'|dx_pos| (m)':>14}  {'pos_err (m)':>12}  "
          f"{'vel_err (m/s)':>14}  {'RMS_rng (m)':>12}")
    print(f"{'─'*60}")

    t_start   = time.time()
    converged = False
    iter_log  = []
    H_final   = None
    last_sol  = None

    for it in range(1, MAX_ITERATIONS + 1):
        # Build normal equations for current estimate
        H_norm, y_norm, res_r, res_rr, obs_t_it, obs_gs_it, sol_it = \
            build_normal_equations(x_est, obs_df, gs_lookup, t_eval)

        # Solve via SVD-based least squares (numerically stable)
        dx, _, _, _ = np.linalg.lstsq(H_norm, y_norm, rcond=None)

        # Apply correction to state estimate
        x_est     += dx
        H_final    = H_norm
        last_sol   = sol_it

        # Convergence metrics
        pos_err = np.linalg.norm(x_est[:3] - x_true[:3]) * 1000   # m
        vel_err = np.linalg.norm(x_est[3:] - x_true[3:]) * 1000   # m/s
        dx_norm = np.linalg.norm(dx[:3]) * 1000                    # m
        rms_rng = np.std(res_r) * 1000                             # m

        print(f"  {it:>4}  {dx_norm:>14.6f}  {pos_err:>12.6f}  "
              f"{vel_err:>14.8f}  {rms_rng:>12.4f}")

        iter_log.append({
            "iteration":    it,
            "dx_norm_m":    dx_norm,
            "pos_error_m":  pos_err,
            "vel_error_ms": vel_err,
            "rms_range_m":  rms_rng,
        })

        if dx_norm < CONV_THRESHOLD * 1000:   # CONV_THRESHOLD in km
            converged = True
            print(f"{'─'*60}")
            print(f"  Converged at iteration {it}  (|dx| = {dx_norm:.6f} m)")
            break

    elapsed = time.time() - t_start
    print(f"{'─'*60}")
    if not converged:
        print(f"  Warning: did not converge in {MAX_ITERATIONS} iterations.")
    print(f"  Elapsed : {elapsed:.1f} s")

    # -----------------------------------------------------------------------
    # Post-fit residuals
    # -----------------------------------------------------------------------
    print("\n  Computing post-fit residuals...")
    res_range_post, res_rr_post = compute_postfit_residuals(
        x_est, obs_df, gs_lookup, t_eval
    )
    print(f"    RMS range      (post-fit): {np.std(res_range_post)*1000:.4f} m  "
          f"(expected ~{SIGMA_RANGE*1000:.0f} m)")
    print(f"    RMS range-rate (post-fit): {np.std(res_rr_post)*1000:.6f} m/s  "
          f"(expected ~{SIGMA_RANGERATE*1000:.4f} m/s)")

    # -----------------------------------------------------------------------
    # Covariance matrix  P = (H^T H)^{-1}  [normalised space = physical units]
    # -----------------------------------------------------------------------
    Lambda = H_final.T @ H_final           # information matrix (normalised)
    P      = np.linalg.inv(Lambda)         # covariance matrix [km^2, (km/s)^2]

    sig_pos = np.sqrt(np.diag(P)[:3]) * 1000   # m
    sig_vel = np.sqrt(np.diag(P)[3:]) * 1000   # m/s

    print(f"\n  1-sigma position uncertainty (m) : "
          f"[{sig_pos[0]:.4f}, {sig_pos[1]:.4f}, {sig_pos[2]:.4f}]")
    print(f"  1-sigma velocity uncertainty (m/s): "
          f"[{sig_vel[0]:.6f}, {sig_vel[1]:.6f}, {sig_vel[2]:.6f}]")

    # -----------------------------------------------------------------------
    # Final error summary
    # -----------------------------------------------------------------------
    final_pos_err = np.linalg.norm(x_est[:3] - x_true[:3]) * 1000
    final_vel_err = np.linalg.norm(x_est[3:] - x_true[3:]) * 1000
    print(f"\n  Final estimation error:")
    print(f"    Position : {final_pos_err:.4f} m")
    print(f"    Velocity : {final_vel_err:.8f} m/s")

    # -----------------------------------------------------------------------
    # Save results
    # -----------------------------------------------------------------------
    result_path = os.path.join(DATA_DIR, "bls_result.npz")
    np.savez(
        result_path,
        x_true              = x_true,
        x_initial           = x_initial,
        x_estimated         = x_est,
        P_covariance        = P,
        perturbation        = PERTURBATION,
        iterations          = np.array(it),
        converged           = np.array(converged),
        elapsed_s           = np.array(elapsed),
        # Pre-fit residuals
        res_range_pre       = res_range_pre,
        res_rr_pre          = res_rr_pre,
        # Post-fit residuals
        res_range_post      = res_range_post,
        res_rr_post         = res_rr_post,
        # Observation metadata
        obs_times           = obs_df["time_s"].values[:len(res_range_post)],
        obs_stations        = obs_df["station"].values[:len(res_range_post)],
        # Noise sigmas (for normalised residual plots)
        sigma_range         = np.array(SIGMA_RANGE),
        sigma_rangerate     = np.array(SIGMA_RANGERATE),
        # Convergence log
        iter_log            = np.array(
            [[d["iteration"], d["dx_norm_m"], d["pos_error_m"],
              d["vel_error_ms"], d["rms_range_m"]]
             for d in iter_log]
        ),
    )
    print(f"\n  Results saved -> {result_path}")

    # Write text summary
    summary_path = os.path.join(DATA_DIR, "bls_summary.txt")
    with open(summary_path, "w") as fh:
        fh.write("Project 2 — Batch Least Squares Summary\n")
        fh.write("=" * 55 + "\n\n")
        fh.write(f"Convergence   : {'YES' if converged else 'NO'} "
                 f"(iter {it} / {MAX_ITERATIONS})\n")
        fh.write(f"Elapsed       : {elapsed:.1f} s\n\n")
        fh.write("Initial Perturbation\n")
        fh.write(f"  Position : {pos_err_init:.1f} m\n")
        fh.write(f"  Velocity : {vel_err_init:.4f} m/s\n\n")
        fh.write("Convergence Table\n")
        fh.write(f"  {'Iter':>4}  {'|dx|(m)':>12}  {'PosErr(m)':>12}  "
                 f"{'VelErr(m/s)':>12}  {'RMS_rng(m)':>12}\n")
        for d in iter_log:
            fh.write(f"  {d['iteration']:>4}  {d['dx_norm_m']:>12.6f}  "
                     f"{d['pos_error_m']:>12.6f}  {d['vel_error_ms']:>12.8f}  "
                     f"{d['rms_range_m']:>12.4f}\n")
        fh.write("\nPost-Fit Residuals\n")
        fh.write(f"  Range      RMS : {np.std(res_range_post)*1000:.4f} m\n")
        fh.write(f"  Range-rate RMS : {np.std(res_rr_post)*1000:.6f} m/s\n\n")
        fh.write("Final Estimation Error\n")
        fh.write(f"  Position : {final_pos_err:.6f} m\n")
        fh.write(f"  Velocity : {final_vel_err:.8f} m/s\n\n")
        fh.write("1-Sigma Covariance (from Lambda^-1)\n")
        fh.write(f"  sigma_x  : {sig_pos[0]:.6f} m\n")
        fh.write(f"  sigma_y  : {sig_pos[1]:.6f} m\n")
        fh.write(f"  sigma_z  : {sig_pos[2]:.6f} m\n")
        fh.write(f"  sigma_vx : {sig_vel[0]:.8f} m/s\n")
        fh.write(f"  sigma_vy : {sig_vel[1]:.8f} m/s\n")
        fh.write(f"  sigma_vz : {sig_vel[2]:.8f} m/s\n")

    print(f"  Summary saved  -> {summary_path}")
    print("\nDone. Next: run 02_residuals_analysis.py\n")


if __name__ == "__main__":
    main()