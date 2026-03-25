import numpy as np
from navigation_utils import get_station_coords
from orbit_models import propagate_state_and_stm
from measurement_models import compute_range_and_doppler, get_measurement_jacobian

def run_batch_least_squares(observations_df, x_initial_guess, time_offset,
                             max_outer=8, max_inner=15, convergence_tol=1e-6):
    """
    Iterated Weighted Batch Least Squares Orbit Determination.

    Uses a two-level iteration loop:
      OUTER loop: re-linearises around the updated estimate after each inner
                  convergence. Required for long arcs (>2h) because the STM
                  phi grows large (~3.4e4 after 17h with correct J2), causing
                  the linear approximation to break down for initial errors >10m.
      INNER loop: standard Gauss-Newton step within one linearisation.

    Normal equations (Bayesian):
        Λ  = inv(P0) + Σ Φᵀ Hᵀ W H Φ
        b  = Σ Φᵀ Hᵀ W dy
        dx = Λ⁻¹ b

    Parameters
    ----------
    max_outer : outer re-linearisation iterations (default 8)
    max_inner : inner Gauss-Newton iterations per outer step (default 15)
    convergence_tol : convergence threshold on ||dx|| in km (default 1e-6 km = 1 mm)
    """
    t0_mjd = observations_df['Time_MJD'].iloc[0]
    x_ref  = x_initial_guess.copy()
    station_coords = get_station_coords()

    sigma_range   = 0.010       # km
    sigma_doppler = 0.000001    # km/s
    w_range   = 1.0 / sigma_range**2
    w_doppler = 1.0 / sigma_doppler**2

    p0_pos = 1.0    # km
    p0_vel = 0.01   # km/s
    P0_diag  = np.array([p0_pos**2]*3 + [p0_vel**2]*3)
    lambda_0 = np.diag(1.0 / P0_diag)

    print(f"\n--- Iterated Batch Least Squares OD ---")
    print(f"  sigma_range={sigma_range*1000:.0f} m  sigma_doppler={sigma_doppler*1e6:.0f} mm/s")
    print(f"  Force model: J2-J6 + Moon + Sun (activated via t_mjd0)")
    print(f"  Outer iterations: max {max_outer}  |  Inner iterations: max {max_inner}")

    raw_range_residuals = []
    p_matrix = None

    for outer in range(max_outer):
        inner_converged = False

        for inner in range(max_inner):
            lambda_matrix = lambda_0.copy()
            b_vector      = np.zeros(6)
            raw_range_residuals = []

            x_current   = x_ref.copy()
            phi_current = np.eye(6)
            t_last      = t0_mjd

            for idx, obs in observations_df.iterrows():
                t_obs = obs['Time_MJD']
                dt    = (t_obs - t_last) * 86400.0
                if dt > 0:
                    x_current, phi_current = propagate_state_and_stm(
                        x_current, phi_current, dt,
                        t_mjd0=t0_mjd, time_offset=time_offset)
                    t_last = t_obs

                r_calc, rr_calc = compute_range_and_doppler(
                    x_current, station_coords[obs['Station']], t_obs, time_offset)

                if obs['Type'] == 'Range':
                    dy = obs['Value'] - r_calc
                    raw_range_residuals.append(dy)
                    w  = w_range
                else:
                    dy = obs['Value'] - rr_calc
                    w  = w_doppler

                H     = get_measurement_jacobian(
                    x_current, station_coords[obs['Station']],
                    t_obs, obs['Type'], time_offset)
                h_vec = (H @ phi_current).flatten()
                lambda_matrix += np.outer(h_vec, h_vec) * w
                b_vector      += h_vec * (w * dy)

            try:
                dx = np.linalg.solve(lambda_matrix, b_vector)
            except np.linalg.LinAlgError:
                print(f"  [WARNING] Singular normal matrix. Stopping.")
                break

            x_ref += dx
            rms   = np.sqrt(np.mean(np.array(raw_range_residuals)**2))
            step  = np.linalg.norm(dx)

            print(f"  Outer {outer+1:2d} / Inner {inner+1:2d} | "
                  f"||dx||={step*1000:.4f} m | "
                  f"RangeRMS={rms*1000:.3f} m | "
                  f"cond={np.linalg.cond(lambda_matrix):.2e}")

            if step < convergence_tol:
                inner_converged = True
                break

        p_matrix = np.linalg.inv(lambda_matrix)

        # Outer convergence: inner converged and step is tiny
        if inner_converged and step < convergence_tol:
            print(f"  Fully converged after outer {outer+1}.")
            break

    return x_ref, p_matrix, raw_range_residuals


def get_sigma_bounds(p_matrix):
    diag = np.diag(p_matrix).copy()
    diag[diag < 0] = 0
    sigmas = np.sqrt(diag)
    return {'pos_sigma': sigmas[0:3], 'vel_sigma': sigmas[3:6]}