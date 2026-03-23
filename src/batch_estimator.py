import numpy as np
from navigation_utils import get_station_coords
from orbit_models import propagate_state_and_stm
from measurement_models import compute_range_and_doppler, get_measurement_jacobian

def run_batch_least_squares(observations_df, x_initial_guess, time_offset,
                             max_iterations=15, convergence_tol=1e-6):
    """
    Weighted Batch Least Squares OD.
    Passes t_mjd0 + time_offset to propagate_state_and_stm so that
    Moon/Sun/Drag/SRP forces are active, matching the GMAT force model.
    """
    t0_mjd = observations_df['Time_MJD'].iloc[0]
    x_ref  = x_initial_guess.copy()
    station_coords = get_station_coords()

    sigma_range   = 0.010
    sigma_doppler = 0.000001
    w_range   = 1.0 / sigma_range**2
    w_doppler = 1.0 / sigma_doppler**2

    p0_pos_sigma = 1.0
    p0_vel_sigma = 0.01
    P0_diag  = np.array([p0_pos_sigma**2]*3 + [p0_vel_sigma**2]*3)
    lambda_0 = np.diag(1.0 / P0_diag)

    print(f"\n--- Batch Least Squares OD (J2-J6 + Moon + Sun + Drag + SRP) ---")
    print(f"  sigma_range={sigma_range*1000:.0f}m  sigma_doppler={sigma_doppler*1e6:.0f}mm/s")

    for iteration in range(max_iterations):
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
                # t_mjd0 activates Moon/Sun/Drag/SRP in orbit_models
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
            print(f"  [WARNING] Singular normal matrix at iter {iteration+1}.")
            break

        x_ref += dx
        rms_range = np.sqrt(np.mean(np.array(raw_range_residuals)**2))
        print(f"  Iter {iteration+1:2d} | ||dx||={np.linalg.norm(dx)*1000:.4f}m "
              f"| RangeRMS={rms_range*1000:.3f}m "
              f"| cond={np.linalg.cond(lambda_matrix):.2e}")

        if np.linalg.norm(dx) < convergence_tol:
            print(f"  Converged in {iteration+1} iterations.")
            break

    return x_ref, np.linalg.inv(lambda_matrix), raw_range_residuals


def get_sigma_bounds(p_matrix):
    diag = np.diag(p_matrix).copy()
    diag[diag < 0] = 0
    sigmas = np.sqrt(diag)
    return {'pos_sigma': sigmas[0:3], 'vel_sigma': sigmas[3:6]}