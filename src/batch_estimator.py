import numpy as np
from navigation_utils import get_station_coords, get_station_eci
from orbit_models import propagate_state_and_stm
from measurement_models import compute_range_and_doppler, get_measurement_jacobian

def run_batch_least_squares(observations_df, x_initial_guess, time_offset, max_iterations=10, convergence_tol=1e-8):
    t0_mjd = observations_df['Time_MJD'].iloc[0]
    x_ref = x_initial_guess.copy()
    station_coords = get_station_coords()
    
    # Balanced weights: Doppler at 1 m/s (0.001) for stability
    sigma_range = 0.050    
    sigma_doppler = 0.001  

    print(f"\n--- Starting Robust High-Fidelity Batch OD ---")

    for iteration in range(max_iterations):
        lambda_matrix = np.zeros((6, 6))
        b_vector = np.zeros(6)
        
        # Regularization: Anchor the state to prevent 800,000km jumps
        lambda_matrix += np.eye(6) * 1e-4
        
        raw_range_residuals = [] 
        x_current = x_ref.copy()
        phi_current = np.eye(6)
        t_last = t0_mjd
        
        for idx, obs in observations_df.iterrows():
            t_obs = obs['Time_MJD']
            dt = (t_obs - t_last) * 86400.0
            if dt > 0:
                x_current, phi_current = propagate_state_and_stm(x_current, phi_current, dt)
                t_last = t_obs
            
            r_calc, rr_calc = compute_range_and_doppler(x_current, station_coords[obs['Station']], t_obs, time_offset)
            dy = obs['Value'] - (r_calc if obs['Type'] == 'Range' else rr_calc)
            
            if obs['Type'] == 'Range': raw_range_residuals.append(dy)
            
            H = get_measurement_jacobian(x_current, station_coords[obs['Station']], t_obs, obs['Type'], time_offset)
            h_vec = (H @ phi_current).flatten()
            w = 1.0/(sigma_range**2) if obs['Type'] == 'Range' else 1.0/(sigma_doppler**2)
            
            lambda_matrix += np.outer(h_vec, h_vec) * w
            b_vector += h_vec * (w * dy)
            
        dx = np.linalg.solve(lambda_matrix, b_vector)
        
        # Trust Region: Max step 20km
        dx_norm_pos = np.linalg.norm(dx[0:3])
        if dx_norm_pos > 20.0:
            dx = dx * (20.0 / dx_norm_pos)
            
        x_ref += dx
        
        # MONITORING ONLY RANGE ERROR (Physical km)
        rms_range = np.sqrt(np.mean(np.array(raw_range_residuals)**2))
        print(f"Iteration {iteration+1:2d} | Step: {np.linalg.norm(dx):.4e} | Range RMS: {rms_range:.4f} km")
        
        if np.linalg.norm(dx) < convergence_tol: break
            
    return x_ref, np.linalg.inv(lambda_matrix), raw_range_residuals

def get_sigma_bounds(p_matrix):
    diag = np.diag(p_matrix).copy()
    diag[diag < 0] = 0
    sigmas = np.sqrt(diag)
    return {'pos_sigma': sigmas[0:3], 'vel_sigma': sigmas[3:6]}