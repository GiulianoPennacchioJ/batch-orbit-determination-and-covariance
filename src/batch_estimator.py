import numpy as np
from navigation_utils import get_station_coords
from orbit_models import propagate_state_and_stm
from measurement_models import compute_range_and_doppler, get_measurement_jacobian

# =============================================================================
# BATCH LEAST SQUARES ESTIMATOR
# =============================================================================

def run_batch_least_squares(observations_df, x_initial_guess, max_iterations=10, convergence_tol=1e-6):
    """
    Performs an Iterative Batch Least Squares estimation to determine 
    the satellite orbital state at the initial epoch (t0).
    
    Inputs:
        observations_df: DataFrame from parse_gmd_file
        x_initial_guess: 6x1 array [x, y, z, vx, vy, vz] in km and km/s
        max_iterations: Safety limit for the loop
        convergence_tol: Minimum correction norm to stop iterations
        
    Returns:
        x_estimated: Optimized state at t0
        P_matrix: Covariance matrix (uncertainty of the estimate)
        residuals_history: List of post-fit residuals for analysis
    """
    
    # 1. SETUP INITIAL CONDITIONS
    # We estimate the state at the time of the very first observation (t0)
    t0_mjd = observations_df['Time_MJD'].iloc[0]
    x_ref = x_initial_guess.copy()
    
    # Get ground station ECEF coordinates
    station_coords = get_station_coords()
    
    # 2. DEFINE WEIGHT MATRIX (W)
    # Based on GMAT noise: 10m (0.010 km) for Range, 1mm/s (0.000001 km/s) for Doppler
    sigma_range = 0.010
    sigma_doppler = 0.001 # 1 mm/s in km/s is 0.000001, but here we use km/s
    
    print(f"\n--- Starting Batch Least Squares Estimation ---")
    print(f"Initial State Guess: {x_ref}")
    
    for iteration in range(max_iterations):
        # Accumulators for the Normal Equations: (H^T * W * H) * dx = (H^T * W * dy)
        # Information Matrix (Lambda) and Information Vector (b)
        lambda_matrix = np.zeros((6, 6))
        b_vector = np.zeros(6)
        
        # Tracking residuals for performance monitoring
        current_residuals = []
        
        # Reset propagation to t0 for each iteration
        x_current = x_ref.copy()
        phi_current = np.eye(6) # State Transition Matrix starts as Identity
        t_last = t0_mjd
        
        # 3. PROCESS EACH OBSERVATION
        for idx, obs in observations_df.iterrows():
            t_obs = obs['Time_MJD']
            dt = (t_obs - t_last) * 86400.0 # Convert MJD difference to seconds
            
            # 3a. Propagate Reference State and STM to the observation time
            # This 'maps' our initial guess to the time the measurement was taken
            if dt > 0:
                x_current, phi_current = propagate_state_and_stm(x_current, phi_current, dt)
                t_last = t_obs
            
            # 3b. Compute Predicted Measurement (h(x))
            station_ecef = station_coords[obs['Station']]
            r_calc, rr_calc = compute_range_and_doppler(x_current, station_ecef, t_obs)
            
            y_calc = r_calc if obs['Type'] == 'Range' else rr_calc
            y_obs = obs['Value']
            
            # 3c. Compute Residual (dy = y_observed - y_calculated)
            dy = y_obs - y_calc
            current_residuals.append(dy)
            
            # 3d. Compute Measurement Jacobian (H) at time t (1x6 matrix)
            H_matrix = get_measurement_jacobian(x_current, station_ecef, t_obs, obs['Type'])
            
            # 3e. Map Jacobian back to the initial epoch t0 using the STM
            # H_tilde is still a 1x6 matrix
            H_tilde = H_matrix @ phi_current
            
            # IMPORTANT: Convert H_tilde to a 1D vector (size 6) for easier calculation
            h_vec = H_tilde.flatten()
            
            # 3f. Weighting (W = 1 / sigma^2)
            weight = 1.0 / (sigma_range**2) if obs['Type'] == 'Range' else 1.0 / (sigma_doppler**2)
            
            # 3g. Accumulate Normal Equations
            # Lambda (6x6) += H^T * W * H  --> We use np.outer for (6,) * (6,) = (6,6)
            lambda_matrix += np.outer(h_vec, h_vec) * weight
            
            # b (6,) += H^T * W * dy --> Standard vector * scalar multiplication
            b_vector += h_vec * (weight * dy)
            
        # 4. SOLVE NORMAL EQUATIONS
        # dx = inv(Lambda) * b
        # This provides the correction to the initial state guess
        try:
            dx = np.linalg.solve(lambda_matrix, b_vector)
        except np.linalg.LinAlgError:
            print("Singular Matrix! Observations might not provide enough observability.")
            return None, None, None
            
        # 5. UPDATE REFERENCE STATE
        x_ref += dx
        
        # 6. CONVERGENCE CHECK
        dx_norm = np.linalg.norm(dx)
        print(f"Iteration {iteration+1:2d} | Correction Norm: {dx_norm:.8e} | RMS Residual: {np.std(current_residuals):.6f}")
        
        if dx_norm < convergence_tol:
            print(f"--- Convergence Reached after {iteration+1} iterations ---")
            break
            
    # 7. COMPUTE FINAL COVARIANCE
    # P = inv(Lambda) -> This represents the uncertainty of the estimated state
    p_matrix = np.linalg.inv(lambda_matrix)
    
    return x_ref, p_matrix, current_residuals

# =============================================================================
# COVARIANCE ANALYSIS UTILITIES
# =============================================================================

def get_sigma_bounds(p_matrix):
    """
    Extracts 1-sigma uncertainties (standard deviations) from the diagonal 
    of the covariance matrix.
    """
    sigmas = np.sqrt(np.diag(p_matrix))
    return {
        'pos_sigma': sigmas[0:3], # Uncertainty in X, Y, Z [km]
        'vel_sigma': sigmas[3:6]  # Uncertainty in VX, VY, VZ [km/s]
    }