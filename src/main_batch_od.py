import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from data_loader import parse_gmd_file, load_ground_truth, get_synchronized_truth
from batch_estimator import run_batch_least_squares, get_sigma_bounds

# =============================================================================
# MAIN ORBIT DETERMINATION WORKFLOW
# =============================================================================

def main():
    # 1. SETUP PATHS
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    obs_path = os.path.join(project_root, 'data', 'satellite_observations.gmd')
    truth_path = os.path.join(project_root, 'data', 'ground_truth.csv')
    
    print("--- Project 2: Batch Orbit Determination ---")
    
    # 2. LOAD DATA
    obs_df = parse_gmd_file(obs_path)
    truth_df = load_ground_truth(truth_path)
    
    # 3. SYNCHRONIZATION
    # Get the time of the first observation (t_start)
    t_start_mjd = obs_df['Time_MJD'].iloc[0]
    
    # Find the matching truth state at t_start from the CSV
    x_true, t0_mjd = get_synchronized_truth(truth_df, t_start_mjd)
    
    # 4. DEFINE INITIAL GUESS
    # We add a controlled error to the true state to simulate an initial guess
    pos_error = np.array([0.6, -0.4, 0.2]) # km
    vel_error = np.array([0.0005, -0.0003, 0.0001]) # km/s
    
    x_initial_guess = x_true + np.concatenate([pos_error, vel_error])

    # 5. RUN BATCH LEAST SQUARES (Mancava questa riga!)
    # This is the "brain" that computes the orbit estimation
    x_estimated, p_matrix, final_residuals = run_batch_least_squares(
        obs_df, 
        x_initial_guess, 
        max_iterations=10, 
        convergence_tol=1e-7
    )

    if x_estimated is None:
        print("Estimation process failed to converge.")
        return

    # 6. ERROR ANALYSIS
    # Compare the estimated state at t0 with the true state at t0
    error = x_estimated - x_true
    pos_err_norm = np.linalg.norm(error[0:3]) * 1000.0 # Convert to meters
    vel_err_norm = np.linalg.norm(error[3:6]) * 1000.0 # Convert to m/s
    
    # Extract 3-sigma uncertainties from the Covariance Matrix
    sigmas = get_sigma_bounds(p_matrix)
    pos_3sigma = np.linalg.norm(sigmas['pos_sigma']) * 3.0 * 1000.0 # meters
    
    print("\n--- FINAL ESTIMATION RESULTS ---")
    print(f"Final Position Error: {pos_err_norm:.3f} meters")
    print(f"Final Velocity Error: {vel_err_norm:.4f} meters/second")
    print(f"Estimated 3-Sigma Position Uncertainty: {pos_3sigma:.3f} meters")
    
    # 7. VISUALIZATION
    # Plotting Post-Fit Residuals
    plt.figure(figsize=(10, 6))
    plt.plot(final_residuals, 'o', markersize=3, alpha=0.6, color='blue')
    plt.axhline(0, color='red', linestyle='--')
    plt.title('Post-Fit Residuals (Optimized Solution)')
    plt.xlabel('Observation Index')
    plt.ylabel('Residual Value [km or km/s]')
    plt.grid(True, alpha=0.3)
    
    # Check if directory exists before saving, or just show
    plt.show()

    print("\nProject 2 completed successfully.")

if __name__ == "__main__":
    main()