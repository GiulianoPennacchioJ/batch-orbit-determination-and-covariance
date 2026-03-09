import numpy as np
from navigation_utils import get_station_eci

# =============================================================================
# NONLINEAR MEASUREMENT FUNCTIONS (h(x))
# =============================================================================

def compute_range_and_doppler(state_eci, station_ecef, mjd):
    """
    Computes the predicted Range and Range-Rate (Doppler) between 
    a satellite in ECI frame and a ground station in ECEF frame.
    
    Inputs:
        state_eci: Satellite state [x, y, z, vx, vy, vz] in km and km/s
        station_ecef: Station position [X, Y, Z] in ECEF frame (fixed to Earth)
        mjd: Current Time in Modified Julian Date
        
    Returns:
        range_val: Scalar distance [km]
        doppler_val: Scalar range-rate [km/s]
    """
    # 1. Transform Station from ECEF (Fixed) to ECI (Inertial)
    # We must account for Earth's rotation at the specific measurement time (MJD).
    # This function provides both inertial position and inertial velocity (due to rotation).
    pos_st_eci, vel_st_eci = get_station_eci(station_ecef, mjd)
    
    # 2. Compute Relative Vectors
    # Rho = R_sat - R_station
    # Rho_dot = V_sat - V_station
    pos_sat = state_eci[0:3]
    vel_sat = state_eci[3:6]
    
    rho_vec = pos_sat - pos_st_eci
    v_rel_vec = vel_sat - vel_st_eci
    
    # 3. Calculate Scalar Range (Euclidean Norm)
    range_val = np.linalg.norm(rho_vec)
    
    # 4. Calculate Scalar Range-Rate (Projection of relative velocity onto line-of-sight)
    # Formula: rho_dot = (rho_vec . v_rel_vec) / rho_scalar
    doppler_val = np.dot(rho_vec, v_rel_vec) / range_val
    
    return range_val, doppler_val

# =============================================================================
# MEASUREMENT JACOBIAN MATRIX (H Matrix)
# =============================================================================

def get_measurement_jacobian(state_eci, station_ecef, mjd, obs_type):
    """
    Computes the Jacobian Matrix H = dh/dx for a single measurement.
    H is a 1x6 vector representing the sensitivity of the measurement 
    with respect to the satellite state [x, y, z, vx, vy, vz].
    
    Inputs:
        obs_type: String, either 'Range' or 'RangeRate'
    """
    # Get inertial coordinates of the station
    pos_st_eci, vel_st_eci = get_station_eci(station_ecef, mjd)
    
    pos_sat = state_eci[0:3]
    vel_sat = state_eci[3:6]
    
    # Relative vectors
    rho_vec = pos_sat - pos_st_eci
    v_rel_vec = vel_sat - vel_st_eci
    rho = np.linalg.norm(rho_vec)
    rho_dot = np.dot(rho_vec, v_rel_vec) / rho
    
    # Initialize 1x6 Jacobian row
    H = np.zeros((1, 6))
    
    if obs_type == 'Range':
        # Sensitivity of Range w.r.t Position: rho_vec / rho
        # Sensitivity of Range w.r.t Velocity: Zero (Range does not depend on instantaneous velocity)
        H[0, 0:3] = rho_vec / rho
        H[0, 3:6] = 0.0
        
    elif obs_type == 'RangeRate':
        # Sensitivity of Range-Rate w.r.t Velocity: rho_vec / rho
        H[0, 3:6] = rho_vec / rho
        
        # Sensitivity of Range-Rate w.r.t Position (More complex derivative):
        # d(rho_dot)/dr = (v_rel / rho) - (rho_dot * rho_vec / rho^2)
        H[0, 0:3] = (v_rel_vec / rho) - (rho_dot * rho_vec / (rho**2))
        
    return H

# =============================================================================
# MODULE TEST BLOCK
# =============================================================================
if __name__ == "__main__":
    from navigation_utils import get_station_coords
    
    # Sample state and time
    test_state = np.array([7000.0, 1000.0, 500.0, -1.0, 7.0, 0.5])
    test_mjd = 30311.380289
    station_coords = get_station_coords()
    dongara_ecef = station_coords['Dongara']
    
    # Test Calculation
    r_calc, rr_calc = compute_range_and_doppler(test_state, dongara_ecef, test_mjd)
    
    # Test Jacobians
    H_range = get_measurement_jacobian(test_state, dongara_ecef, test_mjd, 'Range')
    H_doppler = get_measurement_jacobian(test_state, dongara_ecef, test_mjd, 'RangeRate')
    
    print("--- Measurement Models Module Test ---")
    print(f"Predicted Range: {r_calc:.4f} km")
    print(f"Predicted RangeRate: {rr_calc:.6f} km/s")
    print(f"\nRange Jacobian (H_rho):\n{H_range}")
    print(f"\nRangeRate Jacobian (H_dot_rho):\n{H_doppler}")