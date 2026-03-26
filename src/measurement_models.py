import numpy as np
from navigation_utils import get_station_eci

# =============================================================================
# NONLINEAR MEASUREMENT FUNCTIONS (h(x)) - 2-WAY MODEL
# =============================================================================

def compute_range_and_doppler(state_eci, station_ecef, mjd, time_offset):
    """
    Computes the predicted 2-Way Range and Range-Rate (Doppler) between 
    a satellite and a ground station.
    
    NOTE: GMAT 2025a uses 2-Way tracking by default. The signal travels 
    Station -> Sat -> Station. Therefore, the measurement is 2.0 * 1-Way.
    """
    # 1. Transform Station from ECEF (Fixed) to ECI (Inertial)
    pos_st_eci, vel_st_eci = get_station_eci(station_ecef, mjd, time_offset)
    
    # 2. Extract Satellite Vectors
    pos_sat = state_eci[0:3]
    vel_sat = state_eci[3:6]
    
    # 3. Compute 1-Way Relative Vectors
    rho_vec = pos_sat - pos_st_eci
    v_rel_vec = vel_sat - vel_st_eci
    
    # 4. Calculate 1-Way Scalars
    range_1way = np.linalg.norm(rho_vec)
    doppler_1way = np.dot(rho_vec, v_rel_vec) / range_1way
    
    # 5. CONVERT TO 2-WAY (Round-Trip)
    # To match GMAT's .gmd data, we sum the up-link and down-link
    range_2way = range_1way * 2.0
    doppler_2way = doppler_1way * 2.0

    return range_2way, doppler_2way

# =============================================================================
# MEASUREMENT JACOBIAN MATRIX (H Matrix) - 2-WAY MODEL
# =============================================================================

def get_measurement_jacobian(state_eci, station_ecef, mjd, obs_type, time_offset):
    """
    Computes the Jacobian Matrix H = dh/dx for a 2-Way measurement.
    Since h(x) = 2 * h_1way(x), the Jacobian is also multiplied by 2.0.
    """
    # Get inertial coordinates of the station
    pos_st_eci, vel_st_eci = get_station_eci(station_ecef, mjd, time_offset)
    
    pos_sat = state_eci[0:3]
    vel_sat = state_eci[3:6]
    
    # Relative 1-Way vectors and scalars
    rho_vec = pos_sat - pos_st_eci
    v_rel_vec = vel_sat - vel_st_eci
    rho = np.linalg.norm(rho_vec)
    rho_dot = np.dot(rho_vec, v_rel_vec) / rho
    
    # Initialize 1x6 Jacobian row
    H = np.zeros((1, 6))
    
    if obs_type == 'Range':
        # 1-Way Sensitivity: rho_vec / rho
        # 2-Way Sensitivity: 2.0 * (rho_vec / rho)
        H[0, 0:3] = 2.0 * (rho_vec / rho)
        H[0, 3:6] = 0.0
        
    elif obs_type == 'RangeRate':
        # 1-Way Sensitivity w.r.t Velocity: rho_vec / rho
        # 1-Way Sensitivity w.r.t Position: (v_rel / rho) - (rho_dot * rho_vec / rho^2)
        H_pos_1way = (v_rel_vec / rho) - (rho_dot * rho_vec / (rho**2))
        H_vel_1way = rho_vec / rho
        
        # 2-Way Sensitivity: Multiply 1-Way derivatives by 2.0
        H[0, 0:3] = 2.0 * H_pos_1way
        H[0, 3:6] = 2.0 * H_vel_1way
        
    return H

# =============================================================================
# MODULE TEST BLOCK
# =============================================================================
if __name__ == "__main__":
    from navigation_utils import get_station_coords
    
    # Sample state (LEO-like)
    test_state = np.array([7000.0, 1000.0, 500.0, -1.0, 7.0, 0.5])
    test_mjd = 30311.380289
    station_coords = get_station_coords()
    dongara_ecef = station_coords['Dongara']
    
    # Test Calculation
    r_calc, rr_calc = compute_range_and_doppler(test_state, dongara_ecef, test_mjd)
    
    # Test Jacobians
    H_range = get_measurement_jacobian(test_state, dongara_ecef, test_mjd, 'Range')
    H_doppler = get_measurement_jacobian(test_state, dongara_ecef, test_mjd, 'RangeRate')
    
    print("--- Measurement Models Module Test (2-WAY UPDATED) ---")
    print(f"Predicted 2-Way Range: {r_calc:.4f} km")
    print(f"Predicted 2-Way RangeRate: {rr_calc:.6f} km/s")
    print(f"\n2-Way Range Jacobian (H_rho):\n{H_range}")
    print(f"\n2-Way RangeRate Jacobian (H_dot_rho):\n{H_doppler}")