import numpy as np

# =============================================================================
# PHYSICAL CONSTANTS (WGS-84 & Earth Environment)
# =============================================================================
MU_EARTH    = 398600.4415      # Gravitational parameter [km^3/s^2]
REQ_EARTH   = 6378.137         # Earth equatorial radius [km]
J2_EARTH    = 1.0826263e-3     # J2 perturbation coefficient
OMEGA_EARTH = 7.2921151467e-5  # Earth rotation rate [rad/s]
WGS84_F     = 1 / 298.257223563 # Flattening
WGS84_E2    = WGS84_F * (2 - WGS84_F) # Eccentricity squared

# =============================================================================
# COORDINATE CONVERSIONS
# =============================================================================

def geodetic_to_ecef(lat_deg, lon_deg, alt_km):
    """
    Converts Geodetic coordinates (Lat, Lon, Alt) to ECEF Cartesian (X, Y, Z).
    """
    lat_rad = np.radians(lat_deg)
    lon_rad = np.radians(lon_deg)
    
    # Radius of curvature in the prime vertical
    N = REQ_EARTH / np.sqrt(1 - WGS84_E2 * np.sin(lat_rad)**2)
    
    x = (N + alt_km) * np.cos(lat_rad) * np.cos(lon_rad)
    y = (N + alt_km) * np.cos(lat_rad) * np.sin(lon_rad)
    z = (N * (1 - WGS84_E2) + alt_km) * np.sin(lat_rad)
    
    return np.array([x, y, z])

def get_station_coords():
    """
    Returns ECEF coordinates for the stations used in GMAT Project 2.
    """
    return {
        'Santiago': geodetic_to_ecef(-33.15, 289.33, 0.73),
        'Dongara':  geodetic_to_ecef(-29.04, 114.88, 0.03)
    }

# =============================================================================
# TIME & FRAME TRANSFORMATIONS
# =============================================================================

def mjd_to_gmst(mjd):
    """
    Calculates Greenwich Mean Sidereal Time (GMST) for a given 
    Modified Julian Date (MJD). 
    Formula based on Vallado (Astrodyamics for Engineering Students).
    """
    # JD at J2000.0 epoch
    JD_J2000 = 2451545.0
    # Convert MJD to Julian Date (GMAT A1MJD is relative to 05 Jan 1941)
    # Note: GMAT A1MJD offset is 2430000.5
    jd = mjd + 2430000.5
    
    # Julian centuries from J2000.0
    T = (jd - JD_J2000) / 36525.0
    
    # GMST in degrees (standard polynomial expansion)
    gmst_deg = (280.46061837 + 360.98564736629 * (jd - JD_J2000) + 
                0.000387933 * T**2 - (T**3 / 38710000.0))
    
    # Normalize to [0, 360] degrees and convert to radians
    return np.radians(gmst_deg % 360.0)

def ecef_to_eci_matrix(gmst_rad):
    """
    Rotation matrix from ECEF (Fixed) to ECI (Inertial) at a given GMST.
    The matrix is the transpose of the ECI->ECEF rotation (Rz(gmst)).
    """
    c = np.cos(gmst_rad)
    s = np.sin(gmst_rad)
    
    # Rz(-gmst) transformation
    return np.array([
        [ c, -s,  0],
        [ s,  c,  0],
        [ 0,  0,  1]
    ])

def get_station_eci(station_ecef, mjd):
    """
    Transforms station position and velocity from ECEF to ECI frame.
    Returns: pos_eci (km), vel_eci (km/s)
    """
    gmst = mjd_to_gmst(mjd)
    R = ecef_to_eci_matrix(gmst)
    
    # Position in ECI
    pos_eci = R @ station_ecef
    
    # Velocity in ECI (Cross product omega x R_eci)
    omega_vec = np.array([0, 0, OMEGA_EARTH])
    vel_eci = np.cross(omega_vec, pos_eci)
    
    return pos_eci, vel_eci

# =============================================================================
# TEST BLOCK
# =============================================================================
if __name__ == "__main__":
    # Test with an arbitrary MJD (e.g., from your .gmd file)
    test_mjd = 30311.380289
    stations = get_station_coords()
    
    print(f"--- Navigation Utils Test (MJD: {test_mjd}) ---")
    for name, ecef in stations.items():
        pos_eci, vel_eci = get_station_eci(ecef, test_mjd)
        print(f"\nStation: {name}")
        print(f"  ECEF Pos: {ecef}")
        print(f"  ECI  Pos: {pos_eci}")
        print(f"  ECI  Vel: {vel_eci} (Inertial speed due to Earth rotation)")