import numpy as np

# Physical Constants (Verified with GMAT JGM-2/WGS-84)
MU_EARTH    = 398600.4415
REQ_EARTH   = 6378.137
J2_EARTH    = 1.0826263e-3
OMEGA_EARTH = 7.2921151467e-5
WGS84_F     = 1 / 298.257223563
WGS84_E2    = WGS84_F * (2 - WGS84_F)

def ROTX(a): return np.array([[1, 0, 0], [0, np.cos(a), np.sin(a)], [0, -np.sin(a), np.cos(a)]])
def ROTY(a): return np.array([[np.cos(a), 0, -np.sin(a)], [0, 1, 0], [np.sin(a), 0, np.cos(a)]])
def ROTZ(a): return np.array([[np.cos(a), np.sin(a), 0], [-np.sin(a), np.cos(a), 0], [0, 0, 1]])

def get_precession_matrix(mjd_a1):
    """ MJ2000 -> Mean-of-Date (IAU-76) """
    T = (mjd_a1 + 2430000.0 - 2451545.0) / 36525.0
    zeta = np.radians((2306.2181*T + 0.30188*T**2 + 0.017998*T**3)/3600.0)
    z    = np.radians((2306.2181*T + 1.09468*T**2 + 0.018203*T**3)/3600.0)
    theta = np.radians((2004.3109*T - 0.42665*T**2 - 0.041833*T**3)/3600.0)
    return ROTZ(-z) @ ROTY(theta) @ ROTZ(-zeta)

def get_nutation_matrix(mjd_a1):
    """ Mean-of-Date -> True-of-Date (IAU-80) """
    T = (mjd_a1 + 2430000.0 - 2451545.0) / 36525.0
    Om = np.radians(125.04452 - 1934.13626 * T)
    dPsi = np.radians(-17.1996 * np.sin(Om) / 3600.0)
    dEps = np.radians(9.2025 * np.cos(Om) / 3600.0)
    eps0 = np.radians(23.439291 - 0.0130111 * T)
    eps = eps0 + dEps
    return ROTX(-eps) @ ROTZ(-dPsi) @ ROTX(eps0)

def ecef_to_eci_matrix(mjd_a1, time_offset_days):
    """ 
    TRANSFORMATION ECEF -> ECI (MJ2000Eq)
    ECI = P^T * N^T * R_z(gast)^T * ECEF
    """
    P = get_precession_matrix(mjd_a1)
    N = get_nutation_matrix(mjd_a1)
    
    # GMST from UTC Julian Date
    jd_utc = (mjd_a1 - time_offset_days) + 2430000.0
    d = jd_utc - 2451545.0
    gmst = (18.697374558 + 24.06570982441908 * d) % 24.0
    
    # Equation of Equinoxes (simplified)
    T = d / 36525.0
    Om = np.radians(125.04452 - 1934.13626 * T)
    dPsi = np.radians(-17.1996 * np.sin(Om) / 3600.0)
    eps0 = np.radians(23.439291)
    gast = (gmst * np.pi/12.0) + dPsi * np.cos(eps0)
    
    R_sidereal = ROTZ(gast)
    return P.T @ N.T @ R_sidereal.T

def get_station_eci(station_ecef, mjd_a1, time_offset_days):
    R_total = ecef_to_eci_matrix(mjd_a1, time_offset_days)
    pos_eci = R_total @ station_ecef
    vel_eci = np.cross(np.array([0, 0, OMEGA_EARTH]), pos_eci)
    return pos_eci, vel_eci

def get_station_coords():
    """
    WGS-84 ECEF coordinates recomputed from GMAT script geodetic params.

    FIX (BUG #1): Previous hardcoded values had errors up to 41 km (Dongara X)
    and 32 km (Santiago Y) relative to the GMAT geodetic definition.
    All values now recomputed directly from GMAT script parameters:
      Dongara:  lat=-29.04, lon=114.88, alt=0.03 km
      Santiago: lat=-33.15, lon=289.33, alt=0.73 km
    """
    dongara  = geodetic_to_ecef(-29.04,  114.88, 0.03)
    santiago = geodetic_to_ecef(-33.15,  289.33, 0.73)
    return {
        'Santiago': santiago,
        'Dongara':  dongara
    }

def geodetic_to_ecef(lat_deg, lon_deg, alt_km):
    lat, lon = np.radians(lat_deg), np.radians(lon_deg)
    N = REQ_EARTH / np.sqrt(1 - WGS84_E2 * np.sin(lat)**2)
    x = (N + alt_km) * np.cos(lat) * np.cos(lon)
    y = (N + alt_km) * np.cos(lat) * np.sin(lon)
    z = (N * (1 - WGS84_E2) + alt_km) * np.sin(lat)
    return np.array([x, y, z])