"""
od_utils.py
===========
Shared constants, dynamics, and coordinate-frame utilities for
Project 2: Batch Orbit Determination and Covariance Analysis.

Author : Giuliano Pennacchio
Version: 1.0

Contents
--------
- Physical constants and noise parameters
- J2-perturbed equations of motion
- Jacobian of J2 dynamics (numerical, central differences)
- Augmented ODE for state + STM propagation
- Coordinate transforms: LLA -> ECEF, ECEF <-> ECI
- Ground-station observation functions (range, range-rate, elevation)
- Scipy ODE wrapper with configurable tolerances
"""

import numpy as np
from scipy.integrate import solve_ivp

# ---------------------------------------------------------------------------
# Physical constants
# ---------------------------------------------------------------------------
MU      = 398600.4418    # Earth gravitational parameter      [km^3/s^2]
RE      = 6378.137       # Earth equatorial radius            [km]
J2      = 1.08262668e-3  # Second zonal harmonic coefficient  [–]
WE      = 7.2921150e-5   # Earth rotation rate                [rad/s]
F_EARTH = 1.0 / 298.257223563  # WGS84 flattening factor      [–]

# ---------------------------------------------------------------------------
# Measurement noise (1-sigma)
# ---------------------------------------------------------------------------
SIGMA_RANGE     = 0.010      # Range noise      [km]   (10 m)
SIGMA_RANGERATE = 0.000001   # Range-rate noise [km/s] (1 mm/s)

# ---------------------------------------------------------------------------
# Propagator tolerances
# ---------------------------------------------------------------------------
ODE_RTOL_TRUTH = 1e-12   # Tight: truth trajectory
ODE_ATOL_TRUTH = 1e-14
ODE_RTOL_EST   = 1e-9    # Moderate: estimation trajectory
ODE_ATOL_EST   = 1e-11
ODE_MAX_STEP   = 30.0    # Maximum ODE step [s]


# ===========================================================================
# Equations of Motion
# ===========================================================================

def f_j2(state: np.ndarray) -> np.ndarray:
    """
    J2-perturbed equations of motion (time-invariant).

    Parameters
    ----------
    state : array_like, shape (6,)
        Cartesian ECI state [x, y, z, vx, vy, vz]  [km, km/s].

    Returns
    -------
    ndarray, shape (6,)
        State derivative [vx, vy, vz, ax, ay, az]  [km/s, km/s^2].
    """
    x, y, z, vx, vy, vz = state
    r2 = x**2 + y**2 + z**2
    r  = np.sqrt(r2)
    r3 = r2 * r
    r5 = r2 * r3
    c  = 1.5 * J2 * MU * RE**2 / r5          # J2 prefactor
    f5 = 5.0 * z**2 / r2                       # 5z^2/r^2 factor
    ax = -MU * x / r3 + c * x * (f5 - 1.0)
    ay = -MU * y / r3 + c * y * (f5 - 1.0)
    az = -MU * z / r3 + c * z * (f5 - 3.0)
    return np.array([vx, vy, vz, ax, ay, az])


def eom_scipy(t: float, y: np.ndarray) -> np.ndarray:
    """Scipy-compatible wrapper for f_j2 (accepts time argument)."""
    return f_j2(y)


# ===========================================================================
# Jacobian (A matrix = df/dx)
# ===========================================================================

def jacobian_j2(state: np.ndarray) -> np.ndarray:
    """
    Numerical Jacobian of f_j2 via central differences.

    Central-difference perturbations are chosen to balance truncation
    and round-off error for typical LEO states:
      - Position components: eps = 0.1 km
      - Velocity components: eps = 1e-4 km/s

    Parameters
    ----------
    state : ndarray, shape (6,)
        Current Cartesian ECI state [km, km/s].

    Returns
    -------
    A : ndarray, shape (6, 6)
        Jacobian matrix  A[i,j] = df_i / dx_j.
    """
    eps = np.array([0.1, 0.1, 0.1, 1e-4, 1e-4, 1e-4])
    A   = np.zeros((6, 6))
    for j in range(6):
        e          = np.zeros(6)
        e[j]       = eps[j]
        A[:, j]    = (f_j2(state + e) - f_j2(state - e)) / (2.0 * eps[j])
    return A


# ===========================================================================
# Augmented ODE: state + State Transition Matrix (STM)
# ===========================================================================

def aug_eom(t: float, y: np.ndarray) -> np.ndarray:
    """
    Augmented equations of motion for simultaneous propagation of
    the state vector and the 6x6 State Transition Matrix (STM).

    The augmented state is:
        y = [state (6,), phi.flatten() (36,)]   shape (42,)

    Variational equation:  d(Phi)/dt = A(t) @ Phi,  Phi(t0) = I_6

    Parameters
    ----------
    t : float
        Current time [s]  (required by scipy, not used in time-invariant EOM).
    y : ndarray, shape (42,)
        Augmented state vector.

    Returns
    -------
    dy : ndarray, shape (42,)
        Augmented state derivative.
    """
    state = y[:6]
    phi   = y[6:].reshape(6, 6)
    f     = f_j2(state)
    A     = jacobian_j2(state)
    dphi  = (A @ phi).flatten()
    return np.concatenate([f, dphi])


# ===========================================================================
# Propagation wrappers
# ===========================================================================

def propagate_truth(x0: np.ndarray, t_eval: np.ndarray) -> "solve_ivp result":
    """
    Propagate the true trajectory with tight tolerances (no STM).

    Parameters
    ----------
    x0     : ndarray, shape (6,)  Initial Cartesian state [km, km/s].
    t_eval : ndarray              Output epochs [s].

    Returns
    -------
    scipy ODE solution object.  Access states via sol.y[:, k].
    """
    return solve_ivp(
        eom_scipy,
        [t_eval[0], t_eval[-1]],
        x0,
        method="DOP853",
        t_eval=t_eval,
        rtol=ODE_RTOL_TRUTH,
        atol=ODE_ATOL_TRUTH,
    )


def propagate_with_stm(x0: np.ndarray, t_eval: np.ndarray) -> "solve_ivp result":
    """
    Propagate state + STM using the augmented ODE.

    The STM is initialized to the identity matrix at t = t_eval[0].

    Parameters
    ----------
    x0     : ndarray, shape (6,)  Initial state [km, km/s].
    t_eval : ndarray              Output epochs [s].

    Returns
    -------
    sol : scipy ODE solution, shape (42, N).
        sol.y[:6, :]  — state trajectory.
        sol.y[6:, :]  — STM (flattened row-major, 36 components).
    """
    y0 = np.concatenate([x0, np.eye(6).flatten()])
    return solve_ivp(
        aug_eom,
        [t_eval[0], t_eval[-1]],
        y0,
        method="RK45",
        t_eval=t_eval,
        rtol=ODE_RTOL_EST,
        atol=ODE_ATOL_EST,
        max_step=ODE_MAX_STEP,
    )


# ===========================================================================
# Coordinate Transforms
# ===========================================================================

def lla_to_ecef(lat_deg: float, lon_deg: float, alt_km: float) -> np.ndarray:
    """
    Convert geodetic coordinates (LLA) to ECEF Cartesian (WGS84).

    Parameters
    ----------
    lat_deg : float  Geodetic latitude  [deg]
    lon_deg : float  Geodetic longitude [deg]
    alt_km  : float  Altitude above ellipsoid [km]

    Returns
    -------
    r_ecef : ndarray, shape (3,)  ECEF position [km].
    """
    lat = np.radians(lat_deg)
    lon = np.radians(lon_deg)
    e2  = 2.0 * F_EARTH - F_EARTH**2       # first eccentricity squared
    N   = RE / np.sqrt(1.0 - e2 * np.sin(lat)**2)  # prime vertical radius
    x   = (N + alt_km) * np.cos(lat) * np.cos(lon)
    y   = (N + alt_km) * np.cos(lat) * np.sin(lon)
    z   = (N * (1.0 - e2) + alt_km) * np.sin(lat)
    return np.array([x, y, z])


def ecef_to_eci(r_ecef: np.ndarray, t: float) -> np.ndarray:
    """
    Rotate ECEF vector to ECI using Earth's rotation (simplified GMST).

    Assumes the ECEF and ECI frames are aligned at t = 0.

    Parameters
    ----------
    r_ecef : ndarray, shape (3,)  ECEF position vector [km].
    t      : float                Elapsed time since epoch [s].

    Returns
    -------
    r_eci : ndarray, shape (3,)  ECI position vector [km].
    """
    theta = WE * t
    c, s  = np.cos(theta), np.sin(theta)
    R     = np.array([[c, -s, 0.0],
                      [s,  c, 0.0],
                      [0.0, 0.0, 1.0]])
    return R @ r_ecef


def gs_velocity_eci(r_ecef: np.ndarray, t: float) -> np.ndarray:
    """
    Compute the ECI velocity of a ground station due to Earth rotation.

    v_eci = dR/dt @ r_ecef  where  R = R_z(WE * t).

    Parameters
    ----------
    r_ecef : ndarray, shape (3,)  ECEF position of the station [km].
    t      : float                Elapsed time since epoch [s].

    Returns
    -------
    v_eci : ndarray, shape (3,)  ECI velocity of station [km/s].
    """
    theta = WE * t
    c, s  = np.cos(theta), np.sin(theta)
    # d/dt [R_z(WE*t)] @ r_ecef = WE * [-sin(theta), cos(theta), 0] for x-component, etc.
    vx = WE * (-r_ecef[0] * s - r_ecef[1] * c)
    vy = WE * ( r_ecef[0] * c - r_ecef[1] * s)
    vz = 0.0
    return np.array([vx, vy, vz])


# ===========================================================================
# Observation Model
# ===========================================================================

def compute_observation(
    r_sat_eci: np.ndarray,
    v_sat_eci: np.ndarray,
    gs_ecef: np.ndarray,
    t: float,
) -> tuple:
    """
    Compute two-way range, range-rate, and elevation angle.

    Range and range-rate are computed without light-time correction
    (consistent with the GMAT simulation script setting UseLightTime = True
    only for the GMAT simulator; our Python model omits it for simplicity
    since the correction is <10 ms for LEO altitudes).

    Parameters
    ----------
    r_sat_eci : ndarray, shape (3,)  Satellite ECI position [km].
    v_sat_eci : ndarray, shape (3,)  Satellite ECI velocity [km/s].
    gs_ecef   : ndarray, shape (3,)  Ground station ECEF position [km].
    t         : float                Elapsed time since epoch [s].

    Returns
    -------
    rng      : float  Two-way range [km].
    rng_rate : float  Range-rate (scalar Doppler) [km/s].
    elevation: float  Elevation angle at the ground station [deg].
    """
    gs_eci  = ecef_to_eci(gs_ecef, t)
    gs_vel  = gs_velocity_eci(gs_ecef, t)
    dr      = r_sat_eci - gs_eci       # line-of-sight vector [km]
    dv      = v_sat_eci - gs_vel       # relative velocity    [km/s]
    rng     = np.linalg.norm(dr)
    rhat    = dr / rng                 # unit LOS vector
    rng_rate = np.dot(dr, dv) / rng   # projection of dv on LOS
    # Elevation: angle between LOS and ground plane at station location
    gs_hat  = gs_eci / np.linalg.norm(gs_eci)
    elevation = np.degrees(np.arcsin(np.dot(rhat, gs_hat)))
    return rng, rng_rate, elevation


def measurement_partials(
    r_sat: np.ndarray,
    v_sat: np.ndarray,
    gs_ecef: np.ndarray,
    t: float,
    phi: np.ndarray,
) -> tuple:
    """
    Compute the measurement sensitivity matrix H mapped to epoch 0
    via the State Transition Matrix (STM).

    The local partials (at observation time t_obs) are:
        H_range_local     = [r_hat | 0_3]          (1x6)
        H_rangerate_local = [dv/rng - rr/rng*r_hat | r_hat]  (1x6)

    These are mapped to epoch 0:
        H_range     = H_range_local     @ Phi(t_obs, t_0)
        H_rangerate = H_rangerate_local @ Phi(t_obs, t_0)

    Parameters
    ----------
    r_sat  : ndarray (3,)   Satellite ECI position at t_obs [km].
    v_sat  : ndarray (3,)   Satellite ECI velocity at t_obs [km/s].
    gs_ecef: ndarray (3,)   Ground station ECEF [km].
    t      : float          Observation time [s].
    phi    : ndarray (6,6)  STM: Phi(t_obs, t_0).

    Returns
    -------
    H_range     : ndarray, shape (6,)  Range sensitivity row at epoch 0.
    H_rangerate : ndarray, shape (6,)  Range-rate sensitivity row at epoch 0.
    rng         : float                Computed range [km].
    rng_rate    : float                Computed range-rate [km/s].
    """
    gs_eci = ecef_to_eci(gs_ecef, t)
    gs_vel = gs_velocity_eci(gs_ecef, t)
    dr     = r_sat - gs_eci
    dv     = v_sat - gs_vel
    rng    = np.linalg.norm(dr)
    rhat   = dr / rng
    rng_rate = np.dot(dr, dv) / rng

    # Local partials w.r.t. [r, v] at t_obs
    H_r_loc   = np.concatenate([rhat, np.zeros(3)])
    H_rr_loc  = np.concatenate([
        dv / rng - (rng_rate / rng) * rhat,   # d(rr)/d(r)
        rhat,                                   # d(rr)/d(v)
    ])

    # Map to epoch via STM
    H_range     = H_r_loc  @ phi
    H_rangerate = H_rr_loc @ phi
    return H_range, H_rangerate, rng, rng_rate