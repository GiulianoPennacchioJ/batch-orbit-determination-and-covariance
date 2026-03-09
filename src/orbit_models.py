import numpy as np
from navigation_utils import MU_EARTH, REQ_EARTH, J2_EARTH

# =============================================================================
# SATELLITE DYNAMICS (Equations of Motion)
# =============================================================================

def get_j2_acceleration(state):
    """
    Calculates the acceleration due to the J2 perturbation (Earth's oblateness).
    Input:  state [x, y, z, vx, vy, vz] in km and km/s
    Output: acc_j2 [ax, ay, az] in km/s^2
    """
    r_vec = state[0:3]
    r = np.linalg.norm(r_vec)
    x, y, z = r_vec
    
    # Pre-calculate constant terms for efficiency
    r2 = r**2
    z2 = z**2
    re_r_2 = (REQ_EARTH / r)**2
    
    # Common factor for the J2 perturbation formula
    # Factor = (1.5 * J2 * Mu * Re^2) / r^5
    factor = (1.5 * J2_EARTH * MU_EARTH * REQ_EARTH**2) / (r**5)
    
    # J2 Acceleration components in ECI frame
    ax_j2 = (x / r) * factor * (5 * (z2 / r2) - 1)
    ay_j2 = (y / r) * factor * (5 * (z2 / r2) - 1)
    az_j2 = (z / r) * factor * (5 * (z2 / r2) - 3)
    
    return np.array([ax_j2, ay_j2, az_j2])

def orbit_dynamics(t, state):
    """
    Computes the state derivative (f = dX/dt).
    Includes:
    1. Newtonian Two-Body Gravity (point mass)
    2. J2 Perturbation (Earth's non-spherical shape)
    """
    pos = state[0:3]
    vel = state[3:6]
    r = np.linalg.norm(pos)
    
    # 1. Newtonian Acceleration (2-Body)
    # Acc = - (mu / r^3) * r_vector
    acc_newton = - (MU_EARTH / r**3) * pos
    
    # 2. J2 Perturbation Acceleration
    acc_j2 = get_j2_acceleration(state)
    
    # Combine all accelerations
    acc_total = acc_newton + acc_j2
    
    # The state derivative is [Velocity, Acceleration]
    return np.concatenate([vel, acc_total])

# =============================================================================
# NUMERICAL INTEGRATOR (Runge-Kutta 4th Order)
# =============================================================================

def rk4_step(f, t, state, dt):
    """
    Performs a single integration step using the Runge-Kutta 4 (RK4) algorithm.
    RK4 provides a high degree of accuracy for orbital mechanics problems.
    """
    k1 = f(t, state)
    k2 = f(t + dt/2, state + k1 * dt/2)
    k3 = f(t + dt/2, state + k2 * dt/2)
    k4 = f(t + dt, state + k3 * dt)
    
    return state + (dt/6) * (k1 + 2*k2 + 2*k3 + k4)

# =============================================================================
# DYNAMICS JACOBIAN (Matrix A)
# =============================================================================

def get_dynamics_jacobian(state):
    """
    Computes the Jacobian matrix A = df/dx (the partial derivatives of the dynamics).
    This matrix is essential for propagating the State Transition Matrix (STM).
    
    The structure of A (6x6) is:
    [ 0(3x3)      I(3x3) ]
    [ G(3x3)      0(3x3) ]
    where G is the Gravity Gradient Matrix.
    """
    pos = state[0:3]
    r = np.linalg.norm(pos)
    r2 = r**2
    
    # Identity matrix for position/velocity coupling
    I3 = np.eye(3)
    
    # Newtonian Gravity Gradient Matrix (G)
    # G_ij = d(acc_i) / d(pos_j)
    # Formula: G = (mu/r^3) * [ (3 * r_vec * r_vec_T / r^2) - I ]
    r_outer = np.outer(pos, pos)
    G = (MU_EARTH / r**3) * ((3 * r_outer / r2) - I3)
    
    # Assemble the 6x6 Jacobian Matrix
    A = np.zeros((6, 6))
    A[0:3, 3:6] = I3  # Derivative of Position w.r.t Velocity
    A[3:6, 0:3] = G   # Derivative of Acceleration w.r.t Position
    
    return A

# =============================================================================
# SIMULTANEOUS PROPAGATION (State & STM)
# =============================================================================

def propagate_state_and_stm(state_0, stm_0, dt):
    """
    Propagates both the satellite state and the State Transition Matrix (STM).
    
    The STM (Phi) is used in Batch OD to map errors from the current time 
    back to the initial epoch (t0).
    The evolution is governed by: dPhi/dt = A * Phi.
    """
    # 1. Propagate the State using RK4
    state_next = rk4_step(orbit_dynamics, 0, state_0, dt)
    
    # 2. Propagate the STM (First-order approximation: Phi_new = Phi + A*Phi*dt)
    A = get_dynamics_jacobian(state_0)
    stm_dot = A @ stm_0
    stm_next = stm_0 + stm_dot * dt
    
    return state_next, stm_next

# =============================================================================
# MODULE TEST BLOCK
# =============================================================================
if __name__ == "__main__":
    # Test state (approximate LEO state in km and km/s)
    s0 = np.array([7078.14, 0, 0, 0, 7.5, 0]) 
    
    # The initial STM is always the Identity Matrix
    phi0 = np.eye(6) 
    
    print("--- Orbit Models Module Test ---")
    s1, phi1 = propagate_state_and_stm(s0, phi0, 60.0)
    
    print(f"State after 60s:\n{s1}")
    print(f"\nSTM after 60s (Position-to-Position submatrix):\n{phi1[0:3, 0:3]}")
    print("\nTest completed successfully.")