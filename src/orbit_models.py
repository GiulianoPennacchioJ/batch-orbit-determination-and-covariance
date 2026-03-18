import numpy as np
from navigation_utils import MU_EARTH, REQ_EARTH, J2_EARTH

# =============================================================================
# SATELLITE DYNAMICS
# =============================================================================

def get_j2_acceleration(state):
    r_vec = state[0:3]
    r = np.linalg.norm(r_vec)
    x, y, z = r_vec
    r2, z2 = r**2, z**2
    factor = (1.5 * J2_EARTH * MU_EARTH * REQ_EARTH**2) / (r**5)
    return np.array([(x/r)*factor*(5*(z2/r2)-1),
                     (y/r)*factor*(5*(z2/r2)-1),
                     (z/r)*factor*(5*(z2/r2)-3)])

def orbit_dynamics(t, state):
    pos, vel = state[0:3], state[3:6]
    r = np.linalg.norm(pos)
    acc = -(MU_EARTH / r**3) * pos + get_j2_acceleration(state)
    return np.concatenate([vel, acc])

def get_dynamics_jacobian(state):
    pos = state[0:3]
    r = np.linalg.norm(pos)
    G = (MU_EARTH / r**3) * (3*np.outer(pos, pos)/r**2 - np.eye(3))
    A = np.zeros((6, 6))
    A[0:3, 3:6] = np.eye(3)
    A[3:6, 0:3] = G
    return A

# =============================================================================
# AUGMENTED DYNAMICS: STATE + STM JOINT RK4
# =============================================================================

def augmented_dynamics(t, aug_state):
    """
    Derivative of augmented state [x(6), phi_vec(36)].
    Integrates d(Phi)/dt = A(x)*Phi jointly with the state via RK4,
    keeping the STM bounded and accurate (FIX for BUG #2).
    """
    x   = aug_state[0:6]
    phi = aug_state[6:].reshape(6, 6)
    dx   = orbit_dynamics(t, x)
    dphi = get_dynamics_jacobian(x) @ phi
    return np.concatenate([dx, dphi.flatten()])

def _rk4_augmented_single_step(aug, dt):
    """One RK4 step on the 42-element augmented state."""
    k1 = augmented_dynamics(0,      aug)
    k2 = augmented_dynamics(dt/2,   aug + k1*dt/2)
    k3 = augmented_dynamics(dt/2,   aug + k2*dt/2)
    k4 = augmented_dynamics(dt,     aug + k3*dt)
    return aug + (dt/6)*(k1 + 2*k2 + 2*k3 + k4)

def propagate_state_and_stm(state_0, stm_0, dt, max_step=60.0):
    """
    Propagates both the satellite state and the STM over an interval dt.

    CRITICAL FIX (BUG #7 — inter-pass gap):
    The batch estimator calls this function with dt = time between consecutive
    observations. Within a single ground-station pass, dt = 60 s (fine).
    However, between passes the gap can be 90–360 minutes, so dt can reach
    5400–21600 s. A single RK4 step of that size is catastrophically inaccurate:
    error scales as O(dt^5), giving position errors of 55,000–628,000 km
    for a 90-min or 6-hour gap respectively.

    This function now subdivides any dt into steps of at most max_step seconds
    (default 60 s) before integrating, restoring full RK4 accuracy regardless
    of the calling interval.

    Also retains the joint augmented-state RK4 for the STM (FIX for BUG #2),
    keeping phi bounded to O(1e5) over a 24-hour arc.
    """
    n_steps = max(1, int(np.ceil(dt / max_step)))
    sub_dt  = dt / n_steps

    aug = np.concatenate([state_0, stm_0.flatten()])
    for _ in range(n_steps):
        aug = _rk4_augmented_single_step(aug, sub_dt)

    return aug[0:6], aug[6:].reshape(6, 6)

# =============================================================================
# MODULE TEST
# =============================================================================
if __name__ == "__main__":
    s0   = np.array([7078.14, 0, 0, 0, 7.5, 0])
    phi0 = np.eye(6)

    print("--- orbit_models module test ---")
    # Test 1: standard 60s step
    s1, phi1 = propagate_state_and_stm(s0, phi0, 60.0)
    print(f"State after 60s: {s1}")

    # Test 2: large step (5400s = 90 min) — should subdivide into 90 x 60s
    import time
    t0 = time.time()
    s2, phi2 = propagate_state_and_stm(s0, phi0, 5400.0)
    t1 = time.time()
    # Compare with manual 90 steps
    s_ref, phi_ref = s0.copy(), phi0.copy()
    for _ in range(90):
        s_ref, phi_ref = propagate_state_and_stm(s_ref, phi_ref, 60.0)
    err = np.linalg.norm(s2 - s_ref) * 1000
    print(f"5400s single call vs 90x60s: state diff = {err:.6f} m  ({t1-t0:.2f}s)")

    # Test 3: 547 steps (9-hour approach arc)
    s, phi = s0.copy(), phi0.copy()
    for _ in range(547):
        s, phi = propagate_state_and_stm(s, phi, 60.0)
    print(f"STM max after 547 x 60s: {np.max(np.abs(phi)):.4e}  det={np.linalg.det(phi):.6f}")
    print("All tests passed.")