"""
diagnose.py — Run this FIRST before main_batch_od.py to identify the issue.
"""
import sys, os
import numpy as np

print("=" * 60)
print("BATCH OD ENVIRONMENT DIAGNOSTIC")
print("=" * 60)

# 1. Show which files are actually being loaded
print("\n[1] MODULE FILE LOCATIONS:")
import navigation_utils, orbit_models, batch_estimator, measurement_models
print(f"  navigation_utils : {navigation_utils.__file__}")
print(f"  orbit_models     : {orbit_models.__file__}")
print(f"  batch_estimator  : {batch_estimator.__file__}")
print(f"  measurement_models: {measurement_models.__file__}")

# 2. Verify the STM fix is active
print("\n[2] STM INTEGRATION METHOD CHECK:")
import inspect
src = inspect.getsource(orbit_models.propagate_state_and_stm)
if 'augmented_dynamics' in src or 'aug_0' in src:
    print("  ✓ propagate_state_and_stm uses RK4 augmented state (FIXED)")
elif 'A_mid @ stm_0 * dt' in src or 'stm_0 + stm_dot' in src:
    print("  ✗ propagate_state_and_stm uses Euler step (OLD BUGGY VERSION)")
    print("    -> DELETE __pycache__ and replace orbit_models.py")
else:
    print("  ? Cannot determine STM method. Check orbit_models.py manually.")

# 3. STM numerical test
print("\n[3] STM NUMERICAL BOUNDEDNESS TEST:")
s0 = np.array([7071.061860, 0.0, 0.0, 0.0, -1.073994, 7.434620])
phi0 = np.eye(6)
s, phi = s0.copy(), phi0.copy()
for _ in range(10):  # Only 10 steps for quick check
    s, phi = orbit_models.propagate_state_and_stm(s, phi, 60.0)
max_stm = np.max(np.abs(phi))
print(f"  STM max element after 10 steps (60s each): {max_stm:.4e}")
if max_stm < 1e6:
    print("  ✓ STM is bounded (RK4 fix working)")
else:
    print("  ✗ STM is diverging (old Euler code still active)")

# 4. Station coordinates check
print("\n[4] STATION COORDINATES CHECK:")
from navigation_utils import get_station_coords, geodetic_to_ecef
coords = get_station_coords()
dongara_ref  = geodetic_to_ecef(-29.04,  114.88, 0.03)
santiago_ref = geodetic_to_ecef(-33.15,  289.33, 0.73)
err_d = np.linalg.norm(coords['Dongara']  - dongara_ref)  * 1000
err_s = np.linalg.norm(coords['Santiago'] - santiago_ref) * 1000
print(f"  Dongara  error vs GMAT geodetic: {err_d:.2f} m", "✓" if err_d < 1 else "✗ WRONG")
print(f"  Santiago error vs GMAT geodetic: {err_s:.2f} m", "✓" if err_s < 1 else "✗ WRONG")

# 5. First residual sanity check
print("\n[5] FIRST RESIDUAL SANITY CHECK:")
from measurement_models import compute_range_and_doppler
from orbit_models import propagate_state_and_stm

# Propagate from GMAT initial state to first observation
x0 = np.array([7071.061860, 0.0, 0.0, 0.0, -1.073994254, 7.434619660])
t_truth_epoch = 30311.00042863868  # A1 MJD of truth row 0
t_first_obs   = 30311.3802893518   # A1 MJD of first .gmd observation
dt_total = (t_first_obs - t_truth_epoch) * 86400.0

x = x0.copy()
phi = np.eye(6)
n_steps = int(dt_total / 60.0)
remainder = dt_total - n_steps * 60.0
for _ in range(n_steps):
    x, phi = propagate_state_and_stm(x, phi, 60.0)
if remainder > 0:
    x, phi = propagate_state_and_stm(x, phi, remainder)

time_offset = 37.034382 / 86400.0
r_pred, _ = compute_range_and_doppler(x, coords['Dongara'], t_first_obs, time_offset)
r_obs = 3777.4974811470775  # km from first .gmd line

print(f"  Propagated satellite |r|: {np.linalg.norm(x[:3]):.3f} km (expected ~7085 km)")
print(f"  Predicted 2-Way Range:    {r_pred:.4f} km")
print(f"  Observed  2-Way Range:    {r_obs:.4f} km")
print(f"  Residual:                 {r_obs - r_pred:.4f} km")
if abs(r_pred) > 100000:
    print("  ✗ Predicted range is WILDLY wrong — station ECI or state is broken")
elif abs(r_obs - r_pred) > 1000:
    print("  ✗ Large residual — possible force model accumulation or wrong epoch")
else:
    print("  ✓ Residual is physically plausible (force model mismatch)")

# 6. Check for __pycache__ stale files
print("\n[6] PYCACHE FILES (potential stale .pyc):")
for root, dirs, files in os.walk('.'):
    for f in files:
        if f.endswith('.pyc') and any(m in f for m in ['orbit_models', 'navigation_utils', 'batch_estimator']):
            print(f"  Found: {os.path.join(root, f)}")
            print("  -> Run: find . -name '*.pyc' -delete && find . -name __pycache__ -type d -exec rm -rf {} +")

print("\n" + "=" * 60)
print("DIAGNOSTIC COMPLETE")
print("=" * 60)

# 7. Large-step subdivision test (THE KEY NEW TEST)
print("\n[7] LARGE-STEP SUBDIVISION TEST (inter-pass gap fix):")
s0  = np.array([7071.061860, 0.0, 0.0, 0.0, -1.073994254, 7.434619660])
phi0 = np.eye(6)

# Reference: 90 fine steps of 60s
s_fine, p_fine = s0.copy(), phi0.copy()
for _ in range(90):
    s_fine, p_fine = orbit_models.propagate_state_and_stm(s_fine, p_fine, 60.0)

# Single call with dt=5400s (what happens at inter-pass gaps)
s_coarse, p_coarse = orbit_models.propagate_state_and_stm(s0.copy(), phi0.copy(), 5400.0)

err_m = np.linalg.norm(s_coarse - s_fine) * 1000
print(f"  propagate_state_and_stm(x, phi, dt=5400s) vs 90x60s:")
print(f"  State error: {err_m:.2f} m", "✓ (< 1m, subdivision working)" if err_m < 1 else "✗ CRITICAL BUG — no subdivision, error = 55,000 km!")
if err_m > 1000:
    print("  -> This is the root cause of 816 million m Range RMS.")
    print("  -> Fix: add max_step=60 subdivision inside propagate_state_and_stm.")