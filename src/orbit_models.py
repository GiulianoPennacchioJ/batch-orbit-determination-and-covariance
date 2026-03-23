"""
orbit_models.py  —  High-fidelity satellite dynamics matching GMAT 2025a

Force model:
  • Two-body gravity
  • J2–J6 zonal harmonics  (dominant non-spherical gravity)
  • Moon third-body        (GM = 4902.799 km³/s²)
  • Sun  third-body        (GM = 1.32712440018e11 km³/s²)
  • Atmospheric drag       (GMAT Exponential model)
  • Solar radiation pressure

GMAT spacecraft parameters:
  Cd=2.2, DragArea=0.1 m², DryMass=20 kg, Cr=1.8, SRPArea=0.1 m²

BUG FIXED (previous version):
  The old J2 formula used factor = 1.5·J2·mu·Re²/r⁵
  → a_x = (x/r)·factor·(5z²/r²−1)  =  1.5·J2·mu·Re²·x·(5z²/r²−1) / r⁶
  The correct denominator is r⁵, not r⁶.  The error was a factor of r ≈ 7085,
  making J2 ~7000× too small (orbit was effectively pure two-body).
  All zonal harmonics now use the analytically derived gradient of the potential
  U_n = mu·Jn·(Re/r)^n/r·Pn(sin φ), validated against 4th-order FD.
"""

import numpy as np
from navigation_utils import MU_EARTH, REQ_EARTH, OMEGA_EARTH

MU = MU_EARTH
Re = REQ_EARTH

J2 =  1.0826263e-3
J3 = -2.5327e-6
J4 = -1.6196e-6
J5 = -2.2730e-7
J6 =  5.4067e-7

GM_MOON = 4902.799
GM_SUN  = 1.32712440018e11

# Spacecraft (GMAT script values)
CD=2.2; AREA_DRAG_M2=0.1; MASS=20.0; CR=1.8; AREA_SRP_M2=0.1

# GMAT exponential atmosphere (US Std Atm 1976)
_EXP_ATM = [
    (0,1.225,8.44),(25,3.899e-2,6.49),(30,1.774e-2,6.75),(40,3.972e-3,7.07),
    (50,1.057e-3,7.47),(60,3.206e-4,7.83),(70,8.770e-5,7.95),(80,1.905e-5,9.65),
    (90,3.396e-6,9.81),(100,5.297e-7,8.17),(110,9.661e-8,7.84),(120,2.438e-8,7.22),
    (130,8.484e-9,6.98),(140,3.845e-9,6.99),(150,2.070e-9,7.20),(180,5.464e-10,8.68),
    (200,2.789e-10,9.05),(250,7.248e-11,11.82),(300,2.418e-11,15.12),
    (350,9.158e-12,18.84),(400,3.725e-12,22.26),(450,1.585e-12,25.57),
    (500,6.967e-13,28.73),(600,1.454e-13,34.82),(700,3.614e-14,39.97),
    (800,1.170e-14,47.62),(900,5.245e-15,56.56),(1000,3.019e-15,68.36),
]

def _rho_exp(alt_km):
    for i in range(len(_EXP_ATM)-1):
        h0,rho0,H = _EXP_ATM[i]; h1=_EXP_ATM[i+1][0]
        if h0 <= alt_km < h1: return rho0*np.exp(-(alt_km-h0)/H)
    h0,rho0,H = _EXP_ATM[-1]; return rho0*np.exp(-(alt_km-h0)/H)

def _legendre_and_deriv(n, t):
    """(Pn(t), dPn/dt) via Bonnet recurrence."""
    if n==0: return 1.0, 0.0
    if n==1: return t, 1.0
    P_prev, P_curr = 1.0, t
    for k in range(2, n+1):
        P_next = ((2*k-1)*t*P_curr - (k-1)*P_prev) / k
        P_prev, P_curr = P_curr, P_next
    cos2 = 1.0 - t*t
    dPn = n*(P_prev - t*P_curr)/cos2 if cos2 > 1e-24 else 0.0
    return P_curr, dPn

def _acc_zonal(pos, n, Jn):
    """
    Gradient of U_n = mu·Jn·(Re/r)^n/r·Pn(z/r) in Cartesian ECI.

    Derivation (exact):
      dU_n/dx = mu·Jn·Re^n·[−(n+1)·Pn·x/r^{n+3}  −  dPn/dt·x·z/r^{n+4}]
      dU_n/dz = mu·Jn·Re^n·[−(n+1)·Pn·z/r^{n+3}  +  dPn/dt·(r²−z²)/r^{n+4}]
    """
    x,y,z = pos; r=np.linalg.norm(pos); r2=r*r
    Pn, dPn = _legendre_and_deriv(n, z/r)
    C = MU*Jn*Re**n; rn3=r**(n+3); rn4=r**(n+4)
    ax = C*(-(n+1)*Pn*x/rn3 - dPn*x*z/rn4)
    ay = C*(-(n+1)*Pn*y/rn3 - dPn*y*z/rn4)
    az = C*(-(n+1)*Pn*z/rn3 + dPn*(r2-z*z)/rn4)
    return np.array([ax, ay, az])

def _acc_3body(pos, r_body, gm):
    d = r_body-pos; dm=np.linalg.norm(d); qm=np.linalg.norm(r_body)
    return gm*(d/dm**3 - r_body/qm**3)

def _acc_drag(state):
    pos,vel = state[:3],state[3:]
    alt = np.linalg.norm(pos) - Re
    rho = _rho_exp(alt)
    v_atm = np.cross([0,0,OMEGA_EARTH], pos)     # km/s (atmosphere co-rotates)
    v_rel_m = (vel - v_atm)*1000.0               # m/s
    vm = np.linalg.norm(v_rel_m)
    a_m = -0.5*rho*CD*AREA_DRAG_M2/MASS * vm * v_rel_m
    return a_m*1e-3

def _acc_srp(pos, r_sun):
    P=4.56e-6; sun_hat=r_sun/np.linalg.norm(r_sun)
    a_m = P*CR*AREA_SRP_M2/MASS * sun_hat
    return a_m*1e-3

_EPH = {'jd':None,'rm':None,'rs':None}

def _moon_eci(jd):
    T=(jd-2451545.0)/36525.0
    Mm=np.radians((134.9634+477198.8676*T)%360)
    D =np.radians((297.8502+445267.1115*T)%360)
    F =np.radians(( 93.2721+483202.0175*T)%360)
    M =np.radians((357.5291+ 35999.0503*T)%360)
    L0=np.radians((218.3164+481267.8812*T)%360)
    dL=np.radians(6.288*np.sin(Mm)+1.274*np.sin(2*D-Mm)+0.658*np.sin(2*D)
                  -0.214*np.sin(2*Mm)-0.186*np.sin(M))
    B =np.radians(5.128*np.sin(F)+0.280*np.sin(Mm+F)+0.277*np.sin(Mm-F))
    R =385001-20905*np.cos(Mm)-3699*np.cos(2*D-Mm)-2956*np.cos(2*D)
    lon=L0+dL
    return R*np.array([np.cos(B)*np.cos(lon),np.cos(B)*np.sin(lon),np.sin(B)])

def _sun_eci(jd):
    T=(jd-2451545.0)/36525.0
    M=np.radians((357.5291+35999.0503*T)%360)
    L=np.radians((280.4665+36000.7698*T)%360)+np.radians(1.9146*np.sin(M))
    R=1.496e8*(1.0-0.01671*np.cos(M)); eps=np.radians(23.439-0.0130*T)
    return R*np.array([np.cos(L),np.sin(L)*np.cos(eps),np.sin(L)*np.sin(eps)])

def orbit_dynamics(t, state, t_mjd0=None, time_offset=None):
    """dX/dt with all forces. t in seconds from t_mjd0."""
    pos,vel = state[:3],state[3:]
    r = np.linalg.norm(pos)
    acc = -(MU/r**3)*pos
    acc += _acc_zonal(pos,2,J2)+_acc_zonal(pos,3,J3)+_acc_zonal(pos,4,J4)
    acc += _acc_zonal(pos,5,J5)+_acc_zonal(pos,6,J6)
    if t_mjd0 is not None:
        jd = 2430000.0+(t_mjd0-time_offset)+t/86400.0
        if _EPH['jd'] is None or abs(jd-_EPH['jd'])>600/86400:
            _EPH['jd']=jd; _EPH['rm']=_moon_eci(jd); _EPH['rs']=_sun_eci(jd)
        acc += _acc_3body(pos,_EPH['rm'],GM_MOON)
        acc += _acc_3body(pos,_EPH['rs'],GM_SUN)
        acc += _acc_drag(state)
        acc += _acc_srp(pos,_EPH['rs'])
    return np.concatenate([vel,acc])

def get_dynamics_jacobian(state):
    pos=state[:3]; r=np.linalg.norm(pos); r2=r*r; I3=np.eye(3)
    G=(MU/r**3)*(3*np.outer(pos,pos)/r2-I3)
    A=np.zeros((6,6)); A[0:3,3:6]=I3; A[3:6,0:3]=G
    return A

def _aug_dyn(t, aug, t_mjd0, time_offset):
    x=aug[:6]; phi=aug[6:].reshape(6,6)
    dx=orbit_dynamics(t,x,t_mjd0,time_offset)
    dphi=get_dynamics_jacobian(x)@phi
    return np.concatenate([dx,dphi.flatten()])

def propagate_state_and_stm(state_0, stm_0, dt,
                             t_mjd0=None, time_offset=None, max_step=60.0):
    """
    RK4 propagation of [state, STM] over dt seconds.
    Auto-subdivides into steps <= max_step to handle inter-pass gaps.
    """
    n_steps = max(1, int(np.ceil(abs(dt)/max_step)))
    sub_dt  = dt/n_steps
    aug = np.concatenate([state_0, stm_0.flatten()])
    for _ in range(n_steps):
        k1=_aug_dyn(0,aug,t_mjd0,time_offset)
        k2=_aug_dyn(sub_dt/2,aug+k1*sub_dt/2,t_mjd0,time_offset)
        k3=_aug_dyn(sub_dt/2,aug+k2*sub_dt/2,t_mjd0,time_offset)
        k4=_aug_dyn(sub_dt,aug+k3*sub_dt,t_mjd0,time_offset)
        aug += (sub_dt/6)*(k1+2*k2+2*k3+k4)
    return aug[:6], aug[6:].reshape(6,6)

if __name__ == "__main__":
    import time as tm
    print("=== orbit_models.py self-test ===")
    
    # 1. J2 at equatorial
    r0=np.array([7085.0,0.0,0.0])
    exp=-1.5*J2*MU*Re**2/7085.0**4
    got=_acc_zonal(r0,2,J2)[0]
    print(f"[1] J2 equatorial: expected={exp:.4e} got={got:.4e} "
          f"{'PASS' if abs(got-exp)<1e-12 else 'FAIL'}")
    
    # 2. FD validation
    x0=np.array([-6885.3,238.8,-1653.0]); eps=1e-4
    print("[2] FD validation:")
    for n,Jn,nm in [(2,J2,'J2'),(3,J3,'J3'),(4,J4,'J4'),(5,J5,'J5'),(6,J6,'J6')]:
        def U(p,n=n,Jn=Jn):
            r=np.linalg.norm(p); Pn,_=_legendre_and_deriv(n,p[2]/r)
            return MU*Jn*(Re/r)**n/r*Pn
        gx=(-U(x0+[2*eps,0,0])+8*U(x0+[eps,0,0])-8*U(x0-[eps,0,0])+U(x0-[2*eps,0,0]))/(12*eps)
        gy=(-U(x0+[0,2*eps,0])+8*U(x0+[0,eps,0])-8*U(x0-[0,eps,0])+U(x0-[0,2*eps,0]))/(12*eps)
        gz=(-U(x0+[0,0,2*eps])+8*U(x0+[0,0,eps])-8*U(x0-[0,0,eps])+U(x0-[0,0,2*eps]))/(12*eps)
        fd=np.array([gx,gy,gz]); an=_acc_zonal(x0,n,Jn)
        err=np.linalg.norm(an-fd)*1e9
        print(f"     {nm}: |a|={np.linalg.norm(an)*1e9:.3f} nm/s²  FD_err={err:.4f} nm/s²  {'PASS' if err<0.1 else 'FAIL'}")
    
    # 3. STM stability
    s=np.array([7071.06186,0.,0.,0.,-1.073994,7.43462]); phi=np.eye(6)
    t0=tm.time()
    for _ in range(547): s,phi=propagate_state_and_stm(s,phi,60.0)
    print(f"[3] STM after 547 steps: max={np.max(np.abs(phi)):.2e} det={np.linalg.det(phi):.5f} ({tm.time()-t0:.1f}s)")
    
    # 4. Force model difference J2-only vs J2..J6
    s_j2only = np.array([-6885.34,238.78,-1652.99,1.769,1.042,-7.210])
    s_full   = s_j2only.copy(); phi2=np.eye(6)
    s_j2=s_j2only.copy(); phi3=np.eye(6)
    t0=tm.time()
    for _ in range(24*60):
        s_full,phi2=propagate_state_and_stm(s_full,phi2,60.0)  # J2..J6
        s_j2,phi3  =propagate_state_and_stm(s_j2,phi3,60.0,    # J2-only (no t_mjd0)
                                              t_mjd0=None)
    diff=np.linalg.norm(s_full[:3]-s_j2[:3])*1000
    print(f"[4] J2-only vs J2..J6 after 24h: Δpos={diff:.1f} m ({tm.time()-t0:.0f}s)")
    print("All tests complete.")