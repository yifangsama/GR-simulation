import numpy as np
from numba import jit

# Numba JIT Compiled Kernels

@jit(nopython=True, cache=True)
def _core_timelike(tau, Y, params):
    """
    Time-like geodesic equation. m = 1
    arguments:  Y = [r, u_r, theta, u_theta, t, phi]
                params = [M, a, E, L, C]
    """
    r, u_r, th, u_th, t, phi = Y
    M, a, E, L, C = params
    
    r2 = r * r
    a2 = a * a
    Delta = r2 - 2*M*r + a2
    dDelta_dr = 2*r - 2*M
    
    sin_th = np.sin(th)
    cos_th = np.cos(th)
    sin2 = sin_th * sin_th
    sin2_safe = np.maximum(sin2, 1e-12)

    # R
    P = (r2 + a2) * E - a * L
    K = (L - a * E)**2 + C
    d2r = 2 * r * E * P - 0.5 * (dDelta_dr * (r2 + K) + Delta * 2 * r)
    # Theta
    dTheta_dth = -a2 * (E**2 - 1.0) * np.sin(2*th) + 2 * L**2 * cos_th / (sin2_safe * sin_th)
    d2th = 0.5 * dTheta_dth

    # Time & Phi
    dt_dlam = (r2 + a2) * P / Delta - a * (a * E * sin2 - L)
    dphi_dlam = a * P / Delta + (L / sin2_safe - a * E)

    return np.array([u_r, d2r, u_th, d2th, dt_dlam, dphi_dlam])

@jit(nopython=True, cache=True)
def _core_null(tau, Y, params):
    """
    Null geodesic equation.
    arguments:  Y = [r, u_r, theta, u_theta, t, phi]
                parms = [M, a, lambda, eta]
    """
    r, u_r, th, u_th, t, phi = Y
    M, a, lam, eta = params
    
    r2 = r * r
    a2 = a * a
    Delta = r2 - 2*M*r + a2
    dDelta_dr = 2*r - 2*M
    
    sin_th = np.sin(th)
    cos_th = np.cos(th)
    sin2 = sin_th * sin_th
    sin2_safe = np.maximum(sin2, 1e-12)
    
    # R
    P = r2 + a2 - a * lam
    K = eta + (lam - a)**2
    d2r = 2 * r * P - 0.5 * dDelta_dr * K

    # Theta
    dTheta_dth = -a2 * np.sin(2*th) + 2 * lam**2 * cos_th / (sin2_safe * sin_th)
    d2th = 0.5 * dTheta_dth

    # Time & Phi
    dt_dlam = (r2 + a2) * P / Delta + (a * lam - a2 * sin2)
    dphi_dlam = a * P / Delta + (lam / sin2_safe - a)

    return np.array([u_r, d2r, u_th, d2th, dt_dlam, dphi_dlam])

# Solver Class

class KerrMinoSolver:
    def __init__(self, M, a):
        self.M = float(M)
        self.a = float(a)
        
    def choose_mode(self, mode='null'):
        """return JIT function"""
        if mode == 'timelike':
            return _core_timelike
        elif mode == 'null':
            return _core_null
        else:
            raise ValueError("Mode must be 'timelike' or 'null'")

    def pack_params(self, constants, mode='null'):
        if mode == 'timelike':
            # M, a, E, L, C
            return np.array([self.M, self.a, constants[0], constants[1], constants[2]], dtype=np.float64)
        elif mode == 'null':
            # M, a, lambda, eta
            return np.array([self.M, self.a, constants[0], constants[1]], dtype=np.float64)
        else:
            raise ValueError("Unknown mode")