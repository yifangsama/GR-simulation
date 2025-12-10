import numpy as np
from numba import jit

@jit(nopython=True, cache=True)
def get_kerr_acceleration(x, v, M, a, G=1.0):
    """
    计算 Kerr 度规下的测地线加速度 (d^2x / dtau^2)。
    基于 Sean Carroll Eq 6.70 计算度规及其导数。
    
    Inputs:
        x: [t, r, theta, phi]
        v: [ut, ur, utheta, uphi]
        M: 黑洞质量
        a: 黑洞自旋 (J/M)
        G: 引力常数
        
    Returns:
        acc: [at, ar, atheta, aphi]
    """
    t, r, th, phi = x
    
    # --- 1. 预计算辅助变量 ---
    sin_th = np.sin(th)
    cos_th = np.cos(th)
    sin2 = sin_th**2
    cos2 = cos_th**2
    r2 = r**2
    a2 = a**2

    s2t = 2.0*sin_th*cos_th
    Delta = r2-2.0*G*M*r+a2
    Sigma = r2+a2*cos2
    p1D = 2.0*(r-G*M)
    p1S = 2.0*r
    p2S = -a2*s2t

    g = np.zeros((4, 4))
    g[0, 0] = -(1.0 - 2.0*G*M*r/Sigma)
    g[1, 1] = Sigma / Delta
    g[2, 2] = Sigma
    g[3, 3] = (sin2 / Sigma) * ((r2 + a2)**2 - a2 * Delta * sin2)
    gtphi = -(2.0*G*M*a*r*sin2) / Sigma
    g[0, 3] = gtphi
    g[3, 0] = gtphi

    # p1g = np.zeros((4,4))
    # p2g = np.zeros((4,4))
    """
    p1g[0,0] = 2.0*G*M*(Sigma-r*p1S)/Sigma**2
    p1g[1,1] = (Delta*p1S-Sigma*p1D)/Delta**2
    p1g[2,2] = p1S
    p1g[3,0] = -a*sin2*p1g[0,0]
    p1g[0,3] = p1g[3,0]
    p1g[3,3] = sin2*(Sigma*(4.0*r*(r2+a2)-a2*sin2*p1D)-p1S*((r2+a2)**2-a2*Delta*sin2))/Sigma**2
    p2g[0,0] = -2.0*G*M*r*p2S/Sigma**2
    p2g[1,1] = p2S/Delta
    p2g[2,2] = p2S
    p2g[3,0] = -2.0*G*M*a*r*(Sigma*s2t-sin2*p2S)/Sigma**2
    p2g[0,3] = p2g[3,0]
    p2g[3,3] = (Sigma*(((r2+a2)**2)*s2t-2.0*a2*Delta*sin2*s2t)-sin2*((r2+a2)**2-a2*Delta*sin2)*p2S)/Sigma**2


    """
    Sigma = r2 + a2*cos2
    Delta = r2 - 2*G*M*r + a2
    
    dSigma_dr = 2.0*r
    dSigma_dth = -2.0*a2*cos_th*sin_th
    

    dDelta_dr = 2.0*r - 2.0*G*M
    
    # --- 2. 计算度规矩阵 g (LHS Matrix) ---
    # Carroll Eq 6.70 [cite: 7]
    g = np.zeros((4, 4))
    
    # g_tt
    g[0, 0] = -(1.0 - 2.0*G*M*r/Sigma)
    # g_rr
    g[1, 1] = Sigma / Delta
    # g_th_th
    g[2, 2] = Sigma
    # g_phi_phi
    term1 = (r2 + a2)**2
    term2 = a2 * Delta * sin2
    g[3, 3] = (sin2 / Sigma) * (term1 - term2)
    # g_t_phi
    gtphi = -(2.0*G*M*a*r*sin2) / Sigma
    g[0, 3] = gtphi
    g[3, 0] = gtphi

    # --- 3. 计算度规导数 (dg/dr 和 dg/dtheta) ---
    dg_dr = np.zeros((4, 4))
    dg_dth = np.zeros((4, 4))
    
    # g_tt = -1 + 2GMr * Sigma^-1
    factor_tt = 2.0*G*M / (Sigma**2)
    dg_dr[0, 0] = factor_tt * (Sigma - r*dSigma_dr)
    dg_dth[0, 0] = -factor_tt * r * dSigma_dth
    
    # g_rr = Sigma * Delta^-1
    dg_dr[1, 1] = (dSigma_dr * Delta - Sigma * dDelta_dr) / (Delta**2)
    dg_dth[1, 1] = dSigma_dth / Delta
    
    dg_dr[2, 2] = dSigma_dr
    dg_dth[2, 2] = dSigma_dth
    
    # g_t_phi = -2GMa * (r * sin^2 / Sigma)
    # 令 K = r * sin^2 / Sigma
    factor_tphi = -2.0*G*M*a
    dg_dr[0, 3] = factor_tphi * sin2 * (Sigma - r*dSigma_dr) / (Sigma**2)
    dg_dr[3, 0] = dg_dr[0, 3]
    
    # dK/dth: Numerator = r*(2sin*cos*Sigma - sin^2*dSigma_dth)
    num_dth = r * (2*sin_th*cos_th*Sigma - sin2*dSigma_dth)
    dg_dth[0, 3] = factor_tphi * num_dth / (Sigma**2)
    dg_dth[3, 0] = dg_dth[0, 3]
    
    # g_33 = sin^2/Sigma * A, where A = (r^2+a^2)^2 - a^2*Delta*sin^2
    A = term1 - term2
    dA_dr = 2.0*(r2+a2)*2.0*r - a2*dDelta_dr*sin2
    dA_dth = -a2*Delta*(2.0*sin_th*cos_th)
    
    # g_33 = (sin^2 * Sigma^-1) * A
    Pre = sin2 / Sigma
    dPre_dr = -sin2 * dSigma_dr / (Sigma**2)
    dPre_dth = (2*sin_th*cos_th*Sigma - sin2*dSigma_dth) / (Sigma**2)
    
    dg_dr[3, 3] = dPre_dr * A + Pre * dA_dr
    dg_dth[3, 3] = dPre_dth * A + Pre * dA_dth


    # dg_dr = p1g
    # dg_dth = p2g
    # --- RHS ---
    # F_mu = 0.5 * (dg_ab/dx_mu) * u^a * u^b - (dg_mu_b/dx_a) * u^a * u^b
    
    # 计算 u ⊗ u 矩阵
    uu = np.outer(v, v)
    RHS = np.zeros(4)
    
    # mu=0 (t) 和 mu=3 (phi): 
    # 因为度规对 t 和 phi 没有依赖 (stationary & axisymmetric)，第一项 0.5 * dg/dx_mu 为 0。
    # 只剩下第二项: - (dg_mu_b / dx_a) u^a u^b
    # a 只能取 1(r) 和 2(theta)，其他导数为0
    
    # F_t
    val_t = 0.0
    # a=1 (r): - dg_t_nu/dr * ur * u_nu
    val_t -= np.dot(dg_dr[0, :], v) * v[1]
    # a=2 (th): - dg_t_nu/dth * uth * u_nu
    val_t -= np.dot(dg_dth[0, :], v) * v[2]
    RHS[0] = val_t
    
    # F_phi
    val_phi = 0.0
    val_phi -= np.dot(dg_dr[3, :], v) * v[1]
    val_phi -= np.dot(dg_dth[3, :], v) * v[2]
    RHS[3] = val_phi
    
    # mu=1 (r)
    # Term 1: 0.5 * dG/dr * uu (Scalar contraction)
    term1_r = 0.5 * np.sum(dg_dr * uu)
    # Term 2: - (dg_r_nu / dx_a) * u^a * u^nu
    # Note: dg_r_nu only has non-zero component at nu=1 (g_rr) in diagonal, 
    # BUT g_matrix is not diagonal. However, dg_dr matrix captures all r-dependencies.
    # Be careful: "dx_a" means we sum over a=r and a=theta
    term2_r = -np.dot(dg_dr[1, :], v) * v[1] - np.dot(dg_dth[1, :], v) * v[2]
    RHS[1] = term1_r + term2_r
    
    # mu=2 (theta)
    # Term 1: 0.5 * dG/dth * uu
    term1_th = 0.5 * np.sum(dg_dth * uu)
    # Term 2: - (dg_th_nu / dx_a) * u^a * u^nu
    term2_th = -np.dot(dg_dr[2, :], v) * v[1] - np.dot(dg_dth[2, :], v) * v[2]
    RHS[2] = term1_th + term2_th
    
    acc = np.linalg.solve(g, RHS)
    
    return acc