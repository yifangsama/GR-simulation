import numpy as np

class KerrMetric:
    """
    Kerr Metric in Boyer-Lindquist coordinates.
    Implementation based on Sean Carroll, "Spacetime and Geometry", Chapter 6.
    """
    def __init__(self, M, a, G=1.0):
        self.M = M
        self.a = a  # a = J/M 
        self.G = G  # Newton's constant (usually 1 in geometric units)

    def _calc_auxiliary(self, r, theta):
        """
        Calculate rho^2 and Delta.
        """
        rho2 = r**2 + self.a**2 * np.cos(theta)**2
        Delta = r**2 - 2 * self.G * self.M * r + self.a**2
        
        return rho2, Delta

    def get_metric_matrix(self, x):
        """
        Calculate the covariant metric tensor g_mu_nu at position x.
        Based on Eq 6.70 .
        
        Input:
            x: [t, r, theta, phi]
        Output:
            g: (4, 4) numpy array
        """
        t, r, theta, phi = x
        rho2, Delta = self._calc_auxiliary(r, theta)
        sin_theta = np.sin(theta)
        sin2 = sin_theta**2
        
        g = np.zeros((4, 4))
        
        # g_tt = -(1 - 2GMr / rho^2) 
        g[0, 0] = -(1.0 - (2.0 * self.G * self.M * r) / rho2)
        # g_rr = rho^2 / Delta 
        g[1, 1] = rho2 / Delta
        # g_th_th = rho^2 
        g[2, 2] = rho2
        # g_phi_phi = (sin^2/rho^2) * [(r^2+a^2)^2 - a^2 * Delta * sin^2]
        g[3, 3] = (sin2 / rho2) * ((r**2 + self.a**2)**2 - self.a**2 * Delta * sin2)
        
        # g_t_phi = g_phi_t = - (2GMar sin^2) / rho^2
        val_03 = -(2.0 * self.G * self.M * self.a * r * sin2) / rho2
        g[0, 3] = val_03
        g[3, 0] = val_03
        
        return g

    def get_event_horizons(self):
        """
        Calculate the radii of the inner and outer event horizons.
        Based on Eq 6.82 .
        
        Returns:
            r_plus: Outer horizon (Event Horizon)
            r_minus: Inner horizon (Cauchy Horizon)
        """
        term_sq = (self.G * self.M)**2 - self.a**2
        
        if term_sq < 0:
            # Naked singularity
            return None, None
            
        r_plus = self.G * self.M + np.sqrt(term_sq)
        r_minus = self.G * self.M - np.sqrt(term_sq)
        
        return r_plus, r_minus

    def get_stationary_limit_surface(self, theta):
        """
        Calculate the radius of the stationary limit surface (Ergosphere boundary).
        Based on Eq 6.85 .
        (r - GM)^2 = G^2M^2 - a^2 cos^2 theta
        """
        cos2 = np.cos(theta)**2
        term_sq = (self.G * self.M)**2 - self.a**2 * cos2
        
        if term_sq < 0:
            return None
            
        # We generally care about the outer surface
        r_sls = self.G * self.M + np.sqrt(term_sq)
        return r_sls