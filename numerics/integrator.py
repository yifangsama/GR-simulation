import numpy as np
from numba import jit
import time

@jit(nopython=True, cache=True)
def rkf45_step(f, t, y, h, params):
    k1 = h * f(t, y, params)
    k2 = h * f(t + 0.25*h, y + 0.25*k1, params)
    k3 = h * f(t + 0.375*h, y + 0.09375*k1 + 0.28125*k2, params)
    k4 = h * f(t + 12/13*h, y + 1932/2197*k1 - 7200/2197*k2 + 7296/2197*k3, params)
    k5 = h * f(t + h, y + 439/216*k1 - 8*k2 + 3680/513*k3 - 845/4104*k4, params)
    k6 = h * f(t + 0.5*h, y - 8/27*k1 + 2*k2 - 3544/2565*k3 + 1859/4104*k4 - 11/40*k5, params)

    y_new = y + 16/135*k1 + 6656/12825*k3 + 28561/56430*k4 - 0.18*k5 + 2/55*k6
    y_4th = y + 25/216*k1 + 1408/2565*k3 + 2197/4104*k4 - 0.2*k5
    
    error = np.max(np.abs(y_new - y_4th))
    return y_new, error

class NumericalIntegrator:
    def __init__(self, atol=1e-9, rtol=1e-9):
        self.atol = atol
        self.rtol = rtol
        self.min_step = 1e-16 # should correspond to tolerance
        self.max_step = 10.0

    def solve(self, fun, t_span, y0, args=(), r_bounds=None):
        """
        r_bounds: tuple (r_horizon, r_infinity)
        """
        t = t_span[0]
        t_end = t_span[1]

        if r_bounds is not None:
            r_min = r_bounds[0]*1.01
            r_max = r_bounds[1]
        else:
            r_min = 0.0
            r_max = 1000
        
        direction = np.sign(t_end - t)
        if direction == 0:
            return np.array([t]), np.array([y0])
            
        y = np.array(y0, dtype=np.float64)
        h = 0.1 * direction
        
        t_values = [t]
        y_values = [y]
        
        params = args

        status = "orbit"

        h_sequence = []

        step_count = 0
        last_print_time = time.time()
        sim_time = abs(t_end - t)
        
        while (t < t_end if direction > 0 else t > t_end):
            r_curr = y[0]
            h_sequence.append(h)
            
            if r_curr < r_min:
                print(f"Info: Particle captured by horizon at r={r_curr:.4f}")
                status = "captured"
                break
            
            if r_curr > r_max:
                print(f"Info: Particle escaped to infinity at r={r_curr:.4f}")
                status = "escaped"
                break

            dist_to_end = t_end - t
            if np.abs(h) > np.abs(dist_to_end):
                h = dist_to_end

            y_next, error = rkf45_step(fun, t, y, h, params)

            # Error Control
            scale = self.atol + self.rtol * np.maximum(np.abs(y), np.abs(y_next))
            ratio = np.max(error / (scale + 1e-30))

            if ratio <= 1.0: # accept
                t += h
                y = y_next
                t_values.append(t)
                y_values.append(y)
                
                q = 0.84 * (1.0 / (ratio + 1e-30))**0.25
                h = h * min(q, 4.0)
                
                if np.abs(h) > self.max_step:
                    h = self.max_step * direction

                step_count += 1
                current_wall_time = time.time()
                if current_wall_time - last_print_time > 1.0:
                    progress = abs(t - t_span[0]) / sim_time * 100
                    print(f"\rProgress: {progress:5.1f}% | t={t:.2f}", end="")
                    last_print_time = current_wall_time

            else: # reject
                q = 0.84 * (1.0 / (ratio + 1e-30))**0.25
                h = h * max(q, 0.1)
                
                if np.abs(h) < self.min_step:
                    print(f"Warning: Step size underflow at t={t}, h={h}")
                    break
        
        return np.array(t_values), np.array(y_values), status, h_sequence