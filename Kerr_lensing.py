import numpy as np
from numba import jit, prange
import time
from tqdm import tqdm
from physics import _core_null
from numerics import rkf45_step
from PIL import Image
from scipy.ndimage import map_coordinates

# Initial Conditions

@jit(nopython=True, cache=False)
def get_initial_state(u, v, r_cam, th_cam, M, a, fov_rad):
    """
    Initialization based on LNRF tetrad.
    """
    # (u, v) -> (alpha, beta)
    r_screen = np.sqrt(u*u + v*v)
    theta_local = r_screen * fov_rad
    beta = np.arctan2(v, u)
    
    # LNRF 4-momentum
    k_t_loc = 1.0
    k_r_loc = -1.0 * np.cos(theta_local)
    
    # Screen mapping：
    # Vertical (v, sin(beta)) ->  Theta
    # Horizontal (u, cos(beta)) -> Phi

    # v > 0 (up) -> north (theta decrease) -> k^(th) should be negative
    # u > 0 (right) -> clockwise -> k^(ph) positive
    
    k_th_loc = -np.sin(theta_local) * np.sin(beta) 
    k_ph_loc = -np.sin(theta_local) * np.cos(beta)
    
    sin_th = np.sin(th_cam)
    cos_th = np.cos(th_cam)
    Sigma = r_cam**2 + a**2 * cos_th**2
    Delta = r_cam**2 - 2*M*r_cam + a**2
    A = (r_cam**2 + a**2)**2 - a**2 * Delta * sin_th**2
    
    c_t = np.sqrt(A / (Sigma * Delta))
    Omega = 2*M*r_cam*a / A
    c_r = np.sqrt(Delta / Sigma)
    c_th = np.sqrt(1.0 / Sigma)
    c_ph = np.sqrt(Sigma / A) / sin_th
    
    kt = c_t * k_t_loc
    kr = c_r * k_r_loc
    kth = c_th * k_th_loc
    kphi = c_ph * k_ph_loc + Omega * kt
    
    g_tt = -(1.0 - 2.0*M*r_cam/Sigma)
    g_tphi = -2.0*M*a*r_cam*sin_th**2/Sigma
    g_phiphi = (r_cam**2 + a**2 + 2.0*M*a**2*r_cam*sin_th**2/Sigma)*sin_th**2
    
    E = -(g_tt * kt + g_tphi * kphi)
    L = g_tphi * kt + g_phiphi * kphi
    
    lam = L / E
    
    # Carter Constant C
    p_theta = Sigma * kth
    C = p_theta**2 + cos_th**2 * (L**2/sin_th**2 - a**2 * E**2)
    eta = C / (E**2)
    
    # formalize velocity to Mino Time
    scale_factor = Sigma / E
    
    dr_dtau = kr * scale_factor
    dth_dtau = kth * scale_factor
    
    Y0 = np.array([r_cam, dr_dtau, th_cam, dth_dtau, 0.0, 0.0], dtype=np.float64)
    params = np.array([M, a, lam, eta], dtype=np.float64)
    
    return Y0, params

# Integrator. Here I write a new integrator since it don't have
# to store the trjectory for lensing. I also make it compatible
# to parrallel processing.

@jit(nopython=True, cache=False)
def trace_ray(Y0, params, r_horizon, r_inf):
    t = 0.0
    y = Y0.copy()
    h = 0.5 
    
    max_steps = 10000
    r_min = r_horizon * 1.02
    
    for _ in range(max_steps):
        r_curr = y[0]
        
        if r_curr < r_min:
            return 1, y[2], y[5] # Black Hole
        if r_curr > r_inf:
            return 0, y[2], y[5] # Sky
            
        y_next, error = rkf45_step(_core_null, t, y, h, params)
        
        scale = 1e-6 + 1e-6 * np.max(np.abs(y))
        ratio = error / scale
        
        if ratio <= 1.0:
            t += h
            y = y_next
            h = h * 0.84 / (1e-20 + ratio**0.25)
            h = min(max(h, 1e-4), 2.0)
        else:
            h = h * 0.84 / (1e-20 + ratio**0.25)
            h = max(h, 1e-5)
            
    return 2, y[2], y[5] # Timeout(BH)

# Parallel Rendering

@jit(nopython=True, parallel=True, cache=False)
def main_render(width, height, r_cam, th_cam, M, a, fov_rad, r_inf):
    result_map = np.zeros((height, width, 2), dtype=np.float64)
    result_mask = np.zeros((height, width), dtype=np.int32)
    
    r_horizon = M + np.sqrt(M**2 - a**2)
    aspect_ratio = float(width) / float(height)
    
    for j in prange(height):
        v = 1.0 - 2.0 * j / height 
        
        for i in range(width):
            u = (2.0 * i / width - 1.0) * aspect_ratio # coordinate transformation
            
            Y0, params = get_initial_state(u, v, r_cam, th_cam, M, a, fov_rad)
            
            status, th_inf, ph_inf = trace_ray(Y0, params, r_horizon, r_inf)
            
            if status == 0: 
                # Normalize angles
                ph_final = ph_inf % (2*np.pi)
                result_map[j, i, 0] = th_inf
                result_map[j, i, 1] = ph_final
                result_mask[j, i] = 0
            else:
                result_mask[j, i] = 1 
                
    return result_map, result_mask

@jit(nopython=True, parallel=True, cache=False)
def slice_render(y_start, y_end, width, total_height, r_cam, th_cam, M, a, fov_rad, r_inf):
    """
    Slice rendering to get a progress bar
    """
    height_slice = y_end - y_start
    result_map = np.zeros((height_slice, width, 2), dtype=np.float64)
    result_mask = np.zeros((height_slice, width), dtype=np.int32)
    
    r_horizon = M + np.sqrt(M**2 - a**2)
    aspect_ratio = float(width) / float(total_height)
    
    for j in prange(height_slice):
        abs_j = y_start + j # absolute index
        
        v = 1.0 - 2.0 * abs_j / total_height 
        
        for i in range(width):
            u = (2.0 * i / width - 1.0) * aspect_ratio 
            
            Y0, params = get_initial_state(u, v, r_cam, th_cam, M, a, fov_rad)
            status, th_inf, ph_inf = trace_ray(Y0, params, r_horizon, r_inf)
            
            if status == 0: 
                ph_final = ph_inf % (2*np.pi)
                result_map[j, i, 0] = th_inf
                result_map[j, i, 1] = ph_final
                result_mask[j, i] = 0
            else:
                result_mask[j, i] = 1 
                
    return result_map, result_mask


if __name__ == "__main__":
    # setups are encoded here since argparse is just annoying
    # GUI version should be easier to use
    H, W = 800, 1200 # resolution
    M = 1.0 # BH mass
    a = 0.99 # spin
    r_cam = 15.0 # distance from camera to the blackhole
    th_cam = np.pi/2 - 0.05 # theta angle
    fov_deg = 45.0 # Field of view
    fov_rad = np.deg2rad(fov_deg)
    
    print("Compiling JIT kernels...")

    _ = slice_render(0, 10, 100, 100, r_cam, th_cam, M, a, fov_rad, 100.0)
    
    print(f"Rendering {W}x{H} image...")
    
    full_coords = np.zeros((H, W, 2), dtype=np.float64)
    full_mask = np.zeros((H, W), dtype=np.int32)
    
    d_size = 20
    
    t0 = time.time()
    
    with tqdm(total=H, unit="lines") as pbar:
        for y in range(0, H, d_size):
            y_end = min(y + d_size, H)
            
            d_coords, d_mask = slice_render(
                y, y_end, W, H, 
                r_cam, th_cam, M, a, fov_rad, 1000.0
            )
            
            full_coords[y:y_end, :, :] = d_coords
            full_mask[y:y_end, :] = d_mask
            
            pbar.update(y_end - y)
            
    print(f"Render finished in {time.time()-t0:.2f}s")
    
    import matplotlib.pyplot as plt
    th_map = full_coords[:,:,0]
    ph_map = full_coords[:,:,1]
        
    # sky grid
    grid_lines = (np.sin(ph_map*10)**20 + np.sin(th_map*10)**20) > 0.1
    img = np.zeros((H, W, 3))
    # Sky color (Blueish)
    img[:,:,0] = 0.1
    img[:,:,1] = 0.2
    img[:,:,2] = 0.5
    # Grid lines (White)
    img[grid_lines] = 1.0
        
    # Horizon
    img[full_mask==1] = 0.0
        
    plt.figure(figsize=(10,6))
    plt.imshow(img, origin='upper')
    plt.axis('off')
    plt.title(f"Kerr Black Hole on grid (a={a})")
    plt.show()
    
    # load starmap
    try:
        bg_img = np.array(Image.open("starmap.jpg"))
    except FileNotFoundError:
        print("starmap.jpg not found, will use random noise to simulate starmap")
        bg_img = np.random.randint(0, 50, (1000, 2000, 3), dtype=np.uint8)
        # milkway
        bg_img[450:550, :, :] += 100

    bg_h, bg_w, _ = bg_img.shape

    # mapping
    # full_coords[:,:,0]  theta (0, pi) -> y
    # full_coords[:,:,1]  phi (0, 2pi) -> x
        
    phi_map = (full_coords[:, :, 1] + np.pi) % (2*np.pi)
    theta_map = full_coords[:, :, 0]

    x_coords = phi_map / (2 * np.pi) * (bg_w - 1)
    y_coords = theta_map / np.pi * (bg_h - 1)

    # interpolate RGB
    print("Mapping texture...")
    final_img = np.zeros((H, W, 3), dtype=np.uint8)
        
    #  [2, H, W] -> (row_coords, col_coords)
    coords_for_map = np.array([y_coords, x_coords])

    for i in range(3):
        final_img[:, :, i] = map_coordinates(
            bg_img[:, :, i], 
            coords_for_map, 
            order=1,
            mode='wrap'
        )

    final_img[full_mask == 1] = 0

    plt.figure(figsize=(12, 8))
    plt.imshow(final_img)
    plt.axis('off')
    plt.title(f"Kerr Black Hole (a={a})")
    plt.imsave("kerr_render.png", final_img)
    plt.show()
    print("Image saved to kerr_render.png")
