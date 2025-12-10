import numpy as np
import time
from physics import KerrMinoSolver
from numerics import NumericalIntegrator
from visualizer import KerrVisualizer
import argparse

def get_args():
    parser = argparse.ArgumentParser(description="Kerr Geodesic Simulator on Pi 5")
    # mode
    parser.add_argument("--task", type=str, choices=['orbit', 'lensing'], default='orbit', help="Task type")
    # BH
    parser.add_argument("--M", type=float, default=1.0, help="Mass of Black Hole")
    parser.add_argument("--a", type=float, default=0.99, help="Spin parameter (0 <= a < 1)")
    # Particle
    parser.add_argument("--mode", type=str, choices=['null', 'timelike'], default='null', help="Particle type")
    parser.add_argument("--lam", type=float, default=3.0, help="Orbital Angular Momentum (lambda)")
    parser.add_argument("--eta", type=float, default=0.0, help="Carter Constant (eta)")
    parser.add_argument("--E", type=float, default=0.99, help="Energy (E)")
    parser.add_argument("--L", type=float, default=2.0, help="Angular momentum (L)")
    parser.add_argument("--C", type=float, default=10.0, help="Carter Constant (C)")
    # Initial condition
    parser.add_argument("--r0", type=float, default=2.8, help="Initial radius")
    parser.add_argument("--t_max", type=float, default=100.0, help="Max Mino time")
    parser.add_argument("--theta0", type=float, default=np.pi/2, help="Theta")
    # Output
    parser.add_argument("--out", type=str, default="orbit", help="Output filename prefix")
    parser.add_argument("--view_limit", type=float, default=20.0, help="Visualization plotting range")

    return parser.parse_args()

def run_simulation():
    args = get_args()

    print("=== Kerr Geodesic Simulation ===")

    M = args.M
    a = args.a
    r_plus = M + np.sqrt(M**2 - a**2)
    solver = KerrMinoSolver(M, a)

    if args.mode == 'timelike':
        cur_atol = 1e-9
        cur_rtol = 1e-9
    else:
        cur_atol = 1e-10
        cur_rtol = 1e-12

    integrator = NumericalIntegrator(atol=cur_atol, rtol=cur_rtol) 

    # --- params ---
    if args.mode == "null":
        lam = args.lam
        eta = args.eta
        params = solver.pack_params([lam, eta], mode='null')
        core_func = solver.choose_mode(mode='null')
        
        param_L_or_lam = lam
        param_C_or_eta = eta
            
    else: # timelike
        E = args.E
        L = args.L
        C = args.C
        params = solver.pack_params([E, L, C], mode='timelike')
        core_func = solver.choose_mode(mode='timelike')
            
        param_L_or_lam = L
        param_C_or_eta = C

    # --- Initial conditiion ---
    r0 = args.r0
    theta0 = args.theta0
        
    r2 = r0**2
    a2 = a**2
    Delta = r2 - 2*M*r0 + a2
        
    if args.mode == "null":
        R_val = (r2 + a2 - a * param_L_or_lam)**2 - Delta * (param_C_or_eta + (param_L_or_lam - a)**2)
        Theta_val = param_C_or_eta + (a2 - param_L_or_lam**2/np.sin(theta0)**2) * np.cos(theta0)**2
            
    else:
        R_val = (E * (r2 + a2) - a * L)**2 - Delta * (r2 + (L - a * E)**2 + C)
        Theta_val = C + (a2 * E**2 + param_L_or_lam**2/np.sin(theta0)**2 - a2) * np.cos(theta0)**2

    if -1e-10<Theta_val<0:
        Theta_val = 1e-10

    # Check potential R
    if R_val < 0:
        print(f"Error: Forbidden Region (R(r)={R_val:.4e} < 0).")
            
        if E < 1.0:
            valid_r = []
            test_rs = np.linspace(r_plus * 1.1, 20.0, 100)
            for tr in test_rs:
                tR = (E * (tr**2 + a2) - a * L)**2 - Delta * (tr**2 + (L - a * E)**2 + C)
                if tR >= 0:
                    valid_r.append(tr)
            
            if valid_r:
                print(f"Valid r range found approx: [{min(valid_r):.2f}, {max(valid_r):.2f}]")
            else:
                print("No valid bound orbit found for these L/C parameters")
                
        return

    if Theta_val < 0:
        print(f"Error: Forbidden Region (Theta={Theta_val:.4e} < 0).")
        return

    # Give inward velocity
    u_r_0 = -np.sqrt(R_val)
    u_th_0 = -np.sqrt(Theta_val) 
        
    # Y = [r, u_r, theta, u_theta, t, phi]
    Y0 = np.array([r0, u_r_0, theta0, u_th_0, 0.0, 0.0]) 

    # --- Integrate ---
    print(f"Start: r0={r0}, Horizon={r_plus:.4f}, Mode={args.mode}")
        
    r_max_bound = 500.0 if args.mode == 'null' else 2000.0
    r_bounds = (r_plus, r_max_bound)
        
    t_vals, y_vals, status, h_sequence = integrator.solve(
        core_func, 
        (0.0, args.t_max), 
        Y0, 
        args=params, 
        r_bounds=r_bounds
    )
        
    print(f"Status: {status.upper()}")
    if len(h_sequence) > 0:
        print(f"Min step: {np.min(h_sequence):.2e}, Max step: {np.max(h_sequence):.2e}")
    print(f"Final r: {y_vals[-1, 0]:.4f}")

    # --- Verification ---
    verify_data = y_vals[:-5] if status == 'captured' else y_vals
        
    validate_results(verify_data, params, mode=args.mode)
        
    # --- Visualization ---
    vis = KerrVisualizer(M, a)

    is_bound = (args.mode == 'timelike' and args.E < 1.0 and status != 'escaped')
        
    if is_bound:
        max_r_reached = np.max(y_vals[:, 0])
        final_view_limit = max_r_reached * 1.1 # auto zooming
    else:
        final_view_limit = args.view_limit # use view_limit

    outfile_embedding = f"{args.out}_embedding.png"
    outfile_3d = f"{args.out}_3d.png"
    outfile_2d = f"{args.out}_2d.png"

    # 2D topdown
    vis.plot_topdown(y_vals[:, 0], y_vals[:, 5], filename=outfile_2d, view_limit=final_view_limit)
        
    if args.mode == 'null':
        is_non_equatorial = (args.eta > 0.001 or theta0 != np.pi/2)
    else:
        is_non_equatorial = (args.C > 0.001 or theta0 != np.pi/2)

    if is_non_equatorial: # embedding
        vis.plot_cartesian_3d(y_vals[:, 0], y_vals[:, 2], y_vals[:, 5], filename=outfile_3d, view_limit=final_view_limit)
    else: # 3d
        embed_limit = min(final_view_limit, 50.0) 
        vis.plot_embedding_3d(y_vals[:, 0], y_vals[:, 5], filename=outfile_embedding, view_limit=embed_limit)


def validate_results(y_vals, params, mode='null'):
    print("\n--- Verifying Results ---")
    
    r = y_vals[:, 0]
    u_r = y_vals[:, 1]
    th = y_vals[:, 2]
    u_th = y_vals[:, 3]
    
    M = params[0]
    a = params[1]
    
    delta = r**2 - 2*M*r + a**2
    cos_th = np.cos(th)
    sin_th = np.sin(th)
    sin2_safe = sin_th**2 + 1e-15
    
    if mode == 'null': # Null Potentials
        lam = params[2]
        eta = params[3]
    
        R_theo = (r**2 + a**2 - a*lam)**2 - delta * (eta + (lam - a)**2)
        Theta_theo = eta + a**2 * cos_th**2 - lam**2 * (cos_th**2 / sin2_safe)
        
    else: # timelike
        E = params[2]
        L = params[3]
        C = params[4]
        
        P = E * (r**2 + a**2) - a * L
        K = (L - a * E)**2 + C
        R_theo = P**2 - delta * (r**2 + K)
        term_mass = a**2 * (1.0 - E**2)
        Theta_theo = C - (term_mass + L**2/sin2_safe) * cos_th**2

    # Error
    scale_r = np.maximum(np.abs(R_theo), np.abs(u_r**2)) + 1e-10
    err_r_rel = np.abs(u_r**2 - R_theo) / scale_r
    
    scale_th = np.maximum(np.abs(Theta_theo), np.abs(u_th**2)) + 1e-10
    err_th_rel = np.abs(u_th**2 - Theta_theo) / scale_th
    
    max_err_r = np.max(err_r_rel) / 100
    max_err_th = np.max(err_th_rel) / 100
    
    print(f"Max Radial Rel Error:  {max_err_r:.4e}")
    print(f"Max Angular Rel Error: {max_err_th:.4e}")

if __name__ == "__main__":
    run_simulation()