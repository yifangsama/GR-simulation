import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class KerrVisualizer:
    def __init__(self, M, a):
        self.M = M
        self.a = a
        self.r_plus = M + np.sqrt(M**2 - a**2)

    def _embedding_metric(self, r):
        r2 = r**2
        a2 = self.a**2
        Delta = r2 - 2*self.M*r + a2
        rho = np.sqrt(r2 + a2 + 2*self.M*a2/r)
        d_rho2_dr = 2*r - 2*self.M*a2/r2
        drho_dr = d_rho2_dr / (2 * rho)
        g_rr = r2 / Delta
        term = g_rr - drho_dr**2
        valid_mask = term >= 0
        dz_dr = np.zeros_like(r)
        dz_dr[valid_mask] = np.sqrt(term[valid_mask])
        return rho, dz_dr, valid_mask

    def plot_embedding_3d(self, r_traj, phi_traj, ax=None, filename=None, show=True, view_limit=25.0):
        """
        绘制嵌入图 (Embedding Diagram)
        """
        print(f"Rendering 3D Embedding Diagram...")
        
        mask = r_traj < view_limit
        if not np.any(mask):
            mask[0] = True 
        r_plot = r_traj[mask]
        phi_plot = phi_traj[mask]

        # 计算背景网格
        r_min = self.r_plus * 1.01
        r_max = view_limit 
        r_grid = np.geomspace(r_min, r_max, 150)
        rho_grid, dz_dr_grid, valid = self._embedding_metric(r_grid)
        
        z_integral = np.cumsum(dz_dr_grid * np.gradient(r_grid))
        z_profile = z_integral - z_integral[-1] 
        
        phi_grid = np.linspace(0, 2*np.pi, 60)
        R_mesh, P_mesh = np.meshgrid(rho_grid, phi_grid)
        Z_mesh = np.tile(z_profile, (60, 1))
        X_mesh = R_mesh * np.cos(P_mesh)
        Y_mesh = R_mesh * np.sin(P_mesh)

        # 计算轨迹映射
        rho_traj = np.interp(r_plot, r_grid, rho_grid)
        z_traj = np.interp(r_plot, r_grid, z_profile)
        X_traj = rho_traj * np.cos(phi_plot)
        Y_traj = rho_traj * np.sin(phi_plot)
        Z_traj = z_traj

        # --- 设置绘图对象 ---
        if ax is None:
            fig = plt.figure(figsize=(10, 8))
            ax = fig.add_subplot(111, projection='3d')
            ax.set_title(f"Kerr Embedding (a={self.a})")
            is_standalone = True
        else:
            is_standalone = False
            
        # 绘图
        ax.plot_wireframe(X_mesh, Y_mesh, Z_mesh, color='gray', alpha=0.15, linewidth=0.5)
        
        h_rho = rho_grid[0]
        h_z = z_profile[0]
        ax.plot(h_rho*np.cos(phi_grid), h_rho*np.sin(phi_grid), h_z, color='black', linewidth=2, linestyle='--')
        ax.plot(X_traj, Y_traj, Z_traj, color='orangered', linewidth=2, label='Photon')

        max_range = np.max(rho_grid)
        ax.set_xlim(-max_range, max_range)
        ax.set_ylim(-max_range, max_range)
        ax.set_zlim(h_z * 1.1, 1.0)
        ax.axis('off')
        
        # --- 输出处理 ---
        # 只有在没有外部 ax (独立运行) 时才进行保存或显示
        if is_standalone:
            if filename: 
                plt.savefig(filename, dpi=120, bbox_inches='tight')
                print(f"Saved to {filename}")
            if show:
                plt.show()

    def plot_topdown(self, r_traj, phi_traj, ax=None, filename=None, show=True, view_limit=25.0):
        """
        绘制俯视图 (Top-down View)
        """
        # --- 设置绘图对象 ---
        if ax is None:
            fig, ax = plt.subplots(figsize=(6, 6), subplot_kw={'projection': 'polar'})
            ax.set_title(f"Top-down View")
            is_standalone = True
        else:
            is_standalone = False
        
        # 数据截断
        mask = r_traj < view_limit
        r_plot = r_traj[mask]
        phi_plot = phi_traj[mask]
        
        # 视界
        theta = np.linspace(0, 2*np.pi, 100)
        ax.fill(theta, [self.r_plus]*100, color='black', alpha=0.9, label='Horizon')
        
        # 轨迹
        ax.plot(phi_plot, r_plot, color='red', linewidth=1.5)
        
        # 视野限制
        ax.set_rmax(view_limit)
        
        # --- 输出处理 ---
        if is_standalone:
            if filename:
                plt.savefig(filename, dpi=100)
            if show:
                 plt.show()

    def plot_cartesian_3d(self, r_traj, theta_traj, phi_traj, ax=None, filename=None, show=True, view_limit=30.0):
        """
        绘制 3D 笛卡尔轨迹
        """
        print(f"Rendering 3D Trajectory")
        
        # 数据截断
        mask = r_traj < view_limit
        if not np.any(mask): mask[0] = True
        
        r = r_traj[mask]
        th = theta_traj[mask]
        ph = phi_traj[mask]
        
        sin_th = np.sin(th)
        cos_th = np.cos(th)
        term = np.sqrt(r**2 + self.a**2)
        
        X = term * sin_th * np.cos(ph)
        Y = term * sin_th * np.sin(ph)
        Z = r * cos_th
        
        # --- 设置绘图对象 ---
        if ax is None:
            fig = plt.figure(figsize=(10, 10))
            fig.patch.set_facecolor('black')
            ax = fig.add_subplot(111, projection='3d')
            ax.set_facecolor('black')
            is_standalone = True
        else:
            is_standalone = False
        
        # 视界
        u = np.linspace(0, 2 * np.pi, 60)
        v = np.linspace(0, np.pi, 60)
        rH = self.r_plus
        xH = np.sqrt(rH**2 + self.a**2) * np.outer(np.sin(v), np.cos(u))
        yH = np.sqrt(rH**2 + self.a**2) * np.outer(np.sin(v), np.sin(u))
        zH = rH * np.outer(np.cos(v), np.ones_like(u))
        
        ax.plot_surface(xH, yH, zH, color='black', alpha=1.0, shade=False, zorder=1)
        ax.plot_wireframe(xH, yH, zH, color='darkred', alpha=0.4, linewidth=0.6, zorder=2)
        
        # 轨迹
        ax.plot(X, Y, Z, color='cyan', linewidth=1.2, alpha=0.9, zorder=10)
        ax.scatter(X[0], Y[0], Z[0], color='red', s=20, label='Start', zorder=11)
        
        # 坐标轴比例
        max_range = np.max([np.max(np.abs(X)), np.max(np.abs(Y)), np.max(np.abs(Z))])
        max