import sys
import numpy as np
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
                             QFormLayout, QLineEdit, QPushButton, QLabel, QTabWidget, 
                             QComboBox, QProgressBar, QMessageBox, QSplitter, QFileDialog)
from PyQt5.QtCore import QThread, pyqtSignal, Qt
from PyQt5.QtGui import QImage, QPixmap

from PIL import Image
from scipy.ndimage import map_coordinates

import matplotlib
matplotlib.use('Qt5Agg')
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d import Axes3D


from physics.kerr_dynamics import KerrMinoSolver
from numerics.integrator import NumericalIntegrator
from Kerr_lensing import slice_render
from visualizer import KerrVisualizer



##############################################################################
# Orbit simulation

class Orbit_simulation(QThread):
    result = pyqtSignal(object, dict, str, str) 
    err = pyqtSignal(str)

    def __init__(self, params):
        super().__init__()
        self.p = params

    def run(self):
        try:
            M = self.p['M']
            a = self.p['a']
            solver = KerrMinoSolver(M, a)
            
            mode = self.p['mode']
            if mode == 'timelike':
                E = self.p['E']
                L = self.p['L']
                C = self.p['C']

                params_packed = solver.pack_params([E, L, C], mode='timelike')
                core_func = solver.choose_mode(mode='timelike')
                atol, rtol = 1e-9, 1e-9
            else: 
                lam = self.p['lam']
                eta = self.p['eta']

                params_packed = solver.pack_params([lam, eta], mode='null')
                core_func = solver.choose_mode(mode='null')
                atol, rtol = 1e-10, 1e-12

            integrator = NumericalIntegrator(atol=atol, rtol=rtol)
            integrator.max_step = 1.0

            r0 = self.p['r0']
            theta0 = self.p['theta0']
            r2 = r0**2
            a2 = a**2
            Delta = r2 - 2*M*r0 + a2
            cos2th = np.cos(theta0)**2

            if mode == "null":
                param_L = self.p['lam']
                param_C = self.p['eta']
                R_val = (r2 + a2 - a * param_L)**2 - Delta * (param_C + (param_L - a)**2)
                Theta_val = param_C + (a2 - param_L**2/np.sin(theta0)**2) * cos2th
            else:
                param_L = self.p['L']
                C = self.p['C']
                R_val = (E * (r2 + a2) - a * L)**2 - Delta * (r2 + (L - a * E)**2 + C)
                Theta_val = C + (a2 * E**2 + L**2/np.sin(theta0)**2 - a2) * cos2th
            
            if R_val < 0:
                self.err.emit(f"Error: Forbidden Region R(r)<0.")
                return
            if Theta_val < 0: Theta_val = 0.0

            u_r_0 = -np.sqrt(R_val)
            u_th_0 = -np.sqrt(Theta_val)
            Y0 = np.array([r0, u_r_0, theta0, u_th_0, 0.0, 0.0])

            r_plus = M + np.sqrt(M**2 - a**2)
            if mode == 'null':
                r_max = 300.0
            else:
                r_max = 500.0
            r_bounds = (r_plus, r_max)

            t_vals, y_vals, status, _ = integrator.solve(
                core_func, 
                (0.0, self.p['t_max']), 
                Y0, 
                args=params_packed, 
                r_bounds=r_bounds
            )
            
            msg = f"Status: {status.upper()} | Final r: {y_vals[-1, 0]:.4f}"
            self.result.emit(y_vals, self.p, status, msg)

        except Exception as e:
            self.err.emit(str(e))

#################################################################################
# Gravitational lensing

class Gravitational_lensing(QThread):
    progress = pyqtSignal(int)
    image = pyqtSignal(np.ndarray, np.ndarray)
    
    def __init__(self, params):
        super().__init__()
        self.p = params

    def run(self):
        try:
            H, W = self.p['H'], self.p['W']
            f_coords = np.zeros((H, W, 2), dtype=np.float64)
            f_mask = np.zeros((H, W), dtype=np.int32)
            
            chunk = 20
            for y in range(0, H, chunk):
                y_end = min(y + chunk, H)
                c_chunk, m_chunk = slice_render(
                    y, y_end, W, H, 
                    self.p['r_cam'], np.deg2rad(self.p['inc']), 
                    self.p['M'], self.p['a'], np.deg2rad(self.p['fov']), 1000.0
                )
                f_coords[y:y_end, :, :] = c_chunk
                f_mask[y:y_end, :] = m_chunk
                self.progress.emit(int(y_end / H * 100))
            
            self.image.emit(f_coords, f_mask)
        except Exception as e:
            print(e)

class Canvas(FigureCanvas):
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        self.fig.patch.set_facecolor('#f0f0f0') # grey

        super(Canvas, self).__init__(self.fig)

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Kerr Geodesic & Lensing GUI")
        self.resize(1300, 850)
        
        # Initialize
        self.st_data = None
        
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        layout = QVBoxLayout(main_widget)
        
        self.tabs = QTabWidget()
        layout.addWidget(self.tabs)
        
        self.init_orbit_tab()
        self.init_lensing_tab()
        
        self.lbl_sta = QLabel("Ready.")
        layout.addWidget(self.lbl_sta)

    def init_orbit_tab(self):
        tab = QWidget()
        layout = QHBoxLayout(tab)
        panel = QWidget()
        form = QFormLayout(panel)
        panel.setFixedWidth(300)
        
        self.orb_M = QLineEdit("1.0")
        form.addRow("Mass (M):", self.orb_M)

        self.orb_a = QLineEdit("0.99")
        form.addRow("Spin (a):", self.orb_a)

        self.orb_mode = QComboBox()
        self.orb_mode.addItems(["null", "timelike"])
        self.orb_mode.currentTextChanged.connect(self.toggle_mode)
        form.addRow("Mode:", self.orb_mode)

        form.addRow(QLabel("--- Initial ---"))

        self.orb_r0 = QLineEdit("6.0")
        form.addRow("r0:", self.orb_r0)

        self.orb_theta0 = QLineEdit("90")
        form.addRow("Theta0:", self.orb_theta0)

        self.orb_tmax = QLineEdit("200.0")
        self.orb_lam = QLineEdit("3.0")
        self.orb_eta = QLineEdit("0.0")
        self.orb_E = QLineEdit("0.99")
        self.orb_L = QLineEdit("2.0")
        self.orb_C = QLineEdit("10.0")
        
        self.param_null = [self.orb_lam, self.orb_eta]
        self.labels_null = [QLabel("Lambda:"), QLabel("Eta:")]
        for l, p in zip(self.labels_null, self.param_null): 
            form.addRow(l, p)

        self.param_time = [self.orb_E, self.orb_L, self.orb_C]
        self.labels_time = [QLabel("E:"), QLabel("L:"), QLabel("C:")]
        for l, p in zip(self.labels_time, self.param_time): 
            form.addRow(l, p)

        form.addRow("Max Time:", self.orb_tmax)
        
        btn = QPushButton("Run Orbit")
        btn.clicked.connect(self.run_orbit)
        btn.setStyleSheet("background-color: green; color: white; padding: 6px")
        form.addRow(btn)
        self.toggle_mode("null")

        plot_tabs = QTabWidget()
        self.canvas_2d = Canvas(self)
        plot_tabs.addTab(self.canvas_2d, "Top-Down")
        self.canvas_3d = Canvas(self)
        plot_tabs.addTab(self.canvas_3d, "3D View")
        
        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(panel)
        splitter.addWidget(plot_tabs)
        layout.addWidget(splitter); self.tabs.addTab(tab, "Orbit & Scattering")

    def toggle_mode(self, mode):
        is_null = (mode == 'null')
        for w in self.param_null + self.labels_null:
            w.setVisible(is_null)
        for w in self.param_time + self.labels_time:
            w.setVisible(not is_null)

    def run_orbit(self):
        try:
            params = {
                'M': float(self.orb_M.text()), 
                'a': float(self.orb_a.text()), 
                'mode': self.orb_mode.currentText(),
                'r0': float(self.orb_r0.text()), 
                'theta0': np.deg2rad(float(self.orb_theta0.text())),
                't_max': float(self.orb_tmax.text()), 
                'lam': float(self.orb_lam.text()), 
                'eta': float(self.orb_eta.text()),
                'E': float(self.orb_E.text()), 
                'L': float(self.orb_L.text()), 
                'C': float(self.orb_C.text())
            }
            self.worker = Orbit_simulation(params)
            self.worker.result.connect(self.update_plots)
            self.worker.err.connect(lambda s: QMessageBox.warning(self, "Error", s))
            self.worker.start()
            self.lbl_sta.setText("Calculating Orbit...")

        except ValueError: QMessageBox.warning(self, "Error", "Check inputs.")

    def update_plots(self, y_vals, params, status, msg):
        self.lbl_sta.setText(msg)
        M = params['M']
        a = params['a']
        
        vis = KerrVisualizer(M, a)
        
        is_bound = (params['mode'] == 'timelike' and params['E'] < 1.0 and status != 'escaped')
        view_limit = np.max(y_vals[:, 0]) * 1.1 if is_bound else 25.0
        
        r = y_vals[:, 0]
        th = y_vals[:, 2]
        ph = y_vals[:, 5]
        
        # --- 2D Plot (Top Down) ---
        self.canvas_2d.fig.clf()
        ax2 = self.canvas_2d.fig.add_subplot(111, projection='polar')
        
        vis.plot_topdown(r, ph, ax=ax2, view_limit=view_limit, show=False)
        
        self.canvas_2d.draw()
        
        # --- 3D Plot (Embedding or Cartesian) ---
        self.canvas_3d.fig.clf()
        ax3 = self.canvas_3d.fig.add_subplot(111, projection='3d')
        if params['mode'] == 'null':
            is_eq = (abs(params['theta0'] - np.pi/2) < 1e-4 and abs(params['eta']) < 1e-4)
        else:
            is_eq = (abs(params['theta0'] - np.pi/2) < 1e-4 and abs(params['C']) < 1e-4)

        if is_eq:
            embed_limit = min(view_limit, 50.0)
            vis.plot_embedding_3d(r, ph, ax=ax3, view_limit=embed_limit, show=False)
        else:
            self.canvas_3d.fig.patch.set_facecolor('black')
            ax3.set_facecolor('black')
            vis.plot_cartesian_3d(r, th, ph, ax=ax3, view_limit=view_limit, show=False)
            
            if is_eq:
                 self.canvas_3d.fig.patch.set_facecolor('#f0f0f0')
                 ax3.set_facecolor('white')

        self.canvas_3d.draw()


    # Lensing tab

    def init_lensing_tab(self):
        tab = QWidget()
        layout = QHBoxLayout(tab)
        
        panel = QWidget()
        form = QFormLayout(panel)
        panel.setFixedWidth(300)
        
        self.l_res = QComboBox()
        self.l_res.addItems(["320x240 (Fast)", "640x480 (Normal)", "800x600 (High)"])
        form.addRow("Resolution:", self.l_res)

        self.l_a = QLineEdit("0.99")
        form.addRow("Spin (a):", self.l_a)

        self.l_rcam = QLineEdit("15.0")
        form.addRow("R Cam:", self.l_rcam)

        self.l_inc = QLineEdit("85.0")
        form.addRow("Inclination:", self.l_inc)

        self.fov = QLineEdit("60.0")
        form.addRow("FOV:", self.fov)
        
        # starmap
        self.btn_star = QPushButton("Load Starmap (Image)")
        self.btn_star.clicked.connect(self.load_starmap)
        self.btn_star.setStyleSheet("background-color: green; color: white; padding: 6px")
        form.addRow(self.btn_star)
        
        self.lbl_s_sta = QLabel("No Starmap Loaded, \nUsing Grid")
        self.lbl_s_sta.setStyleSheet("color: gray; font-size: 25px;")
        form.addRow(self.lbl_s_sta)
        
        btn_rend = QPushButton("Render Image")
        btn_rend.clicked.connect(self.run_lensing)
        btn_rend.setStyleSheet("background-color: green; color: white; padding: 6px")
        form.addRow(btn_rend)
        
        self.pbar = QProgressBar()
        form.addRow(self.pbar)

        self.img_lbl = QLabel("Output Image")
        self.img_lbl.setAlignment(Qt.AlignCenter)
        self.img_lbl.setStyleSheet("background: black; border: 1px;")
        self.img_lbl.setScaledContents(True)
        
        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(panel)
        splitter.addWidget(self.img_lbl)
        layout.addWidget(splitter)
        self.tabs.addTab(tab, "Gravitational Lensing")

    def load_starmap(self):
        fname, _ = QFileDialog.getOpenFileName(self, "Load Starmap", "", "Image Files (*.jpg *.png *.jpeg)")
        if fname:
            try:
                img = Image.open(fname).convert('RGB')
                self.st_data = np.array(img)
                self.lbl_s_sta.setText(f"Loaded: {fname.split('/')[-1]}")
                self.lbl_s_sta.setStyleSheet("color: black; font-weight: bold; font-size: 25px;")
            except Exception as e:
                QMessageBox.warning(self, "Load Error", str(e))

    def run_lensing(self):
        w, h = map(int, self.l_res.currentText().split()[0].split('x'))
        params = {
            'W': w, 
            'H': h, 
            'M': 1.0, 
            'a': float(self.l_a.text()),
            'r_cam': float(self.l_rcam.text()), 
            'th_cam': np.deg2rad(float(self.l_inc.text())),
            'inc': float(self.l_inc.text()), 
            'fov': float(self.fov.text())
        }
        self.l_sim = Gravitational_lensing(params)
        self.l_sim.progress.connect(self.pbar.setValue)
        self.l_sim.image.connect(self.show_render)
        self.l_sim.start()

    def show_render(self, coords, mask):
        H, W, _ = coords.shape
        img_buffer = np.zeros((H, W, 3), dtype=np.uint8)
        
        if self.st_data is not None:
            # Similar to Kerr_lensing.py
            self.lbl_sta.setText("Mapping Starmap Texture...")
            QApplication.processEvents()
            
            bg_h, bg_w, _ = self.st_data.shape
            
            phi_map = (coords[:, :, 1] + np.pi) % (2*np.pi)
            theta_map = coords[:, :, 0]

            x_coords = phi_map / (2 * np.pi) * (bg_w - 1)
            y_coords = theta_map / np.pi * (bg_h - 1)
            
            # [2, H, W]
            coords_for_map = np.array([y_coords, x_coords])

            for i in range(3):
                img_buffer[:, :, i] = map_coordinates(
                    self.st_data[:, :, i], 
                    coords_for_map, 
                    order=1,
                    mode='wrap'
                )
        else:
            th, ph = coords[:,:,0], coords[:,:,1]
            grid = (np.sin(ph*12)**20 + np.sin(th*12)**20) > 0.05
            img_buffer[:,:,0] = 20
            img_buffer[:,:,1] = 40
            img_buffer[:,:,2] = 80
            img_buffer[grid] = 255
            
        img_buffer[mask==1] = 0
        
        qimg = QImage(img_buffer.data, W, H, 3*W, QImage.Format_RGB888)
        self.img_lbl.setPixmap(QPixmap.fromImage(qimg))
        self.lbl_sta.setText("Render Finished.")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    w = MainWindow()
    w.show()
    sys.exit(app.exec_())