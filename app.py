"""
CPU Temperature Monitor - Desktop Application
GUI for training models and monitoring CPU temperature anomalies.
"""

import os
import sys
import threading
import time
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from pathlib import Path

import yaml

from src.core_regressor import CoreTempRegressor, CoreTempPCA
from src.data_extractor import ComputerInfoExtractor
from src.tray_monitor import TrayMonitor
from src.cpu_temp_bundled import HardwareMonitor

import pystray
from PIL import Image, ImageDraw

import matplotlib
matplotlib.use('TkAgg')
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from collections import deque


class ConfigManager:
    """Manage application configuration."""

    DEFAULT_CONFIG = {
        'model_path': 'models/cpu_temp_model_lightgbm.joblib',
        'threshold_std': 1.5,
        'check_interval': 5,
        'notifications_enabled': True,
        'model_approach': 'regressor',
        'last_model_type': 'lightgbm',
        'pca_type': 'pca',
        'pca_components': 5,
        'last_scaler': 'standard',
        'multi_variable': True,
        'collection_interval': 5.0,
        'mean_time': 0,
        'monitor_mean_time': 0,
        'monitor_anomaly_window': 1
    }

    def __init__(self, config_path='config.yaml'):
        self.config_path = config_path
        self.config = self.load()

    def load(self):
        """Load config from file or create default."""
        if os.path.exists(self.config_path):
            try:
                with open(self.config_path, 'r') as f:
                    loaded = yaml.safe_load(f) or {}
                    # Merge with defaults
                    config = self.DEFAULT_CONFIG.copy()
                    config.update(loaded)
                    return config
            except Exception:
                pass
        return self.DEFAULT_CONFIG.copy()

    def save(self):
        """Save config to file."""
        with open(self.config_path, 'w') as f:
            yaml.dump(self.config, f, default_flow_style=False)

    def get(self, key, default=None):
        return self.config.get(key, default)

    def set(self, key, value):
        self.config[key] = value


class CPUTempMonitorApp:
    """Main application window."""

    def __init__(self):
        self.root = tk.Tk()
        self.root.title("CPU Temperature Monitor")
        self.root.geometry("600x800")
        self.root.resizable(True, True)

        # Config
        self.config = ConfigManager()

        # State
        self.regressor = None
        self.tray_monitor = None
        self.training_thread = None
        self.is_training = False
        self.is_monitoring = False
        self.pc_info = None

        # Data collection state
        self.is_collecting = False
        self.collection_thread = None
        self.collected_background_data = []
        self.collection_time_counter = 0
        self.training_data_df = None

        # Simple tray icon (when not monitoring)
        self.simple_tray_icon = None
        self.simple_tray_thread = None

        # Graph data history (for real-time plotting)
        self.graph_max_points = 100  # Show last 100 points
        self.graph_times = deque(maxlen=self.graph_max_points)
        self.graph_actual = deque(maxlen=self.graph_max_points)
        self.graph_predicted = deque(maxlen=self.graph_max_points)
        self.graph_diff = deque(maxlen=self.graph_max_points)
        self.graph_counter = 0
        self.last_prediction_id = -1

        # Setup UI
        self.setup_ui()

        # Handle window close
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

    def setup_ui(self):
        """Setup the main UI with tabs."""
        # Create notebook (tabs)
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill='both', expand=True, padx=10, pady=10)

        # Create scrollable tabs
        self.train_tab = self.create_scrollable_frame(self.notebook)
        self.monitor_tab = self.create_scrollable_frame(self.notebook)
        self.settings_tab = self.create_scrollable_frame(self.notebook)

        self.notebook.add(self.train_tab['outer'], text='  Train  ')
        self.notebook.add(self.monitor_tab['outer'], text='  Monitor  ')
        self.notebook.add(self.settings_tab['outer'], text='  Settings  ')

        self.setup_train_tab()
        self.setup_monitor_tab()
        self.setup_settings_tab()

    def create_scrollable_frame(self, parent):
        """Create a scrollable frame with scrollbar."""
        # Outer frame
        outer_frame = ttk.Frame(parent)

        # Canvas for scrolling
        canvas = tk.Canvas(outer_frame, highlightthickness=0)
        scrollbar = ttk.Scrollbar(outer_frame, orient="vertical", command=canvas.yview)

        # Inner frame that will hold content
        inner_frame = ttk.Frame(canvas)

        # Configure canvas
        canvas.configure(yscrollcommand=scrollbar.set)

        # Pack scrollbar and canvas
        scrollbar.pack(side="right", fill="y")
        canvas.pack(side="left", fill="both", expand=True)

        # Create window in canvas
        canvas_frame = canvas.create_window((0, 0), window=inner_frame, anchor="nw")

        # Update scroll region when frame size changes
        def on_frame_configure(event):
            canvas.configure(scrollregion=canvas.bbox("all"))

        inner_frame.bind("<Configure>", on_frame_configure)

        # Bind mousewheel for scrolling
        def on_mousewheel(event):
            canvas.yview_scroll(int(-1*(event.delta/120)), "units")

        def bind_mousewheel(event):
            canvas.bind_all("<MouseWheel>", on_mousewheel)

        def unbind_mousewheel(event):
            canvas.unbind_all("<MouseWheel>")

        outer_frame.bind("<Enter>", bind_mousewheel)
        outer_frame.bind("<Leave>", unbind_mousewheel)

        return {'outer': outer_frame, 'inner': inner_frame, 'canvas': canvas}

    def setup_train_tab(self):
        """Setup the training tab."""
        frame = self.train_tab['inner']

        # Model selection
        model_frame = ttk.LabelFrame(frame, text="Model Configuration", padding=10)
        model_frame.pack(fill='x', padx=10, pady=5)

        # Model Approach Selection (Regressor vs PCA)
        ttk.Label(model_frame, text="Approach:").grid(row=0, column=0, sticky='w', pady=2)
        self.model_approach_var = tk.StringVar(value=self.config.get('model_approach', 'regressor'))
        approach_frame = ttk.Frame(model_frame)
        approach_frame.grid(row=0, column=1, columnspan=3, sticky='w', pady=2)

        regressor_radio = ttk.Radiobutton(approach_frame, text="Regressor", variable=self.model_approach_var,
                                          value='regressor', command=self.on_approach_change)
        regressor_radio.pack(side='left', padx=(0, 15))
        pca_radio = ttk.Radiobutton(approach_frame, text="PCA", variable=self.model_approach_var,
                                    value='pca', command=self.on_approach_change)
        pca_radio.pack(side='left')

        # Regressor model type
        ttk.Label(model_frame, text="Regressor:").grid(row=1, column=0, sticky='w', pady=2)
        self.model_type_var = tk.StringVar(value=self.config.get('last_model_type', 'lightgbm'))
        self.regressor_combo = ttk.Combobox(model_frame, textvariable=self.model_type_var,
                                            values=['linear', 'xgb', 'lightgbm'], state='readonly', width=12)
        self.regressor_combo.grid(row=1, column=1, pady=2, padx=5, sticky='w')

        # PCA model type
        ttk.Label(model_frame, text="PCA Type:").grid(row=1, column=2, sticky='w', pady=2, padx=(10, 0))
        self.pca_type_var = tk.StringVar(value=self.config.get('pca_type', 'pca'))
        self.pca_combo = ttk.Combobox(model_frame, textvariable=self.pca_type_var,
                                      values=['pca', 'kernel_pca'], state='readonly', width=12)
        self.pca_combo.grid(row=1, column=3, pady=2, padx=5, sticky='w')

        # PCA components
        ttk.Label(model_frame, text="PCA Components:").grid(row=2, column=2, sticky='w', pady=2, padx=(10, 0))
        self.pca_components_var = tk.IntVar(value=self.config.get('pca_components', 5))
        self.pca_components_spin = ttk.Spinbox(model_frame, from_=2, to=50, increment=1,
                                               textvariable=self.pca_components_var, width=10)
        self.pca_components_spin.grid(row=2, column=3, pady=2, padx=5, sticky='w')

        # Scaler
        ttk.Label(model_frame, text="Scaler:").grid(row=2, column=0, sticky='w', pady=2)
        self.scaler_var = tk.StringVar(value=self.config.get('last_scaler', 'standard'))
        scaler_combo = ttk.Combobox(model_frame, textvariable=self.scaler_var,
                                    values=['standard', 'minmax', 'robust'], state='readonly', width=12)
        scaler_combo.grid(row=2, column=1, pady=2, padx=5, sticky='w')

        # Multi-variable checkbox
        self.multi_variable_var = tk.BooleanVar(value=self.config.get('multi_variable', True))
        multi_var_check = ttk.Checkbutton(model_frame, text="Use all sensors (multi-variable mode)",
                                          variable=self.multi_variable_var)
        multi_var_check.grid(row=3, column=0, columnspan=4, sticky='w', pady=5)

        # Initialize UI state based on current approach
        self.on_approach_change()

        # Training parameters
        param_frame = ttk.LabelFrame(frame, text="Training Parameters", padding=10)
        param_frame.pack(fill='x', padx=10, pady=5)

        ttk.Label(param_frame, text="Mean Time (s):").grid(row=0, column=0, sticky='w', pady=2)
        self.mean_time_var = tk.IntVar(value=self.config.get('mean_time', 0))
        mean_time_spin = ttk.Spinbox(param_frame, from_=0, to=300, increment=5,
                                     textvariable=self.mean_time_var, width=10)
        mean_time_spin.grid(row=0, column=1, pady=2, padx=5)

        ttk.Label(param_frame, text="(0 = no resampling, >0 = average over window)",
                 font=('TkDefaultFont', 8), foreground='gray').grid(row=1, column=0, columnspan=2, sticky='w', pady=2)

        # Training data management
        data_frame = ttk.LabelFrame(frame, text="Training Data Collection", padding=10)
        data_frame.pack(fill='x', padx=10, pady=5)

        # Background auto-collection
        ttk.Label(data_frame, text="Background Data Collection:",
                 font=('TkDefaultFont', 9, 'bold')).pack(anchor='w', pady=(0, 5))

        ttk.Label(data_frame,
                 text="Continuously collect sensor data in background. Leave running for hours/days.",
                 font=('TkDefaultFont', 8),
                 foreground='gray').pack(anchor='w', pady=(0, 5))

        # Collection interval
        interval_frame = ttk.Frame(data_frame)
        interval_frame.pack(fill='x', pady=(0, 5))

        ttk.Label(interval_frame, text="Collection Interval (s):").pack(side='left', padx=(0, 5))
        self.collection_interval_var = tk.DoubleVar(value=self.config.get('collection_interval', 5.0))
        interval_spin = ttk.Spinbox(interval_frame, from_=1, to=60, increment=1,
                                    textvariable=self.collection_interval_var, width=10)
        interval_spin.pack(side='left')

        # Collection status
        self.collection_status_label = ttk.Label(data_frame, text="Not collecting", foreground='gray')
        self.collection_status_label.pack(anchor='w', pady=5)

        # Collection control buttons
        collection_ctrl_frame = ttk.Frame(data_frame)
        collection_ctrl_frame.pack(fill='x', pady=(0, 10))

        self.start_collection_btn = ttk.Button(collection_ctrl_frame, text="Start Collection",
                                               command=self.start_background_collection)
        self.start_collection_btn.pack(side='left', padx=5)

        self.stop_collection_btn = ttk.Button(collection_ctrl_frame, text="Stop Collection",
                                              command=self.stop_background_collection, state='disabled')
        self.stop_collection_btn.pack(side='left', padx=5)

        # Save/Clear collected data
        data_mgmt_frame = ttk.Frame(data_frame)
        data_mgmt_frame.pack(fill='x', pady=(0, 10))

        self.save_collected_btn = ttk.Button(data_mgmt_frame, text="Save Collected Data",
                                             command=self.save_background_collected_data, state='disabled')
        self.save_collected_btn.pack(side='left', padx=5)

        self.clear_collected_btn = ttk.Button(data_mgmt_frame, text="Clear Collected Data",
                                              command=self.clear_background_collected_data, state='disabled')
        self.clear_collected_btn.pack(side='left', padx=5)

        # Separator
        ttk.Separator(data_frame, orient='horizontal').pack(fill='x', pady=10)

        # Manual training data buttons
        ttk.Label(data_frame, text="Manual Training Data:",
                 font=('TkDefaultFont', 9, 'bold')).pack(anchor='w', pady=(0, 5))

        manual_data_frame = ttk.Frame(data_frame)
        manual_data_frame.pack(fill='x')

        self.save_data_btn = ttk.Button(manual_data_frame, text="Save Training Data", command=self.save_training_data, state='disabled')
        self.save_data_btn.pack(side='left', padx=5)

        self.load_data_btn = ttk.Button(manual_data_frame, text="Load Training Data", command=self.load_training_data)
        self.load_data_btn.pack(side='left', padx=5)

        self.load_multiple_btn = ttk.Button(manual_data_frame, text="Load & Combine CSVs", command=self.load_and_combine_csvs)
        self.load_multiple_btn.pack(side='left', padx=5)

        # Progress
        progress_frame = ttk.LabelFrame(frame, text="Progress", padding=10)
        progress_frame.pack(fill='x', padx=10, pady=5)

        self.progress_var = tk.DoubleVar(value=0)
        self.progress_bar = ttk.Progressbar(progress_frame, variable=self.progress_var,
                                            maximum=100, length=400)
        self.progress_bar.pack(fill='x', pady=5)

        self.progress_label = ttk.Label(progress_frame, text="Ready to train")
        self.progress_label.pack()

        # Buttons
        btn_frame = ttk.Frame(frame)
        btn_frame.pack(fill='x', padx=10, pady=10)

        self.train_from_data_btn = ttk.Button(
            btn_frame,
            text="Train From Data",
            command=self.train_from_existing_data,
            state='disabled'
        )
        self.train_from_data_btn.pack(side='left', padx=5)

        self.save_btn = ttk.Button(btn_frame, text="Save Model", command=self.save_model, state='disabled')
        self.save_btn.pack(side='left', padx=5)

        self.stop_train_btn = ttk.Button(btn_frame, text="Stop", command=self.stop_training, state='disabled')
        self.stop_train_btn.pack(side='left', padx=5)

        # Metrics display
        metrics_frame = ttk.LabelFrame(frame, text="Training Metrics", padding=10)
        metrics_frame.pack(fill='x', padx=10, pady=5)

        self.metrics_label = ttk.Label(metrics_frame, text="Train a model to see metrics")
        self.metrics_label.pack()

    def setup_monitor_tab(self):
        """Setup the monitoring tab."""
        frame = self.monitor_tab['inner']

        # Model selection
        model_frame = ttk.LabelFrame(frame, text="Model Selection", padding=10)
        model_frame.pack(fill='x', padx=10, pady=5)

        ttk.Label(model_frame, text="Model File:").grid(row=0, column=0, sticky='w', pady=2)
        self.model_path_var = tk.StringVar(value=self.config.get('model_path', ''))
        model_entry = ttk.Entry(model_frame, textvariable=self.model_path_var, width=35)
        model_entry.grid(row=0, column=1, pady=2, padx=5)

        browse_btn = ttk.Button(model_frame, text="Browse", command=self.browse_model)
        browse_btn.grid(row=0, column=2, pady=2, padx=5)

        # Threshold
        ttk.Label(model_frame, text="Threshold (std):").grid(row=1, column=0, sticky='w', pady=2)
        self.threshold_var = tk.DoubleVar(value=self.config.get('threshold_std', 1.5))
        threshold_spin = ttk.Spinbox(model_frame, from_=0.5, to=5.0, increment=0.1,
                                     textvariable=self.threshold_var, width=10)
        threshold_spin.grid(row=1, column=1, pady=2, padx=5, sticky='w')

        self.apply_threshold_btn = ttk.Button(model_frame, text="Apply Threshold", command=self.apply_threshold, state='disabled')
        self.apply_threshold_btn.grid(row=1, column=2, pady=2, padx=5)

        # Mean time for monitor predictions
        ttk.Label(model_frame, text="Mean Time (s):").grid(row=2, column=0, sticky='w', pady=2)
        self.monitor_mean_time_var = tk.IntVar(value=self.config.get('monitor_mean_time', 0))
        monitor_mean_spin = ttk.Spinbox(model_frame, from_=0, to=300, increment=1,
                                        textvariable=self.monitor_mean_time_var, width=10)
        monitor_mean_spin.grid(row=2, column=1, pady=2, padx=5, sticky='w')
        ttk.Label(model_frame, text="(0 = no averaging; >0 = mean of X seconds)",
                 font=('TkDefaultFont', 8), foreground='gray').grid(row=3, column=0, columnspan=3, sticky='w', pady=2)

        # Anomaly window (consecutive points)
        ttk.Label(model_frame, text="Anomaly Window (points):").grid(row=4, column=0, sticky='w', pady=2)
        self.monitor_anomaly_window_var = tk.IntVar(value=self.config.get('monitor_anomaly_window', 1))
        anomaly_window_spin = ttk.Spinbox(model_frame, from_=1, to=300, increment=1,
                                          textvariable=self.monitor_anomaly_window_var, width=10)
        anomaly_window_spin.grid(row=4, column=1, pady=2, padx=5, sticky='w')
        self.anomaly_window_time_label = ttk.Label(model_frame, text="", foreground='gray')
        self.anomaly_window_time_label.grid(row=4, column=2, sticky='w', pady=2)
        ttk.Label(model_frame, text="(Requires N consecutive points over threshold)",
                 font=('TkDefaultFont', 8), foreground='gray').grid(row=5, column=0, columnspan=3, sticky='w', pady=2)

        # Dynamic window time label updates
        self.monitor_mean_time_var.trace_add('write', self.update_anomaly_window_time)
        self.monitor_anomaly_window_var.trace_add('write', self.update_anomaly_window_time)
        self.update_anomaly_window_time()

        # Buttons
        btn_frame = ttk.Frame(frame)
        btn_frame.pack(fill='x', padx=10, pady=5)

        self.monitor_btn = ttk.Button(btn_frame, text="Start Monitoring", command=self.start_monitoring)
        self.monitor_btn.pack(side='left', padx=5)

        self.stop_monitor_btn = ttk.Button(btn_frame, text="Stop", command=self.stop_monitoring, state='disabled')
        self.stop_monitor_btn.pack(side='left', padx=5)

        self.minimize_btn = ttk.Button(btn_frame, text="Minimize to Tray", command=self.minimize_to_tray, state='disabled')
        self.minimize_btn.pack(side='left', padx=5)

        # Real-time Graph
        graph_frame = ttk.LabelFrame(frame, text="Real-time Temperature Graph", padding=5)
        graph_frame.pack(fill='both', expand=True, padx=10, pady=5)

        # Create matplotlib figure with 2 subplots
        self.fig = Figure(figsize=(6, 4), dpi=80)
        self.fig.set_facecolor('#f0f0f0')

        # Subplot 1: Actual vs Predicted Temperature
        self.ax1 = self.fig.add_subplot(211)
        self.ax1.set_ylabel('Temperature (°C)')
        self.ax1.set_title('Actual vs Predicted', fontsize=9)
        self.ax1.grid(True, alpha=0.3)
        self.line_actual, = self.ax1.plot([], [], 'b-', label='Actual', linewidth=1.5)
        self.line_predicted, = self.ax1.plot([], [], 'r--', label='Predicted', linewidth=1.5)
        self.ax1.legend(loc='upper left', fontsize=8)

        # Subplot 2: Difference with Thresholds
        self.ax2 = self.fig.add_subplot(212)
        self.ax2.set_xlabel('Time')
        self.ax2.set_ylabel('Diff (°C)')
        self.ax2.set_title('Prediction Error (Diff)', fontsize=9)
        self.ax2.grid(True, alpha=0.3)
        self.line_diff, = self.ax2.plot([], [], 'orange', label='Diff', linewidth=1.5)
        self.line_low_thresh = self.ax2.axhline(y=0, color='green', linestyle='--', linewidth=1.5, alpha=0.7, label='Low')
        self.line_high_thresh = self.ax2.axhline(y=0, color='red', linestyle='--', linewidth=1.5, alpha=0.7, label='High')
        self.ax2.legend(loc='upper left', fontsize=8)

        self.fig.tight_layout()

        # Embed in tkinter
        self.canvas = FigureCanvasTkAgg(self.fig, master=graph_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill='both', expand=True)

        # Status (compact)
        status_frame = ttk.LabelFrame(frame, text="Status", padding=5)
        status_frame.pack(fill='x', padx=10, pady=5)

        self.status_text = tk.Text(status_frame, height=5, width=50, state='disabled', font=('Consolas', 9))
        self.status_text.pack(fill='x', pady=2)

    def _format_duration(self, total_seconds: int) -> str:
        """Format seconds into human-friendly text."""
        total_seconds = max(0, int(round(total_seconds)))

        if total_seconds < 60:
            return f"{total_seconds} second" + ("s" if total_seconds != 1 else "")

        minutes, seconds = divmod(total_seconds, 60)
        if minutes < 60:
            minutes_text = f"{minutes} minute" + ("s" if minutes != 1 else "")
            if seconds:
                seconds_text = f"{seconds} second" + ("s" if seconds != 1 else "")
                return f"{minutes_text} {seconds_text}"
            return minutes_text

        hours, minutes = divmod(minutes, 60)
        hours_text = f"{hours} hour" + ("s" if hours != 1 else "")
        if minutes:
            minutes_text = f"{minutes} minute" + ("s" if minutes != 1 else "")
            return f"{hours_text} {minutes_text}"
        return hours_text

    def update_anomaly_window_time(self, *_):
        """Update the displayed time for the anomaly window."""
        if not hasattr(self, 'anomaly_window_time_label'):
            return

        try:
            points = int(self.monitor_anomaly_window_var.get() or 1)
        except Exception:
            points = 1

        try:
            mean_time = int(self.monitor_mean_time_var.get() or 0)
        except Exception:
            mean_time = 0

        points = max(1, points)
        seconds_per_point = mean_time if mean_time > 0 else 1
        total_seconds = points * seconds_per_point
        formatted = self._format_duration(total_seconds)
        self.anomaly_window_time_label.config(text=f"= {formatted}")

    def setup_settings_tab(self):
        """Setup the settings tab."""
        frame = self.settings_tab['inner']

        # Hardware Information
        hw_frame = ttk.LabelFrame(frame, text="Hardware Information", padding=10)
        hw_frame.pack(fill='x', padx=10, pady=5)

        # Hardware info display
        self.hw_info_text = tk.Text(hw_frame, height=6, width=50, state='disabled',
                                     font=('Consolas', 9), background='#f8f9fa')
        self.hw_info_text.pack(fill='x', pady=5)

        # Button to detect hardware
        detect_hw_btn = ttk.Button(hw_frame, text="Detect Hardware", command=self.detect_hardware)
        detect_hw_btn.pack(pady=5)

        # General settings
        general_frame = ttk.LabelFrame(frame, text="General", padding=10)
        general_frame.pack(fill='x', padx=10, pady=5)

        ttk.Label(general_frame, text="Check Interval (s):").grid(row=0, column=0, sticky='w', pady=2)
        self.check_interval_var = tk.DoubleVar(value=self.config.get('check_interval', 5))
        interval_spin = ttk.Spinbox(general_frame, from_=1, to=60, increment=1,
                                    textvariable=self.check_interval_var, width=10)
        interval_spin.grid(row=0, column=1, pady=2, padx=5, sticky='w')

        self.notifications_var = tk.BooleanVar(value=self.config.get('notifications_enabled', True))
        notif_check = ttk.Checkbutton(general_frame, text="Enable Windows Notifications",
                                      variable=self.notifications_var)
        notif_check.grid(row=1, column=0, columnspan=2, sticky='w', pady=5)

        ttk.Label(general_frame,
                 text="Note: App ALWAYS minimizes to tray (never closes from X button).\nEverything continues running in background (training, collection, monitoring).",
                 font=('TkDefaultFont', 8),
                 foreground='gray').grid(row=2, column=0, columnspan=2, sticky='w', pady=5)

        # Save button
        save_frame = ttk.Frame(frame)
        save_frame.pack(fill='x', padx=10, pady=20)

        save_btn = ttk.Button(save_frame, text="Save Settings", command=self.save_settings)
        save_btn.pack()

        # Info
        info_frame = ttk.LabelFrame(frame, text="Info", padding=10)
        info_frame.pack(fill='x', padx=10, pady=5)

        info_text = ("CPU Temperature Monitor\n\n"
                     "1. Train, collect data, or monitor\n"
                     "2. Close window (X) → minimizes to tray\n"
                     "3. Everything keeps running in background!\n"
                     "4. Click tray icon to reopen window\n"
                     "5. To quit: use tray menu 'Quit'")
        ttk.Label(info_frame, text=info_text, justify='left').pack()

    def on_approach_change(self):
        """Handle approach change between Regressor and PCA."""
        approach = self.model_approach_var.get()
        if approach == 'regressor':
            # Enable regressor combo, disable PCA options
            self.regressor_combo.config(state='readonly')
            self.pca_combo.config(state='disabled')
            self.pca_components_spin.config(state='disabled')
        else:
            # Disable regressor combo, enable PCA options
            self.regressor_combo.config(state='disabled')
            self.pca_combo.config(state='readonly')
            self.pca_components_spin.config(state='normal')

    def browse_model(self):
        """Open file dialog to select a model."""
        initial_dir = 'models' if os.path.exists('models') else '.'
        path = filedialog.askopenfilename(
            initialdir=initial_dir,
            title="Select Model",
            filetypes=[("Joblib files", "*.joblib"), ("All files", "*.*")]
        )
        if path:
            self.model_path_var.set(path)

    def update_train_from_data_state(self):
        """Enable/disable training from existing data based on availability."""
        has_loaded = self.training_data_df is not None and len(self.training_data_df) > 0
        has_collected = len(self.collected_background_data) > 0
        state = 'normal' if (has_loaded or has_collected) else 'disabled'
        if hasattr(self, 'train_from_data_btn'):
            self.train_from_data_btn.config(state=state)

    def train_from_existing_data(self):
        """Train a model using already collected or loaded data."""
        if self.is_training:
            return

        data_df = self.training_data_df
        if data_df is None or len(data_df) == 0:
            if self.collected_background_data:
                import pandas as pd
                data_df = pd.DataFrame(list(self.collected_background_data))
                self.training_data_df = data_df
                self.update_train_from_data_state()
            else:
                messagebox.showwarning(
                    "Warning",
                    "No training data available.\n\nLoad data or collect samples first."
                )
                return

        self.is_training = True
        self.train_from_data_btn.config(state='disabled')
        self.stop_train_btn.config(state='normal')
        self.save_btn.config(state='disabled')
        self.progress_var.set(0)
        self.progress_label.config(text="Training model...")

        def train_thread():
            try:
                # Create extractor and set existing data
                extractor = ComputerInfoExtractor(
                    scaler_type=self.scaler_var.get(),
                    use_time_features=True,
                    lag_steps=[1, 2, 3],
                    rolling_windows=[3, 5, 7]
                )

                # Set raw data and preprocess
                extractor.data = data_df.copy()
                if 'timestamp' in extractor.data.columns:
                    import pandas as pd
                    extractor.data['timestamp'] = pd.to_datetime(extractor.data['timestamp'])

                # Apply preprocessing pipeline
                if extractor.use_time_features:
                    processed_data = extractor.create_time_features_on_df()
                else:
                    processed_data = extractor.data.copy()
                processed_data = processed_data.ffill().bfill().reset_index(drop=True)

                # Get selected approach
                approach = self.model_approach_var.get()

                if approach == 'regressor':
                    # Create regressor with extractor
                    self.regressor = CoreTempRegressor(extractor=extractor)
                    self.regressor.set_data(processed_data)
                    self.regressor.configure_model(
                        model=self.model_type_var.get(),
                        multi_variable=self.multi_variable_var.get(),
                        scaler=self.scaler_var.get(),
                        use_time_features=True
                    )

                    self.regressor.fit_predict(train_size=0.8, threshold_std=self.threshold_var.get())

                    metrics = self.regressor.evaluate_metrics()
                    metrics_text = (f"RMSE: {metrics['rmse']:.2f}\n"
                                   f"MAE: {metrics['mae']:.2f}\n"
                                   f"R2: {metrics['r2']:.3f}\n"
                                   f"MAPE: {metrics['mape']:.1f}%")
                else:
                    # Create PCA model with extractor
                    self.regressor = CoreTempPCA(extractor=extractor)
                    self.regressor.set_data(processed_data)
                    self.regressor.configure_model(
                        model=self.pca_type_var.get(),
                        multi_variable=self.multi_variable_var.get(),
                        scaler=self.scaler_var.get(),
                        use_time_features=True,
                        n_components=self.pca_components_var.get()
                    )

                    self.regressor.fit_predict(train_size=0.8, threshold_std=self.threshold_var.get())

                    metrics = self.regressor.evaluate_metrics()
                    metrics_text = (f"Mean Recon Error: {metrics['mean_reconstruction_error']:.4f}\n"
                                   f"Std Recon Error: {metrics['std_reconstruction_error']:.4f}\n"
                                   f"Anomalies: {metrics['n_anomalies']}\n"
                                   f"Anomaly Rate: {metrics['anomaly_rate_percent']:.1f}%")

                self.root.after(0, lambda: self.metrics_label.config(text=metrics_text))
                self.root.after(0, lambda: self.progress_label.config(text="Training complete!"))
                self.root.after(0, lambda: self.progress_var.set(100))
                self.root.after(0, lambda: self.save_btn.config(state='normal'))
                self.root.after(0, lambda: self.save_data_btn.config(state='normal'))
                self.root.after(0, self.update_train_from_data_state)

            except Exception as e:
                self.root.after(0, lambda: messagebox.showerror("Error", f"Training failed: {e}"))
                self.root.after(0, lambda: self.progress_label.config(text="Training failed"))

            finally:
                self.is_training = False
                self.root.after(0, self.update_train_from_data_state)
                self.root.after(0, lambda: self.stop_train_btn.config(state='disabled'))

        threading.Thread(target=train_thread, daemon=True).start()

    def stop_training(self):
        """Stop the training process."""
        self.is_training = False
        self.progress_label.config(text="Training stopped")

    def save_model(self):
        """Save the trained model."""
        if not self.regressor:
            messagebox.showwarning("Warning", "No trained model to save")
            return

        # Ensure directory exists
        os.makedirs('models', exist_ok=True)

        # Generate filename based on approach
        approach = self.model_approach_var.get()
        if approach == 'regressor':
            model_type = self.model_type_var.get()
        else:
            model_type = self.pca_type_var.get()
        default_name = f"cpu_temp_model_{model_type}.joblib"

        path = filedialog.asksaveasfilename(
            initialdir='models',
            initialfile=default_name,
            title="Save Model",
            filetypes=[("Joblib files", "*.joblib")],
            defaultextension=".joblib"
        )

        if path:
            self.regressor.save_model(path)
            messagebox.showinfo("Success", f"Model saved to:\n{path}")
            self.model_path_var.set(path)

            # Update config
            self.config.set('model_approach', approach)
            self.config.set('last_model_type', self.model_type_var.get())
            self.config.set('pca_type', self.pca_type_var.get())
            self.config.set('pca_components', self.pca_components_var.get())
            self.config.set('last_scaler', self.scaler_var.get())
            self.config.set('multi_variable', self.multi_variable_var.get())
            self.config.set('mean_time', self.mean_time_var.get())
            self.config.set('model_path', path)
            self.config.save()

    def save_training_data(self):
        """Save the collected training data to CSV."""
        data_df = None
        if self.training_data_df is not None and len(self.training_data_df) > 0:
            data_df = self.training_data_df
        elif self.regressor and self.regressor.data is not None and len(self.regressor.data) > 0:
            data_df = self.regressor.data

        if data_df is None:
            messagebox.showwarning("Warning", "No training data to save")
            return

        # Ensure directory exists
        os.makedirs('data', exist_ok=True)

        # Generate filename with timestamp
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        default_name = f"training_data_{timestamp}.csv"

        path = filedialog.asksaveasfilename(
            initialdir='data',
            initialfile=default_name,
            title="Save Training Data",
            filetypes=[("CSV files", "*.csv")],
            defaultextension=".csv"
        )

        if path:
            data_df.to_csv(path, index=False)
            messagebox.showinfo("Success", f"Training data saved to:\n{path}\n\n{len(data_df)} rows saved")

    def load_training_data(self):
        """Load training data from CSV and train model."""
        path = filedialog.askopenfilename(
            initialdir='data' if os.path.exists('data') else '.',
            title="Load Training Data",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )

        if not path:
            return

        try:
            # Load data from CSV
            import pandas as pd
            data_df = pd.read_csv(path)

            # Convert timestamp if present
            if 'timestamp' in data_df.columns:
                data_df['timestamp'] = pd.to_datetime(data_df['timestamp'])

            # Apply mean_time resampling if specified
            mean_time = self.mean_time_var.get()
            if mean_time > 0 and 'timestamp' in data_df.columns:
                original_rows = len(data_df)
                data_df = data_df.resample(f"{mean_time}s", on='timestamp').mean().reset_index()
                # Recreate sequential time column
                if 'time' in data_df.columns:
                    data_df['time'] = range(len(data_df))
                rows = len(data_df)
                resample_msg = f"\n(Resampled from {original_rows} to {rows} rows using {mean_time}s window)"
            else:
                rows = len(data_df)
                resample_msg = ""

            self.training_data_df = data_df
            self.update_train_from_data_state()

            # Ask if should train immediately
            should_train = messagebox.askyesno(
                "Data Loaded",
                f"Loaded {rows} rows from:\n{path}{resample_msg}\n\nTrain model with this data now?"
            )

            if should_train:
                self.train_from_existing_data()
            else:
                self.progress_label.config(text=f"Loaded {rows} rows - ready to train")
                self.save_data_btn.config(state='normal')

        except Exception as e:
            messagebox.showerror("Error", f"Failed to load data:\n{e}")

    def load_and_combine_csvs(self):
        """Load and combine multiple CSV files for training."""
        paths = filedialog.askopenfilenames(
            initialdir='data' if os.path.exists('data') else '.',
            title="Select Multiple CSV Files to Combine",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )

        if not paths:
            return

        try:
            import pandas as pd

            # Load all CSV files
            dfs = []
            total_rows = 0
            file_info = []

            for path in paths:
                df = pd.read_csv(path)

                # Convert timestamp if present
                if 'timestamp' in df.columns:
                    df['timestamp'] = pd.to_datetime(df['timestamp'])

                dfs.append(df)
                rows = len(df)
                total_rows += rows
                file_info.append(f"  - {Path(path).name}: {rows} rows")

            # Combine all dataframes
            combined_df = pd.concat(dfs, ignore_index=True)

            # Sort by timestamp if available
            if 'timestamp' in combined_df.columns:
                combined_df = combined_df.sort_values('timestamp').reset_index(drop=True)
                # Recreate sequential time column
                if 'time' in combined_df.columns:
                    combined_df['time'] = range(len(combined_df))

            # Apply mean_time resampling if specified
            mean_time = self.mean_time_var.get()
            if mean_time > 0 and 'timestamp' in combined_df.columns:
                original_rows = len(combined_df)
                combined_df = combined_df.resample(f"{mean_time}s", on='timestamp').mean().reset_index()
                # Recreate sequential time column after resampling
                if 'time' in combined_df.columns:
                    combined_df['time'] = range(len(combined_df))
                final_rows = len(combined_df)
                resample_msg = f"\n\nResampled from {original_rows} to {final_rows} rows using {mean_time}s window"
            else:
                final_rows = len(combined_df)
                resample_msg = ""

            self.training_data_df = combined_df
            self.update_train_from_data_state()

            # Build info message
            info_msg = (f"Combined {len(paths)} CSV files:\n" +
                       "\n".join(file_info) +
                       f"\n\nTotal: {total_rows} rows combined" +
                       (f" → {final_rows} rows after sorting" if final_rows != total_rows else "") +
                       resample_msg)

            # Ask if should train immediately
            should_train = messagebox.askyesno(
                "Data Combined",
                info_msg + "\n\nTrain model with this combined data now?"
            )

            if should_train:
                self.train_from_existing_data()
            else:
                self.progress_label.config(text=f"Combined {len(paths)} files ({final_rows} rows) - ready to train")
                self.save_data_btn.config(state='normal')

        except Exception as e:
            messagebox.showerror("Error", f"Failed to combine CSV files:\n{e}")

    def start_monitoring(self):
        """Start the monitoring process."""
        model_path = self.model_path_var.get()
        if not model_path or not os.path.exists(model_path):
            messagebox.showwarning("Warning", "Please select a valid model file")
            return

        try:
            # Stop simple tray icon if exists (monitoring has its own tray)
            if self.simple_tray_icon:
                self.simple_tray_icon.stop()
                self.simple_tray_icon = None

            self.tray_monitor = TrayMonitor(
                model_path=model_path,
                check_interval=self.check_interval_var.get(),
                threshold_std=self.threshold_var.get(),
                notifications_enabled=self.notifications_var.get(),
                on_open_gui=self.show_window,
                on_quit_app=self.quit_application,
                mean_time=self.monitor_mean_time_var.get(),
                anomaly_window=self.monitor_anomaly_window_var.get()
            )
            self.tray_monitor.load_model()
            self.tray_monitor.start_monitoring()

            self.is_monitoring = True
            self.monitor_btn.config(state='disabled')
            self.stop_monitor_btn.config(state='normal')
            self.minimize_btn.config(state='normal')
            self.apply_threshold_btn.config(state='normal')
            self.last_prediction_id = -1

            # Update status periodically
            self.update_status()

            # Start tray in background
            self.tray_monitor.run_tray_detached()

        except Exception as e:
            messagebox.showerror("Error", f"Failed to start monitoring: {e}")

    def update_status(self):
        """Update the status display and graph."""
        if not self.is_monitoring or not self.tray_monitor:
            return

        # Update text status
        status = self.tray_monitor.get_status_text()
        self.status_text.config(state='normal')
        self.status_text.delete(1.0, tk.END)
        self.status_text.insert(tk.END, status)
        self.status_text.config(state='disabled')

        prediction_id = getattr(self.tray_monitor, 'prediction_counter', 0)
        new_prediction = prediction_id != self.last_prediction_id
        if new_prediction:
            self.last_prediction_id = prediction_id

        # Update graph data only when new prediction is available
        if new_prediction and self.tray_monitor.current_temp > 0:
            self.graph_times.append(self.graph_counter)
            self.graph_actual.append(self.tray_monitor.current_temp)
            self.graph_predicted.append(self.tray_monitor.predicted_temp)
            self.graph_diff.append(self.tray_monitor.last_diff)
            self.graph_counter += 1

            # Update graph
            self.update_graph()

        # Schedule next update
        self.root.after(1000, self.update_status)

    def update_graph(self):
        """Update the real-time matplotlib graph."""
        if not self.graph_times:
            return

        times = list(self.graph_times)
        actual = list(self.graph_actual)
        predicted = list(self.graph_predicted)
        diff = list(self.graph_diff)

        # Update subplot 1: Actual vs Predicted
        self.line_actual.set_data(times, actual)
        self.line_predicted.set_data(times, predicted)

        # Adjust axes for subplot 1
        if times:
            self.ax1.set_xlim(min(times), max(times) + 1)
            all_temps = actual + predicted
            if all_temps:
                min_temp = min(all_temps) - 2
                max_temp = max(all_temps) + 2
                self.ax1.set_ylim(min_temp, max_temp)

        # Update subplot 2: Diff with Thresholds
        self.line_diff.set_data(times, diff)

        # Update threshold lines
        if self.tray_monitor:
            low_thresh = self.tray_monitor.low_threshold or 0.0
            high_thresh = self.tray_monitor.high_threshold or 0.0
            self.line_low_thresh.set_ydata([float(low_thresh), float(low_thresh)])
            self.line_high_thresh.set_ydata([float(high_thresh), float(high_thresh)])

        # Adjust axes for subplot 2
        if times:
            self.ax2.set_xlim(min(times), max(times) + 1)
            if diff:
                low_thresh = float(self.tray_monitor.low_threshold or 0) if self.tray_monitor else 0.0
                high_thresh = float(self.tray_monitor.high_threshold or 0) if self.tray_monitor else 0.0
                min_diff = min(min(diff), low_thresh) - 1
                max_diff = max(max(diff), high_thresh) + 1
                self.ax2.set_ylim(min_diff, max_diff)

        # Redraw canvas
        self.canvas.draw_idle()

    def clear_graph(self):
        """Clear the graph data."""
        self.graph_times.clear()
        self.graph_actual.clear()
        self.graph_predicted.clear()
        self.graph_diff.clear()
        self.graph_counter = 0

        # Clear plot lines
        self.line_actual.set_data([], [])
        self.line_predicted.set_data([], [])
        self.line_diff.set_data([], [])
        self.canvas.draw_idle()

    def stop_monitoring(self):
        """Stop the monitoring process."""
        if self.tray_monitor:
            self.tray_monitor.stop_monitoring()
            self.tray_monitor.stop_tray()
            self.tray_monitor = None

        self.is_monitoring = False
        self.monitor_btn.config(state='normal')
        self.stop_monitor_btn.config(state='disabled')
        self.minimize_btn.config(state='disabled')
        self.apply_threshold_btn.config(state='disabled')

        self.status_text.config(state='normal')
        self.status_text.delete(1.0, tk.END)
        self.status_text.insert(tk.END, "Monitoring stopped")
        self.status_text.config(state='disabled')

        # Clear graph for next session
        self.clear_graph()

        # If window is minimized, create simple tray icon
        if not self.root.winfo_viewable():
            self.create_simple_tray()

    def start_background_collection(self):
        """Start background data collection without monitoring."""
        if self.is_collecting:
            return

        self.is_collecting = True
        self.start_collection_btn.config(state='disabled')
        self.stop_collection_btn.config(state='normal')
        self.save_collected_btn.config(state='normal')
        self.clear_collected_btn.config(state='normal')

        def collection_loop():
            from datetime import datetime
            from src.cpu_temp_bundled import HardwareMonitor

            monitor = HardwareMonitor()
            interval = self.collection_interval_var.get()

            while self.is_collecting:
                try:
                    # Get sensor data
                    data = monitor.get_essential_fast()
                    data['timestamp'] = datetime.now()
                    data['time'] = self.collection_time_counter
                    self.collected_background_data.append(data)
                    self.collection_time_counter += 1
                    if len(self.collected_background_data) == 1:
                        self.root.after(0, self.update_train_from_data_state)

                    # Update status
                    self.root.after(0, lambda: self.collection_status_label.config(
                        text=f"Collecting... {len(self.collected_background_data)} samples",
                        foreground='green'
                    ))

                except Exception as e:
                    print(f"Collection error: {e}")

                time.sleep(interval)

            # Cleanup
            monitor.close()
            self.root.after(0, lambda: self.collection_status_label.config(
                text=f"Stopped - {len(self.collected_background_data)} samples collected",
                foreground='orange'
            ))

        self.collection_thread = threading.Thread(target=collection_loop, daemon=True)
        self.collection_thread.start()

    def stop_background_collection(self):
        """Stop background data collection."""
        self.is_collecting = False
        self.start_collection_btn.config(state='normal')
        self.stop_collection_btn.config(state='disabled')

    def save_background_collected_data(self):
        """Save background collected data to CSV."""
        if not self.collected_background_data:
            messagebox.showwarning("Warning", "No collected data to save")
            return

        # Ensure directory exists
        os.makedirs('data', exist_ok=True)

        # Generate filename with timestamp
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        default_name = f"background_data_{timestamp}.csv"

        path = filedialog.asksaveasfilename(
            initialdir='data',
            initialfile=default_name,
            title="Save Background Collected Data",
            filetypes=[("CSV files", "*.csv")],
            defaultextension=".csv"
        )

        if path:
            import pandas as pd
            df = pd.DataFrame(self.collected_background_data)
            df.to_csv(path, index=False)
            messagebox.showinfo("Success",
                f"Background data saved to:\n{path}\n\n"
                f"{len(self.collected_background_data)} samples saved")

    def clear_background_collected_data(self):
        """Clear background collected data."""
        if self.collected_background_data:
            result = messagebox.askyesno(
                "Clear Data",
                f"Are you sure you want to clear {len(self.collected_background_data)} collected samples?\n\n"
                "This cannot be undone!"
            )
            if result:
                self.collected_background_data = []
                self.collection_time_counter = 0
                self.collection_status_label.config(text="Not collecting", foreground='gray')
                self.update_train_from_data_state()
                messagebox.showinfo("Cleared", "Collected data has been cleared")
        else:
            messagebox.showinfo("Info", "No collected data to clear")

    def apply_threshold(self):
        """Apply new threshold to running monitor."""
        if not self.tray_monitor or not self.tray_monitor.regressor:
            messagebox.showwarning("Warning", "No model loaded")
            return

        try:
            new_threshold = self.threshold_var.get()

            # Recalculate thresholds in the model
            self.tray_monitor.regressor.recalculate_thresholds(new_threshold)

            # Update tray monitor's cached thresholds
            self.tray_monitor.low_threshold = self.tray_monitor.regressor.low_threshold
            self.tray_monitor.high_threshold = self.tray_monitor.regressor.high_threshold

            # Update status display
            self.update_status()

            messagebox.showinfo("Success",
                f"Threshold updated to {new_threshold:.1f} std\n"
                f"New diff range: [{self.tray_monitor.low_threshold:.1f}, {self.tray_monitor.high_threshold:+.1f}]°C")

        except Exception as e:
            messagebox.showerror("Error", f"Failed to apply threshold:\n{e}")

    def minimize_to_tray(self):
        """Minimize the window to system tray."""
        self.root.withdraw()

        # If no tray monitor exists (not monitoring), create a simple tray icon
        if not self.tray_monitor:
            self.create_simple_tray()

    def show_window(self):
        """Show the window from tray."""
        # Stop simple tray icon if it exists (not needed when window is visible)
        if self.simple_tray_icon:
            self.simple_tray_icon.stop()
            self.simple_tray_icon = None

        self.root.deiconify()
        self.root.lift()
        self.root.focus_force()

    def create_simple_tray_icon(self):
        """Create a simple colored circle icon for tray."""
        size = 64
        image = Image.new('RGBA', (size, size), (0, 0, 0, 0))
        draw = ImageDraw.Draw(image)

        # Determine color based on state
        if self.is_training:
            color = '#3b82f6'  # Blue - training
        elif self.is_collecting:
            color = '#10b981'  # Green - collecting
        else:
            color = '#6b7280'  # Gray - idle

        # Draw circle
        margin = 4
        draw.ellipse([margin, margin, size - margin, size - margin], fill=color)

        # Draw thermometer shape inside
        center_x, center_y = size // 2, size // 2
        draw.ellipse([center_x - 8, center_y + 8, center_x + 8, center_y + 20], fill='white')
        draw.rectangle([center_x - 4, center_y - 16, center_x + 4, center_y + 12], fill='white')
        draw.ellipse([center_x - 6, center_y - 18, center_x + 6, center_y - 12], fill='white')

        return image

    def create_simple_tray(self):
        """Create simple tray icon when not monitoring."""
        if self.simple_tray_icon:
            return  # Already exists

        def on_open(icon, item):
            self.show_window()

        def on_quit(icon, item):
            icon.stop()
            self.quit_application()

        # Get status text
        status_lines = []
        if self.is_training:
            status_lines.append("Training in progress...")
        if self.is_collecting:
            status_lines.append(f"Collecting: {len(self.collected_background_data)} samples")
        if not status_lines:
            status_lines.append("Idle - Click to open")

        menu = pystray.Menu(
            pystray.MenuItem("Open GUI", on_open, default=True),
            pystray.Menu.SEPARATOR,
            pystray.MenuItem(lambda text: status_lines[0], None, enabled=False),
            pystray.Menu.SEPARATOR,
            pystray.MenuItem("Quit", on_quit)
        )

        self.simple_tray_icon = pystray.Icon(
            "cpu_temp_monitor",
            self.create_simple_tray_icon(),
            "CPU Temp Monitor",
            menu=menu
        )

        # Run in background thread
        def run_tray():
            self.simple_tray_icon.run()

        self.simple_tray_thread = threading.Thread(target=run_tray, daemon=True)
        self.simple_tray_thread.start()

    def detect_hardware(self):
        """Detect and display PC hardware information."""
        def detect_thread():
            try:
                # Show loading message
                self.root.after(0, lambda: self._update_hw_info_display("Detecting hardware..."))

                # Extract PC info using ComputerInfoExtractor
                extractor = ComputerInfoExtractor()
                pc_dict = extractor.extract_PC_info()

                self.pc_info = pc_dict

                # Update display
                self.root.after(0, lambda: self._update_hw_info_display(self._format_hw_info(pc_dict)))

            except Exception as e:
                error_msg = f"Failed to detect hardware:\n{str(e)}"
                self.root.after(0, lambda: self._update_hw_info_display(error_msg))

        threading.Thread(target=detect_thread, daemon=True).start()

    def _format_hw_info(self, pc_dict):
        """Format PC info dictionary as readable text."""
        if not pc_dict:
            return "No hardware information available.\nClick 'Detect Hardware' to scan."

        lines = []
        for key, value in pc_dict.items():
            lines.append(f"{key:12}: {value}")

        return "\n".join(lines)

    def _update_hw_info_display(self, text):
        """Update the hardware info text widget."""
        self.hw_info_text.config(state='normal')
        self.hw_info_text.delete(1.0, tk.END)
        self.hw_info_text.insert(tk.END, text)
        self.hw_info_text.config(state='disabled')

    def quit_application(self):
        """Completely quit the application."""
        # Stop any active processes
        if self.is_training:
            self.is_training = False
        if self.is_collecting:
            self.is_collecting = False

        # Stop tray icons
        if self.simple_tray_icon:
            self.simple_tray_icon.stop()
        if self.tray_monitor:
            self.tray_monitor.stop_tray()

        # Destroy the window (will close the app)
        self.root.quit()
        self.root.destroy()

    def save_settings(self):
        """Save current settings to config file."""
        self.config.set('check_interval', self.check_interval_var.get())
        self.config.set('notifications_enabled', self.notifications_var.get())
        self.config.set('threshold_std', self.threshold_var.get())
        self.config.set('model_approach', self.model_approach_var.get())
        self.config.set('last_model_type', self.model_type_var.get())
        self.config.set('pca_type', self.pca_type_var.get())
        self.config.set('pca_components', self.pca_components_var.get())
        self.config.set('multi_variable', self.multi_variable_var.get())
        self.config.set('collection_interval', self.collection_interval_var.get())
        self.config.set('mean_time', self.mean_time_var.get())
        self.config.set('monitor_mean_time', self.monitor_mean_time_var.get())
        self.config.set('monitor_anomaly_window', self.monitor_anomaly_window_var.get())
        self.config.save()
        messagebox.showinfo("Settings", "Settings saved successfully")

    def on_close(self):
        """Handle window close event - ALWAYS minimize to tray, never close."""
        # Always minimize to tray - everything keeps running in background
        self.minimize_to_tray()

        # Show notification if first time
        if not hasattr(self, '_tray_notification_shown'):
            self._tray_notification_shown = True
            # Could show a system notification here if needed

    def run(self):
        """Run the application."""
        self.root.mainloop()


def main():
    app = CPUTempMonitorApp()
    app.run()


if __name__ == "__main__":
    main()
