"""
CPU Temperature Monitor - Desktop Application
GUI for training models and monitoring CPU temperature anomalies.
"""

import os
import sys
import threading
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from pathlib import Path

import yaml

from src.core_regressor import CoreTempRegressor
from src.tray_monitor import TrayMonitor
from src.cpu_temp_bundled import HardwareMonitor


class ConfigManager:
    """Manage application configuration."""

    DEFAULT_CONFIG = {
        'model_path': 'models/cpu_temp_model_lightgbm.joblib',
        'threshold_std': 1.5,
        'check_interval': 5,
        'notifications_enabled': True,
        'minimize_to_tray': True,
        'last_model_type': 'lightgbm',
        'last_scaler': 'standard',
        'training_iterations': 1000,
        'training_interval': 1
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
        self.root.geometry("500x450")
        self.root.resizable(False, False)

        # Config
        self.config = ConfigManager()

        # State
        self.regressor = None
        self.tray_monitor = None
        self.training_thread = None
        self.is_training = False
        self.is_monitoring = False

        # Setup UI
        self.setup_ui()

        # Handle window close
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

    def setup_ui(self):
        """Setup the main UI with tabs."""
        # Create notebook (tabs)
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill='both', expand=True, padx=10, pady=10)

        # Create tabs
        self.train_tab = ttk.Frame(self.notebook)
        self.monitor_tab = ttk.Frame(self.notebook)
        self.settings_tab = ttk.Frame(self.notebook)

        self.notebook.add(self.train_tab, text='  Train  ')
        self.notebook.add(self.monitor_tab, text='  Monitor  ')
        self.notebook.add(self.settings_tab, text='  Settings  ')

        self.setup_train_tab()
        self.setup_monitor_tab()
        self.setup_settings_tab()

    def setup_train_tab(self):
        """Setup the training tab."""
        frame = self.train_tab

        # Model selection
        model_frame = ttk.LabelFrame(frame, text="Model Configuration", padding=10)
        model_frame.pack(fill='x', padx=10, pady=5)

        ttk.Label(model_frame, text="Model Type:").grid(row=0, column=0, sticky='w', pady=2)
        self.model_type_var = tk.StringVar(value=self.config.get('last_model_type', 'lightgbm'))
        model_combo = ttk.Combobox(model_frame, textvariable=self.model_type_var,
                                   values=['linear', 'xgb', 'lightgbm'], state='readonly', width=15)
        model_combo.grid(row=0, column=1, pady=2, padx=5)

        ttk.Label(model_frame, text="Scaler:").grid(row=1, column=0, sticky='w', pady=2)
        self.scaler_var = tk.StringVar(value=self.config.get('last_scaler', 'standard'))
        scaler_combo = ttk.Combobox(model_frame, textvariable=self.scaler_var,
                                    values=['standard', 'minmax', 'robust'], state='readonly', width=15)
        scaler_combo.grid(row=1, column=1, pady=2, padx=5)

        # Training parameters
        param_frame = ttk.LabelFrame(frame, text="Training Parameters", padding=10)
        param_frame.pack(fill='x', padx=10, pady=5)

        ttk.Label(param_frame, text="Iterations:").grid(row=0, column=0, sticky='w', pady=2)
        self.iterations_var = tk.IntVar(value=self.config.get('training_iterations', 1000))
        iterations_spin = ttk.Spinbox(param_frame, from_=100, to=100000, increment=100,
                                      textvariable=self.iterations_var, width=10)
        iterations_spin.grid(row=0, column=1, pady=2, padx=5)

        ttk.Label(param_frame, text="Interval (s):").grid(row=1, column=0, sticky='w', pady=2)
        self.interval_var = tk.DoubleVar(value=self.config.get('training_interval', 1))
        interval_spin = ttk.Spinbox(param_frame, from_=0, to=60, increment=0.5,
                                    textvariable=self.interval_var, width=10)
        interval_spin.grid(row=1, column=1, pady=2, padx=5)

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

        self.train_btn = ttk.Button(btn_frame, text="Start Training", command=self.start_training)
        self.train_btn.pack(side='left', padx=5)

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
        frame = self.monitor_tab

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

        # Status
        status_frame = ttk.LabelFrame(frame, text="Status", padding=10)
        status_frame.pack(fill='x', padx=10, pady=5)

        self.status_text = tk.Text(status_frame, height=8, width=50, state='disabled')
        self.status_text.pack(fill='x', pady=5)

        # Buttons
        btn_frame = ttk.Frame(frame)
        btn_frame.pack(fill='x', padx=10, pady=10)

        self.monitor_btn = ttk.Button(btn_frame, text="Start Monitoring", command=self.start_monitoring)
        self.monitor_btn.pack(side='left', padx=5)

        self.stop_monitor_btn = ttk.Button(btn_frame, text="Stop", command=self.stop_monitoring, state='disabled')
        self.stop_monitor_btn.pack(side='left', padx=5)

        self.minimize_btn = ttk.Button(btn_frame, text="Minimize to Tray", command=self.minimize_to_tray, state='disabled')
        self.minimize_btn.pack(side='left', padx=5)

    def setup_settings_tab(self):
        """Setup the settings tab."""
        frame = self.settings_tab

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

        self.minimize_var = tk.BooleanVar(value=self.config.get('minimize_to_tray', True))
        minimize_check = ttk.Checkbutton(general_frame, text="Minimize to Tray on Close",
                                         variable=self.minimize_var)
        minimize_check.grid(row=2, column=0, columnspan=2, sticky='w', pady=5)

        # Save button
        save_frame = ttk.Frame(frame)
        save_frame.pack(fill='x', padx=10, pady=20)

        save_btn = ttk.Button(save_frame, text="Save Settings", command=self.save_settings)
        save_btn.pack()

        # Info
        info_frame = ttk.LabelFrame(frame, text="Info", padding=10)
        info_frame.pack(fill='x', padx=10, pady=5)

        info_text = ("CPU Temperature Monitor\n\n"
                     "1. Train a model using your CPU data\n"
                     "2. Save the trained model\n"
                     "3. Start monitoring to detect anomalies\n"
                     "4. Minimize to system tray for background monitoring")
        ttk.Label(info_frame, text=info_text, justify='left').pack()

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

    def start_training(self):
        """Start the training process in a background thread."""
        if self.is_training:
            return

        self.is_training = True
        self.train_btn.config(state='disabled')
        self.stop_train_btn.config(state='normal')
        self.save_btn.config(state='disabled')
        self.progress_var.set(0)

        # Create regressor
        self.regressor = CoreTempRegressor()
        self.regressor.configure_model(
            model=self.model_type_var.get(),
            scaler=self.scaler_var.get(),
            use_time_features=True
        )

        def progress_callback(current, total):
            progress = (current / total) * 100
            self.root.after(0, lambda: self.progress_var.set(progress))
            self.root.after(0, lambda: self.progress_label.config(
                text=f"Collecting data: {current}/{total}"
            ))

        def train_thread():
            try:
                # Collect data
                self.regressor.extract_CPU_data(
                    iterations=self.iterations_var.get(),
                    interval=self.interval_var.get(),
                    progress_callback=progress_callback,
                    should_stop_callback=lambda: not self.is_training
                )

                if not self.is_training:  # Stopped
                    return

                # Train
                self.root.after(0, lambda: self.progress_label.config(text="Training model..."))
                self.regressor.fit_predict(train_size=0.8, threshold_std=self.threshold_var.get())

                # Show metrics
                metrics = self.regressor.evaluate_metrics()
                metrics_text = (f"RMSE: {metrics['rmse']:.2f}\n"
                               f"MAE: {metrics['mae']:.2f}\n"
                               f"R2: {metrics['r2']:.3f}\n"
                               f"MAPE: {metrics['mape']:.1f}%")

                self.root.after(0, lambda: self.metrics_label.config(text=metrics_text))
                self.root.after(0, lambda: self.progress_label.config(text="Training complete!"))
                self.root.after(0, lambda: self.save_btn.config(state='normal'))

            except Exception as e:
                self.root.after(0, lambda: messagebox.showerror("Error", f"Training failed: {e}"))
                self.root.after(0, lambda: self.progress_label.config(text="Training failed"))

            finally:
                self.is_training = False
                self.root.after(0, lambda: self.train_btn.config(state='normal'))
                self.root.after(0, lambda: self.stop_train_btn.config(state='disabled'))

        self.training_thread = threading.Thread(target=train_thread, daemon=True)
        self.training_thread.start()

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

        # Generate filename
        model_type = self.model_type_var.get()
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
            self.config.set('last_model_type', model_type)
            self.config.set('last_scaler', self.scaler_var.get())
            self.config.set('model_path', path)
            self.config.save()

    def start_monitoring(self):
        """Start the monitoring process."""
        model_path = self.model_path_var.get()
        if not model_path or not os.path.exists(model_path):
            messagebox.showwarning("Warning", "Please select a valid model file")
            return

        try:
            self.tray_monitor = TrayMonitor(
                model_path=model_path,
                check_interval=self.check_interval_var.get(),
                threshold_std=self.threshold_var.get(),
                notifications_enabled=self.notifications_var.get(),
                on_open_gui=self.show_window
            )
            self.tray_monitor.load_model()
            self.tray_monitor.start_monitoring()

            self.is_monitoring = True
            self.monitor_btn.config(state='disabled')
            self.stop_monitor_btn.config(state='normal')
            self.minimize_btn.config(state='normal')

            # Update status periodically
            self.update_status()

            # Start tray in background
            self.tray_monitor.run_tray_detached()

        except Exception as e:
            messagebox.showerror("Error", f"Failed to start monitoring: {e}")

    def update_status(self):
        """Update the status display."""
        if not self.is_monitoring or not self.tray_monitor:
            return

        status = self.tray_monitor.get_status_text()
        self.status_text.config(state='normal')
        self.status_text.delete(1.0, tk.END)
        self.status_text.insert(tk.END, status)
        self.status_text.config(state='disabled')

        # Schedule next update
        self.root.after(1000, self.update_status)

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

        self.status_text.config(state='normal')
        self.status_text.delete(1.0, tk.END)
        self.status_text.insert(tk.END, "Monitoring stopped")
        self.status_text.config(state='disabled')

    def minimize_to_tray(self):
        """Minimize the window to system tray."""
        self.root.withdraw()

    def show_window(self):
        """Show the window from tray."""
        self.root.deiconify()
        self.root.lift()
        self.root.focus_force()

    def save_settings(self):
        """Save current settings to config file."""
        self.config.set('check_interval', self.check_interval_var.get())
        self.config.set('notifications_enabled', self.notifications_var.get())
        self.config.set('minimize_to_tray', self.minimize_var.get())
        self.config.set('threshold_std', self.threshold_var.get())
        self.config.set('training_iterations', self.iterations_var.get())
        self.config.set('training_interval', self.interval_var.get())
        self.config.save()
        messagebox.showinfo("Settings", "Settings saved successfully")

    def on_close(self):
        """Handle window close event."""
        if self.minimize_var.get() and self.is_monitoring:
            self.minimize_to_tray()
        else:
            if self.is_monitoring:
                self.stop_monitoring()
            if self.is_training:
                self.is_training = False
            self.root.destroy()

    def run(self):
        """Run the application."""
        self.root.mainloop()


def main():
    app = CPUTempMonitorApp()
    app.run()


if __name__ == "__main__":
    main()
