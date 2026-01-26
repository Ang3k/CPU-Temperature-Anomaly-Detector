"""
System Tray Monitor for CPU Temperature Anomaly Detection.
Provides background monitoring with Windows notifications.
"""

import threading
import time
from collections import deque
from io import BytesIO

from PIL import Image, ImageDraw
from plyer import notification
import pystray

from .cpu_temp_bundled import HardwareMonitor
from .core_regressor import CoreTempRegressor, CoreTempPCA
from .data_extractor import ComputerInfoExtractor


def create_icon(color='green', size=64):
    """Create a simple colored circle icon."""
    image = Image.new('RGBA', (size, size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(image)

    colors = {
        'green': '#22c55e',
        'red': '#ef4444',
        'yellow': '#eab308',
        'gray': '#6b7280'
    }
    fill_color = colors.get(color, colors['gray'])

    # Draw circle
    margin = 4
    draw.ellipse([margin, margin, size - margin, size - margin], fill=fill_color)

    # Draw thermometer shape inside
    center_x, center_y = size // 2, size // 2
    draw.ellipse([center_x - 8, center_y + 8, center_x + 8, center_y + 20], fill='white')
    draw.rectangle([center_x - 4, center_y - 16, center_x + 4, center_y + 12], fill='white')
    draw.ellipse([center_x - 6, center_y - 18, center_x + 6, center_y - 12], fill='white')

    return image


class TrayMonitor:
    """System tray monitor for CPU temperature anomaly detection."""

    def __init__(self, model_path: str = None, check_interval: float = 5.0,
                 threshold_std: float = 1.5, notifications_enabled: bool = True,
                 on_open_gui=None, on_quit_app=None, collect_data: bool = False,
                 mean_time: int = 0, anomaly_window: int = 1):
        self.model_path = model_path
        self.check_interval = check_interval
        self.threshold_std = threshold_std
        self.notifications_enabled = notifications_enabled
        self.on_open_gui = on_open_gui
        self.on_quit_app = on_quit_app
        self.collect_data = collect_data
        self.mean_time = max(0, int(mean_time or 0))
        self.sample_interval = 1.0 if self.mean_time > 0 else self.check_interval
        self.anomaly_window = max(1, int(anomaly_window or 1))

        self.regressor = None
        self.monitor = None
        self.running = False
        self.paused = False
        self.monitor_thread = None
        self.icon = None

        # Status info
        self.current_temp = 0.0
        self.predicted_temp = 0.0
        self.last_diff = 0.0
        self.reconstruction_error = 0.0
        self.anomaly_count = 0
        self.last_anomaly_time = None
        self.low_threshold = 0.0
        self.high_threshold = 0.0

        # Model type detection
        self.is_pca_model = False

        # Background data collection
        self.collected_data = []
        self.time_counter = 0

        # Mean prediction buffering
        self.mean_buffer = []
        self.prediction_counter = 0
        self.anomaly_window_buffer = deque(maxlen=self.anomaly_window)

    def _average_buffer(self, buffer: list) -> dict:
        """Average numeric values in a list of sensor dicts."""
        if not buffer:
            return {}

        sums = {}
        counts = {}
        for sample in buffer:
            for key, value in sample.items():
                if isinstance(value, (int, float)) and value is not None:
                    sums[key] = sums.get(key, 0.0) + float(value)
                    counts[key] = counts.get(key, 0) + 1

        return {key: (sums[key] / counts[key]) for key in sums}

    def _process_prediction(self, data: dict):
        """Run anomaly detection on provided data and update state."""
        if self.is_pca_model:
            # PCA returns: (is_anomaly, reconstruction_error, threshold, actual)
            is_anomaly_raw, recon_error, threshold, actual = self.regressor.detect_anomaly(data)
            self.reconstruction_error = recon_error
            self.last_diff = recon_error  # Use recon error as "diff" for display
            self.predicted_temp = 0.0  # PCA doesn't predict temperature
        else:
            # Regressor returns: (is_anomaly, diff, predicted, actual)
            is_anomaly_raw, diff, predicted, actual = self.regressor.detect_anomaly(data)
            self.last_diff = diff
            self.predicted_temp = predicted

        self.current_temp = actual
        self.prediction_counter += 1

        self.anomaly_window_buffer.append(bool(is_anomaly_raw))
        window_ready = len(self.anomaly_window_buffer) >= self.anomaly_window
        is_anomaly_confirmed = window_ready and all(self.anomaly_window_buffer)

        # Update icon
        self.update_icon(is_anomaly_confirmed)

        if is_anomaly_confirmed:
            self.anomaly_count += 1
            self.last_anomaly_time = time.strftime('%H:%M:%S')
            if self.is_pca_model:
                self.send_notification(
                    'CPU Temperature Anomaly!',
                    f'Temp: {actual:.1f}Â°C\n'
                    f'Reconstruction Error: {recon_error:.4f}\n'
                    f'Threshold: {self.high_threshold:.4f}'
                )
            else:
                self.send_notification(
                    'CPU Temperature Anomaly!',
                    f'Temp: {actual:.1f}Â°C (expected ~{predicted:.1f}Â°C)\n'
                    f'Diff: {diff:+.1f}Â°C'
                )

    def load_model(self, path: str = None):
        """Load the trained model."""
        if path:
            self.model_path = path
        if self.model_path:
            # Check if it's a PCA model or regressor
            loaded_model = None
            try:
                import joblib
                loaded_model = joblib.load(self.model_path)
            except Exception as e:
                print(f"Error loading model: {e}")
                return False

            # Backward compatibility: Add extractor if not present
            if not hasattr(loaded_model, 'extractor') or loaded_model.extractor is None:
                # Create extractor with default parameters
                extractor = ComputerInfoExtractor(
                    scaler_type='standard',
                    use_time_features=True,
                    lag_steps=[1, 2, 3],
                    rolling_windows=[3, 5, 7]
                )
                loaded_model.extractor = extractor

            self.regressor = loaded_model

            # Detect if this is a PCA model
            self.is_pca_model = isinstance(loaded_model, CoreTempPCA)

            # Initialize buffer for real-time detection
            if hasattr(self.regressor, 'init_realtime_buffer'):
                self.regressor.init_realtime_buffer()

            # Get thresholds from model
            if self.regressor.low_threshold is not None:
                self.low_threshold = self.regressor.low_threshold
                self.high_threshold = self.regressor.high_threshold

            return True
        return False

    def send_notification(self, title: str, message: str, timeout: int = 10):
        """Send a Windows notification."""
        if not self.notifications_enabled:
            return

        try:
            notification.notify(
                title=title,
                message=message,
                app_name='CPU Temp Monitor',
                timeout=timeout
            )
        except Exception as e:
            print(f"Notification error: {e}")

    def update_icon(self, is_anomaly: bool):
        """Update the tray icon color based on status."""
        if self.icon:
            color = 'red' if is_anomaly else 'green'
            self.icon.icon = create_icon(color)

    def monitor_loop(self):
        """Main monitoring loop running in background thread."""
        from datetime import datetime
        self.monitor = HardwareMonitor()

        while self.running:
            if self.paused:
                time.sleep(0.5)
                continue

            try:
                # Get current sensor data
                data = self.monitor.get_essential_fast()
                sample_time = datetime.now()

                # Collect data in background if enabled
                if self.collect_data:
                    data_point = data.copy()
                    data_point['timestamp'] = sample_time
                    data_point['time'] = self.time_counter
                    self.collected_data.append(data_point)
                    self.time_counter += 1

                # Detect anomaly using raw or averaged data
                if self.mean_time > 0:
                    self.mean_buffer.append(data)
                    if len(self.mean_buffer) >= self.mean_time:
                        mean_data = self._average_buffer(self.mean_buffer)
                        self.mean_buffer = []
                        self._process_prediction(mean_data)
                else:
                    self._process_prediction(data)
            except Exception as e:
                print(f"Monitor error: {e}")

            time.sleep(self.sample_interval)

        # Cleanup
        if self.monitor:
            self.monitor.close()
            self.monitor = None

    def start_monitoring(self):
        """Start the background monitoring thread."""
        if not self.regressor:
            return False

        self.running = True
        self.paused = False
        self.monitor_thread = threading.Thread(target=self.monitor_loop, daemon=True)
        self.monitor_thread.start()
        return True

    def stop_monitoring(self):
        """Stop the monitoring thread."""
        self.running = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=2)
            self.monitor_thread = None

    def pause_monitoring(self):
        """Pause/resume monitoring."""
        self.paused = not self.paused
        if self.paused:
            self.update_icon_to_gray()

    def update_icon_to_gray(self):
        """Set icon to gray when paused."""
        if self.icon:
            self.icon.icon = create_icon('gray')

    def get_status_text(self):
        """Get current status as text."""
        status = "Paused" if self.paused else "Monitoring"
        model_type = "PCA" if self.is_pca_model else "Regressor"

        if self.is_pca_model:
            # PCA model status
            status_text = (f"Status: {status} ({model_type})\n"
                           f"CPU Temp: {self.current_temp:.1f}°C\n"
                           f"Recon Error: {self.reconstruction_error:.4f}\n"
                           f"Error Threshold: [{self.low_threshold:.4f}, {self.high_threshold:.4f}]\n"
                           f"Anomalies: {self.anomaly_count}")
        else:
            # Regressor model status
            # Calculate absolute temperature thresholds
            low_temp = self.predicted_temp + self.low_threshold if self.predicted_temp else 0.0
            high_temp = self.predicted_temp + self.high_threshold if self.predicted_temp else 0.0

            status_text = (f"Status: {status} ({model_type})\n"
                           f"CPU Temp: {self.current_temp:.1f}°C\n"
                           f"Predicted: {self.predicted_temp:.1f}°C\n"
                           f"Diff: {self.last_diff:+.1f}°C\n"
                           f"Temp Range: [{low_temp:.1f}, {high_temp:.1f}]°C\n"
                           f"Diff Range: [{self.low_threshold:.1f}, {self.high_threshold:+.1f}]°C\n"
                           f"Anomalies: {self.anomaly_count}")

        if self.collect_data:
            status_text += f"\nCollected Data: {len(self.collected_data)} samples"

        return status_text

    def save_collected_data(self, filepath: str):
        """Save collected data to CSV file."""
        import pandas as pd
        if not self.collected_data:
            return False

        df = pd.DataFrame(self.collected_data)
        df.to_csv(filepath, index=False)
        return True

    def clear_collected_data(self):
        """Clear collected data and reset counter."""
        self.collected_data = []
        self.time_counter = 0

    def toggle_data_collection(self):
        """Toggle background data collection on/off."""
        self.collect_data = not self.collect_data
        return self.collect_data

    def create_menu(self):
        """Create the system tray menu."""
        def on_open(icon, item):
            if self.on_open_gui:
                self.on_open_gui()

        def on_pause(icon, item):
            self.pause_monitoring()
            # Update menu item text
            self.icon.menu = self.create_menu()

        def on_toggle_collection(icon, item):
            self.toggle_data_collection()
            # Update menu item text
            self.icon.menu = self.create_menu()

        def on_quit(icon, item):
            self.stop_monitoring()
            icon.stop()
            if self.on_quit_app:
                self.on_quit_app()

        pause_text = "Resume" if self.paused else "Pause"
        collection_text = "Stop Data Collection" if self.collect_data else "Start Data Collection"

        # Build status menu based on model type
        if self.is_pca_model:
            # PCA model menu
            status_items = [
                pystray.MenuItem(lambda text: f"Model: PCA", None, enabled=False),
                pystray.MenuItem(lambda text: f"Temp: {self.current_temp:.1f}°C", None, enabled=False),
                pystray.MenuItem(lambda text: f"Recon Error: {self.reconstruction_error:.4f}", None, enabled=False),
                pystray.Menu.SEPARATOR,
                pystray.MenuItem(lambda text: f"Error Threshold: [{(self.low_threshold or 0):.4f}, {(self.high_threshold or 0):.4f}]", None, enabled=False),
                pystray.Menu.SEPARATOR,
                pystray.MenuItem(lambda text: f"Anomalies: {self.anomaly_count}", None, enabled=False),
            ]
        else:
            # Regressor model menu
            status_items = [
                pystray.MenuItem(lambda text: f"Model: Regressor", None, enabled=False),
                pystray.MenuItem(lambda text: f"Temp: {self.current_temp:.1f}°C", None, enabled=False),
                pystray.MenuItem(lambda text: f"Predicted: {self.predicted_temp:.1f}°C", None, enabled=False),
                pystray.MenuItem(lambda text: f"Diff: {self.last_diff:+.1f}°C", None, enabled=False),
                pystray.Menu.SEPARATOR,
                pystray.MenuItem(lambda text: f"Range: [{(self.predicted_temp or 0) + (self.low_threshold or 0):.1f}, {(self.predicted_temp or 0) + (self.high_threshold or 0):.1f}]°C", None, enabled=False),
                pystray.MenuItem(lambda text: f"Diff Range: [{(self.low_threshold or 0):.1f}, {(self.high_threshold or 0):+.1f}]°C", None, enabled=False),
                pystray.Menu.SEPARATOR,
                pystray.MenuItem(lambda text: f"Anomalies: {self.anomaly_count}", None, enabled=False),
            ]

        # Add data collection info if enabled
        if self.collect_data:
            status_items.append(pystray.MenuItem(lambda text: f"Collected: {len(self.collected_data)} samples", None, enabled=False))

        status_menu = pystray.Menu(*status_items)

        return pystray.Menu(
            pystray.MenuItem("Open GUI", on_open, default=True),
            pystray.MenuItem(pause_text, on_pause),
            pystray.MenuItem(collection_text, on_toggle_collection),
            pystray.Menu.SEPARATOR,
            pystray.MenuItem("Status", status_menu),
            pystray.Menu.SEPARATOR,
            pystray.MenuItem("Quit", on_quit)
        )

    def run_tray(self):
        """Run the system tray icon (blocking)."""
        self.icon = pystray.Icon(
            "cpu_temp_monitor",
            create_icon('gray'),
            "CPU Temp Monitor",
            menu=self.create_menu()
        )
        self.icon.run()

    def run_tray_detached(self):
        """Run the system tray icon in a separate thread."""
        tray_thread = threading.Thread(target=self.run_tray, daemon=True)
        tray_thread.start()
        return tray_thread

    def stop_tray(self):
        """Stop the tray icon."""
        if self.icon:
            self.icon.stop()


def run_monitor_standalone(model_path: str, check_interval: float = 5.0,
                           threshold_std: float = 1.5):
    """Run the monitor as a standalone application."""
    monitor = TrayMonitor(
        model_path=model_path,
        check_interval=check_interval,
        threshold_std=threshold_std,
        notifications_enabled=True
    )

    if not monitor.load_model():
        print(f"Failed to load model from {model_path}")
        return

    print(f"Starting CPU Temperature Monitor...")
    print(f"Model: {model_path}")
    print(f"Check interval: {check_interval}s")
    print(f"Threshold: {threshold_std} std")

    monitor.start_monitoring()
    monitor.run_tray()  # Blocking


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        model_path = sys.argv[1]
    else:
        model_path = "models/cpu_temp_model_lightgbm.joblib"

    run_monitor_standalone(model_path)
