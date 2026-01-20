"""
System Tray Monitor for CPU Temperature Anomaly Detection.
Provides background monitoring with Windows notifications.
"""

import threading
import time
from io import BytesIO

from PIL import Image, ImageDraw
from plyer import notification
import pystray

from .cpu_temp_bundled import HardwareMonitor
from .core_regressor import CoreTempRegressor


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
                 on_open_gui=None):
        self.model_path = model_path
        self.check_interval = check_interval
        self.threshold_std = threshold_std
        self.notifications_enabled = notifications_enabled
        self.on_open_gui = on_open_gui

        self.regressor = None
        self.monitor = None
        self.running = False
        self.paused = False
        self.monitor_thread = None
        self.icon = None

        # Status info
        self.current_temp = 0.0
        self.last_diff = 0.0
        self.anomaly_count = 0
        self.last_anomaly_time = None

    def load_model(self, path: str = None):
        """Load the trained model."""
        if path:
            self.model_path = path
        if self.model_path:
            self.regressor = CoreTempRegressor.load_model(self.model_path)
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
        self.monitor = HardwareMonitor()

        while self.running:
            if self.paused:
                time.sleep(0.5)
                continue

            try:
                # Get current sensor data
                data = self.monitor.get_essential_fast()
                self.current_temp = data.get('cpu_temp', 0.0)

                # Detect anomaly
                is_anomaly, diff, predicted, actual = self.regressor.detect_anomaly(data)
                self.last_diff = diff

                # Update icon
                self.update_icon(is_anomaly)

                if is_anomaly:
                    self.anomaly_count += 1
                    self.last_anomaly_time = time.strftime('%H:%M:%S')
                    self.send_notification(
                        'CPU Temperature Anomaly!',
                        f'Temp: {actual:.1f}°C (expected ~{predicted:.1f}°C)\n'
                        f'Diff: {diff:+.1f}°C'
                    )

            except Exception as e:
                print(f"Monitor error: {e}")

            time.sleep(self.check_interval)

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
        return (f"Status: {status}\n"
                f"CPU Temp: {self.current_temp:.1f}°C\n"
                f"Last Diff: {self.last_diff:+.1f}°C\n"
                f"Anomalies: {self.anomaly_count}")

    def create_menu(self):
        """Create the system tray menu."""
        def on_open(icon, item):
            if self.on_open_gui:
                self.on_open_gui()

        def on_pause(icon, item):
            self.pause_monitoring()
            # Update menu item text
            self.icon.menu = self.create_menu()

        def on_quit(icon, item):
            self.stop_monitoring()
            icon.stop()

        pause_text = "Resume" if self.paused else "Pause"

        return pystray.Menu(
            pystray.MenuItem("Open GUI", on_open, default=True),
            pystray.MenuItem(pause_text, on_pause),
            pystray.Menu.SEPARATOR,
            pystray.MenuItem(
                "Status",
                pystray.Menu(
                    pystray.MenuItem(lambda text: f"Temp: {self.current_temp:.1f}°C", None, enabled=False),
                    pystray.MenuItem(lambda text: f"Diff: {self.last_diff:+.1f}°C", None, enabled=False),
                    pystray.MenuItem(lambda text: f"Anomalies: {self.anomaly_count}", None, enabled=False),
                )
            ),
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
