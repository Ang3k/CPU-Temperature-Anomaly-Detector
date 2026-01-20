"""
CPU Temperature Detector Package
Anomaly detection for CPU temperature using machine learning.
"""

from .cpu_temp_bundled import HardwareMonitor
from .core_regressor import CoreTempRegressor
from .tray_monitor import TrayMonitor

__all__ = ['HardwareMonitor', 'CoreTempRegressor', 'TrayMonitor']
__version__ = '1.0.0'
