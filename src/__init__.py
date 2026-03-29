"""
CPU Temperature Detector Package
Anomaly detection for CPU temperature using machine learning.
"""

from .cpu_temp_bundled import HardwareMonitor
from .core_regressor import CoreTempRegressor, CoreTempPCA
from .conv_autoencoder import ConvAutoencoder
from .data_extractor import ComputerInfoExtractor
from .tray_monitor import TrayMonitor

__all__ = ['HardwareMonitor', 'CoreTempRegressor', 'CoreTempPCA', 'ConvAutoencoder', 'ComputerInfoExtractor', 'TrayMonitor']
__version__ = '2.0.0'
