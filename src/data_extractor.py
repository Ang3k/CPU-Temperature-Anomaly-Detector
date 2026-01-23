# Standard library imports
from datetime import datetime
from time import sleep

# Third-party imports
import pandas as pd
import plotly.express as px

# Local imports
from src.cpu_temp_bundled import HardwareMonitor

# Background color white
px.defaults.template = "plotly_white"


class ComputerInfoExtractor:
    """
    Handles data extraction and preprocessing for CPU temperature monitoring.
    Separates data collection concerns from model training.
    """

    def __init__(self, scaler_type='standard', use_time_features=True, lag_steps=[1,2,3], rolling_windows=[3,5,7]):
        self.data = pd.DataFrame()
        self.pc_info = {}
        self.scaler_type = scaler_type
        self.use_time_features = use_time_features
        self.lag_steps = lag_steps
        self.rolling_windows = rolling_windows

    def extract_CPU_data(self, iterations=100, interval=5, mean_time=None, csv=False, progress_callback=None, should_stop_callback=None):
        """
        Extract CPU data for training.

        Args:
            iterations: Number of data points to collect
            interval: Time between collections (seconds)
            mean_time: Optional time window (seconds) for resampling and averaging data. None = no resampling
            csv: If True, load from '../data/new_latest.csv'
            progress_callback: Optional callback function(current, total) for progress updates
            should_stop_callback: Optional callback function() that returns True to stop collection
        """
        if csv:
            self.data = pd.read_csv('../data/new_latest.csv')
            if 'timestamp' in self.data.columns:
                self.data['timestamp'] = pd.to_datetime(self.data['timestamp'])
        else:
            data_list = []
            with HardwareMonitor() as monitor:
                for t in range(iterations):
                    # Check if should stop
                    if should_stop_callback and should_stop_callback():
                        break

                    row = monitor.get_essential_fast()
                    row['timestamp'] = datetime.now()
                    row['time'] = t
                    data_list.append(row)

                    if progress_callback:
                        progress_callback(t + 1, iterations)

                    if interval > 0:
                        sleep(interval)

            self.data = pd.DataFrame(data_list)

        if mean_time not in (None, 0):
            self.data = self.data.resample(f"{mean_time}s", on='timestamp').mean().reset_index()
            # Recreate sequential time column after resampling
            if 'time' in self.data.columns:
                self.data['time'] = range(len(self.data))

        return self.data

    def extract_PC_info(self):
        """Extract PC hardware information."""
        with HardwareMonitor() as monitor:
            info = monitor.get_hardware_info()
            pc_dict = {}
            # Access information
            if info['cpu']:
                pc_dict['CPU'] = info['cpu']['name']

            if info['gpu']:
                pc_dict['GPU'] = info['gpu']['name']

            if info['motherboard']:
                pc_dict['Motherboard'] = info['motherboard']['name']

            if info['ram']:
                pc_dict['RAM'] = info['ram']['name']

            for disk in info['storage']:
                pc_dict["Storage"] = disk['name']

            self.pc_info = pc_dict
            return pc_dict

    def plot_CPU_data(self):
        """Plot CPU data with Plotly."""
        if self.pc_info == {}:
            self.extract_PC_info()
        px.line(self.data, x='timestamp', y=['cpu_temp','cpu_load','cpu_power','cpu_clock','cpu_volt','gpu_temp','gpu_load','gpu_power','mb_temp','ram_load','fan_rpm'], title=f'{self.pc_info.get("CPU")} Temperature').show()

    def create_time_features_on_df(self):
        """Create lag, rolling, and diff features for better predictions."""

        if self.data.empty:
            self.extract_CPU_data()

        exclude_cols = ['cpu_temp', 'time', 'fan_rpm', 'timestamp']
        feature_cols = [col for col in self.data.columns if col not in exclude_cols and self.data[col].dtype in ['float64', 'int64', 'float32', 'int32']]

        new_features = {}

        for col in feature_cols:
            # Lag features
            for lag in self.lag_steps:
                new_features[f'{col}_lag_{lag}'] = self.data[col].shift(lag)

            # Rolling statistics
            for window in self.rolling_windows:
                new_features[f'{col}_rolling_mean_{window}'] = self.data[col].rolling(window=window).mean()
                new_features[f'{col}_rolling_std_{window}'] = self.data[col].rolling(window=window).std()

            # Rate of change
            new_features[f'{col}_diff_1'] = self.data[col].diff(1)

        if new_features:
            new_cols_df = pd.DataFrame(new_features, index=self.data.index)
            df = pd.concat([self.data, new_cols_df], axis=1)
        else:
            df = self.data.copy()

        return df

    def extract_data_pipeline(self, csv=False, iterations=100, interval=5, mean_time=None, progress_callback=None, should_stop_callback=None):
        """Full data extraction and preprocessing pipeline."""
        if self.data.empty:
            self.extract_CPU_data(iterations=iterations, interval=interval, mean_time=mean_time, csv=csv, progress_callback=progress_callback, should_stop_callback=should_stop_callback)

        if self.use_time_features:
            df = self.create_time_features_on_df()
        else:
            df = self.data.copy()

        # Fill missing values but DO NOT scale here to avoid data leakage
        # Scaling will be done in CoreTempRegressor.fit_predict() after train/test split
        self.processed_data = df.ffill().bfill().reset_index(drop=True)
        return self.processed_data
