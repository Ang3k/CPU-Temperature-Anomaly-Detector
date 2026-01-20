# Standard library imports
from collections import deque
from datetime import datetime
from time import sleep

# Third-party imports
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from xgboost import XGBRegressor
import plotly.express as px
import joblib

# Local imports
from .cpu_temp_bundled import HardwareMonitor


class CoreTempRegressor:
    """
    CPU Temperature anomaly detection using regression models.
    Can be used for training, saving/loading models, and real-time anomaly detection.
    """

    def __init__(self):
        self.scaler = None
        self.model = None
        self.data = None
        self.predict_data = None

        # Store for predictions
        self.x_test = None
        self.y_test = None
        self.y_pred = None
        self.test_data = None

        # Store feature engineering parameters
        self.lag_steps = None
        self.rolling_windows = None
        self.use_time_features = None

        # Store model name
        self.model_name = None

        # Anomaly detection thresholds
        self.low_threshold = None
        self.high_threshold = None
        self.threshold_std = 1.5

        # Real-time monitoring buffer
        self.data_buffer = None
        self.time_counter = 0

        # Feature columns (set after training)
        self.feature_columns = None

    def configure_model(self, model='linear', scaler='standard', use_time_features=True, lag_steps=[1, 2, 3], rolling_windows=[3, 5, 7]):
        """Configure the model and scaler to use."""
        scaler_dict = {
            'standard': StandardScaler(),
            'minmax': MinMaxScaler(),
            'robust': RobustScaler()
        }

        model_dict = {
            'xgb': XGBRegressor(
                n_estimators=100,
                max_depth=3,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                min_child_weight=3,
                reg_alpha=0.1,
                reg_lambda=1.0,
                random_state=42
            ),
            'linear': Ridge(
                alpha=1.0,
                solver='auto',
                random_state=42
            ),
            'lightgbm': LGBMRegressor(
                n_estimators=100,
                max_depth=3,
                learning_rate=0.1,
                num_leaves=7,
                subsample=0.8,
                colsample_bytree=0.8,
                min_child_samples=20,
                reg_alpha=0.1,
                reg_lambda=1.0,
                random_state=42,
                verbose=-1
            )
        }

        self.model = model_dict.get(model)
        self.scaler = scaler_dict.get(scaler)
        self.model_name = model
        self.lag_steps = lag_steps
        self.rolling_windows = rolling_windows
        self.use_time_features = use_time_features

    def extract_CPU_data(self, iterations=100, interval=5, csv=False, progress_callback=None, should_stop_callback=None):
        """
        Extract CPU data for training.

        Args:
            iterations: Number of data points to collect
            interval: Time between collections (seconds)
            csv: If True, load from data.csv instead
            progress_callback: Optional callback function(current, total) for progress updates
            should_stop_callback: Optional callback that returns True if should stop collection
        """
        if csv:
            self.data = pd.read_csv('data.csv')
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
        return self.data
    
    def plot_CPU_data(self):
        px.line(self.data, x='timestamp', y=['cpu_temp','cpu_load','cpu_power','cpu_clock','cpu_volt','gpu_temp','gpu_load','gpu_power','mb_temp','ram_load','fan_rpm'], title='CPU Temperature').show()

    def create_time_features_on_df(self, df, lag_steps, rolling_windows):
        """Create lag, rolling, and diff features for better predictions."""
        exclude_cols = ['cpu_temp', 'time', 'fan_rpm', 'timestamp']
        feature_cols = [col for col in df.columns if col not in exclude_cols and df[col].dtype in ['float64', 'int64', 'float32', 'int32']]

        new_features = {}

        for col in feature_cols:
            # Lag features
            for lag in lag_steps:
                new_features[f'{col}_lag_{lag}'] = df[col].shift(lag)

            # Rolling statistics
            for window in rolling_windows:
                new_features[f'{col}_rolling_mean_{window}'] = df[col].rolling(window=window).mean()
                new_features[f'{col}_rolling_std_{window}'] = df[col].rolling(window=window).std()

            # Rate of change
            new_features[f'{col}_diff_1'] = df[col].diff(1)

        if new_features:
            new_cols_df = pd.DataFrame(new_features, index=df.index)
            df = pd.concat([df, new_cols_df], axis=1)

        return df

    def fit_predict(self, train_size=0.8, threshold_std=1.5):
        """
        Train the model and make predictions on test set.
        Also calculates anomaly thresholds.
        """
        self.threshold_std = threshold_std

        # Split the raw data FIRST (before feature engineering)
        split_idx = int(len(self.data) * train_size)
        train_data = self.data[:split_idx].copy()
        test_data = self.data[split_idx:].copy()

        # Apply time features SEPARATELY to train and test
        if self.use_time_features:
            train_data = self.create_time_features_on_df(train_data, self.lag_steps, self.rolling_windows)
            test_data = self.create_time_features_on_df(test_data, self.lag_steps, self.rolling_windows)

            train_data = train_data.dropna()
            test_data = test_data.dropna()

        # Prepare X and y
        x_train = train_data.drop(['cpu_temp', 'time', 'fan_rpm', 'timestamp'], axis=1)
        y_train = train_data['cpu_temp']

        self.x_test = test_data.drop(['cpu_temp', 'time', 'fan_rpm', 'timestamp'], axis=1)
        self.y_test = test_data['cpu_temp']

        # Store feature columns for later use
        self.feature_columns = list(x_train.columns)

        # Fit scaler ONLY on training data
        x_train_scaled = self.scaler.fit_transform(x_train)
        x_test_scaled = self.scaler.transform(self.x_test)

        # Train model
        self.model.fit(x_train_scaled, y_train)

        # Make predictions
        self.y_pred = self.model.predict(x_test_scaled)

        # Store test_data for plotting
        self.test_data = test_data

        # Calculate anomaly thresholds
        diff = self.y_test.values - self.y_pred
        mean_diff = np.mean(diff)
        std_diff = np.std(diff)
        self.low_threshold = mean_diff - threshold_std * std_diff
        self.high_threshold = mean_diff + threshold_std * std_diff

    def predict(self, X):
        """Make predictions on new data."""
        X_scaled = self.scaler.transform(X)
        self.predict_data = self.model.predict(X_scaled)
        return self.predict_data

    def get_thresholds(self):
        """Return current anomaly thresholds."""
        return {
            'low': self.low_threshold,
            'high': self.high_threshold,
            'std_multiplier': self.threshold_std
        }

    def init_realtime_buffer(self):
        """Initialize the buffer for real-time anomaly detection."""
        max_window = max(self.rolling_windows) if self.rolling_windows else 7
        max_lag = max(self.lag_steps) if self.lag_steps else 3
        buffer_size = max(max_window, max_lag) + 5  # Extra buffer
        self.data_buffer = deque(maxlen=buffer_size)
        self.time_counter = 0

    def detect_anomaly(self, current_data: dict) -> tuple:
        """
        Detect if current data point is an anomaly.

        Args:
            current_data: Dict with sensor readings from HardwareMonitor.get_essential_fast()

        Returns:
            tuple: (is_anomaly: bool, diff: float, predicted: float, actual: float)
        """
        if self.data_buffer is None:
            self.init_realtime_buffer()

        # Add timestamp and time
        current_data = current_data.copy()
        current_data['timestamp'] = datetime.now()
        current_data['time'] = self.time_counter
        self.time_counter += 1

        # Add to buffer
        self.data_buffer.append(current_data)

        # Need enough history for features
        min_required = max(max(self.rolling_windows), max(self.lag_steps)) + 1
        if len(self.data_buffer) < min_required:
            return False, 0.0, 0.0, current_data.get('cpu_temp', 0.0)

        # Create DataFrame from buffer
        df = pd.DataFrame(list(self.data_buffer))

        # Apply time features
        if self.use_time_features:
            df = self.create_time_features_on_df(df, self.lag_steps, self.rolling_windows)
            df = df.dropna()

        if len(df) == 0:
            return False, 0.0, 0.0, current_data.get('cpu_temp', 0.0)

        # Get last row for prediction
        last_row = df.iloc[[-1]]

        # Prepare features
        X = last_row.drop(['cpu_temp', 'time', 'fan_rpm', 'timestamp'], axis=1, errors='ignore')

        # Ensure columns match training
        if self.feature_columns:
            missing_cols = set(self.feature_columns) - set(X.columns)
            for col in missing_cols:
                X[col] = 0
            X = X[self.feature_columns]

        # Predict
        predicted = self.predict(X)[0]
        actual = current_data.get('cpu_temp', 0.0)
        diff = actual - predicted

        # Check anomaly
        is_anomaly = diff < self.low_threshold or diff > self.high_threshold

        return is_anomaly, diff, predicted, actual

    def evaluate_metrics(self):
        """Calculate and return evaluation metrics."""
        rmse = np.sqrt(mean_squared_error(self.y_test, self.y_pred))
        mape = np.mean(np.abs((self.y_test - self.y_pred) / self.y_test)) * 100
        r2 = r2_score(self.y_test, self.y_pred)
        mae = mean_absolute_error(self.y_test, self.y_pred)

        return {
            'rmse': rmse,
            'mape': mape,
            'r2': r2,
            'mae': mae
        }

    def save_model(self, path: str):
        """Save the trained model to file."""
        joblib.dump(self, path)

    @staticmethod
    def load_model(path: str) -> 'CoreTempRegressor':
        """Load a trained model from file."""
        model = joblib.load(path)
        # Initialize buffer for real-time detection
        model.init_realtime_buffer()
        return model
