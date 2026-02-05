# Standard library imports
from collections import deque
from datetime import datetime

# Third-party imports
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.decomposition import PCA, KernelPCA
from xgboost import XGBRegressor
import plotly.express as px
import joblib


class CoreTempRegressor:
    """
    CPU Temperature anomaly detection using regression models.
    Receives pre-processed data from ComputerInfoExtractor.
    Can be used for training, saving/loading models, and real-time anomaly detection.

    IMPORTANT: Scaling is performed AFTER train/test split to prevent data leakage.
    """

    PLOT_STYLE = {
        'line_width': 3,
        'marker_size': 8,
        'threshold_line_width': 5,
        'threshold_opacity': 0.5,
        'error_color': 'orange',
        'lower_threshold_color': 'green',
        'upper_threshold_color': 'red',
        'line_dash': 'dash'
    }

    def __init__(self, extractor=None):
        self.extractor = extractor
        self.scaler = None
        self.model = None
        self.data = None
        self.predict_data = None

        # Store for predictions
        self.x_train = None
        self.y_train = None
        self.x_test = None
        self.y_test = None
        self.y_pred = None
        self.test_data = None

        # Store feature engineering parameters
        self.use_time_features = None

        # Store model name
        self.model_name = None
        self.multi_variable = True

        # Anomaly detection thresholds
        self.low_threshold = None
        self.high_threshold = None
        self.threshold_std = 1.5

        # Real-time monitoring buffer
        self.data_buffer = None
        self.time_counter = 0

        # Feature columns (set after training)
        self.feature_columns = None

    def set_data(self, data: pd.DataFrame):
        """Set pre-processed data from ComputerInfoExtractor."""
        self.data = data.copy()
        return self

    def configure_model(self, model='linear', multi_variable=True, scaler='standard', use_time_features=True):
        """Configure the model and scaler to use."""
        scaler_dict = {
            'standard': StandardScaler(),
            'minmax': MinMaxScaler(),
            'robust': RobustScaler()
        }

        model_dict = {
            'xgb': XGBRegressor(
                n_estimators=300,
                max_depth=6,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                min_child_weight=10,
                reg_alpha=0.5,
                reg_lambda=2.0,
                random_state=42,
                tree_method='hist'
            ),
            'linear': Ridge(
                alpha=10.0,
                solver='auto',
                random_state=42
            ),
            'lightgbm': LGBMRegressor(
                n_estimators=300,
                max_depth=6,
                learning_rate=0.05,
                num_leaves=31,
                subsample=0.8,
                colsample_bytree=0.8,
                min_child_samples=100,
                reg_alpha=0.5,
                reg_lambda=2.0,
                random_state=42,
                verbose=-1
            ),
            'pca': 'pca',  # Placeholder for backwards compatibility
            'kernel_pca': 'kernel_pca'  # Placeholder for backwards compatibility
        }

        self.model = model_dict.get(model)
        self.scaler = scaler_dict.get(scaler)
        self.model_name = model
        self.use_time_features = use_time_features
        self.multi_variable = multi_variable

    def fit_predict(self, train_size=0.8, threshold_std=1.5):
        """
        Train the model and make predictions on test set.
        Data should already be pre-processed by ComputerInfoExtractor.

        CRITICAL: Scaling is done HERE after train/test split to prevent data leakage.
        The scaler is fit ONLY on training data, then applied to test data.
        """
        if self.data is None:
            raise ValueError("No data set. Use set_data() with pre-processed data from ComputerInfoExtractor.")

        self.threshold_std = threshold_std

        # Data is already processed (features created, missing values filled)
        df = self.data.copy()

        # Split the data BEFORE scaling
        split_idx = int(len(df) * train_size)
        train_data = df[:split_idx].copy()
        test_data = df[split_idx:].copy()

        # Drop NaN rows (from lag/rolling features)
        train_data = train_data.dropna()
        test_data = test_data.dropna()

        # Prepare X and y
        exclude_cols = ['cpu_temp', 'time', 'fan_rpm', 'timestamp']
        if self.multi_variable:
            feature_cols = [col for col in train_data.columns if col not in exclude_cols]
            self.x_train = train_data[feature_cols]
            self.y_train = train_data['cpu_temp']

            self.x_test = test_data[feature_cols]
            self.y_test = test_data['cpu_temp']
        else:
            self.x_train = train_data[['time']]
            self.y_train = train_data['cpu_temp']

            self.x_test = test_data[['time']]
            self.y_test = test_data['cpu_temp']

        # Store feature columns for later use
        self.feature_columns = list(self.x_train.columns)

        # Fit scaler ONLY on training data (prevents data leakage)
        x_train_scaled = self.scaler.fit_transform(self.x_train)
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

    def recalculate_thresholds(self, threshold_std: float):
        """Recalculate thresholds with new std multiplier without retraining."""
        if self.y_test is None or self.y_pred is None:
            raise ValueError("Model must be trained before recalculating thresholds")

        self.threshold_std = threshold_std
        diff = self.y_test.values - self.y_pred
        mean_diff = np.mean(diff)
        std_diff = np.std(diff)
        self.low_threshold = mean_diff - threshold_std * std_diff
        self.high_threshold = mean_diff + threshold_std * std_diff

    def calculate_PSI(self, n_bins=11, plot=False):
        """
        Calculate Population Stability Index (PSI) for drift detection.

        PSI measures distribution shift between training and test data.
        Values interpretation:
        - PSI < 0.1: No significant change
        - 0.1 <= PSI < 0.2: Moderate change
        - PSI >= 0.2: Significant change (model may need retraining)

        Args:
            n_bins: Number of bins for histogram (default: 11)
            plot: Whether to show bar plot of PSI values (default: False)

        Returns:
            dict: PSI value for each original feature
        """
        original_features = [
            'cpu_load', 'cpu_power', 'cpu_clock', 'cpu_volt',
            'gpu_temp', 'gpu_load', 'gpu_power',
            'mb_temp', 'ram_load'
        ]

        if self.x_train is None or self.x_test is None:
            raise ValueError("Model not trained. Run fit_predict() first.")

        columns = [col for col in original_features if col in self.x_train.columns]
        psi_cols_dict = {}

        for column in columns:
            list_train = self.x_train[column].to_list()
            list_test = self.x_test[column].to_list()

            bins = np.linspace(min(list_train), max(list_train), num=n_bins)

            hist_train, _ = np.histogram(list_train, bins=bins)
            hist_test, _ = np.histogram(list_test, bins=bins)

            pct_train = hist_train / hist_train.sum()
            pct_test = hist_test / hist_test.sum()

            eps = 0.0001
            pct_train = pct_train + eps
            pct_test = pct_test + eps

            psi_per_bin = (pct_test - pct_train) * np.log(pct_test / pct_train)
            psi_value = np.sum(psi_per_bin)
            psi_cols_dict[column] = psi_value

        if plot:
            psi_df = pd.DataFrame(list(psi_cols_dict.items()), columns=['Feature', 'PSI'])
            fig = px.bar(psi_df, x='Feature', y='PSI',
                        title='Population Stability Index (PSI) for Each Feature',
                        color='PSI',
                        color_continuous_scale='RdYlGn_r')
            fig.add_hline(y=0.1, line_dash='dash', line_color='orange',
                         annotation_text='Moderate Change (0.1)',
                         annotation_position='top left')
            fig.add_hline(y=0.2, line_dash='dash', line_color='red',
                         annotation_text='Significant Change (0.2)',
                         annotation_position='top left')
            fig.show()

        return psi_cols_dict

    def init_realtime_buffer(self):
        """Initialize the buffer for real-time anomaly detection."""
        if self.extractor is None:
            raise ValueError("Extractor required for real-time detection. Pass ComputerInfoExtractor to constructor.")
        max_window = max(self.extractor.rolling_windows) if self.extractor.rolling_windows else 7
        max_lag = max(self.extractor.lag_steps) if self.extractor.lag_steps else 3
        buffer_size = max(max_window, max_lag) + 5
        self.data_buffer = deque(maxlen=buffer_size)
        self.time_counter = 0

    def detect_anomaly(self, current_data: dict) -> tuple:
        """
        Detect if current data point is an anomaly.
        Uses extractor to create time features from buffer.

        Args:
            current_data: Dict with sensor readings from HardwareMonitor.get_essential_fast()

        Returns:
            tuple: (is_anomaly: bool, diff: float, predicted: float, actual: float)
        """
        if self.extractor is None:
            raise ValueError("Extractor required for real-time detection. Pass ComputerInfoExtractor to constructor.")

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
        min_required = max(max(self.extractor.rolling_windows), max(self.extractor.lag_steps)) + 1
        if len(self.data_buffer) < min_required:
            return False, 0.0, 0.0, current_data.get('cpu_temp', 0.0)

        # Create DataFrame from buffer and use extractor for feature engineering
        self.extractor.data = pd.DataFrame(list(self.data_buffer))

        if self.use_time_features and self.multi_variable:
            df = self.extractor.create_time_features_on_df()
            df = df.dropna()
        else:
            df = self.extractor.data.copy()

        if len(df) == 0:
            return False, 0.0, 0.0, current_data.get('cpu_temp', 0.0)

        # Get last row for prediction
        last_row = df.iloc[[-1]]

        # Prepare features
        if self.multi_variable:
            X = last_row.drop(['cpu_temp', 'time', 'fan_rpm', 'timestamp'], axis=1, errors='ignore')
        else:
            X = last_row[['time']]

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

    def plot_predictions(self):
        """Plot predictions and anomaly detection results."""
        # Use stored test_data time values
        time_test = self.test_data['time'].values
        timestamp_test = self.test_data['timestamp'].values

        # Get model name for titles
        model_display_name = self.model_name.upper() if self.model_name else "Model"

        # Prepare data
        df_test = self.x_test.copy()
        df_test['time'] = time_test
        df_test['timestamp'] = timestamp_test
        df_test["Actual"] = self.y_test.values
        df_test["Predicted"] = self.y_pred
        df_test = df_test.sort_values("time")

        df_test["diff"] = df_test["Actual"] - df_test["Predicted"]
        mean_diff = df_test["diff"].mean()
        stardard_dev = df_test["diff"].std()
        low_lim = mean_diff - self.threshold_std * stardard_dev
        high_lim = mean_diff + self.threshold_std * stardard_dev

        # Transform to long format (better for px.line)
        df_plot = df_test.melt(id_vars="timestamp", value_vars=["diff"], var_name="Type", value_name="Temperature Diff")

        # Predictions plot
        fig_pred = px.line(df_test, x="timestamp", y=["Actual", "Predicted"], title=f"{model_display_name} Regression: CPU Temp: Predicted vs Actual")
        fig_pred.update_traces(
            line_width=self.PLOT_STYLE['line_width'],
            marker={'size': self.PLOT_STYLE['marker_size']}
        )
        fig_pred.show()

        # Error plot
        fig = px.line(df_plot, x="timestamp", y="Temperature Diff", color="Type", title=f"{model_display_name} Regression: CPU Temp Difference: Predicted vs Actual")
        fig.update_traces(
            line_color=self.PLOT_STYLE['error_color'],
            line_width=self.PLOT_STYLE['line_width'],
            marker={'size': self.PLOT_STYLE['marker_size']}
        )
        fig.add_hline(
            y=low_lim,
            line_width=self.PLOT_STYLE['threshold_line_width'],
            line_color=self.PLOT_STYLE['lower_threshold_color'],
            line_dash=self.PLOT_STYLE['line_dash'],
            opacity=self.PLOT_STYLE['threshold_opacity']
        )
        fig.add_hline(
            y=high_lim,
            line_width=self.PLOT_STYLE['threshold_line_width'],
            line_color=self.PLOT_STYLE['upper_threshold_color'],
            line_dash=self.PLOT_STYLE['line_dash'],
            opacity=self.PLOT_STYLE['threshold_opacity']
        )
        fig.show()

        # Anomaly detection plot
        df_test["anomaly"] = (df_test["diff"] > high_lim) | (df_test["diff"] < low_lim)
        fig_anom = px.line(df_test, x="timestamp", y="anomaly", color="anomaly", title=f"{model_display_name} Regression: Anomaly Detection", markers=True)
        fig_anom.show()

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
        from src.data_extractor import ComputerInfoExtractor

        model = joblib.load(path)

        # Backward compatibility: Add extractor if not present (for old models)
        if not hasattr(model, 'extractor') or model.extractor is None:
            # Create extractor with default parameters
            extractor = ComputerInfoExtractor(
                scaler_type='standard',
                use_time_features=getattr(model, 'use_time_features', True),
                lag_steps=getattr(model, 'lag_steps', [1, 2, 4, 8, 12]),
                rolling_windows=getattr(model, 'rolling_windows', [6, 12, 24, 36])
            )
            model.extractor = extractor
            
        # Initialize buffer for real-time detection if extractor exists
        if model.extractor and hasattr(model, 'init_realtime_buffer'):
            try:
                model.init_realtime_buffer()
            except:
                pass  # Buffer initialization might fail if extractor params are missing

        return model


class CoreTempPCA:
    """
    CPU Temperature anomaly detection using PCA reconstruction error.
    Trains PCA on healthy data, then detects anomalies based on reconstruction error.
    """

    PLOT_STYLE = {
        'line_width': 3,
        'marker_size': 8,
        'threshold_line_width': 5,
        'threshold_opacity': 0.5,
        'error_color': 'orange',
        'lower_threshold_color': 'green',
        'upper_threshold_color': 'red',
        'line_dash': 'dash'
    }

    def __init__(self, extractor=None):
        self.extractor = extractor
        self.scaler = None
        self.pca = None
        self.data = None

        # Store for predictions/reconstruction
        self.x_train = None
        self.y_train = None
        self.x_test = None
        self.y_test = None
        self.reconstruction_error = None
        self.test_data = None

        # Store feature engineering parameters
        self.use_time_features = None

        # Store model name
        self.model_name = None
        self.multi_variable = True

        # Anomaly detection thresholds
        self.low_threshold = None
        self.high_threshold = None
        self.threshold_std = 1.5

        # Real-time monitoring buffer
        self.data_buffer = None
        self.time_counter = 0

        # Feature columns (set after training)
        self.feature_columns = None

    def set_data(self, data: pd.DataFrame):
        """Set pre-processed data from ComputerInfoExtractor."""
        self.data = data.copy()
        return self

    def configure_model(self, model='pca', multi_variable=True, scaler='standard', use_time_features=True, n_components=0.95):
        """Configure the PCA model and scaler to use."""
        scaler_dict = {
            'standard': StandardScaler(),
            'minmax': MinMaxScaler(),
            'robust': RobustScaler()
        }

        model_dict = {
            'pca': PCA(
                n_components=n_components,
                random_state=42
            ),
            'kernel_pca': KernelPCA(
                n_components=n_components if isinstance(n_components, int) else 10,
                kernel='rbf',
                random_state=42,
                fit_inverse_transform=True
            )
        }

        self.pca = model_dict.get(model)
        self.scaler = scaler_dict.get(scaler)
        self.model_name = model
        self.use_time_features = use_time_features
        self.multi_variable = multi_variable

    def fit_predict(self, train_size=0.8, threshold_std=1.5):
        """
        Train PCA on healthy data and detect anomalies based on reconstruction error.

        Workflow:
        1. Split data into train (healthy) and test
        2. Scale features (fit on train only)
        3. Fit PCA on scaled training data
        4. Transform test data through PCA
        5. Inverse transform to reconstruct original features
        6. Calculate reconstruction error
        7. Anomalies have high reconstruction error

        CRITICAL: Scaling and PCA are fit ONLY on training data to prevent data leakage.
        """
        if self.data is None:
            raise ValueError("No data set. Use set_data() with pre-processed data from ComputerInfoExtractor.")

        self.threshold_std = threshold_std

        # Data is already processed (features created, missing values filled)
        df = self.data.copy()

        # Split the data BEFORE scaling
        split_idx = int(len(df) * train_size)
        train_data = df[:split_idx].copy()
        test_data = df[split_idx:].copy()

        # Drop NaN rows (from lag/rolling features)
        train_data = train_data.dropna()
        test_data = test_data.dropna()

        # Prepare X and y
        exclude_cols = ['cpu_temp', 'time', 'fan_rpm', 'timestamp']
        if self.multi_variable:
            feature_cols = [col for col in train_data.columns if col not in exclude_cols]
            self.x_train = train_data[feature_cols]
            self.y_train = train_data['cpu_temp']

            self.x_test = test_data[feature_cols]
            self.y_test = test_data['cpu_temp']
        else:
            self.x_train = train_data[['time']]
            self.y_train = train_data['cpu_temp']

            self.x_test = test_data[['time']]
            self.y_test = test_data['cpu_temp']

        # Store feature columns for later use
        self.feature_columns = list(self.x_train.columns)

        # Step 1: Fit scaler ONLY on training data (prevents data leakage)
        x_train_scaled = self.scaler.fit_transform(self.x_train)
        x_test_scaled = self.scaler.transform(self.x_test)

        # Step 2: Fit PCA ONLY on scaled training data (healthy data)
        self.pca.fit(x_train_scaled)

        # Step 3: Transform test data through PCA (reduce dimensions)
        x_test_pca = self.pca.transform(x_test_scaled)

        # Step 4: Inverse transform to reconstruct original features
        x_test_reconstructed = self.pca.inverse_transform(x_test_pca)

        # Step 5: Calculate reconstruction error (MSE per sample)
        self.reconstruction_error = np.mean((x_test_scaled - x_test_reconstructed) ** 2, axis=1)

        if hasattr(self.pca, 'n_components_'):
            print(f"PCA reduced dimensions from {x_train_scaled.shape[1]} to {self.pca.n_components_}")
            print(f"Explained variance ratio: {self.pca.explained_variance_ratio_.sum():.4f}")
        else:
            print(f"Using {self.model_name} with configured components")

        # Store test_data for plotting
        self.test_data = test_data

        # Calculate anomaly thresholds based on reconstruction error
        mean_error = np.mean(self.reconstruction_error)
        std_error = np.std(self.reconstruction_error)
        self.low_threshold = mean_error - threshold_std * std_error
        self.high_threshold = mean_error + threshold_std * std_error

    def predict(self, X):
        """Calculate reconstruction error for new data."""
        X_scaled = self.scaler.transform(X)
        X_pca = self.pca.transform(X_scaled)
        X_reconstructed = self.pca.inverse_transform(X_pca)
        reconstruction_error = np.mean((X_scaled - X_reconstructed) ** 2, axis=1)
        return reconstruction_error

    def detect_anomaly(self, current_data: dict) -> tuple:
        """
        Detect if current data point is an anomaly using PCA reconstruction error.
        Uses extractor to create time features from buffer.

        Args:
            current_data: Dict with sensor readings from HardwareMonitor.get_essential_fast()

        Returns:
            tuple: (is_anomaly: bool, reconstruction_error: float, threshold: float, actual: float)
        """
        if self.extractor is None:
            raise ValueError("Extractor required for real-time detection. Pass ComputerInfoExtractor to constructor.")

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
        min_required = max(max(self.extractor.rolling_windows), max(self.extractor.lag_steps)) + 1
        if len(self.data_buffer) < min_required:
            return False, 0.0, self.high_threshold or 0.0, current_data.get('cpu_temp', 0.0)

        # Create DataFrame from buffer and use extractor for feature engineering
        self.extractor.data = pd.DataFrame(list(self.data_buffer))

        if self.use_time_features and self.multi_variable:
            df = self.extractor.create_time_features_on_df()
            df = df.dropna()
        else:
            df = self.extractor.data.copy()

        if len(df) == 0:
            return False, 0.0, self.high_threshold or 0.0, current_data.get('cpu_temp', 0.0)

        # Get last row for prediction
        last_row = df.iloc[[-1]]

        # Prepare features
        if self.multi_variable:
            X = last_row.drop(['cpu_temp', 'time', 'fan_rpm', 'timestamp'], axis=1, errors='ignore')
        else:
            X = last_row[['time']]

        # Ensure columns match training
        if self.feature_columns:
            missing_cols = set(self.feature_columns) - set(X.columns)
            for col in missing_cols:
                X[col] = 0
            X = X[self.feature_columns]

        # Calculate reconstruction error
        error = self.predict(X)[0]
        actual = current_data.get('cpu_temp', 0.0)

        # Check anomaly
        is_anomaly = error < self.low_threshold or error > self.high_threshold

        return is_anomaly, error, self.high_threshold, actual

    def init_realtime_buffer(self):
        """Initialize the buffer for real-time anomaly detection."""
        if self.extractor is None:
            raise ValueError("Extractor required for real-time detection. Pass ComputerInfoExtractor to constructor.")
        max_window = max(self.extractor.rolling_windows) if self.extractor.rolling_windows else 7
        max_lag = max(self.extractor.lag_steps) if self.extractor.lag_steps else 3
        buffer_size = max(max_window, max_lag) + 5
        self.data_buffer = deque(maxlen=buffer_size)
        self.time_counter = 0

    def recalculate_thresholds(self, threshold_std: float):
        """Recalculate thresholds with new std multiplier without retraining."""
        if self.reconstruction_error is None:
            raise ValueError("Model must be trained before recalculating thresholds")

        self.threshold_std = threshold_std
        mean_error = np.mean(self.reconstruction_error)
        std_error = np.std(self.reconstruction_error)
        self.low_threshold = mean_error - threshold_std * std_error
        self.high_threshold = mean_error + threshold_std * std_error

    def calculate_PSI(self, n_bins=11, plot=False):
        """
        Calculate Population Stability Index (PSI) for drift detection.

        PSI measures distribution shift between training and test data.
        Values interpretation:
        - PSI < 0.1: No significant change
        - 0.1 <= PSI < 0.2: Moderate change
        - PSI >= 0.2: Significant change (model may need retraining)

        Args:
            n_bins: Number of bins for histogram (default: 11)
            plot: Whether to show bar plot of PSI values (default: False)

        Returns:
            dict: PSI value for each original feature
        """
        original_features = [
            'cpu_load', 'cpu_power', 'cpu_clock', 'cpu_volt',
            'gpu_temp', 'gpu_load', 'gpu_power',
            'mb_temp', 'ram_load'
        ]

        if self.x_train is None or self.x_test is None:
            raise ValueError("Model not trained. Run fit_predict() first.")

        columns = self.x_train[original_features].columns.to_list()
        psi_cols_dict = {}

        for column in columns:
            list_train = self.x_train[column].to_list()
            list_test = self.x_test[column].to_list()

            bins = np.linspace(min(list_train), max(list_train), num=n_bins)

            hist_train, _ = np.histogram(list_train, bins=bins)
            hist_test, _ = np.histogram(list_test, bins=bins)

            pct_train = hist_train / hist_train.sum()
            pct_test = hist_test / hist_test.sum()

            eps = 0.0001
            pct_train = pct_train + eps
            pct_test = pct_test + eps

            psi_per_bin = (pct_test - pct_train) * np.log(pct_test / pct_train)
            psi_value = np.sum(psi_per_bin)
            psi_cols_dict[column] = psi_value

        if plot:
            psi_df = pd.DataFrame(list(psi_cols_dict.items()), columns=['Feature', 'PSI'])
            fig = px.bar(psi_df, x='Feature', y='PSI',
                        title='Population Stability Index (PSI) for Each Feature',
                        color='PSI',
                        color_continuous_scale='RdYlGn_r')
            fig.add_hline(y=0.1, line_dash='dash', line_color='orange',
                         annotation_text='Moderate Change (0.1)',
                         annotation_position='top left')
            fig.add_hline(y=0.2, line_dash='dash', line_color='red',
                         annotation_text='Significant Change (0.2)',
                         annotation_position='top left')
            fig.show()

        return psi_cols_dict

    def plot_predictions(self):
        """Plot reconstruction error and anomaly detection results."""
        # Use stored test_data time values
        time_test = self.test_data['time'].values
        timestamp_test = self.test_data['timestamp'].values

        # Get model name for titles
        model_display_name = self.model_name.upper() if self.model_name else "PCA"

        # Prepare data
        df_test = pd.DataFrame()
        df_test['time'] = time_test
        df_test['timestamp'] = timestamp_test
        df_test["Actual Temperature"] = self.y_test.values
        df_test["Reconstruction Error"] = self.reconstruction_error
        df_test = df_test.sort_values("time")

        # Calculate thresholds
        mean_error = np.mean(self.reconstruction_error)
        std_error = np.std(self.reconstruction_error)
        low_lim = mean_error - self.threshold_std * std_error
        high_lim = mean_error + self.threshold_std * std_error

        # Reconstruction error plot
        fig = px.line(df_test, x="timestamp", y="Reconstruction Error",
                      title=f"{model_display_name}: Reconstruction Error Over Time")
        fig.update_traces(
            line_color=self.PLOT_STYLE['error_color'],
            line_width=self.PLOT_STYLE['line_width'],
            marker={'size': self.PLOT_STYLE['marker_size']}
        )
        fig.add_hline(
            y=low_lim,
            line_width=self.PLOT_STYLE['threshold_line_width'],
            line_color=self.PLOT_STYLE['lower_threshold_color'],
            line_dash=self.PLOT_STYLE['line_dash'],
            opacity=self.PLOT_STYLE['threshold_opacity']
        )
        fig.add_hline(
            y=high_lim,
            line_width=self.PLOT_STYLE['threshold_line_width'],
            line_color=self.PLOT_STYLE['upper_threshold_color'],
            line_dash=self.PLOT_STYLE['line_dash'],
            opacity=self.PLOT_STYLE['threshold_opacity']
        )
        fig.show()

        # Temperature with anomaly overlay
        df_test["is_anomaly"] = (df_test["Reconstruction Error"] > high_lim) | (df_test["Reconstruction Error"] < low_lim)
        fig_temp = px.line(df_test, x="timestamp", y="Actual Temperature",
                           title=f"{model_display_name}: CPU Temperature with Anomalies Highlighted")

        # Highlight anomalies
        anomaly_points = df_test[df_test["is_anomaly"]]
        if len(anomaly_points) > 0:
            fig_temp.add_scatter(
                x=anomaly_points["timestamp"],
                y=anomaly_points["Actual Temperature"],
                mode='markers',
                marker=dict(color='red', size=12, symbol='x'),
                name='Anomaly'
            )
        fig_temp.show()

        # Anomaly detection plot
        fig_anom = px.line(df_test, x="timestamp", y="is_anomaly",
                          title=f"{model_display_name}: Anomaly Detection", markers=True)
        fig_anom.show()

    def evaluate_metrics(self):
        """Calculate and return evaluation metrics for reconstruction."""
        mean_error = np.mean(self.reconstruction_error)
        std_error = np.std(self.reconstruction_error)
        max_error = np.max(self.reconstruction_error)
        min_error = np.min(self.reconstruction_error)

        # Count anomalies
        anomalies = (self.reconstruction_error > self.high_threshold) | (self.reconstruction_error < self.low_threshold)
        n_anomalies = np.sum(anomalies)
        anomaly_rate = n_anomalies / len(self.reconstruction_error) * 100

        return {
            'mean_reconstruction_error': mean_error,
            'std_reconstruction_error': std_error,
            'max_reconstruction_error': max_error,
            'min_reconstruction_error': min_error,
            'n_anomalies': n_anomalies,
            'anomaly_rate_percent': anomaly_rate,
            'low_threshold': self.low_threshold,
            'high_threshold': self.high_threshold
        }

    def save_model(self, path: str):
        """Save the trained model to file."""
        joblib.dump(self, path)

    @staticmethod
    def load_model(path: str) -> 'CoreTempPCA':
        """Load a trained model from file."""
        from src.data_extractor import ComputerInfoExtractor

        model = joblib.load(path)

        # Backward compatibility: Add extractor if not present
        if not hasattr(model, 'extractor') or model.extractor is None:
            # Create extractor with default parameters
            extractor = ComputerInfoExtractor(
                scaler_type='standard',
                use_time_features=getattr(model, 'use_time_features', True), lag_steps = [1, 2, 4, 8, 12], rolling_windows = [6, 12, 24, 36]
            )
            model.extractor = extractor

        # Initialize buffer for real-time detection if extractor exists
        if model.extractor and hasattr(model, 'init_realtime_buffer'):
            try:
                model.init_realtime_buffer()
            except:
                pass  # Buffer initialization might fail if extractor params are missing

        return model
