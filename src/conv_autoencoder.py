"""
CNN Autoencoder for CPU Temperature Anomaly Detection.
Uses reconstruction error on sliding windows of sensor data to detect anomalies.
"""

from collections import deque

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import plotly.express as px


class ConvAutoencoder(nn.Module):

    FEATURE_COLUMNS = ['cpu_temp', 'cpu_load', 'cpu_power', 'gpu_temp', 'gpu_load', 'gpu_power', 'ram_load']

    def __init__(
        self,
        input_dim=7,
        encoder_channels=[64, 32, 16],
        decoder_channels=[16, 32, 64],
        window_size=60,
        epochs=20,
        batch_size=32,
        learning_rate=1e-4,
        test_size=0.2,
        threshold_std=2,
        *args, **kwargs
    ):
        super().__init__(*args, **kwargs)

        self.input_dim = input_dim
        self.window_size = window_size
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.test_size = test_size
        self.threshold_std = threshold_std

        self.data = None
        self.train_data = None
        self.train_loader = None
        self.test_data = None
        self.test_loader = None
        self.scaler = MinMaxScaler()
        self.feature_columns = list(self.FEATURE_COLUMNS)
        self.erros = None
        self.threshold = None

        # Thresholds compatible with app/tray_monitor interface
        self.low_threshold = None
        self.high_threshold = None

        # Per-feature training errors and thresholds
        self.erros_per_feature = None
        self.per_feature_thresholds = {}

        # Training data reference for PSI
        self.x_train = None

        # Real-time detection buffer
        self.realtime_buffer = None

        # Build encoder
        encoder_layers = []
        encoder_current = input_dim
        for out_channels in encoder_channels:
            encoder_layers.append(nn.Conv1d(encoder_current, out_channels, kernel_size=3, stride=1, padding=1))
            encoder_layers.append(nn.ReLU())
            encoder_current = out_channels
        self.encoder = nn.Sequential(*encoder_layers)

        # Build decoder
        decoder_layers = []
        decoder_current = encoder_current
        for out_channels in decoder_channels:
            decoder_layers.append(nn.Conv1d(decoder_current, out_channels, kernel_size=3, stride=1, padding=1))
            decoder_layers.append(nn.ReLU())
            decoder_current = out_channels
        decoder_layers.append(nn.Conv1d(decoder_current, input_dim, kernel_size=3, stride=1, padding=1))
        decoder_layers.append(nn.Sigmoid())
        self.decoder = nn.Sequential(*decoder_layers)

    def make_windows(self, data):
        return torch.tensor(
            np.array([data[i:i + self.window_size] for i in range(len(data) - self.window_size)]),
            dtype=torch.float32
        )

    def process_data(self, processed_data):
        self.data = processed_data
        self.feature_columns = [c for c in self.FEATURE_COLUMNS if c in processed_data.columns]
        self.input_dim = len(self.feature_columns)

        data = processed_data[self.feature_columns].values
        data = self.scaler.fit_transform(data)

        # Store raw training data for PSI
        split_idx = int(len(data) * (1 - self.test_size))
        self.x_train = pd.DataFrame(
            processed_data[self.feature_columns].iloc[:split_idx].values,
            columns=self.feature_columns
        )

        windows = self.make_windows(data)
        train_data, test_data = train_test_split(windows, test_size=self.test_size, shuffle=False)
        self.train_data = train_data
        self.test_data = test_data

    def _get_test_timestamps(self, use_window_end=True):
        if self.data is None or self.test_data is None or 'timestamp' not in self.data.columns:
            return None

        n_windows = max(len(self.data) - self.window_size, 0)
        if n_windows == 0:
            return None

        split_idx = int((1 - self.test_size) * n_windows)
        offset = self.window_size - 1 if use_window_end else 0
        start_idx = split_idx + offset
        end_idx = start_idx + len(self.test_data)
        timestamps = self.data['timestamp'].iloc[start_idx:end_idx].reset_index(drop=True)

        if len(timestamps) != len(self.test_data):
            return None

        return pd.to_datetime(timestamps)

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

    def fit(self):
        if self.train_data is None:
            print("No training data. Run process_data() first.")
            return

        train_dataset = TensorDataset(self.train_data)
        self.train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)

        test_dataset = TensorDataset(self.test_data)
        self.test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)

        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        criterion = nn.MSELoss()

        train_losses = []
        test_losses = []

        for epoch in range(self.epochs):
            self.train()
            epoch_train_loss = 0

            for batch in self.train_loader:
                original_data = batch[0].permute(0, 2, 1)
                optimizer.zero_grad()
                reconstruction = self(original_data)
                loss = criterion(reconstruction, original_data)
                loss.backward()
                optimizer.step()
                epoch_train_loss += loss.item()

            train_losses.append(epoch_train_loss)

            self.eval()
            epoch_val_loss = 0

            with torch.no_grad():
                for batch in self.test_loader:
                    original_data = batch[0].permute(0, 2, 1)
                    reconstruction = self(original_data)
                    loss = criterion(reconstruction, original_data)
                    epoch_val_loss += loss.item()

            test_losses.append(epoch_val_loss)
            print(f"Epoch: {epoch + 1}/{self.epochs} | Train Loss: {epoch_train_loss:.4f} | Val Loss: {epoch_val_loss:.4f}")

        return train_losses, test_losses

    def reconstruct(self):
        if self.test_loader is None:
            print("No test data. Run process_data() and fit() first.")
            return

        self.eval()
        with torch.no_grad():
            all_original = []
            all_reconstructed = []

            for batch in self.test_loader:
                x = batch[0].permute(0, 2, 1)
                output = self(x)
                all_original.append(x.permute(0, 2, 1).numpy())
                all_reconstructed.append(output.permute(0, 2, 1).numpy())

        all_original = np.concatenate(all_original, axis=0)
        all_reconstructed = np.concatenate(all_reconstructed, axis=0)

        original = all_original[:, 0, :]
        reconstruido = all_reconstructed[:, 0, :]

        original_denorm = self.scaler.inverse_transform(original)
        reconstruido_denorm = self.scaler.inverse_transform(reconstruido)

        df_orig = pd.DataFrame(original_denorm, columns=self.feature_columns)
        df_rec = pd.DataFrame(reconstruido_denorm, columns=[f'{col}_rec' for col in self.feature_columns])

        timestamp = self._get_test_timestamps(use_window_end=False)

        df = pd.concat([df_orig, df_rec], axis=1)
        if timestamp is not None:
            df['timestamp'] = timestamp
            x_col = 'timestamp'
        else:
            df['sample'] = range(len(df))
            x_col = 'sample'

        colunas_plot = self.feature_columns + [f'{col}_rec' for col in self.feature_columns]
        px.line(df, x=x_col, y=colunas_plot, title='Original vs Reconstructed').show()

    def reconstruction_error(self):
        if self.test_loader is None:
            print("No test data. Run process_data() and fit() first.")
            return None, None, None

        self.eval()
        erros_global = []
        erros_per_feature = []

        with torch.no_grad():
            for batch in self.test_loader:
                x = batch[0].permute(0, 2, 1)  # (batch, features, window)
                output = self(x)
                # Global: mean over features and window
                erro_global = ((output - x) ** 2).mean(dim=(1, 2))
                erros_global.extend(erro_global.numpy())
                # Per-feature: mean over window only, keep feature dim
                erro_feat = ((output - x) ** 2).mean(dim=2)  # (batch, features)
                erros_per_feature.append(erro_feat.numpy())

        erros_global = np.array(erros_global)
        erros_per_feature = np.concatenate(erros_per_feature, axis=0)  # (n_samples, n_features)
        threshold = erros_global.mean() + self.threshold_std * erros_global.std()

        return erros_global, threshold, erros_per_feature

    def fit_reconstruct(self, processed_data):
        self.process_data(processed_data)
        self.fit()
        self.reconstruct()
        erros, threshold, erros_per_feature = self.reconstruction_error()

        self.erros = erros
        self.threshold = threshold
        self.erros_per_feature = erros_per_feature

        # Global thresholds
        mean_err = erros.mean()
        std_err = erros.std()
        self.low_threshold = max(0, mean_err - self.threshold_std * std_err)
        self.high_threshold = threshold

        # Per-feature thresholds: {feature_name: threshold_value}
        self.per_feature_thresholds = {}
        for i, col in enumerate(self.feature_columns):
            feat_errs = erros_per_feature[:, i]
            self.per_feature_thresholds[col] = float(feat_errs.mean() + self.threshold_std * feat_errs.std())

    def plot_anomaly_detection(self):
        if self.erros is None or self.threshold is None:
            self.erros, self.threshold = self.reconstruction_error()

        timestamp = self._get_test_timestamps(use_window_end=True)

        if timestamp is not None:
            df_plot = pd.DataFrame({'timestamp': timestamp, 'reconstruction_error': self.erros})
            fig = px.line(df_plot, x='timestamp', y='reconstruction_error', title='Reconstruction Error Over Time', color_discrete_sequence=['dimgray'])
        else:
            df_plot = pd.DataFrame({'sample': range(len(self.erros)), 'reconstruction_error': self.erros})
            fig = px.line(df_plot, x='sample', y='reconstruction_error', title='Reconstruction Error Over Time', color_discrete_sequence=['dimgray'])

        fig.add_hline(y=self.threshold, line_dash='dash', line_width=4, line_color='orange', annotation_text='Anomaly Threshold', annotation_position='top left')
        fig.update_traces(line=dict(width=3))
        fig.show()

    # ---- App integration methods ----

    def evaluate_metrics(self):
        """Calculate and return evaluation metrics for reconstruction."""
        if self.erros is None:
            self.erros, self.threshold = self.reconstruction_error()

        mean_error = np.mean(self.erros)
        std_error = np.std(self.erros)

        anomalies = self.erros > self.high_threshold
        n_anomalies = int(np.sum(anomalies))
        anomaly_rate = n_anomalies / len(self.erros) * 100

        return {
            'mean_reconstruction_error': mean_error,
            'std_reconstruction_error': std_error,
            'n_anomalies': n_anomalies,
            'anomaly_rate_percent': anomaly_rate,
        }

    def recalculate_thresholds(self, threshold_std: float):
        """Recalculate thresholds with new std multiplier without retraining."""
        if self.erros is None:
            raise ValueError("Model must be trained before recalculating thresholds")

        self.threshold_std = threshold_std
        mean_err = np.mean(self.erros)
        std_err = np.std(self.erros)
        self.low_threshold = max(0, mean_err - threshold_std * std_err)
        self.high_threshold = mean_err + threshold_std * std_err
        self.threshold = self.high_threshold

        # Recalculate per-feature thresholds
        if hasattr(self, 'erros_per_feature') and self.erros_per_feature is not None:
            self.per_feature_thresholds = {}
            for i, col in enumerate(self.feature_columns):
                feat_errs = self.erros_per_feature[:, i]
                self.per_feature_thresholds[col] = float(feat_errs.mean() + threshold_std * feat_errs.std())

    def init_realtime_buffer(self):
        """Initialize the sliding window buffer for real-time anomaly detection."""
        self.realtime_buffer = deque(maxlen=self.window_size)

    def detect_anomaly(self, current_data: dict) -> tuple:
        """
        Detect if current data point is an anomaly using reconstruction error
        on a sliding window of raw sensor values.

        Args:
            current_data: Dict with sensor readings from HardwareMonitor.get_essential_fast()

        Returns:
            tuple: (is_anomaly, global_error, threshold, actual, per_feature_errors)
                per_feature_errors: dict {feature_name: error_value} or None if buffer not full
        """
        if self.realtime_buffer is None:
            self.init_realtime_buffer()

        actual = current_data.get('cpu_temp', 0.0)

        # Build feature vector from current reading
        row = [current_data.get(col, 0.0) for col in self.feature_columns]
        self.realtime_buffer.append(row)

        # Not enough data yet to fill a window
        if len(self.realtime_buffer) < self.window_size:
            return False, 0.0, self.high_threshold or 0.0, actual, None

        # Build window, scale, and run through model
        window = np.array(list(self.realtime_buffer), dtype=np.float32)
        window_scaled = self.scaler.transform(window)
        # Clip to [0, 1] to match training conditions — the decoder uses Sigmoid
        # which can only output [0, 1], so out-of-range inputs create irreducible
        # error that causes false anomalies
        window_scaled = np.clip(window_scaled, 0.0, 1.0)

        # Shape: (1, window_size, features) -> permute to (1, features, window_size) for Conv1d
        x = torch.tensor(window_scaled, dtype=torch.float32).unsqueeze(0).permute(0, 2, 1)

        self.eval()
        with torch.no_grad():
            output = self(x)
            # Global error
            error = float(((output - x) ** 2).mean())
            # Per-feature error: mean over window dim only
            feat_errors = ((output - x) ** 2).mean(dim=2).squeeze(0).numpy()  # (n_features,)

        per_feature_errors = {col: float(feat_errors[i]) for i, col in enumerate(self.feature_columns)}

        is_anomaly = error > self.high_threshold

        return is_anomaly, error, self.high_threshold, actual, per_feature_errors

    def save_model(self, path: str):
        """Save the trained model (weights + metadata) to file."""
        save_dict = {
            'state_dict': self.state_dict(),
            'scaler': self.scaler,
            'feature_columns': self.feature_columns,
            'input_dim': self.input_dim,
            'window_size': self.window_size,
            'threshold_std': self.threshold_std,
            'threshold': self.threshold,
            'low_threshold': self.low_threshold,
            'high_threshold': self.high_threshold,
            'erros': self.erros,
            'erros_per_feature': getattr(self, 'erros_per_feature', None),
            'per_feature_thresholds': getattr(self, 'per_feature_thresholds', {}),
            'x_train': self.x_train,
            'epochs': self.epochs,
            'batch_size': self.batch_size,
            'learning_rate': self.learning_rate,
            # Save architecture params for reconstruction
            'encoder_channels': [layer.out_channels for layer in self.encoder if isinstance(layer, nn.Conv1d)],
            'decoder_channels': [layer.out_channels for layer in self.decoder if isinstance(layer, nn.Conv1d)][:-1],
        }
        torch.save(save_dict, path)

    @staticmethod
    def load_model(path: str) -> 'ConvAutoencoder':
        """Load a trained model from file."""
        save_dict = torch.load(path, map_location='cpu', weights_only=False)

        model = ConvAutoencoder(
            input_dim=save_dict['input_dim'],
            encoder_channels=save_dict['encoder_channels'],
            decoder_channels=save_dict['decoder_channels'],
            window_size=save_dict['window_size'],
            epochs=save_dict.get('epochs', 20),
            batch_size=save_dict.get('batch_size', 32),
            learning_rate=save_dict.get('learning_rate', 1e-4),
            threshold_std=save_dict['threshold_std'],
        )
        model.load_state_dict(save_dict['state_dict'])
        model.scaler = save_dict['scaler']
        model.feature_columns = save_dict['feature_columns']
        model.threshold = save_dict['threshold']
        model.low_threshold = save_dict['low_threshold']
        model.high_threshold = save_dict['high_threshold']
        model.erros = save_dict.get('erros')
        model.erros_per_feature = save_dict.get('erros_per_feature')
        model.per_feature_thresholds = save_dict.get('per_feature_thresholds', {})
        model.x_train = save_dict.get('x_train')

        # Recompute per-feature thresholds if missing but training errors exist
        if not model.per_feature_thresholds and model.erros_per_feature is not None:
            for i, col in enumerate(model.feature_columns):
                feat_errs = model.erros_per_feature[:, i]
                model.per_feature_thresholds[col] = float(
                    feat_errs.mean() + model.threshold_std * feat_errs.std()
                )

        model.eval()
        return model

    def calculate_PSI(self, train_data=None, test_data=None, n_bins=11, plot=False):
        """
        Calculate Population Stability Index (PSI) for drift detection.
        Compatible with TrayMonitor.calculate_psi() interface.
        """
        from src.core_regressor import _calculate_feature_psi

        if train_data is None:
            train_data = self.x_train
        if train_data is None:
            return None

        results = {}
        for col in train_data.columns:
            if col in test_data.columns:
                psi_val = _calculate_feature_psi(train_data[col], test_data[col], n_bins=n_bins)
                if psi_val is not None:
                    results[col] = psi_val

        return results if results else None
