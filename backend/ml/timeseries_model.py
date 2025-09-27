"""
LSTM Time-Series Model for Outbreak Prediction
==============================================

This module implements an LSTM-based time-series model for outbreak prediction
using sensor data, environmental factors, and case counts over time.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc
import joblib
import logging
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime, timedelta
import os

from schemas.ml_models import TimeSeriesWindow, ModelEvaluation

logger = logging.getLogger(__name__)

class TimeSeriesDataset(Dataset):
    """PyTorch Dataset for time-series data."""
    
    def __init__(self, sequences: np.ndarray, labels: np.ndarray):
        """
        Initialize dataset.
        
        Parameters:
        -----------
        sequences : np.ndarray
            Time-series sequences of shape (N, T, F)
        labels : np.ndarray
            Binary labels for outbreak prediction
        """
        self.sequences = torch.FloatTensor(sequences)
        self.labels = torch.FloatTensor(labels)
        
    def __len__(self) -> int:
        return len(self.sequences)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.sequences[idx], self.labels[idx]

class LSTMModel(nn.Module):
    """
    LSTM model for time-series outbreak prediction.
    
    Architecture:
    - LSTM layers for temporal pattern recognition
    - Attention mechanism for important timesteps
    - Dense layers for final prediction
    """
    
    def __init__(self, input_dim: int, hidden_dim: int = 64, num_layers: int = 2,
                 dropout: float = 0.2, attention: bool = True):
        """
        Initialize LSTM model.
        
        Parameters:
        -----------
        input_dim : int
            Number of input features per timestep
        hidden_dim : int
            LSTM hidden dimension
        num_layers : int
            Number of LSTM layers
        dropout : float
            Dropout rate
        attention : bool
            Whether to use attention mechanism
        """
        super(LSTMModel, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.attention = attention
        
        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True
        )
        
        # Attention mechanism
        if attention:
            self.attention_layer = nn.Sequential(
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.Tanh(),
                nn.Linear(hidden_dim, 1)
            )
        
        # Output layers
        self.output_layers = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the model.
        
        Parameters:
        -----------
        x : torch.Tensor
            Input sequences of shape (batch_size, seq_len, input_dim)
            
        Returns:
        --------
        torch.Tensor
            Output predictions of shape (batch_size, 1)
        """
        # LSTM forward pass
        lstm_out, (hidden, cell) = self.lstm(x)
        # lstm_out shape: (batch_size, seq_len, hidden_dim * 2)
        
        if self.attention:
            # Apply attention mechanism
            attention_weights = self.attention_layer(lstm_out)
            attention_weights = torch.softmax(attention_weights, dim=1)
            
            # Weighted sum of LSTM outputs
            attended_output = torch.sum(lstm_out * attention_weights, dim=1)
        else:
            # Use last timestep output
            attended_output = lstm_out[:, -1, :]
        
        # Final prediction
        output = self.output_layers(attended_output)
        
        return output

class TimeSeriesOutbreakPredictor:
    """
    LSTM-based time-series model for outbreak prediction.
    
    This model processes time-series data including:
    - Sensor readings (water quality, environmental)
    - Case counts over time
    - Environmental factors
    - Temporal patterns and trends
    """
    
    def __init__(self, model_path: str = "models/ts_lstm.pth",
                 sequence_length: int = 30, input_dim: int = 10):
        """
        Initialize the time-series predictor.
        
        Parameters:
        -----------
        model_path : str
            Path to save/load the model
        sequence_length : int
            Length of input sequences
        input_dim : int
            Number of input features per timestep
        """
        self.model_path = model_path
        self.sequence_length = sequence_length
        self.input_dim = input_dim
        self.model = None
        self.scaler = StandardScaler()
        self.feature_names = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    def prepare_sequences(self, data: pd.DataFrame, 
                          target_column: str = 'outbreak_flag') -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare time-series sequences from data.
        
        Parameters:
        -----------
        data : pd.DataFrame
            Time-series data with features and target
        target_column : str
            Name of the target column
            
        Returns:
        --------
        Tuple[np.ndarray, np.ndarray]
            Sequences and labels
        """
        logger.info("Preparing time-series sequences...")
        
        # Sort by date if available
        if 'date' in data.columns:
            data = data.sort_values('date')
        
        # Select feature columns (exclude target and date)
        feature_columns = [col for col in data.columns 
                          if col not in [target_column, 'date', 'timestamp']]
        
        # Store feature names
        self.feature_names = feature_columns
        
        # Extract features and target
        X = data[feature_columns].values
        y = data[target_column].values
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Create sequences
        sequences = []
        labels = []
        
        for i in range(len(X_scaled) - self.sequence_length + 1):
            seq = X_scaled[i:i + self.sequence_length]
            label = y[i + self.sequence_length - 1]  # Label for the last timestep
            
            sequences.append(seq)
            labels.append(label)
        
        sequences = np.array(sequences)
        labels = np.array(labels)
        
        logger.info(f"Created {len(sequences)} sequences of length {self.sequence_length}")
        return sequences, labels
    
    def train(self, data: pd.DataFrame, target_column: str = 'outbreak_flag',
              test_size: float = 0.2, validation_size: float = 0.2,
              epochs: int = 100, batch_size: int = 32, learning_rate: float = 0.001,
              patience: int = 10) -> Dict[str, Any]:
        """
        Train the LSTM model.
        
        Parameters:
        -----------
        data : pd.DataFrame
            Training data
        target_column : str
            Name of the target column
        test_size : float
            Proportion of data for testing
        validation_size : float
            Proportion of data for validation
        epochs : int
            Number of training epochs
        batch_size : int
            Batch size for training
        learning_rate : float
            Learning rate for optimizer
        patience : int
            Early stopping patience
            
        Returns:
        --------
        Dict[str, Any]
            Training results and metrics
        """
        logger.info("Training LSTM time-series model...")
        
        # Prepare sequences
        sequences, labels = self.prepare_sequences(data, target_column)
        
        # Split data
        n_samples = len(sequences)
        train_size = int(n_samples * (1 - test_size - validation_size))
        val_size = int(n_samples * validation_size)
        
        # Shuffle indices
        indices = np.random.permutation(n_samples)
        
        train_indices = indices[:train_size]
        val_indices = indices[train_size:train_size + val_size]
        test_indices = indices[train_size + val_size:]
        
        # Create datasets
        train_dataset = TimeSeriesDataset(sequences[train_indices], labels[train_indices])
        val_dataset = TimeSeriesDataset(sequences[val_indices], labels[val_indices])
        test_dataset = TimeSeriesDataset(sequences[test_indices], labels[test_indices])
        
        # Create data loaders
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        
        # Initialize model
        self.model = LSTMModel(
            input_dim=self.input_dim,
            hidden_dim=64,
            num_layers=2,
            dropout=0.2,
            attention=True
        ).to(self.device)
        
        # Loss and optimizer
        criterion = nn.BCELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
        
        # Training loop
        best_val_loss = float('inf')
        patience_counter = 0
        train_losses = []
        val_losses = []
        val_aucs = []
        
        for epoch in range(epochs):
            # Training
            self.model.train()
            train_loss = 0.0
            
            for batch_x, batch_y in train_loader:
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device).unsqueeze(1)
                
                optimizer.zero_grad()
                outputs = self.model(batch_x)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
            
            # Validation
            self.model.eval()
            val_loss = 0.0
            val_predictions = []
            val_targets = []
            
            with torch.no_grad():
                for batch_x, batch_y in val_loader:
                    batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device).unsqueeze(1)
                    
                    outputs = self.model(batch_x)
                    loss = criterion(outputs, batch_y)
                    val_loss += loss.item()
                    
                    val_predictions.extend(outputs.cpu().numpy())
                    val_targets.extend(batch_y.cpu().numpy())
            
            # Calculate metrics
            val_predictions = np.array(val_predictions).flatten()
            val_targets = np.array(val_targets).flatten()
            val_auc = roc_auc_score(val_targets, val_predictions)
            
            train_losses.append(train_loss / len(train_loader))
            val_losses.append(val_loss / len(val_loader))
            val_aucs.append(val_auc)
            
            # Learning rate scheduling
            scheduler.step(val_loss)
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                # Save best model
                torch.save(self.model.state_dict(), self.model_path)
            else:
                patience_counter += 1
            
            if patience_counter >= patience:
                logger.info(f"Early stopping at epoch {epoch + 1}")
                break
            
            if epoch % 10 == 0:
                logger.info(f"Epoch {epoch + 1}/{epochs} - Train Loss: {train_losses[-1]:.4f}, "
                           f"Val Loss: {val_losses[-1]:.4f}, Val AUC: {val_auc:.4f}")
        
        # Load best model
        self.model.load_state_dict(torch.load(self.model_path))
        
        # Evaluate on test set
        test_auc, test_predictions = self.evaluate_model(test_loader)
        
        # Save scaler and feature names
        scaler_path = self.model_path.replace('.pth', '_scaler.joblib')
        joblib.dump(self.scaler, scaler_path)
        
        features_path = self.model_path.replace('.pth', '_features.joblib')
        joblib.dump(self.feature_names, features_path)
        
        results = {
            "test_auc": test_auc,
            "best_val_loss": best_val_loss,
            "train_losses": train_losses,
            "val_losses": val_losses,
            "val_aucs": val_aucs,
            "n_epochs": epoch + 1,
            "sequence_length": self.sequence_length,
            "input_dim": self.input_dim
        }
        
        logger.info(f"Training completed. Test AUC: {test_auc:.4f}")
        return results
    
    def evaluate_model(self, data_loader: DataLoader) -> Tuple[float, np.ndarray]:
        """
        Evaluate the model on a data loader.
        
        Parameters:
        -----------
        data_loader : DataLoader
            Data loader for evaluation
            
        Returns:
        --------
        Tuple[float, np.ndarray]
            AUC score and predictions
        """
        self.model.eval()
        predictions = []
        targets = []
        
        with torch.no_grad():
            for batch_x, batch_y in data_loader:
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                
                outputs = self.model(batch_x)
                predictions.extend(outputs.cpu().numpy())
                targets.extend(batch_y.cpu().numpy())
        
        predictions = np.array(predictions).flatten()
        targets = np.array(targets).flatten()
        
        auc_score = roc_auc_score(targets, predictions)
        return auc_score, predictions
    
    def predict(self, time_series_window: TimeSeriesWindow) -> float:
        """
        Make prediction for a time-series window.
        
        Parameters:
        -----------
        time_series_window : TimeSeriesWindow
            Time-series data for prediction
            
        Returns:
        --------
        float
            Prediction probability
        """
        if self.model is None:
            self.load_model()
        
        # Convert to numpy array
        sequences = np.array(time_series_window.sensor_readings)
        
        # Ensure correct shape
        if sequences.shape[0] != self.sequence_length:
            logger.warning(f"Sequence length {sequences.shape[0]} doesn't match expected {self.sequence_length}")
            # Pad or truncate as needed
            if sequences.shape[0] < self.sequence_length:
                # Pad with zeros
                padding = np.zeros((self.sequence_length - sequences.shape[0], sequences.shape[1]))
                sequences = np.vstack([sequences, padding])
            else:
                # Truncate
                sequences = sequences[:self.sequence_length]
        
        # Scale features
        sequences_scaled = self.scaler.transform(sequences)
        
        # Convert to tensor
        sequences_tensor = torch.FloatTensor(sequences_scaled).unsqueeze(0).to(self.device)
        
        # Make prediction
        self.model.eval()
        with torch.no_grad():
            prediction = self.model(sequences_tensor)
            prediction = prediction.cpu().numpy()[0, 0]
        
        return float(prediction)
    
    def predict_batch(self, time_series_windows: List[TimeSeriesWindow]) -> List[float]:
        """
        Make predictions for multiple time-series windows.
        
        Parameters:
        -----------
        time_series_windows : List[TimeSeriesWindow]
            List of time-series windows
            
        Returns:
        --------
        List[float]
            List of prediction probabilities
        """
        if self.model is None:
            self.load_model()
        
        predictions = []
        
        for window in time_series_windows:
            pred = self.predict(window)
            predictions.append(pred)
        
        return predictions
    
    def load_model(self):
        """Load the trained model and preprocessing objects."""
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model file not found: {self.model_path}")
        
        # Load model
        self.model = LSTMModel(
            input_dim=self.input_dim,
            hidden_dim=64,
            num_layers=2,
            dropout=0.2,
            attention=True
        ).to(self.device)
        
        self.model.load_state_dict(torch.load(self.model_path, map_location=self.device))
        self.model.eval()
        
        # Load scaler
        scaler_path = self.model_path.replace('.pth', '_scaler.joblib')
        if os.path.exists(scaler_path):
            self.scaler = joblib.load(scaler_path)
        
        # Load feature names
        features_path = self.model_path.replace('.pth', '_features.joblib')
        if os.path.exists(features_path):
            self.feature_names = joblib.load(features_path)
        
        logger.info("Time-series model loaded successfully")

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def create_sample_timeseries_data(n_samples: int = 1000, 
                                sequence_length: int = 30) -> pd.DataFrame:
    """
    Create sample time-series data for testing.
    
    Parameters:
    -----------
    n_samples : int
        Number of samples to generate
    sequence_length : int
        Length of time series
        
    Returns:
    --------
    pd.DataFrame
        Sample time-series data
    """
    np.random.seed(42)
    
    # Generate timestamps
    start_date = datetime.now() - timedelta(days=n_samples)
    timestamps = [start_date + timedelta(days=i) for i in range(n_samples)]
    
    # Generate sample data
    data = {
        'timestamp': timestamps,
        'water_ph': np.random.normal(7.0, 0.5, n_samples),
        'water_turbidity': np.random.exponential(1, n_samples),
        'water_temperature': np.random.normal(25, 5, n_samples),
        'water_conductivity': np.random.normal(500, 100, n_samples),
        'rainfall': np.random.exponential(2, n_samples),
        'temperature': np.random.normal(25, 5, n_samples),
        'humidity': np.random.uniform(40, 90, n_samples),
        'case_count': np.random.poisson(5, n_samples),
        'outbreak_flag': np.random.choice([0, 1], n_samples, p=[0.7, 0.3])
    }
    
    return pd.DataFrame(data)

def main():
    """Example usage of the time-series model."""
    # Create sample data
    data = create_sample_timeseries_data(1000, 30)
    
    # Initialize model
    model = TimeSeriesOutbreakPredictor(sequence_length=30, input_dim=9)
    
    # Train model
    results = model.train(data)
    print("Training results:", results)
    
    # Test prediction
    sample_window = TimeSeriesWindow(
        timestamps=[datetime.now() - timedelta(days=i) for i in range(30)],
        sensor_readings=[[7.0, 1.0, 25.0, 500.0, 2.0, 25.0, 60.0, 5.0] for _ in range(30)],
        case_counts=[5] * 30,
        environmental_factors=[[2.0, 25.0, 60.0] for _ in range(30)],
        window_size=30
    )
    
    prediction = model.predict(sample_window)
    print(f"Sample prediction: {prediction}")

if __name__ == "__main__":
    main()
