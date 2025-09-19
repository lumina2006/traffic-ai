"""
Base Traffic Predictor Class

This module provides the base class for all traffic prediction implementations
in the Traffic AI system.
"""

import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
from typing import List, Dict, Optional, Tuple, Any, Union
from dataclasses import dataclass
import time
from datetime import datetime, timedelta

from ..analysis.analyzer import TrafficMetrics

@dataclass
class PredictionInput:
    """Input data for traffic prediction"""
    timestamp: float
    vehicle_count: int
    density: float
    average_speed: float
    flow_rate: float
    congestion_level: str
    weather_condition: Optional[str] = None
    time_of_day: Optional[int] = None  # Hour of day (0-23)
    day_of_week: Optional[int] = None  # Day of week (0-6)

@dataclass
class PredictionResult:
    """Result from traffic prediction"""
    timestamp: float
    prediction_horizon: int  # Minutes into future
    predicted_vehicle_count: float
    predicted_density: float
    predicted_speed: float
    predicted_congestion: str
    confidence: float
    prediction_interval: Optional[Tuple[float, float]] = None  # (lower, upper)

class TrafficPredictor(ABC):
    """
    Abstract base class for traffic predictors
    
    This class defines the interface that all traffic predictors must implement.
    It provides common functionality for processing historical data and generating predictions.
    """
    
    def __init__(self,
                 prediction_horizon: int = 30,  # minutes
                 lookback_window: int = 60,     # minutes
                 update_frequency: int = 5):    # minutes
        """
        Initialize traffic predictor
        
        Args:
            prediction_horizon: How far into future to predict (minutes)
            lookback_window: How much historical data to use (minutes)
            update_frequency: How often to update predictions (minutes)
        """
        self.prediction_horizon = prediction_horizon
        self.lookback_window = lookback_window
        self.update_frequency = update_frequency
        
        # Historical data storage
        self.historical_data: List[PredictionInput] = []
        self.max_history_size = 10000  # Maximum number of historical records
        
        # Model state
        self.is_trained = False
        self.last_update_time = 0
        
        # Feature engineering parameters
        self.feature_columns = [
            'vehicle_count', 'density', 'average_speed', 'flow_rate',
            'hour_sin', 'hour_cos', 'day_sin', 'day_cos'
        ]
    
    @abstractmethod
    def train(self, historical_data: List[PredictionInput]) -> bool:
        """
        Train the prediction model
        
        Args:
            historical_data: Historical traffic data for training
            
        Returns:
            True if training successful, False otherwise
        """
        pass
    
    @abstractmethod
    def predict(self, 
               current_data: Union[PredictionInput, List[PredictionInput]],
               horizon_minutes: Optional[int] = None) -> Union[PredictionResult, List[PredictionResult]]:
        """
        Generate traffic predictions
        
        Args:
            current_data: Current traffic state(s)
            horizon_minutes: Optional override for prediction horizon
            
        Returns:
            Prediction result(s)
        """
        pass
    
    def add_historical_data(self, data: PredictionInput):
        """Add new data point to historical dataset"""
        self.historical_data.append(data)
        
        # Maintain maximum history size
        if len(self.historical_data) > self.max_history_size:
            self.historical_data = self.historical_data[-self.max_history_size:]
    
    def load_historical_data(self, data_path: str) -> bool:
        """
        Load historical data from file
        
        Args:
            data_path: Path to historical data file (CSV, JSON, etc.)
            
        Returns:
            True if loaded successfully, False otherwise
        """
        try:
            if data_path.endswith('.csv'):
                df = pd.read_csv(data_path)
                self._dataframe_to_prediction_inputs(df)
            elif data_path.endswith('.json'):
                df = pd.read_json(data_path)
                self._dataframe_to_prediction_inputs(df)
            else:
                print(f"Unsupported file format: {data_path}")
                return False
                
            print(f"Loaded {len(self.historical_data)} historical data points")
            return True
            
        except Exception as e:
            print(f"Error loading historical data: {str(e)}")
            return False
    
    def _dataframe_to_prediction_inputs(self, df: pd.DataFrame):
        """Convert pandas DataFrame to PredictionInput objects"""
        self.historical_data = []
        
        for _, row in df.iterrows():
            data = PredictionInput(
                timestamp=row.get('timestamp', time.time()),
                vehicle_count=row.get('vehicle_count', 0),
                density=row.get('density', 0.0),
                average_speed=row.get('average_speed', 0.0),
                flow_rate=row.get('flow_rate', 0.0),
                congestion_level=row.get('congestion_level', 'low'),
                weather_condition=row.get('weather_condition'),
                time_of_day=row.get('time_of_day'),
                day_of_week=row.get('day_of_week')
            )
            self.historical_data.append(data)
    
    def metrics_to_prediction_input(self, 
                                   metrics: TrafficMetrics,
                                   weather_condition: Optional[str] = None) -> PredictionInput:
        """
        Convert TrafficMetrics to PredictionInput
        
        Args:
            metrics: Traffic metrics from analyzer
            weather_condition: Optional weather information
            
        Returns:
            PredictionInput object
        """
        # Extract time features
        dt = datetime.fromtimestamp(metrics.timestamp)
        time_of_day = dt.hour
        day_of_week = dt.weekday()
        
        return PredictionInput(
            timestamp=metrics.timestamp,
            vehicle_count=metrics.vehicle_count,
            density=metrics.density,
            average_speed=metrics.average_speed,
            flow_rate=metrics.flow_rate,
            congestion_level=metrics.congestion_level,
            weather_condition=weather_condition,
            time_of_day=time_of_day,
            day_of_week=day_of_week
        )
    
    def prepare_features(self, data: List[PredictionInput]) -> np.ndarray:
        """
        Prepare feature matrix from prediction inputs
        
        Args:
            data: List of prediction inputs
            
        Returns:
            Feature matrix as numpy array
        """
        features = []
        
        for item in data:
            # Basic traffic features
            feature_vector = [
                item.vehicle_count,
                item.density,
                item.average_speed,
                item.flow_rate
            ]
            
            # Time-based features (cyclical encoding)
            if item.time_of_day is not None:
                hour_sin = np.sin(2 * np.pi * item.time_of_day / 24)
                hour_cos = np.cos(2 * np.pi * item.time_of_day / 24)
                feature_vector.extend([hour_sin, hour_cos])
            else:
                feature_vector.extend([0.0, 1.0])  # Default to midnight
            
            if item.day_of_week is not None:
                day_sin = np.sin(2 * np.pi * item.day_of_week / 7)
                day_cos = np.cos(2 * np.pi * item.day_of_week / 7)
                feature_vector.extend([day_sin, day_cos])
            else:
                feature_vector.extend([0.0, 1.0])  # Default to Monday
            
            # Congestion level encoding
            congestion_encoding = {
                'low': [1, 0, 0],
                'medium': [0, 1, 0], 
                'high': [0, 0, 1]
            }
            feature_vector.extend(congestion_encoding.get(item.congestion_level, [1, 0, 0]))
            
            features.append(feature_vector)
        
        return np.array(features)
    
    def create_sequences(self, 
                        data: np.ndarray,
                        sequence_length: int,
                        target_columns: List[int] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create sequences for time series prediction
        
        Args:
            data: Input data matrix
            sequence_length: Length of input sequences
            target_columns: Columns to use as targets (default: first 4 columns)
            
        Returns:
            Tuple of (X, y) arrays for training
        """
        if target_columns is None:
            target_columns = [0, 1, 2, 3]  # vehicle_count, density, speed, flow_rate
        
        X, y = [], []
        
        for i in range(sequence_length, len(data)):
            # Input sequence
            X.append(data[i-sequence_length:i])
            
            # Target values
            y.append(data[i][target_columns])
        
        return np.array(X), np.array(y)
    
    def evaluate_predictions(self, 
                           true_values: List[PredictionResult],
                           predictions: List[PredictionResult]) -> Dict[str, float]:
        """
        Evaluate prediction accuracy
        
        Args:
            true_values: Actual traffic values
            predictions: Predicted traffic values
            
        Returns:
            Dictionary with evaluation metrics
        """
        if len(true_values) != len(predictions):
            raise ValueError("Lengths of true_values and predictions must match")
        
        # Extract values for evaluation
        true_counts = [v.predicted_vehicle_count for v in true_values]
        pred_counts = [p.predicted_vehicle_count for p in predictions]
        
        true_densities = [v.predicted_density for v in true_values]
        pred_densities = [p.predicted_density for p in predictions]
        
        true_speeds = [v.predicted_speed for v in true_values]
        pred_speeds = [p.predicted_speed for p in predictions]
        
        # Calculate metrics
        def mean_absolute_error(y_true, y_pred):
            return np.mean(np.abs(np.array(y_true) - np.array(y_pred)))
        
        def root_mean_square_error(y_true, y_pred):
            return np.sqrt(np.mean((np.array(y_true) - np.array(y_pred))**2))
        
        def mean_absolute_percentage_error(y_true, y_pred):
            y_true, y_pred = np.array(y_true), np.array(y_pred)
            return np.mean(np.abs((y_true - y_pred) / np.maximum(y_true, 1e-8))) * 100
        
        metrics = {
            'count_mae': mean_absolute_error(true_counts, pred_counts),
            'count_rmse': root_mean_square_error(true_counts, pred_counts),
            'count_mape': mean_absolute_percentage_error(true_counts, pred_counts),
            
            'density_mae': mean_absolute_error(true_densities, pred_densities),
            'density_rmse': root_mean_square_error(true_densities, pred_densities),
            'density_mape': mean_absolute_percentage_error(true_densities, pred_densities),
            
            'speed_mae': mean_absolute_error(true_speeds, pred_speeds),
            'speed_rmse': root_mean_square_error(true_speeds, pred_speeds),
            'speed_mape': mean_absolute_percentage_error(true_speeds, pred_speeds)
        }
        
        return metrics
    
    def should_update_model(self) -> bool:
        """Check if model should be updated based on time and new data"""
        current_time = time.time()
        time_since_update = (current_time - self.last_update_time) / 60  # minutes
        
        return (time_since_update >= self.update_frequency and 
                len(self.historical_data) > 0)
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the prediction model"""
        return {
            'prediction_horizon': self.prediction_horizon,
            'lookback_window': self.lookback_window,
            'update_frequency': self.update_frequency,
            'is_trained': self.is_trained,
            'historical_data_points': len(self.historical_data),
            'last_update_time': self.last_update_time,
            'feature_columns': self.feature_columns
        }