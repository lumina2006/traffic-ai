"""
Traffic Prediction Module

This module provides AI-powered traffic prediction capabilities including
flow forecasting, congestion prediction, and pattern recognition for
intelligent traffic management systems.
"""

from .predictor import TrafficPredictor
from .lstm_predictor import LSTMPredictor
from .models import load_prediction_model, create_prediction_model

__all__ = [
    "TrafficPredictor",
    "LSTMPredictor",
    "load_prediction_model",
    "create_prediction_model"
]