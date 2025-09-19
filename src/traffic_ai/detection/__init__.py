"""
Vehicle Detection Module

This module provides advanced vehicle detection capabilities using state-of-the-art
YOLO models for real-time traffic monitoring applications.
"""

from .detector import VehicleDetector
from .yolo_detector import YOLODetector
from .models import load_model, get_available_models

__all__ = [
    "VehicleDetector",
    "YOLODetector", 
    "load_model",
    "get_available_models"
]