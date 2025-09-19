"""
Traffic AI - Intelligent Traffic Analysis System

A comprehensive AI-powered traffic monitoring and analysis system that provides
real-time vehicle detection, tracking, flow analysis, and predictive insights.

Author: Traffic AI Team
Version: 1.0.0
License: MIT
"""

__version__ = "1.0.0"
__author__ = "Traffic AI Team"
__email__ = "contact@traffic-ai.com"

from .detection import VehicleDetector
from .tracking import VehicleTracker
from .analysis import TrafficAnalyzer
from .prediction import TrafficPredictor

__all__ = [
    "VehicleDetector",
    "VehicleTracker", 
    "TrafficAnalyzer",
    "TrafficPredictor"
]