"""
Multi-Object Tracking Module

This module provides advanced vehicle tracking capabilities for maintaining
consistent identities of vehicles across video frames in traffic monitoring.
"""

from .tracker import VehicleTracker
from .sort_tracker import SORTTracker
from .deep_sort_tracker import DeepSORTTracker

__all__ = [
    "VehicleTracker",
    "SORTTracker",
    "DeepSORTTracker"
]