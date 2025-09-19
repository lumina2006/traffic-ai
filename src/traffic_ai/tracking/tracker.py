"""
Base Vehicle Tracker Class

This module provides the base class for all vehicle tracking implementations
in the Traffic AI system.
"""

import numpy as np
from abc import ABC, abstractmethod
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
import time

from ..detection.detector import Detection, DetectionResult

@dataclass
class Track:
    """Represents a tracked vehicle"""
    track_id: int
    bbox: Tuple[int, int, int, int]  # (x1, y1, x2, y2)
    confidence: float
    class_id: int
    class_name: str
    age: int  # Number of frames since first detection
    hits: int  # Number of successful detections
    time_since_update: int  # Frames since last update
    velocity: Optional[Tuple[float, float]] = None  # (vx, vy)
    trajectory: Optional[List[Tuple[float, float]]] = None  # Path history
    
    def __post_init__(self):
        if self.trajectory is None:
            self.trajectory = []

@dataclass  
class TrackingResult:
    """Container for tracking results"""
    tracks: List[Track]
    frame_id: int
    timestamp: float
    processing_time: float
    total_tracks: int  # Total number of tracks created so far

class VehicleTracker(ABC):
    """
    Abstract base class for vehicle trackers
    
    This class defines the interface that all vehicle trackers must implement.
    It provides common functionality for maintaining vehicle identities across frames.
    """
    
    def __init__(self,
                 max_age: int = 30,
                 min_hits: int = 3,
                 iou_threshold: float = 0.3):
        """
        Initialize the vehicle tracker
        
        Args:
            max_age: Maximum number of frames to keep a track alive without updates
            min_hits: Minimum number of hits before a track is confirmed
            iou_threshold: IoU threshold for data association
        """
        self.max_age = max_age
        self.min_hits = min_hits  
        self.iou_threshold = iou_threshold
        
        self.tracks = []
        self.track_id_counter = 0
        self.frame_count = 0
        
    @abstractmethod
    def update(self, detections: List[Detection]) -> TrackingResult:
        """
        Update tracker with new detections
        
        Args:
            detections: List of detections from current frame
            
        Returns:
            TrackingResult containing active tracks
        """
        pass
    
    def reset(self):
        """Reset tracker state"""
        self.tracks = []
        self.track_id_counter = 0
        self.frame_count = 0
    
    def _get_next_track_id(self) -> int:
        """Get next available track ID"""
        self.track_id_counter += 1
        return self.track_id_counter
    
    def _calculate_iou(self, 
                      bbox1: Tuple[int, int, int, int],
                      bbox2: Tuple[int, int, int, int]) -> float:
        """
        Calculate Intersection over Union (IoU) between two bounding boxes
        
        Args:
            bbox1: First bounding box (x1, y1, x2, y2)
            bbox2: Second bounding box (x1, y1, x2, y2)
            
        Returns:
            IoU score between 0 and 1
        """
        x1_1, y1_1, x2_1, y2_1 = bbox1
        x1_2, y1_2, x2_2, y2_2 = bbox2
        
        # Calculate intersection area
        x1_i = max(x1_1, x1_2)
        y1_i = max(y1_1, y1_2)
        x2_i = min(x2_1, x2_2)
        y2_i = min(y2_1, y2_2)
        
        if x2_i <= x1_i or y2_i <= y1_i:
            return 0.0
            
        intersection = (x2_i - x1_i) * (y2_i - y1_i)
        
        # Calculate union area
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        union = area1 + area2 - intersection
        
        if union == 0:
            return 0.0
            
        return intersection / union
    
    def _calculate_velocity(self, 
                          current_bbox: Tuple[int, int, int, int],
                          previous_bbox: Tuple[int, int, int, int]) -> Tuple[float, float]:
        """
        Calculate velocity between two bounding boxes
        
        Args:
            current_bbox: Current bounding box
            previous_bbox: Previous bounding box
            
        Returns:
            Velocity as (vx, vy) tuple
        """
        # Calculate center points
        cx1 = (current_bbox[0] + current_bbox[2]) / 2
        cy1 = (current_bbox[1] + current_bbox[3]) / 2
        
        cx2 = (previous_bbox[0] + previous_bbox[2]) / 2
        cy2 = (previous_bbox[1] + previous_bbox[3]) / 2
        
        # Calculate velocity (pixels per frame)
        vx = cx1 - cx2
        vy = cy1 - cy2
        
        return (vx, vy)
    
    def get_track_statistics(self) -> Dict:
        """
        Get statistics about current tracks
        
        Returns:
            Dictionary with tracking statistics
        """
        if not self.tracks:
            return {
                'active_tracks': 0,
                'confirmed_tracks': 0,
                'total_tracks_created': self.track_id_counter
            }
        
        confirmed_tracks = [t for t in self.tracks if t.hits >= self.min_hits]
        
        # Calculate average track age and hits
        avg_age = np.mean([t.age for t in self.tracks])
        avg_hits = np.mean([t.hits for t in self.tracks])
        
        # Count tracks by vehicle class
        class_distribution = {}
        for track in confirmed_tracks:
            class_name = track.class_name
            class_distribution[class_name] = class_distribution.get(class_name, 0) + 1
        
        return {
            'active_tracks': len(self.tracks),
            'confirmed_tracks': len(confirmed_tracks),
            'total_tracks_created': self.track_id_counter,
            'average_track_age': avg_age,
            'average_track_hits': avg_hits,
            'class_distribution': class_distribution,
            'frame_count': self.frame_count
        }
    
    def get_track_by_id(self, track_id: int) -> Optional[Track]:
        """
        Get track by ID
        
        Args:
            track_id: Track ID to search for
            
        Returns:
            Track object if found, None otherwise
        """
        for track in self.tracks:
            if track.track_id == track_id:
                return track
        return None
    
    def export_tracks(self) -> List[Dict]:
        """
        Export all tracks to dictionary format
        
        Returns:
            List of track dictionaries
        """
        exported_tracks = []
        
        for track in self.tracks:
            track_dict = {
                'track_id': track.track_id,
                'bbox': track.bbox,
                'confidence': track.confidence,
                'class_id': track.class_id,
                'class_name': track.class_name,
                'age': track.age,
                'hits': track.hits,
                'time_since_update': track.time_since_update,
                'velocity': track.velocity,
                'trajectory': track.trajectory
            }
            exported_tracks.append(track_dict)
        
        return exported_tracks