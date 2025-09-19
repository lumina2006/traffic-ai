"""
Traffic Analyzer

This module provides comprehensive traffic analysis capabilities for processing
detection and tracking results to extract meaningful traffic insights.
"""

import numpy as np
import cv2
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass
from collections import defaultdict, deque
import time

from ..detection.detector import DetectionResult
from ..tracking.tracker import TrackingResult, Track

@dataclass
class TrafficZone:
    """Defines a region of interest for traffic analysis"""
    name: str
    polygon: List[Tuple[int, int]]  # List of (x, y) points defining the zone
    zone_type: str = "counting"  # "counting", "speed", "density"
    
@dataclass
class TrafficMetrics:
    """Container for traffic analysis results"""
    frame_id: int
    timestamp: float
    vehicle_count: int
    density: float  # vehicles per unit area
    average_speed: float  # pixels per frame
    flow_rate: float  # vehicles per minute
    congestion_level: str  # "low", "medium", "high"
    zone_counts: Dict[str, int]  # vehicle counts per zone
    class_distribution: Dict[str, int]  # vehicle counts per class
    
class TrafficAnalyzer:
    """
    Comprehensive traffic analyzer for processing vehicle detection and tracking data
    
    Provides various traffic analysis capabilities including flow measurement,
    density calculation, speed estimation, and congestion detection.
    """
    
    def __init__(self,
                 frame_buffer_size: int = 300,  # 10 seconds at 30 FPS
                 fps: float = 30.0):
        """
        Initialize traffic analyzer
        
        Args:
            frame_buffer_size: Number of frames to keep in history buffer
            fps: Video frame rate for temporal analysis
        """
        self.frame_buffer_size = frame_buffer_size
        self.fps = fps
        
        # Traffic zones for region-specific analysis
        self.zones: List[TrafficZone] = []
        
        # Historical data buffers
        self.detection_history = deque(maxlen=frame_buffer_size)
        self.tracking_history = deque(maxlen=frame_buffer_size)
        self.metrics_history = deque(maxlen=frame_buffer_size)
        
        # Counting lines for flow measurement
        self.counting_lines: List[Dict] = []
        
        # Speed measurement zones
        self.speed_zones: List[Dict] = []
        
        # Vehicle counting state
        self.vehicle_counts = defaultdict(int)
        self.crossed_vehicles = set()
        
    def add_zone(self, zone: TrafficZone):
        """Add a traffic analysis zone"""
        self.zones.append(zone)
    
    def add_counting_line(self, 
                         name: str,
                         start_point: Tuple[int, int],
                         end_point: Tuple[int, int],
                         direction: str = "both"):
        """
        Add a counting line for vehicle flow measurement
        
        Args:
            name: Unique name for the counting line
            start_point: Starting point (x, y) of the line
            end_point: Ending point (x, y) of the line
            direction: Count direction ("up", "down", "left", "right", "both")
        """
        counting_line = {
            'name': name,
            'start': start_point,
            'end': end_point,
            'direction': direction,
            'count': 0,
            'crossed_tracks': set()
        }
        self.counting_lines.append(counting_line)
    
    def analyze_frame(self, 
                     detection_result: DetectionResult,
                     tracking_result: Optional[TrackingResult] = None,
                     frame: Optional[np.ndarray] = None) -> TrafficMetrics:
        """
        Analyze a single frame for traffic metrics
        
        Args:
            detection_result: Detection results from the frame
            tracking_result: Optional tracking results
            frame: Optional frame image for visualization
            
        Returns:
            TrafficMetrics containing analysis results
        """
        # Store in history buffers
        self.detection_history.append(detection_result)
        if tracking_result:
            self.tracking_history.append(tracking_result)
        
        # Calculate basic metrics
        vehicle_count = len(detection_result.detections)
        
        # Calculate density (vehicles per 1000 square pixels)
        frame_area = detection_result.frame_shape[0] * detection_result.frame_shape[1]
        density = (vehicle_count * 1000) / frame_area if frame_area > 0 else 0
        
        # Calculate class distribution
        class_distribution = defaultdict(int)
        for detection in detection_result.detections:
            class_distribution[detection.class_name] += 1
        
        # Calculate zone-specific counts
        zone_counts = {}
        for zone in self.zones:
            zone_counts[zone.name] = self._count_vehicles_in_zone(
                detection_result.detections, zone)
        
        # Calculate average speed from tracking data
        average_speed = 0.0
        if tracking_result and tracking_result.tracks:
            speeds = []
            for track in tracking_result.tracks:
                if track.velocity:
                    speed = np.sqrt(track.velocity[0]**2 + track.velocity[1]**2)
                    speeds.append(speed)
            average_speed = np.mean(speeds) if speeds else 0.0
        
        # Calculate flow rate (vehicles per minute)
        flow_rate = self._calculate_flow_rate()
        
        # Update counting lines
        if tracking_result:
            self._update_counting_lines(tracking_result.tracks)
        
        # Determine congestion level
        congestion_level = self._determine_congestion_level(
            density, average_speed, vehicle_count)
        
        # Create metrics object
        metrics = TrafficMetrics(
            frame_id=detection_result.frame_id,
            timestamp=detection_result.timestamp,
            vehicle_count=vehicle_count,
            density=density,
            average_speed=average_speed,
            flow_rate=flow_rate,
            congestion_level=congestion_level,
            zone_counts=zone_counts,
            class_distribution=dict(class_distribution)
        )
        
        # Store metrics in history
        self.metrics_history.append(metrics)
        
        return metrics
    
    def _count_vehicles_in_zone(self, 
                               detections: List,
                               zone: TrafficZone) -> int:
        """Count vehicles within a specific zone"""
        count = 0
        
        for detection in detections:
            # Get vehicle center point
            x1, y1, x2, y2 = detection.bbox
            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2
            
            # Check if point is inside zone polygon
            if self._point_in_polygon((center_x, center_y), zone.polygon):
                count += 1
        
        return count
    
    def _point_in_polygon(self, 
                         point: Tuple[int, int], 
                         polygon: List[Tuple[int, int]]) -> bool:
        """Check if a point is inside a polygon using ray casting algorithm"""
        x, y = point
        n = len(polygon)
        inside = False
        
        p1x, p1y = polygon[0]
        for i in range(1, n + 1):
            p2x, p2y = polygon[i % n]
            if y > min(p1y, p2y):
                if y <= max(p1y, p2y):
                    if x <= max(p1x, p2x):
                        if p1y != p2y:
                            xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                        if p1x == p2x or x <= xinters:
                            inside = not inside
            p1x, p1y = p2x, p2y
        
        return inside
    
    def _calculate_flow_rate(self) -> float:
        """Calculate vehicle flow rate in vehicles per minute"""
        if len(self.metrics_history) < 2:
            return 0.0
        
        # Get vehicle counts from recent history (last 60 frames ≈ 2 seconds at 30fps)
        recent_frames = min(60, len(self.metrics_history))
        recent_counts = [m.vehicle_count for m in list(self.metrics_history)[-recent_frames:]]
        
        if not recent_counts:
            return 0.0
        
        # Calculate average vehicles and convert to per-minute rate
        avg_vehicles = np.mean(recent_counts)
        flow_rate = avg_vehicles * self.fps  # vehicles per second
        flow_rate = flow_rate * 60  # vehicles per minute
        
        return flow_rate
    
    def _update_counting_lines(self, tracks: List[Track]):
        """Update vehicle counts for counting lines"""
        for line in self.counting_lines:
            for track in tracks:
                if track.track_id not in line['crossed_tracks']:
                    if self._track_crossed_line(track, line):
                        line['count'] += 1
                        line['crossed_tracks'].add(track.track_id)
    
    def _track_crossed_line(self, track: Track, line: Dict) -> bool:
        """Check if a track has crossed a counting line"""
        if len(track.trajectory) < 2:
            return False
        
        # Get the last two positions
        prev_pos = track.trajectory[-2]
        curr_pos = track.trajectory[-1]
        
        # Check if the line segment crosses the counting line
        return self._line_intersection(
            prev_pos, curr_pos,
            line['start'], line['end']
        )
    
    def _line_intersection(self, 
                          p1: Tuple[float, float], p2: Tuple[float, float],
                          p3: Tuple[int, int], p4: Tuple[int, int]) -> bool:
        """Check if two line segments intersect"""
        x1, y1 = p1
        x2, y2 = p2  
        x3, y3 = p3
        x4, y4 = p4
        
        denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
        if abs(denom) < 1e-10:
            return False
        
        t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denom
        u = -((x1 - x2) * (y1 - y3) - (y1 - y2) * (x1 - x3)) / denom
        
        return 0 <= t <= 1 and 0 <= u <= 1
    
    def _determine_congestion_level(self, 
                                   density: float,
                                   speed: float,
                                   count: int) -> str:
        """Determine traffic congestion level based on metrics"""
        # Define thresholds (these can be calibrated based on specific scenarios)
        high_density_threshold = 0.5  # vehicles per 1000 pixels
        medium_density_threshold = 0.2
        
        low_speed_threshold = 2.0  # pixels per frame
        medium_speed_threshold = 5.0
        
        # Congestion logic
        if density > high_density_threshold or speed < low_speed_threshold:
            return "high"
        elif density > medium_density_threshold or speed < medium_speed_threshold:
            return "medium"
        else:
            return "low"
    
    def get_summary_statistics(self, time_window: int = 300) -> Dict[str, Any]:
        """
        Get summary statistics for a time window
        
        Args:
            time_window: Number of recent frames to analyze
            
        Returns:
            Dictionary with summary statistics
        """
        if not self.metrics_history:
            return {}
        
        # Get recent metrics
        recent_metrics = list(self.metrics_history)[-time_window:]
        
        # Calculate aggregated statistics
        avg_vehicle_count = np.mean([m.vehicle_count for m in recent_metrics])
        avg_density = np.mean([m.density for m in recent_metrics])
        avg_speed = np.mean([m.average_speed for m in recent_metrics])
        avg_flow_rate = np.mean([m.flow_rate for m in recent_metrics])
        
        # Congestion level distribution
        congestion_levels = [m.congestion_level for m in recent_metrics]
        congestion_distribution = {
            level: congestion_levels.count(level) / len(congestion_levels)
            for level in set(congestion_levels)
        }
        
        # Peak traffic detection
        max_count = max([m.vehicle_count for m in recent_metrics])
        min_count = min([m.vehicle_count for m in recent_metrics])
        
        return {
            'time_window_frames': len(recent_metrics),
            'average_vehicle_count': avg_vehicle_count,
            'average_density': avg_density,
            'average_speed': avg_speed,
            'average_flow_rate': avg_flow_rate,
            'congestion_distribution': congestion_distribution,
            'peak_vehicle_count': max_count,
            'min_vehicle_count': min_count,
            'counting_line_totals': {line['name']: line['count'] for line in self.counting_lines}
        }
    
    def visualize_analysis(self, 
                          frame: np.ndarray,
                          metrics: TrafficMetrics) -> np.ndarray:
        """
        Add analysis visualization to frame
        
        Args:
            frame: Input frame
            metrics: Traffic metrics to visualize
            
        Returns:
            Annotated frame with analysis overlay
        """
        annotated = frame.copy()
        
        # Draw zones
        for zone in self.zones:
            pts = np.array(zone.polygon, np.int32)
            cv2.polylines(annotated, [pts], True, (0, 255, 255), 2)
            
            # Add zone label
            if zone.polygon:
                label_pos = zone.polygon[0]
                cv2.putText(annotated, f"{zone.name}: {metrics.zone_counts.get(zone.name, 0)}",
                           label_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        
        # Draw counting lines
        for line in self.counting_lines:
            cv2.line(annotated, line['start'], line['end'], (255, 0, 255), 3)
            cv2.putText(annotated, f"{line['name']}: {line['count']}",
                       (line['start'][0], line['start'][1] - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
        
        # Add metrics overlay
        y_offset = 30
        metrics_text = [
            f"Vehicles: {metrics.vehicle_count}",
            f"Density: {metrics.density:.3f}",
            f"Speed: {metrics.average_speed:.1f}px/f",
            f"Flow: {metrics.flow_rate:.1f}v/min",
            f"Congestion: {metrics.congestion_level}"
        ]
        
        for text in metrics_text:
            cv2.putText(annotated, text, (10, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            y_offset += 25
        
        return annotated