"""
Base Vehicle Detector Class

This module provides the base class for all vehicle detection implementations
in the Traffic AI system.
"""

import cv2
import numpy as np
from abc import ABC, abstractmethod
from typing import List, Dict, Optional, Tuple, Union
from dataclasses import dataclass
import time

@dataclass
class Detection:
    """Represents a single vehicle detection"""
    bbox: Tuple[int, int, int, int]  # (x1, y1, x2, y2)
    confidence: float
    class_id: int
    class_name: str
    timestamp: float

@dataclass
class DetectionResult:
    """Container for detection results"""
    detections: List[Detection]
    frame_id: int
    timestamp: float
    processing_time: float
    frame_shape: Tuple[int, int, int]

class VehicleDetector(ABC):
    """
    Abstract base class for vehicle detectors
    
    This class defines the interface that all vehicle detectors must implement.
    It provides common functionality and ensures consistent behavior across
    different detection algorithms.
    """
    
    def __init__(self, 
                 model_path: Optional[str] = None,
                 confidence_threshold: float = 0.5,
                 device: str = "cpu"):
        """
        Initialize the vehicle detector
        
        Args:
            model_path: Path to the model file
            confidence_threshold: Minimum confidence score for detections
            device: Device to run inference on ('cpu', 'cuda', 'mps')
        """
        self.model_path = model_path
        self.confidence_threshold = confidence_threshold
        self.device = device
        self.model = None
        self.is_loaded = False
        
        # Vehicle classes (COCO dataset classes)
        self.vehicle_classes = {
            2: 'car',
            3: 'motorcycle', 
            5: 'bus',
            7: 'truck'
        }
        
    @abstractmethod
    def load_model(self) -> bool:
        """Load the detection model"""
        pass
        
    @abstractmethod
    def detect_frame(self, frame: np.ndarray) -> DetectionResult:
        """
        Detect vehicles in a single frame
        
        Args:
            frame: Input image/frame as numpy array
            
        Returns:
            DetectionResult containing all detections
        """
        pass
        
    def detect_video(self, 
                    video_path: str,
                    output_path: Optional[str] = None,
                    show_progress: bool = True) -> List[DetectionResult]:
        """
        Detect vehicles in a video file
        
        Args:
            video_path: Path to input video
            output_path: Optional path to save annotated video
            show_progress: Whether to show processing progress
            
        Returns:
            List of DetectionResult for each frame
        """
        if not self.is_loaded:
            if not self.load_model():
                raise RuntimeError("Failed to load detection model")
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video file: {video_path}")
            
        results = []
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        # Setup video writer if output path is provided
        writer = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            writer = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))
        
        frame_id = 0
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                    
                # Detect vehicles in current frame
                detection_result = self.detect_frame(frame)
                detection_result.frame_id = frame_id
                results.append(detection_result)
                
                # Annotate frame if saving output
                if writer:
                    annotated_frame = self.annotate_frame(frame, detection_result)
                    writer.write(annotated_frame)
                
                # Show progress
                if show_progress and frame_id % 30 == 0:
                    progress = (frame_id / frame_count) * 100
                    print(f"Processing: {progress:.1f}% ({frame_id}/{frame_count})")
                
                frame_id += 1
                
        finally:
            cap.release()
            if writer:
                writer.release()
                
        return results
    
    def detect_realtime(self, 
                       source: Union[str, int] = 0,
                       display: bool = True) -> None:
        """
        Real-time vehicle detection from camera or stream
        
        Args:
            source: Camera index or stream URL
            display: Whether to display the video feed
        """
        if not self.is_loaded:
            if not self.load_model():
                raise RuntimeError("Failed to load detection model")
        
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video source: {source}")
        
        frame_id = 0
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Detect vehicles
                result = self.detect_frame(frame)
                result.frame_id = frame_id
                
                # Annotate and display frame
                if display:
                    annotated_frame = self.annotate_frame(frame, result)
                    cv2.imshow('Traffic AI - Vehicle Detection', annotated_frame)
                    
                    # Exit on 'q' key press
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                
                frame_id += 1
                
        except KeyboardInterrupt:
            print("\nStopping real-time detection...")
        finally:
            cap.release()
            if display:
                cv2.destroyAllWindows()
    
    def annotate_frame(self, 
                      frame: np.ndarray, 
                      result: DetectionResult) -> np.ndarray:
        """
        Annotate frame with detection results
        
        Args:
            frame: Original frame
            result: Detection results
            
        Returns:
            Annotated frame
        """
        annotated = frame.copy()
        
        for detection in result.detections:
            x1, y1, x2, y2 = detection.bbox
            
            # Draw bounding box
            color = self._get_class_color(detection.class_id)
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            
            # Draw label
            label = f"{detection.class_name}: {detection.confidence:.2f}"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
            cv2.rectangle(annotated, (x1, y1 - label_size[1] - 10), 
                         (x1 + label_size[0], y1), color, -1)
            cv2.putText(annotated, label, (x1, y1 - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        # Add stats
        stats_text = f"Vehicles: {len(result.detections)} | Time: {result.processing_time:.3f}s"
        cv2.putText(annotated, stats_text, (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        return annotated
    
    def _get_class_color(self, class_id: int) -> Tuple[int, int, int]:
        """Get color for vehicle class"""
        colors = {
            2: (0, 255, 0),    # car - green
            3: (255, 0, 0),    # motorcycle - blue  
            5: (0, 0, 255),    # bus - red
            7: (255, 255, 0)   # truck - cyan
        }
        return colors.get(class_id, (128, 128, 128))
    
    def get_stats(self, results: List[DetectionResult]) -> Dict:
        """
        Calculate detection statistics
        
        Args:
            results: List of detection results
            
        Returns:
            Dictionary with various statistics
        """
        if not results:
            return {}
        
        total_detections = sum(len(r.detections) for r in results)
        avg_processing_time = np.mean([r.processing_time for r in results])
        
        # Count detections by class
        class_counts = {}
        for result in results:
            for detection in result.detections:
                class_name = detection.class_name
                class_counts[class_name] = class_counts.get(class_name, 0) + 1
        
        return {
            'total_frames': len(results),
            'total_detections': total_detections,
            'avg_detections_per_frame': total_detections / len(results),
            'avg_processing_time': avg_processing_time,
            'fps': 1.0 / avg_processing_time if avg_processing_time > 0 else 0,
            'class_distribution': class_counts
        }