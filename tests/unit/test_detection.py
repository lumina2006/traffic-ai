"""
Unit tests for vehicle detection module
"""

import pytest
import numpy as np
import sys
import os

# Add src to path for testing
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from traffic_ai.detection.detector import Detection, DetectionResult, VehicleDetector
from traffic_ai.detection.yolo_detector import YOLODetector


class TestDetection:
    """Test cases for Detection dataclass"""
    
    def test_detection_creation(self):
        """Test Detection object creation"""
        detection = Detection(
            bbox=(100, 100, 200, 200),
            confidence=0.85,
            class_id=2,
            class_name="car",
            timestamp=1234567890.0
        )
        
        assert detection.bbox == (100, 100, 200, 200)
        assert detection.confidence == 0.85
        assert detection.class_id == 2
        assert detection.class_name == "car"
        assert detection.timestamp == 1234567890.0


class TestDetectionResult:
    """Test cases for DetectionResult dataclass"""
    
    def test_detection_result_creation(self):
        """Test DetectionResult object creation"""
        detections = [
            Detection((100, 100, 200, 200), 0.85, 2, "car", 1234567890.0),
            Detection((300, 150, 400, 250), 0.75, 7, "truck", 1234567890.0)
        ]
        
        result = DetectionResult(
            detections=detections,
            frame_id=42,
            timestamp=1234567890.0,
            processing_time=0.05,
            frame_shape=(480, 640, 3)
        )
        
        assert len(result.detections) == 2
        assert result.frame_id == 42
        assert result.timestamp == 1234567890.0
        assert result.processing_time == 0.05
        assert result.frame_shape == (480, 640, 3)


class TestVehicleDetector:
    """Test cases for VehicleDetector base class"""
    
    def test_detector_initialization(self):
        """Test detector initialization"""
        # Create mock detector since VehicleDetector is abstract
        class MockDetector(VehicleDetector):
            def load_model(self):
                return True
            
            def detect_frame(self, frame):
                return DetectionResult([], 0, 0.0, 0.0, frame.shape)
        
        detector = MockDetector(
            confidence_threshold=0.6,
            device="cpu"
        )
        
        assert detector.confidence_threshold == 0.6
        assert detector.device == "cpu"
        assert not detector.is_loaded
        assert 2 in detector.vehicle_classes  # car
        assert detector.vehicle_classes[2] == "car"
    
    def test_get_class_color(self):
        """Test vehicle class color mapping"""
        class MockDetector(VehicleDetector):
            def load_model(self):
                return True
            
            def detect_frame(self, frame):
                return DetectionResult([], 0, 0.0, 0.0, frame.shape)
        
        detector = MockDetector()
        
        # Test known vehicle classes
        assert detector._get_class_color(2) == (0, 255, 0)  # car - green
        assert detector._get_class_color(3) == (255, 0, 0)  # motorcycle - blue
        assert detector._get_class_color(5) == (0, 0, 255)  # bus - red
        assert detector._get_class_color(7) == (255, 255, 0)  # truck - cyan
        
        # Test unknown class
        assert detector._get_class_color(99) == (128, 128, 128)  # default gray


class TestYOLODetector:
    """Test cases for YOLO detector (requires mocking YOLO model)"""
    
    def test_yolo_detector_initialization(self):
        """Test YOLO detector initialization"""
        # This test would require mocking the YOLO model
        # For now, just test the initialization parameters
        detector = YOLODetector(
            model_variant="yolov8n.pt",
            confidence_threshold=0.7,
            iou_threshold=0.5,
            device="cpu",
            img_size=640
        )
        
        assert detector.model_variant == "yolov8n.pt"
        assert detector.confidence_threshold == 0.7
        assert detector.iou_threshold == 0.5
        assert detector.img_size == 640
    
    def test_set_parameters(self):
        """Test parameter updating"""
        detector = YOLODetector(device="cpu")
        
        detector.set_parameters(
            confidence_threshold=0.8,
            iou_threshold=0.3,
            img_size=512
        )
        
        assert detector.confidence_threshold == 0.8
        assert detector.iou_threshold == 0.3
        assert detector.img_size == 512
    
    def test_get_model_info(self):
        """Test model information retrieval"""
        detector = YOLODetector(device="cpu")
        
        info = detector.get_model_info()
        
        assert "model_variant" in info
        assert "device" in info
        assert "confidence_threshold" in info
        assert "is_loaded" in info
        assert info["model_variant"] == "yolov8n.pt"
        assert info["device"] == "cpu"


if __name__ == "__main__":
    pytest.main([__file__])