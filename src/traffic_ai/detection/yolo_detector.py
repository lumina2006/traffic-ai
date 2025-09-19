"""
YOLO-based Vehicle Detector

This module implements vehicle detection using YOLO (You Only Look Once) models
from Ultralytics, providing high-performance real-time detection capabilities.
"""

import cv2
import numpy as np
import time
from typing import List, Optional
from ultralytics import YOLO
import torch

from .detector import VehicleDetector, Detection, DetectionResult

class YOLODetector(VehicleDetector):
    """
    YOLO-based vehicle detector using Ultralytics implementation
    
    Supports various YOLO model variants (YOLOv8n, YOLOv8s, YOLOv8m, YOLOv8l, YOLOv8x)
    with optimized performance for vehicle detection tasks.
    """
    
    def __init__(self,
                 model_path: Optional[str] = None,
                 model_variant: str = "yolov8n.pt",
                 confidence_threshold: float = 0.5,
                 iou_threshold: float = 0.4,
                 device: str = "auto",
                 img_size: int = 640):
        """
        Initialize YOLO detector
        
        Args:
            model_path: Path to custom model file (optional)
            model_variant: YOLO model variant to use if no custom path provided
            confidence_threshold: Minimum confidence score for detections
            iou_threshold: IoU threshold for Non-Maximum Suppression
            device: Device for inference ('auto', 'cpu', 'cuda', '0', '1', etc.)
            img_size: Input image size for model inference
        """
        super().__init__(model_path, confidence_threshold, device)
        
        self.model_variant = model_variant
        self.iou_threshold = iou_threshold
        self.img_size = img_size
        
        # Auto-detect device if not specified
        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Load model on initialization
        self.load_model()
    
    def load_model(self) -> bool:
        """
        Load YOLO model for vehicle detection
        
        Returns:
            True if model loaded successfully, False otherwise
        """
        try:
            # Use custom model path if provided, otherwise use variant
            model_source = self.model_path or self.model_variant
            
            print(f"Loading YOLO model: {model_source}")
            self.model = YOLO(model_source)
            
            # Move model to specified device
            if self.device != "cpu":
                self.model.to(self.device)
            
            # Warm up the model with a dummy prediction
            dummy_img = np.zeros((self.img_size, self.img_size, 3), dtype=np.uint8)
            self.model(dummy_img, verbose=False)
            
            self.is_loaded = True
            print(f"✅ YOLO model loaded successfully on {self.device}")
            return True
            
        except Exception as e:
            print(f"❌ Error loading YOLO model: {str(e)}")
            self.is_loaded = False
            return False
    
    def detect_frame(self, frame: np.ndarray) -> DetectionResult:
        """
        Detect vehicles in a single frame using YOLO
        
        Args:
            frame: Input image/frame as numpy array
            
        Returns:
            DetectionResult containing all vehicle detections
        """
        if not self.is_loaded:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        start_time = time.time()
        
        # Run YOLO inference
        results = self.model(
            frame,
            conf=self.confidence_threshold,
            iou=self.iou_threshold,
            imgsz=self.img_size,
            verbose=False,
            device=self.device
        )
        
        # Extract detections
        detections = []
        if results and len(results) > 0:
            result = results[0]  # YOLO returns list, take first result
            
            if result.boxes is not None:
                boxes = result.boxes.xyxy.cpu().numpy()  # x1, y1, x2, y2
                scores = result.boxes.conf.cpu().numpy()
                classes = result.boxes.cls.cpu().numpy().astype(int)
                
                for i in range(len(boxes)):
                    class_id = classes[i]
                    
                    # Filter for vehicle classes only
                    if class_id in self.vehicle_classes:
                        bbox = tuple(map(int, boxes[i]))  # Convert to int tuple
                        confidence = float(scores[i])
                        class_name = self.vehicle_classes[class_id]
                        
                        detection = Detection(
                            bbox=bbox,
                            confidence=confidence,
                            class_id=class_id,
                            class_name=class_name,
                            timestamp=time.time()
                        )
                        detections.append(detection)
        
        processing_time = time.time() - start_time
        
        return DetectionResult(
            detections=detections,
            frame_id=0,  # Will be set by caller
            timestamp=time.time(),
            processing_time=processing_time,
            frame_shape=frame.shape
        )
    
    def detect_batch(self, 
                    frames: List[np.ndarray],
                    batch_size: int = 4) -> List[DetectionResult]:
        """
        Detect vehicles in multiple frames using batch processing
        
        Args:
            frames: List of input frames
            batch_size: Number of frames to process in each batch
            
        Returns:
            List of DetectionResult for each frame
        """
        if not self.is_loaded:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        results = []
        
        # Process frames in batches
        for i in range(0, len(frames), batch_size):
            batch_frames = frames[i:i + batch_size]
            start_time = time.time()
            
            # Run batch inference
            batch_results = self.model(
                batch_frames,
                conf=self.confidence_threshold,
                iou=self.iou_threshold,
                imgsz=self.img_size,
                verbose=False,
                device=self.device
            )
            
            processing_time = time.time() - start_time
            
            # Process each result in the batch
            for j, result in enumerate(batch_results):
                detections = []
                
                if result.boxes is not None:
                    boxes = result.boxes.xyxy.cpu().numpy()
                    scores = result.boxes.conf.cpu().numpy() 
                    classes = result.boxes.cls.cpu().numpy().astype(int)
                    
                    for k in range(len(boxes)):
                        class_id = classes[k]
                        
                        if class_id in self.vehicle_classes:
                            bbox = tuple(map(int, boxes[k]))
                            confidence = float(scores[k])
                            class_name = self.vehicle_classes[class_id]
                            
                            detection = Detection(
                                bbox=bbox,
                                confidence=confidence,
                                class_id=class_id,
                                class_name=class_name,
                                timestamp=time.time()
                            )
                            detections.append(detection)
                
                detection_result = DetectionResult(
                    detections=detections,
                    frame_id=i + j,
                    timestamp=time.time(),
                    processing_time=processing_time / len(batch_frames),
                    frame_shape=batch_frames[j].shape
                )
                results.append(detection_result)
        
        return results
    
    def set_parameters(self, 
                      confidence_threshold: Optional[float] = None,
                      iou_threshold: Optional[float] = None,
                      img_size: Optional[int] = None):
        """
        Update detection parameters
        
        Args:
            confidence_threshold: New confidence threshold
            iou_threshold: New IoU threshold  
            img_size: New input image size
        """
        if confidence_threshold is not None:
            self.confidence_threshold = confidence_threshold
        
        if iou_threshold is not None:
            self.iou_threshold = iou_threshold
            
        if img_size is not None:
            self.img_size = img_size
    
    def get_model_info(self) -> dict:
        """
        Get information about the loaded model
        
        Returns:
            Dictionary with model information
        """
        if not self.is_loaded:
            return {"status": "Model not loaded"}
        
        info = {
            "model_variant": self.model_variant,
            "model_path": self.model_path,
            "device": self.device,
            "confidence_threshold": self.confidence_threshold,
            "iou_threshold": self.iou_threshold,
            "img_size": self.img_size,
            "is_loaded": self.is_loaded
        }
        
        # Add model-specific info if available
        if hasattr(self.model, 'names'):
            info["total_classes"] = len(self.model.names)
            info["vehicle_classes"] = list(self.vehicle_classes.values())
        
        return info