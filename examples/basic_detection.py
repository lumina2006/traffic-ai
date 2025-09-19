#!/usr/bin/env python3
"""
Basic Vehicle Detection Example

This script demonstrates basic vehicle detection using the Traffic AI system.
It can process a video file or use a live camera feed to detect vehicles.

Usage:
    python examples/basic_detection.py --input video.mp4
    python examples/basic_detection.py --camera 0
    python examples/basic_detection.py --rtmp rtmp://localhost/stream
"""

import argparse
import sys
import os
import time
import cv2

# Add the src directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from traffic_ai.detection import YOLODetector
from traffic_ai.utils.config import load_config
from traffic_ai.utils.logger import setup_logger

def main():
    parser = argparse.ArgumentParser(description='Traffic AI - Basic Vehicle Detection')
    parser.add_argument('--input', '-i', type=str, help='Input video file path')
    parser.add_argument('--camera', '-c', type=int, help='Camera index (e.g., 0 for webcam)')
    parser.add_argument('--rtmp', '-r', type=str, help='RTMP stream URL')
    parser.add_argument('--output', '-o', type=str, help='Output video file path')
    parser.add_argument('--config', type=str, default='config/config.yaml',
                       help='Configuration file path')
    parser.add_argument('--model', '-m', type=str, help='YOLO model variant (e.g., yolov8n.pt)')
    parser.add_argument('--confidence', type=float, default=0.5,
                       help='Confidence threshold (0.0-1.0)')
    parser.add_argument('--device', type=str, default='auto',
                       help='Device for inference (auto, cpu, cuda)')
    parser.add_argument('--no-display', action='store_true',
                       help='Disable video display window')
    
    args = parser.parse_args()
    
    # Validate arguments
    if not any([args.input, args.camera is not None, args.rtmp]):
        print("Error: Please specify --input, --camera, or --rtmp")
        sys.exit(1)
    
    # Setup logging
    logger = setup_logger('basic_detection', level='INFO')
    logger.info("Starting Traffic AI Basic Detection")
    
    try:
        # Load configuration
        config = load_config(args.config)
        detection_config = config.get('detection', {})
        
        # Initialize detector
        model_variant = args.model or detection_config.get('model', 'yolov8n.pt')
        confidence = args.confidence or detection_config.get('confidence_threshold', 0.5)
        device = args.device or detection_config.get('device', 'auto')
        
        logger.info(f"Initializing YOLO detector: {model_variant}")
        detector = YOLODetector(
            model_variant=model_variant,
            confidence_threshold=confidence,
            device=device
        )
        
        # Determine video source
        if args.input:
            source = args.input
            source_type = "file"
            logger.info(f"Processing video file: {source}")
        elif args.camera is not None:
            source = args.camera
            source_type = "camera"
            logger.info(f"Using camera: {source}")
        else:  # args.rtmp
            source = args.rtmp
            source_type = "stream"
            logger.info(f"Connecting to RTMP stream: {source}")
        
        # Process video
        if source_type == "file":
            # Process video file
            results = detector.detect_video(
                video_path=source,
                output_path=args.output,
                show_progress=True
            )
            
            # Display statistics
            stats = detector.get_stats(results)
            logger.info("Detection completed!")
            logger.info(f"Total frames processed: {stats.get('total_frames', 0)}")
            logger.info(f"Total detections: {stats.get('total_detections', 0)}")
            logger.info(f"Average detections per frame: {stats.get('avg_detections_per_frame', 0):.2f}")
            logger.info(f"Average processing time: {stats.get('avg_processing_time', 0):.3f}s")
            logger.info(f"Average FPS: {stats.get('fps', 0):.1f}")
            
            # Display class distribution
            class_dist = stats.get('class_distribution', {})
            if class_dist:
                logger.info("Vehicle class distribution:")
                for vehicle_class, count in class_dist.items():
                    logger.info(f"  {vehicle_class}: {count}")
        
        else:
            # Real-time processing
            display = not args.no_display
            logger.info(f"Starting real-time detection (display: {display})")
            logger.info("Press 'q' to quit")
            
            detector.detect_realtime(source=source, display=display)
            
        logger.info("Detection finished successfully!")
    
    except KeyboardInterrupt:
        logger.info("Detection interrupted by user")
    except Exception as e:
        logger.error(f"Error during detection: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()