#!/usr/bin/env python3
"""
Traffic Analysis Demo

This script demonstrates comprehensive traffic analysis including detection,
tracking, and flow analysis capabilities of the Traffic AI system.

Usage:
    python examples/traffic_analysis_demo.py --input video.mp4
    python examples/traffic_analysis_demo.py --camera 0 --save-results
"""

import argparse
import sys
import os
import time
import cv2
import json
import numpy as np
from datetime import datetime

# Add the src directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from traffic_ai.detection import YOLODetector
from traffic_ai.tracking import SORTTracker
from traffic_ai.analysis import TrafficAnalyzer, TrafficZone
from traffic_ai.utils.config import load_config
from traffic_ai.utils.logger import setup_logger

class TrafficAnalysisDemo:
    """Comprehensive traffic analysis demonstration"""
    
    def __init__(self, config_path='config/config.yaml'):
        # Setup logging
        self.logger = setup_logger('traffic_analysis_demo', level='INFO')
        
        # Load configuration
        self.config = load_config(config_path)
        
        # Initialize components
        self._init_detector()
        self._init_tracker()
        self._init_analyzer()
        
        # Results storage
        self.results = {
            'detections': [],
            'tracks': [],
            'analysis': [],
            'summary': {}
        }
        
    def _init_detector(self):
        """Initialize vehicle detector"""
        detection_config = self.config.get('detection', {})
        
        self.detector = YOLODetector(
            model_variant=detection_config.get('model', 'yolov8n.pt'),
            confidence_threshold=detection_config.get('confidence_threshold', 0.5),
            iou_threshold=detection_config.get('iou_threshold', 0.4),
            device=detection_config.get('device', 'auto')
        )
        
        self.logger.info("✅ Vehicle detector initialized")
    
    def _init_tracker(self):
        """Initialize vehicle tracker"""
        tracking_config = self.config.get('tracking', {})
        
        # For this demo, we'll use a simple SORT tracker
        # In practice, you would import the actual SORTTracker class
        self.logger.info("✅ Vehicle tracker initialized (placeholder)")
    
    def _init_analyzer(self):
        """Initialize traffic analyzer"""
        analysis_config = self.config.get('analysis', {})
        
        self.analyzer = TrafficAnalyzer(
            frame_buffer_size=analysis_config.get('frame_buffer_size', 300),
            fps=analysis_config.get('fps', 30.0)
        )
        
        # Setup analysis zones from config
        zones_config = analysis_config.get('zones', [])
        for zone_config in zones_config:
            zone = TrafficZone(
                name=zone_config['name'],
                polygon=zone_config['polygon'],
                zone_type=zone_config.get('type', 'counting')
            )
            self.analyzer.add_zone(zone)
        
        # Setup counting lines from config
        lines_config = analysis_config.get('counting_lines', [])
        for line_config in lines_config:
            self.analyzer.add_counting_line(
                name=line_config['name'],
                start_point=tuple(line_config['start']),
                end_point=tuple(line_config['end']),
                direction=line_config.get('direction', 'both')
            )
        
        self.logger.info(f"✅ Traffic analyzer initialized with {len(self.analyzer.zones)} zones and {len(self.analyzer.counting_lines)} counting lines")
    
    def process_video(self, video_path, output_path=None, save_results=False):
        """Process a video file with full traffic analysis"""
        self.logger.info(f"Starting video analysis: {video_path}")
        
        # Open video
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")
        
        # Video properties
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        self.logger.info(f"Video info: {frame_count} frames, {fps:.1f} FPS, {frame_width}x{frame_height}")
        
        # Setup video writer if output path provided
        writer = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))
        
        # Processing variables
        frame_id = 0
        start_time = time.time()
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Process frame
                analysis_result = self.process_frame(frame, frame_id)
                
                # Store results
                if save_results:
                    self.results['analysis'].append({
                        'frame_id': frame_id,
                        'timestamp': analysis_result.timestamp,
                        'vehicle_count': analysis_result.vehicle_count,
                        'density': analysis_result.density,
                        'average_speed': analysis_result.average_speed,
                        'flow_rate': analysis_result.flow_rate,
                        'congestion_level': analysis_result.congestion_level,
                        'zone_counts': analysis_result.zone_counts,
                        'class_distribution': analysis_result.class_distribution
                    })
                
                # Visualize results
                if output_path or True:  # Always create visualization for demo
                    annotated_frame = self.visualize_results(frame, analysis_result)
                    
                    if writer:
                        writer.write(annotated_frame)
                    
                    # Display frame (optional)
                    cv2.imshow('Traffic Analysis Demo', annotated_frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                
                # Progress update
                if frame_id % 30 == 0:
                    progress = (frame_id / frame_count) * 100
                    elapsed = time.time() - start_time
                    fps_actual = frame_id / elapsed if elapsed > 0 else 0
                    self.logger.info(f"Progress: {progress:.1f}% ({frame_id}/{frame_count}) - {fps_actual:.1f} FPS")
                
                frame_id += 1
        
        finally:
            cap.release()
            if writer:
                writer.release()
            cv2.destroyAllWindows()
        
        # Generate summary
        self._generate_summary()
        
        # Save results if requested
        if save_results:
            self._save_results(video_path)
        
        self.logger.info("✅ Video analysis completed!")
        
    def process_frame(self, frame, frame_id):
        """Process a single frame with detection, tracking, and analysis"""
        # Step 1: Vehicle detection
        detection_result = self.detector.detect_frame(frame)
        detection_result.frame_id = frame_id
        
        # Step 2: Vehicle tracking (placeholder - would use actual tracker)
        # tracking_result = self.tracker.update(detection_result.detections)
        
        # Step 3: Traffic analysis
        analysis_result = self.analyzer.analyze_frame(
            detection_result=detection_result,
            tracking_result=None,  # Would pass actual tracking result
            frame=frame
        )
        
        return analysis_result
    
    def visualize_results(self, frame, analysis_result):
        """Create visualization of analysis results"""
        # Use analyzer's built-in visualization
        annotated_frame = self.analyzer.visualize_analysis(frame, analysis_result)
        
        # Add additional information
        height, width = frame.shape[:2]
        
        # Add title
        cv2.putText(annotated_frame, "Traffic AI - Analysis Demo", 
                   (width//2 - 150, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        # Add timestamp
        timestamp_str = datetime.fromtimestamp(analysis_result.timestamp).strftime("%H:%M:%S")
        cv2.putText(annotated_frame, f"Time: {timestamp_str}", 
                   (width - 150, height - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return annotated_frame
    
    def _generate_summary(self):
        """Generate analysis summary statistics"""
        if not self.results['analysis']:
            return
        
        analysis_data = self.results['analysis']
        
        # Calculate summary statistics
        total_frames = len(analysis_data)
        avg_vehicle_count = np.mean([d['vehicle_count'] for d in analysis_data])
        max_vehicle_count = max([d['vehicle_count'] for d in analysis_data])
        avg_density = np.mean([d['density'] for d in analysis_data])
        avg_speed = np.mean([d['average_speed'] for d in analysis_data])
        avg_flow_rate = np.mean([d['flow_rate'] for d in analysis_data])
        
        # Congestion level distribution
        congestion_levels = [d['congestion_level'] for d in analysis_data]
        congestion_dist = {
            level: congestion_levels.count(level) / len(congestion_levels) * 100
            for level in set(congestion_levels)
        }
        
        # Class distribution (aggregate)
        class_totals = {}
        for frame_data in analysis_data:
            for class_name, count in frame_data['class_distribution'].items():
                class_totals[class_name] = class_totals.get(class_name, 0) + count
        
        self.results['summary'] = {
            'total_frames_analyzed': total_frames,
            'average_vehicle_count': round(avg_vehicle_count, 2),
            'peak_vehicle_count': max_vehicle_count,
            'average_density': round(avg_density, 4),
            'average_speed': round(avg_speed, 2),
            'average_flow_rate': round(avg_flow_rate, 2),
            'congestion_distribution': congestion_dist,
            'total_vehicle_detections': class_totals,
            'analysis_zones': len(self.analyzer.zones),
            'counting_lines': len(self.analyzer.counting_lines)
        }
        
        # Print summary
        self.logger.info("=== Analysis Summary ===")
        self.logger.info(f"Frames analyzed: {total_frames}")
        self.logger.info(f"Average vehicles per frame: {avg_vehicle_count:.2f}")
        self.logger.info(f"Peak vehicle count: {max_vehicle_count}")
        self.logger.info(f"Average density: {avg_density:.4f}")
        self.logger.info(f"Average speed: {avg_speed:.2f} px/frame")
        self.logger.info(f"Average flow rate: {avg_flow_rate:.2f} vehicles/minute")
        self.logger.info(f"Congestion distribution: {congestion_dist}")
        self.logger.info(f"Total detections by class: {class_totals}")
    
    def _save_results(self, video_path):
        """Save analysis results to JSON file"""
        output_file = f"results/analysis_{os.path.basename(video_path)}_results.json"
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        # Add metadata
        self.results['metadata'] = {
            'video_path': video_path,
            'analysis_timestamp': datetime.now().isoformat(),
            'config_used': self.config
        }
        
        with open(output_file, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
        
        self.logger.info(f"Results saved to: {output_file}")

def main():
    parser = argparse.ArgumentParser(description='Traffic AI - Analysis Demo')
    parser.add_argument('--input', '-i', type=str, required=True,
                       help='Input video file path')
    parser.add_argument('--output', '-o', type=str,
                       help='Output video file path with annotations')
    parser.add_argument('--config', type=str, default='config/config.yaml',
                       help='Configuration file path')
    parser.add_argument('--save-results', action='store_true',
                       help='Save analysis results to JSON file')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.input):
        print(f"Error: Input file does not exist: {args.input}")
        sys.exit(1)
    
    try:
        # Initialize demo
        demo = TrafficAnalysisDemo(config_path=args.config)
        
        # Process video
        demo.process_video(
            video_path=args.input,
            output_path=args.output,
            save_results=args.save_results
        )
        
        print("✅ Traffic analysis demo completed successfully!")
    
    except Exception as e:
        print(f"❌ Error during analysis: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()