#!/usr/bin/env python3
"""
Traffic AI Web Application
Main Flask application with editing capabilities and real-time preview
"""

import os
import sys
import json
import logging
from datetime import datetime
from flask import Flask, render_template, request, jsonify, redirect, url_for, send_from_directory
from flask_cors import CORS
import yaml

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), 'src'))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('traffic_ai_web.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
app.config['SECRET_KEY'] = 'traffic-ai-secret-key-2024'
CORS(app)

# Global configuration
CONFIG_FILE = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'config', 'web_config.yaml')
DEMO_DATA = {
    'cameras': [
        {
            'id': 1,
            'name': 'Main Street Intersection',
            'location': 'Downtown',
            'status': 'active',
            'url': 'rtmp://demo-camera-1.local/stream',
            'current_traffic': 'moderate'
        },
        {
            'id': 2,
            'name': 'Highway Junction A',
            'location': 'North Highway',
            'status': 'active',
            'url': 'rtmp://demo-camera-2.local/stream',
            'current_traffic': 'heavy'
        },
        {
            'id': 3,
            'name': 'School Zone Camera',
            'location': 'School District',
            'status': 'maintenance',
            'url': 'rtmp://demo-camera-3.local/stream',
            'current_traffic': 'light'
        }
    ],
    'traffic_stats': {
        'total_vehicles': 1247,
        'avg_speed': 45.2,
        'congestion_level': 'moderate',
        'incidents': 2,
        'last_updated': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    },
    'recent_detections': [
        {'time': '14:23:45', 'camera': 'Main Street', 'vehicles': 12, 'type': 'car'},
        {'time': '14:23:30', 'camera': 'Highway Junction', 'vehicles': 8, 'type': 'truck'},
        {'time': '14:23:15', 'camera': 'School Zone', 'vehicles': 3, 'type': 'car'},
    ]
}

def load_config():
    """Load configuration from file or create default"""
    try:
        if os.path.exists(CONFIG_FILE):
            with open(CONFIG_FILE, 'r') as f:
                return yaml.safe_load(f)
    except Exception as e:
        logger.warning(f"Could not load config: {e}")
    
    # Default configuration
    return {
        'app': {
            'title': 'Traffic AI Dashboard',
            'version': '1.0.0',
            'debug': True
        },
        'detection': {
            'model': 'yolov8n',
            'confidence_threshold': 0.5,
            'device': 'cpu'
        },
        'tracking': {
            'max_age': 30,
            'min_hits': 3,
            'iou_threshold': 0.3
        }
    }

def save_config(config):
    """Save configuration to file"""
    try:
        os.makedirs(os.path.dirname(CONFIG_FILE), exist_ok=True)
        with open(CONFIG_FILE, 'w') as f:
            yaml.dump(config, f, default_flow_style=False, indent=2)
        return True
    except Exception as e:
        logger.error(f"Could not save config: {e}")
        return False

# Load initial configuration
app_config = load_config()

@app.route('/')
def index():
    """Main dashboard page"""
    return render_template('index.html', 
                         config=app_config,
                         demo_data=DEMO_DATA)

@app.route('/editor')
def editor():
    """Configuration editor page"""
    return render_template('editor.html', 
                         config=app_config)

@app.route('/cameras')
def cameras():
    """Camera management page"""
    return render_template('cameras.html', 
                         cameras=DEMO_DATA['cameras'],
                         config=app_config)

@app.route('/analytics')
def analytics():
    """Analytics and reporting page"""
    return render_template('analytics.html', 
                         stats=DEMO_DATA['traffic_stats'],
                         config=app_config)

@app.route('/api/config', methods=['GET'])
def get_config():
    """API endpoint to get current configuration"""
    return jsonify(app_config)

@app.route('/api/config', methods=['POST'])
def update_config():
    """API endpoint to update configuration"""
    try:
        new_config = request.get_json()
        if new_config:
            app_config.update(new_config)
            if save_config(app_config):
                logger.info("Configuration updated successfully")
                return jsonify({'status': 'success', 'message': 'Configuration updated'})
            else:
                return jsonify({'status': 'error', 'message': 'Failed to save configuration'}), 500
        else:
            return jsonify({'status': 'error', 'message': 'No configuration data provided'}), 400
    except Exception as e:
        logger.error(f"Error updating configuration: {e}")
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/api/traffic/current', methods=['GET'])
def get_current_traffic():
    """API endpoint for current traffic data"""
    return jsonify(DEMO_DATA['traffic_stats'])

@app.route('/api/cameras', methods=['GET'])
def get_cameras():
    """API endpoint for camera data"""
    return jsonify(DEMO_DATA['cameras'])

@app.route('/api/cameras/<int:camera_id>', methods=['PUT'])
def update_camera(camera_id):
    """API endpoint to update camera configuration"""
    try:
        camera_data = request.get_json()
        # Find and update camera
        for camera in DEMO_DATA['cameras']:
            if camera['id'] == camera_id:
                camera.update(camera_data)
                logger.info(f"Camera {camera_id} updated")
                return jsonify({'status': 'success', 'camera': camera})
        
        return jsonify({'status': 'error', 'message': 'Camera not found'}), 404
    except Exception as e:
        logger.error(f"Error updating camera: {e}")
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/api/detection/start', methods=['POST'])
def start_detection():
    """API endpoint to start traffic detection"""
    try:
        data = request.get_json()
        camera_id = data.get('camera_id')
        
        logger.info(f"Starting detection for camera {camera_id}")
        
        # Simulate detection start
        return jsonify({
            'status': 'success',
            'message': f'Detection started for camera {camera_id}',
            'detection_id': f'det_{camera_id}_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
        })
    except Exception as e:
        logger.error(f"Error starting detection: {e}")
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/api/detection/stop', methods=['POST'])
def stop_detection():
    """API endpoint to stop traffic detection"""
    try:
        data = request.get_json()
        detection_id = data.get('detection_id')
        
        logger.info(f"Stopping detection {detection_id}")
        
        return jsonify({
            'status': 'success',
            'message': f'Detection {detection_id} stopped'
        })
    except Exception as e:
        logger.error(f"Error stopping detection: {e}")
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/static/<path:filename>')
def static_files(filename):
    """Serve static files"""
    return send_from_directory('static', filename)

@app.errorhandler(404)
def not_found(error):
    """404 error handler"""
    return render_template('error.html', 
                         error_code=404, 
                         error_message="Page not found",
                         config=app_config), 404

@app.errorhandler(500)
def internal_error(error):
    """500 error handler"""
    return render_template('error.html', 
                         error_code=500, 
                         error_message="Internal server error",
                         config=app_config), 500

if __name__ == '__main__':
    logger.info("Starting Traffic AI Web Application")
    logger.info(f"Configuration loaded: {app_config['app']['title']}")
    
    # Run the Flask app
    app.run(
        host='0.0.0.0',
        port=5000,
        debug=app_config['app'].get('debug', False),
        threaded=True
    )