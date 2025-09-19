#!/usr/bin/env python3
"""
Model Download Script

This script downloads pre-trained models for the Traffic AI system.
It downloads YOLO models and other required model files.

Usage:
    python scripts/download_models.py
    python scripts/download_models.py --model yolov8s.pt
    python scripts/download_models.py --all
"""

import argparse
import os
import sys
import urllib.request
from pathlib import Path

def download_file(url: str, filepath: str, show_progress: bool = True) -> bool:
    """
    Download a file from URL with progress indicator
    
    Args:
        url: Source URL
        filepath: Destination file path
        show_progress: Whether to show download progress
        
    Returns:
        True if successful, False otherwise
    """
    try:
        def progress_hook(block_num, block_size, total_size):
            if show_progress and total_size > 0:
                downloaded = block_num * block_size
                percent = min(100, downloaded * 100 / total_size)
                sys.stdout.write(f'\rDownloading: {percent:.1f}%')
                sys.stdout.flush()
        
        urllib.request.urlretrieve(url, filepath, progress_hook if show_progress else None)
        if show_progress:
            print()  # New line after progress
        return True
        
    except Exception as e:
        print(f"Error downloading {url}: {str(e)}")
        return False

def main():
    parser = argparse.ArgumentParser(description='Download Traffic AI models')
    parser.add_argument('--model', type=str, help='Specific model to download')
    parser.add_argument('--all', action='store_true', help='Download all available models')
    parser.add_argument('--models-dir', type=str, default='data/models',
                       help='Directory to save models')
    
    args = parser.parse_args()
    
    # Create models directory
    models_dir = Path(args.models_dir)
    models_dir.mkdir(parents=True, exist_ok=True)
    
    # Available models with their URLs
    models = {
        'yolov8n.pt': 'https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt',
        'yolov8s.pt': 'https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8s.pt',
        'yolov8m.pt': 'https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8m.pt',
        'yolov8l.pt': 'https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8l.pt',
        'yolov8x.pt': 'https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8x.pt'
    }
    
    print("🚗 Traffic AI - Model Download Tool")
    print("=" * 40)
    
    # Determine which models to download
    if args.model:
        if args.model not in models:
            print(f"Error: Model '{args.model}' not available.")
            print(f"Available models: {', '.join(models.keys())}")
            sys.exit(1)
        models_to_download = {args.model: models[args.model]}
    elif args.all:
        models_to_download = models
    else:
        # Default: download yolov8n.pt (smallest, fastest)
        models_to_download = {'yolov8n.pt': models['yolov8n.pt']}
    
    print(f"Downloading {len(models_to_download)} model(s) to {models_dir}")
    print()
    
    success_count = 0
    
    for model_name, model_url in models_to_download.items():
        model_path = models_dir / model_name
        
        # Check if model already exists
        if model_path.exists():
            print(f"✅ {model_name} already exists, skipping...")
            success_count += 1
            continue
        
        print(f"📥 Downloading {model_name}...")
        
        if download_file(model_url, str(model_path)):
            print(f"✅ {model_name} downloaded successfully!")
            success_count += 1
        else:
            print(f"❌ Failed to download {model_name}")
    
    print()
    print(f"Download complete: {success_count}/{len(models_to_download)} models")
    
    if success_count > 0:
        print("\n🎉 Models are ready! You can now run:")
        print("  python examples/basic_detection.py --input your_video.mp4")
        print("  python examples/traffic_analysis_demo.py --input your_video.mp4")
    
    return success_count == len(models_to_download)

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)