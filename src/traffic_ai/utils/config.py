"""
Configuration management utilities
"""

import yaml
import os
from typing import Dict, Any, Optional

def load_config(config_path: str = "config/config.yaml") -> Dict[str, Any]:
    """
    Load configuration from YAML file
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Configuration dictionary
    """
    try:
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
        return config or {}
    except FileNotFoundError:
        print(f"Warning: Configuration file not found: {config_path}")
        return {}
    except yaml.YAMLError as e:
        print(f"Error parsing configuration file: {e}")
        return {}

def save_config(config: Dict[str, Any], config_path: str = "config/config.yaml") -> bool:
    """
    Save configuration to YAML file
    
    Args:
        config: Configuration dictionary
        config_path: Path to save configuration file
        
    Returns:
        True if saved successfully, False otherwise
    """
    try:
        os.makedirs(os.path.dirname(config_path), exist_ok=True)
        with open(config_path, 'w') as file:
            yaml.dump(config, file, default_flow_style=False, indent=2)
        return True
    except Exception as e:
        print(f"Error saving configuration: {e}")
        return False

def get_config_value(config: Dict[str, Any], key_path: str, default: Any = None) -> Any:
    """
    Get nested configuration value using dot notation
    
    Args:
        config: Configuration dictionary
        key_path: Dot-separated key path (e.g., 'detection.confidence_threshold')
        default: Default value if key not found
        
    Returns:
        Configuration value or default
    """
    keys = key_path.split('.')
    value = config
    
    try:
        for key in keys:
            value = value[key]
        return value
    except (KeyError, TypeError):
        return default