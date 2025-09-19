"""
Unit tests for utility modules
"""

import pytest
import tempfile
import os
import sys
import yaml

# Add src to path for testing
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from traffic_ai.utils.config import load_config, save_config, get_config_value
from traffic_ai.utils.logger import setup_logger, get_logger


class TestConfigUtils:
    """Test cases for configuration utilities"""
    
    def test_load_config_valid_file(self):
        """Test loading valid configuration file"""
        config_data = {
            'detection': {
                'model': 'yolov8n.pt',
                'confidence_threshold': 0.5
            },
            'tracking': {
                'max_age': 30
            }
        }
        
        # Create temporary config file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(config_data, f)
            temp_path = f.name
        
        try:
            # Test loading
            loaded_config = load_config(temp_path)
            
            assert loaded_config == config_data
            assert loaded_config['detection']['model'] == 'yolov8n.pt'
            assert loaded_config['detection']['confidence_threshold'] == 0.5
            assert loaded_config['tracking']['max_age'] == 30
            
        finally:
            os.unlink(temp_path)
    
    def test_load_config_nonexistent_file(self):
        """Test loading non-existent configuration file"""
        config = load_config('nonexistent_config.yaml')
        assert config == {}
    
    def test_save_config(self):
        """Test saving configuration to file"""
        config_data = {
            'test': {
                'value': 42,
                'name': 'test_config'
            }
        }
        
        # Create temporary directory
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = os.path.join(temp_dir, 'test_config.yaml')
            
            # Save config
            result = save_config(config_data, config_path)
            assert result is True
            
            # Verify file was created and content is correct
            assert os.path.exists(config_path)
            
            with open(config_path, 'r') as f:
                loaded_data = yaml.safe_load(f)
            
            assert loaded_data == config_data
    
    def test_get_config_value(self):
        """Test getting nested configuration values"""
        config = {
            'detection': {
                'model': 'yolov8n.pt',
                'params': {
                    'confidence': 0.5,
                    'iou': 0.4
                }
            },
            'tracking': {
                'max_age': 30
            }
        }
        
        # Test existing values
        assert get_config_value(config, 'detection.model') == 'yolov8n.pt'
        assert get_config_value(config, 'detection.params.confidence') == 0.5
        assert get_config_value(config, 'tracking.max_age') == 30
        
        # Test non-existent values with default
        assert get_config_value(config, 'nonexistent.key', 'default') == 'default'
        assert get_config_value(config, 'detection.nonexistent', None) is None
        
        # Test non-existent values without default
        assert get_config_value(config, 'nonexistent.key') is None


class TestLoggerUtils:
    """Test cases for logging utilities"""
    
    def test_setup_logger_basic(self):
        """Test basic logger setup"""
        logger = setup_logger('test_logger', level='INFO')
        
        assert logger.name == 'test_logger'
        assert logger.level == 20  # INFO level
        assert len(logger.handlers) >= 1  # At least console handler
    
    def test_setup_logger_with_file(self):
        """Test logger setup with file handler"""
        with tempfile.TemporaryDirectory() as temp_dir:
            log_file = os.path.join(temp_dir, 'test.log')
            
            logger = setup_logger('test_file_logger', log_file=log_file, level='DEBUG')
            
            assert logger.name == 'test_file_logger'
            assert logger.level == 10  # DEBUG level
            assert len(logger.handlers) == 2  # Console + file handlers
            
            # Test logging
            logger.info('Test message')
            
            # Verify file was created
            assert os.path.exists(log_file)
    
    def test_setup_logger_custom_format(self):
        """Test logger setup with custom format"""
        custom_format = '%(levelname)s - %(message)s'
        logger = setup_logger('test_format_logger', format_string=custom_format)
        
        assert logger.name == 'test_format_logger'
        # Verify format was applied to handlers
        for handler in logger.handlers:
            assert handler.formatter._fmt == custom_format
    
    def test_get_logger(self):
        """Test getting existing logger"""
        # Create a logger first
        original_logger = setup_logger('test_get_logger')
        
        # Get the same logger
        retrieved_logger = get_logger('test_get_logger')
        
        assert retrieved_logger is original_logger
        assert retrieved_logger.name == 'test_get_logger'


class TestIntegration:
    """Integration tests for utilities"""
    
    def test_config_and_logger_integration(self):
        """Test using config to setup logger"""
        config_data = {
            'logging': {
                'level': 'WARNING',
                'format': '%(name)s - %(levelname)s - %(message)s'
            }
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(config_data, f)
            temp_path = f.name
        
        try:
            # Load config
            config = load_config(temp_path)
            
            # Use config to setup logger
            log_level = get_config_value(config, 'logging.level', 'INFO')
            log_format = get_config_value(config, 'logging.format')
            
            logger = setup_logger('integration_test', level=log_level, format_string=log_format)
            
            assert logger.level == 30  # WARNING level
            
        finally:
            os.unlink(temp_path)


if __name__ == "__main__":
    pytest.main([__file__])