"""
Logging utilities
"""
import logging
import os
from pathlib import Path
from logging.handlers import RotatingFileHandler
from typing import Dict, Any


def setup_logger(config: Dict[str, Any], name: str = "app_scraper") -> logging.Logger:
    """Setup logger with configuration"""
    
    # Get logging configuration
    log_config = config.get("logging", {})
    
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, log_config.get("level", "INFO")))
    
    # Clear existing handlers
    logger.handlers.clear()
    
    # Create formatter
    formatter = logging.Formatter(
        log_config.get("format", "%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    )
    
    # Console handler
    if log_config.get("console", True):
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    
    # File handler
    log_file = log_config.get("file", "logs/app_scraper.log")
    if log_file:
        # Ensure log directory exists
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        
        # Parse file size
        max_bytes = _parse_file_size(log_config.get("max_file_size", "10MB"))
        backup_count = log_config.get("backup_count", 5)
        
        file_handler = RotatingFileHandler(
            log_file, 
            maxBytes=max_bytes, 
            backupCount=backup_count,
            encoding='utf-8'
        )
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


def _parse_file_size(size_str: str) -> int:
    """Parse file size string to bytes"""
    size_str = size_str.upper()
    
    if size_str.endswith('KB'):
        return int(size_str[:-2]) * 1024
    elif size_str.endswith('MB'):
        return int(size_str[:-2]) * 1024 * 1024
    elif size_str.endswith('GB'):
        return int(size_str[:-2]) * 1024 * 1024 * 1024
    else:
        # Assume bytes
        return int(size_str)


def get_logger(name: str = "app_scraper") -> logging.Logger:
    """Get existing logger"""
    return logging.getLogger(name) 