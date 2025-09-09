"""
Centralized Logging Configuration

This module provides a single point of logging configuration to prevent conflicts
between different services and modules. It ensures consistent logging behavior
across the entire application.
"""

import logging
import os
from pathlib import Path
from datetime import datetime
from typing import Optional

# Global flag to track if logging has been initialized
_logging_initialized = False

def initialize_logging_once(
    log_level: str = "INFO",
    log_dir: str = "logs",
    console_output: bool = True,
    max_file_size: int = 10 * 1024 * 1024,  # 10MB
    backup_count: int = 5,
    service_name: str = "youtube_optimizer"
) -> logging.Logger:
    """
    Initialize logging configuration only once to prevent conflicts.
    
    This function ensures that logging is configured only once, even if called
    multiple times from different modules or services.
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_dir: Directory to store log files
        console_output: Whether to output logs to console
        max_file_size: Maximum size of log file before rotation
        backup_count: Number of backup files to keep
        service_name: Name of the service for log file naming
        
    Returns:
        Configured root logger
    """
    global _logging_initialized
    
    if _logging_initialized:
        # Logging already initialized, return existing logger
        return logging.getLogger()
    
    # Create log directory if it doesn't exist
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)
    
    # Generate log filename with service name
    timestamp = datetime.now().strftime("%Y-%m-%d")
    log_filename = f"{service_name}_{timestamp}.log"
    log_file_path = log_path / log_filename
    
    # Get root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, log_level.upper()))
    
    # Clear any existing handlers to prevent duplicates
    root_logger.handlers.clear()
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # File handler with rotation
    try:
        from logging.handlers import RotatingFileHandler
        file_handler = RotatingFileHandler(
            log_file_path,
            maxBytes=max_file_size,
            backupCount=backup_count,
            encoding='utf-8'
        )
        file_handler.setLevel(getattr(logging, log_level.upper()))
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)
        
        root_logger.info(f"File logging initialized: {log_file_path}")
        
    except Exception as e:
        # Fallback to basic file handler
        try:
            file_handler = logging.FileHandler(log_file_path, encoding='utf-8')
            file_handler.setLevel(getattr(logging, log_level.upper()))
            file_handler.setFormatter(formatter)
            root_logger.addHandler(file_handler)
            root_logger.info(f"Basic file logging initialized: {log_file_path}")
        except Exception as fallback_e:
            root_logger.warning(f"Could not set up file logging: {fallback_e}")
    
    # Console handler
    if console_output:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(getattr(logging, log_level.upper()))
        console_handler.setFormatter(formatter)
        root_logger.addHandler(console_handler)
    
    # Suppress noisy third-party loggers
    logging.getLogger("googleapiclient.discovery_cache").setLevel(logging.ERROR)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("requests").setLevel(logging.WARNING)
    
    # Mark as initialized
    _logging_initialized = True
    
    root_logger.info(f"Centralized logging initialized for {service_name}")
    return root_logger

def get_safe_logger(name: str) -> logging.Logger:
    """
    Get a logger instance that's safe to use in any module.
    
    This function ensures that logging is properly initialized before
    returning a logger instance.
    
    Args:
        name: Logger name (usually __name__)
        
    Returns:
        Logger instance
    """
    # Initialize logging if not already done
    if not _logging_initialized:
        # Use environment variables for configuration
        log_level = os.getenv("LOG_LEVEL", "INFO")
        console_output = os.getenv("LOG_CONSOLE", "true").lower() == "true"
        service_name = os.getenv("SERVICE_NAME", "youtube_optimizer")
        
        initialize_logging_once(
            log_level=log_level,
            console_output=console_output,
            service_name=service_name
        )
    
    return logging.getLogger(name)

def is_logging_initialized() -> bool:
    """
    Check if logging has been initialized.
    
    Returns:
        True if logging is initialized, False otherwise
    """
    return _logging_initialized

# Backward compatibility functions
def setup_logging(
    log_level: str = "INFO",
    log_dir: str = "logs",
    log_filename: Optional[str] = None,
    console_output: bool = True,
    max_file_size: int = 10 * 1024 * 1024,  # 10MB
    backup_count: int = 5
) -> logging.Logger:
    """
    Backward compatibility wrapper for setup_logging.
    
    This function maintains compatibility with existing code that uses
    the old setup_logging function.
    """
    return initialize_logging_once(
        log_level=log_level,
        log_dir=log_dir,
        console_output=console_output,
        max_file_size=max_file_size,
        backup_count=backup_count,
        service_name="youtube_optimizer"
    )

def get_logger(name: str, log_dir: str = "logs") -> logging.Logger:
    """
    Backward compatibility wrapper for get_logger.
    
    This function maintains compatibility with existing code that uses
    the old get_logger function.
    """
    return get_safe_logger(name)

def setup_module_logging(
    module_name: str,
    log_level: str = "INFO",
    log_dir: str = "logs",
    console_output: bool = True
) -> logging.Logger:
    """
    Backward compatibility wrapper for setup_module_logging.
    
    This function maintains compatibility with existing code that uses
    the old setup_module_logging function.
    """
    return get_safe_logger(module_name)
