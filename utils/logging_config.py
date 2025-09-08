"""
Logging configuration utilities with automatic folder creation.
"""
import logging
from pathlib import Path
from datetime import datetime
from typing import Optional


def setup_logging(
    log_level: str = "INFO",
    log_dir: str = "logs",
    log_filename: Optional[str] = None,
    console_output: bool = True,
    max_file_size: int = 10 * 1024 * 1024,  # 10MB
    backup_count: int = 5
) -> logging.Logger:
    """
    Set up logging with automatic folder creation.
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_dir: Directory to store log files (will be created if it doesn't exist)
        log_filename: Name of the log file (defaults to app_YYYY-MM-DD.log)
        console_output: Whether to output logs to console
        max_file_size: Maximum size of log file before rotation (bytes)
        backup_count: Number of backup files to keep
    
    Returns:
        Configured logger instance
    """
    # Create log directory if it doesn't exist
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)
    
    # Generate log filename if not provided
    if not log_filename:
        timestamp = datetime.now().strftime("%Y-%m-%d")
        log_filename = f"app_{timestamp}.log"
    
    # Full path to log file
    log_file_path = log_path / log_filename
    
    # Create logger
    logger = logging.getLogger()
    logger.setLevel(getattr(logging, log_level.upper()))
    
    # Clear any existing handlers
    logger.handlers.clear()
    
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
        logger.addHandler(file_handler)
        
        # Log that file logging is set up
        logger.info(f"File logging initialized: {log_file_path}")
        
    except Exception as e:
        # Fallback to basic file handler if RotatingFileHandler fails
        try:
            file_handler = logging.FileHandler(log_file_path, encoding='utf-8')
            file_handler.setLevel(getattr(logging, log_level.upper()))
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
            logger.info(f"Basic file logging initialized: {log_file_path}")
        except Exception as fallback_e:
            logger.warning(f"Could not set up file logging: {fallback_e}")
    
    # Console handler
    if console_output:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(getattr(logging, log_level.upper()))
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    
    return logger


def get_logger(name: str, log_dir: str = "logs") -> logging.Logger:
    """
    Get a logger instance with automatic folder creation.
    
    Args:
        name: Logger name (usually __name__)
        log_dir: Directory to store log files
    
    Returns:
        Logger instance
    """
    # Ensure log directory exists
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)
    
    return logging.getLogger(name)


def setup_module_logging(
    module_name: str,
    log_level: str = "INFO",
    log_dir: str = "logs",
    console_output: bool = True
) -> logging.Logger:
    """
    Set up logging for a specific module with automatic folder creation.
    
    Args:
        module_name: Name of the module (e.g., 'services.youtube')
        log_level: Logging level
        log_dir: Directory to store log files
        console_output: Whether to output logs to console
    
    Returns:
        Configured logger for the module
    """
    # Create log directory if it doesn't exist
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)
    
    # Create module-specific log file
    timestamp = datetime.now().strftime("%Y-%m-%d")
    log_filename = f"{module_name.replace('.', '_')}_{timestamp}.log"
    log_file_path = log_path / log_filename
    
    # Create logger for the module
    logger = logging.getLogger(module_name)
    logger.setLevel(getattr(logging, log_level.upper()))
    
    # Clear existing handlers for this logger
    logger.handlers.clear()
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # File handler
    try:
        file_handler = logging.FileHandler(log_file_path, encoding='utf-8')
        file_handler.setLevel(getattr(logging, log_level.upper()))
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        
        logger.info(f"Module logging initialized: {log_file_path}")
        
    except Exception as e:
        logger.warning(f"Could not set up file logging for {module_name}: {e}")
    
    # Console handler
    if console_output:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(getattr(logging, log_level.upper()))
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    
    return logger
