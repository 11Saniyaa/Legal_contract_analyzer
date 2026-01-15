"""
Logging configuration for Legal Contract Analyzer
Provides centralized logging setup
"""

import logging
import sys
from typing import Optional

# Log format
LOG_FORMAT = "[%(levelname)s] %(name)s - %(message)s"
DETAILED_LOG_FORMAT = "%(asctime)s [%(levelname)s] %(name)s:%(lineno)d - %(message)s"

# Log levels
LOG_LEVELS = {
    "DEBUG": logging.DEBUG,
    "INFO": logging.INFO,
    "WARNING": logging.WARNING,
    "ERROR": logging.ERROR,
    "CRITICAL": logging.CRITICAL,
}


def setup_logging(
    level: str = "INFO",
    detailed: bool = False,
    log_file: Optional[str] = None
) -> logging.Logger:
    """
    Set up logging configuration for the application
    
    Args:
        level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        detailed: If True, use detailed format with timestamps and line numbers
        log_file: Optional file path to write logs to
    
    Returns:
        Configured logger instance
    """
    # Get log level
    log_level = LOG_LEVELS.get(level.upper(), logging.INFO)
    
    # Choose format
    formatter = logging.Formatter(
        DETAILED_LOG_FORMAT if detailed else LOG_FORMAT
    )
    
    # Create root logger
    logger = logging.getLogger()
    logger.setLevel(log_level)
    
    # Remove existing handlers to avoid duplicates
    logger.handlers.clear()
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler (if specified)
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(log_level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance for a specific module
    
    Args:
        name: Logger name (typically __name__)
    
    Returns:
        Logger instance
    """
    return logging.getLogger(name)


# Initialize default logging on import
# Can be overridden by calling setup_logging() with different parameters
setup_logging(level="INFO", detailed=False)


