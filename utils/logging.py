"""
Logging utilities for the Kimchi GitHub Assistant.
"""

import logging
import sys
from typing import Optional


def setup_logging(level: str = "INFO", format_string: Optional[str] = None) -> None:
    """
    Set up logging for the application.
    
    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        format_string: Custom format string for log messages
    """
    if format_string is None:
        format_string = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format=format_string,
        handlers=[
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    # Always configure third-party loggers when setting up logging
    configure_third_party_loggers()


def setup_quiet_logging() -> None:
    """
    Set up quiet logging that suppresses verbose third-party library messages.
    This is the recommended setup for end users.
    """
    # Set up basic logging at WARNING level to reduce noise
    logging.basicConfig(
        level=logging.WARNING,
        format="%(message)s",
        handlers=[logging.StreamHandler(sys.stdout)]
    )
    
    # Configure third-party loggers to be even quieter
    configure_third_party_loggers()


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance.
    
    Args:
        name: Name of the logger (usually __name__)
        
    Returns:
        Configured logger instance
    """
    return logging.getLogger(name)


def configure_third_party_loggers():
    """Configure third-party library loggers to reduce noise."""
    # Reduce noise from common libraries
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("elastic_transport").setLevel(logging.WARNING)
    logging.getLogger("elasticsearch").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("openai").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("asyncio").setLevel(logging.WARNING)
