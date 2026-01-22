"""
Logging utilities.
"""

from loguru import logger
import sys


def setup_logger(
    log_file: str = None,
    level: str = "INFO",
    rotation: str = "1 day",
    retention: str = "30 days"
):
    """
    Setup logger configuration.
    
    Args:
        log_file: Path to log file
        level: Logging level
        rotation: Log rotation period
        retention: Log retention period
    """
    # Remove default handler
    logger.remove()
    
    # Add console handler
    logger.add(
        sys.stderr,
        level=level,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>"
    )
    
    # Add file handler if specified
    if log_file:
        logger.add(
            log_file,
            rotation=rotation,
            retention=retention,
            level="DEBUG",
            format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}"
        )

