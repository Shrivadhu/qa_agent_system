"""Logging utilities for QA Agent system."""

import sys
from pathlib import Path
from loguru import logger
from config import config


def setup_logger():
    """Configure logger with settings from config."""
    
    # Remove default handler
    logger.remove()
    
    # Console handler
    logger.add(
        sys.stderr,
        level=config.get("logging.level", "INFO"),
        format=config.get(
            "logging.format",
            "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
            "<level>{message}</level>"
        )
    )
    
    # File handler
    log_file = config.get("logging.file", "logs/qa_agent.log")
    log_path = Path(log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    
    logger.add(
        log_file,
        rotation=config.get("logging.rotation", "10 MB"),
        retention=config.get("logging.retention", "30 days"),
        level=config.get("logging.level", "INFO"),
        format=config.get("logging.format")
    )
    
    return logger


# Initialize logger
log = setup_logger()
