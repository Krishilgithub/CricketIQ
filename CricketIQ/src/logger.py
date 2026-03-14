"""
CricketIQ — Shared structured logger.

Usage:
    from src.logger import get_logger
    log = get_logger(__name__)
    log.info("Starting ingestion...")
"""

from __future__ import annotations

import sys
from loguru import logger


def get_logger(name: str):
    """Return a loguru logger bound with the given module name."""
    logger.remove()
    logger.add(
        sys.stderr,
        format=(
            "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{extra[module]}</cyan> | "
            "<level>{message}</level>"
        ),
        level="INFO",
        colorize=True,
    )
    logger.add(
        "artifacts/reports/cricketiq.log",
        rotation="10 MB",
        retention="14 days",
        level="DEBUG",
        format="{time} | {level} | {extra[module]} | {message}",
    )
    return logger.bind(module=name)
