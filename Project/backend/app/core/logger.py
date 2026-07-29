from __future__ import annotations

import logging
import sys


def setup_logging(level: str = "INFO") -> None:
    """Configure the root logger for the entire application.

    Call this once at application startup before any other code runs.
    All module-level loggers obtained via get_logger() will inherit
    this configuration automatically.
    """
    logging.basicConfig(
        level=level.upper(),
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        stream=sys.stdout,
        force=True,  # override any handlers already attached by imported libs
    )


def get_logger(name: str) -> logging.Logger:
    """Return a module-scoped logger.

    Usage (at module top level)::

        from app.core.logger import get_logger
        logger = get_logger(__name__)
    """
    return logging.getLogger(name)
