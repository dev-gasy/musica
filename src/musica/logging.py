"""Shared Loguru configuration."""

from __future__ import annotations

import sys

from loguru import logger

_CONFIGURED = False


def configure_logging(level: str = "INFO") -> None:
    global _CONFIGURED
    if _CONFIGURED:
        return

    logger.remove()
    logger.add(
        sys.stderr,
        level=level,
        format="{time:HH:mm:ss} | {level} | {message}",
    )
    _CONFIGURED = True
