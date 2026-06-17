"""Logging configuration helpers."""

from __future__ import annotations

import logging
from typing import Optional

_CONFIGURED = False


def setup_logging(level: Optional[str] = None) -> None:
    """Configure root logging once, idempotently."""
    global _CONFIGURED
    if _CONFIGURED:
        return

    from config import config

    logging.basicConfig(
        level=getattr(logging, (level or config.log_level).upper(), logging.INFO),
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    # Quiet down noisy third-party loggers.
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("chromadb").setLevel(logging.WARNING)
    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    """Return a module logger, ensuring logging is configured."""
    setup_logging()
    return logging.getLogger(name)
