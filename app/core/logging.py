"""Structured logging configuration."""

import logging
import sys


class ContextFilter(logging.Filter):
    """Context filter providing default correlation attributes."""

    def filter(self, record: logging.LogRecord) -> bool:
        if not hasattr(record, "request_id"):
            record.request_id = "SYSTEM"
        return True


class SafeStreamHandler(logging.StreamHandler):
    """StreamHandler that ignores I/O errors during interpreter shutdown."""

    def emit(self, record: logging.LogRecord) -> None:
        try:
            if self.stream is None or getattr(self.stream, "closed", False):
                return
            super().emit(record)
        except Exception:
            pass

    def handleError(self, record: logging.LogRecord) -> None:
        # Silently ignore shutdown stream write failures
        pass


def setup_logger(name: str = "htr_service", level: str = "INFO") -> logging.Logger:
    """Configure and return an application logger."""
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))

    if not logger.handlers:
        handler = SafeStreamHandler(sys.stdout)
        handler.addFilter(ContextFilter())
        formatter = logging.Formatter(
            fmt="[%(asctime)s] [%(levelname)s] [req:%(request_id)s] %(name)s: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    logger.propagate = False
    return logger


logger = setup_logger()
