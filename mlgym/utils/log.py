"""
Copyright (c) Meta Platforms, Inc. and affiliates.

Logging utility for MLGym.

Adapted from SWE-agent/sweagent/utils/log.py
"""
from __future__ import annotations

import logging
import os
from pathlib import PurePath
from typing import Optional

from rich.logging import RichHandler

_SET_UP_LOGGERS = set()
_ADDITIONAL_HANDLERS = []

logging.TRACE = 5  # type: ignore
logging.addLevelName(logging.TRACE, "TRACE")  # type: ignore


def _interpret_level_from_env(level: str | None, *, default=logging.DEBUG) -> int:
    if not level:
        return default
    if level.isnumeric():
        return int(level)
    return getattr(logging, level.upper())


_STREAM_LEVEL = _interpret_level_from_env(os.environ.get("MLGYM_LOG_STREAM_LEVEL"))
_FILE_LEVEL = _interpret_level_from_env(os.environ.get("MLGYM_LOG_FILE_LEVEL"), default=logging.TRACE)  # type: ignore


def get_logger(name: str) -> logging.Logger:
    """Get logger. Use this instead of `logging.getLogger` to ensure
    that the logger is set up with the correct handlers.
    """
    print(f"Setting up logger for {name=}")
    logger = logging.getLogger(name)
    if name in _SET_UP_LOGGERS:
        # Already set up
        return logger
    handler = RichHandler(
        show_time=bool(os.environ.get("MLGYM_LOG_TIME", False)),
        show_path=False,
    )
    handler.setLevel(_STREAM_LEVEL)
    logger.setLevel(min(_STREAM_LEVEL, _FILE_LEVEL))
    logger.addHandler(handler)
    logger.propagate = False
    _SET_UP_LOGGERS.add(name)
    for handler in _ADDITIONAL_HANDLERS:
        print(f"Registering {handler.baseFilename} to logger {name=}")
        logger.addHandler(handler)
    return logger


def add_file_handler(path: PurePath | str, logger_names: Optional[List[str]] = None) -> None:
    """Adds a file handler to all loggers that we have set up
    and all future loggers that will be set up with `get_logger`.
    """
    print(f"Adding file_handler for {path=}")
    handler = logging.FileHandler(path)
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    handler.setFormatter(formatter)
    handler.setLevel(_FILE_LEVEL)
    if logger_names is None:
        for name in _SET_UP_LOGGERS:
            logger = logging.getLogger(name)
            print(f"Registering {path=} to logger {name=}")
            logger.addHandler(handler)
    else:
        for name in logger_names:
            logger = logging.getLogger(name)
            print(f"Registering {path=} to logger {name=}")
            logger.addHandler(handler)
    _ADDITIONAL_HANDLERS.append(handler)


default_logger = get_logger("MLGym")
