import logging
import sys
from colorlog import ColoredFormatter

def setup_colored_logging():
    logger = logging.getLogger()
    if not logger.hasHandlers():
        formatter = ColoredFormatter(
            "%(log_color)s%(asctime)s %(levelname)-8s%(reset)s %(message)s",
            datefmt="%H:%M:%S",
            reset=True,
            log_colors={
                'DEBUG': 'cyan',
                'INFO': 'green',
                'WARNING': 'yellow',
                'ERROR': 'red',
                'CRITICAL': 'red,bg_white',
            },
            style='%'
        )
        handler = logging.StreamHandler(sys.stderr)  # Use stderr
        handler.setFormatter(formatter)
        logger.setLevel(logging.DEBUG)  # Show info and above by default
        logger.addHandler(handler)
    return logger

# Do NOT call setup_colored_logging() here.