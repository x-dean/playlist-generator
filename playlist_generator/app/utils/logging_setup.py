import logging
import sys
import threading
import queue
import time
from colorlog import ColoredFormatter

log_queue = None
log_consumer_thread = None

def setup_queue_colored_logging(logfile_path=None):
    global log_queue, log_consumer_thread
    if log_queue is not None:
        return  # Already set up in this process
    log_queue = queue.Queue()

    class QueueHandler(logging.Handler):
        def emit(self, record):
            try:
                msg = self.format(record)
                log_queue.put(msg)
            except Exception:
                pass

    def log_consumer():
        log_file = None
        if logfile_path:
            log_file = open(logfile_path, 'a', encoding='utf-8')
        while True:
            try:
                msg = log_queue.get(timeout=0.5)
                if log_file:
                    log_file.write(msg + '\n')
                    log_file.flush()
                # Do not print to terminal
                time.sleep(0.05)
            except queue.Empty:
                continue

    queue_handler = QueueHandler()
    color_formatter = ColoredFormatter(
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
    queue_handler.setFormatter(color_formatter)
    root_logger = logging.getLogger()
    root_logger.handlers = [queue_handler]

    log_consumer_thread = threading.Thread(
        target=log_consumer, daemon=True)
    log_consumer_thread.start()


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
        logger.addHandler(handler)
    return logger

# Do NOT call setup_colored_logging() here.