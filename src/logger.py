import logging
import os
import time
from logging.handlers import TimedRotatingFileHandler


class TimedStructuredFormatter(logging.Formatter):
    """
    Custom formatter for structured, block-style logging.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.operation_start_times = {}

    def start_timing(self, operation_id):
        """
        Record the start time for a specific operation.
        """
        self.operation_start_times[operation_id] = time.time()

    def stop_timing(self, operation_id):
        """
        Compute and return the elapsed time for a specific operation.
        """
        if operation_id in self.operation_start_times:
            start_time = self.operation_start_times.pop(operation_id)
            return time.time() - start_time
        return None
    
    def format(self, record):
        """
        Format the log entry and include elapsed time if available.
        """
        elapsed_time = record.__dict__.get("elapsed_time", None)
        log_entry = f"\n{'-'*40}\n"
        log_entry += f"Operation - {record.msg}\n"
        log_entry += f"Timestamp: {self.formatTime(record, self.datefmt)}\n"
        log_entry += f"Level: {record.levelname}\n"
        if elapsed_time is not None:
            log_entry += f"Elapsed Time: {elapsed_time:.4f} seconds\n"
        log_entry += f"{'-'*40}"
        return log_entry

def setup_logger(log_file: str, when: str = "midnight", interval: int = 1, backup_count: int = 7):
    """
    Set up the logger to write logs to the specified file and stream (console).

    Args:
    - log_file (str): Path to the log file.
    - when (str): The type of time interval for rotation (e.g., 'S', 'M', 'H', 'D', 'midnight').
    - interval (int): The number of intervals before rotating the log.
    - backup_count (int): The number of backup logs to keep.
    
    Returns:
    - logger: Configured logger instance.
    """
    # Ensure the directory exists
    os.makedirs(os.path.dirname(log_file), exist_ok=True)

    # Create and configure logger
    logger = logging.getLogger(log_file)
    logger.setLevel(logging.DEBUG)

    formatter = TimedStructuredFormatter()

    file_handler = TimedRotatingFileHandler(log_file, when="midnight", backupCount=7)
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)

    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.INFO) 
    stream_handler.setFormatter(formatter)

    # Add the handler to the logger
    if not logger.hasHandlers():
        logger.addHandler(file_handler)
        logger.addHandler(stream_handler)

    return logger, formatter


def log_message(logger, message: str, level: str = "INFO", elapsed_time: float = None):
    """
    Log a message using the provided logger at the specified log level.

    Args:
    - logger: Logger instance created by setup_logger.
    - message (str): The message to log.
    - level (str): The log level ('INFO', 'DEBUG', 'WARNING', 'ERROR', 'CRITICAL').
    """
    extra = {"elapsed_time": elapsed_time}
    if level == "DEBUG":
        logger.debug(message, extra=extra)
    elif level == "INFO":
        logger.info(message, extra=extra)
    elif level == "WARNING":
        logger.warning(message, extra=extra)
    elif level == "ERROR":
        logger.error(message, extra=extra)
    elif level == "CRITICAL":
        logger.critical(message, extra=extra)
    else:
        logger.info(message, extra=extra)
