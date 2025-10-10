import logging
import logging.handlers
import os

from rich.logging import RichHandler

from utils.utils import get_envvar

LOG_DATE_FMT = "[%d-%m-%Y %H:%M:%S]"

log_dir = get_envvar("LOG_DIR")
if not os.path.exists(log_dir):
    os.mkdir(log_dir)

log = logging.getLogger(get_envvar("LOG_NAME"))
log.propagate = False
log.setLevel(get_envvar("LOG_LEVEL"))

msg_formatter = logging.Formatter(
    fmt="%(asctime)s | %(levelname)s | %(message)s", datefmt=LOG_DATE_FMT
)
rich_msg_formatter = logging.Formatter(fmt="%(message)s", datefmt=LOG_DATE_FMT)


def create_time_rotating_file_handler(log_level, filename, formatter):
    handler = logging.handlers.TimedRotatingFileHandler(
        f"{log_dir}/{filename}.log", when="midnight", backupCount=30
    )
    handler.setLevel(log_level)
    handler.setFormatter(formatter)
    return handler


class DebugFilter(logging.Filter):
    def filter(self, record):
        return record.levelno == logging.DEBUG


# debug_handler
debug_handler = create_time_rotating_file_handler(logging.DEBUG, "debug", msg_formatter)
debug_handler.addFilter(DebugFilter())

# error_handler
error_handler = create_time_rotating_file_handler(
    logging.WARNING, "error", msg_formatter
)

# info_handler
info_handler = create_time_rotating_file_handler(logging.INFO, "info", msg_formatter)

info_stdout_handler = RichHandler(rich_tracebacks=True)
info_stdout_handler.setLevel(logging.INFO)
info_stdout_handler.setFormatter(rich_msg_formatter)

log.addHandler(debug_handler)
log.addHandler(error_handler)
log.addHandler(info_handler)
log.addHandler(info_stdout_handler)
