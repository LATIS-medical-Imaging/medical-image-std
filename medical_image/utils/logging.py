import logging

logger = logging.getLogger("medical_image")
logger.addHandler(logging.NullHandler())


def configure_logging(level=logging.DEBUG, log_file=None):
    """
    Optional convenience function for users who want console/file logging.

    Args:
        level: Logging level (default DEBUG).
        log_file: Path to a log file. If None, only console output.
    """
    logger.setLevel(level)

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y%m%d-%H:%M:%S",
    )

    if not any(isinstance(h, logging.StreamHandler) for h in logger.handlers):
        console = logging.StreamHandler()
        console.setFormatter(formatter)
        logger.addHandler(console)

    if log_file and not any(
        isinstance(h, logging.FileHandler) for h in logger.handlers
    ):
        fh = logging.FileHandler(log_file, mode="a")
        fh.setFormatter(formatter)
        logger.addHandler(fh)
