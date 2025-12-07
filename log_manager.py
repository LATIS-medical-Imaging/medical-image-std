import logging
from pathlib import Path


class LogManager:
    def __init__(self):
        self.logger = logging.getLogger("medical_image")
        self.logger.setLevel(logging.DEBUG)

        # Avoid adding multiple handlers if pytest reloads modules
        if not self.logger.handlers:
            log_path = Path("./file.log")

            file_handler = logging.FileHandler(log_path, mode="w")
            console_handler = logging.StreamHandler()

            formatter = logging.Formatter(
                "%(asctime)s | %(levelname)s | %(message)s",
                datefmt="%Y%m%d-%H:%M:%S",
            )

            file_handler.setFormatter(formatter)
            console_handler.setFormatter(formatter)

            self.logger.addHandler(file_handler)
            self.logger.addHandler(console_handler)

    # Convenience: allow log_manager.info("message")
    def info(self, msg):
        self.logger.info(msg)

    def debug(self, msg):
        self.logger.debug(msg)

    def warning(self, msg):
        self.logger.warning(msg)

    def error(self, msg):
        self.logger.error(msg)


# Create a global instance
logger = LogManager()
