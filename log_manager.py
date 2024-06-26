import logging
from pathlib import Path

# Logger
logger: logging.Logger = logging.getLogger("Logger")
logger.setLevel(logging.DEBUG)
home = str(Path.home())
handler = logging.FileHandler("./file.log", mode="w")
formatter = logging.Formatter(
    "%(asctime)s | %(funcName)s | PID:%(process)d | %(levelname)s | %(message)s",
    datefmt="%Y%m%d-%H:%M:%S",
)
handler.setFormatter(formatter)
logger.addHandler(handler)
consoleHandler = logging.StreamHandler()
consoleHandler.setFormatter(formatter)
logger.addHandler(consoleHandler)