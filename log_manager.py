import logging

# Configure global logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)  # Set the logging level to INFO or desired level

# Create a formatter
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")

# Create a console handler and set level and formatter
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(formatter)

# Add the console handler to the logger
logger.addHandler(console_handler)
