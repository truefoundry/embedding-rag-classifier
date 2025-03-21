import logging
import sys

from src.config import settings

# Create logger
logger = logging.getLogger(__name__)
logger.setLevel(settings.LOG_LEVEL)

# Create formatter
formatter = logging.Formatter(
    "%(levelname)s:    %(asctime)s - %(module)s:%(funcName)s:%(lineno)d - %(message)s"
)

# Create default handler
handler = logging.StreamHandler(stream=sys.stdout)
handler.setFormatter(formatter)
logger.addHandler(handler)
