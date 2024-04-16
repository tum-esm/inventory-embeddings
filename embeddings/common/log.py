import logging

logger = logging.getLogger("ConsoleLogger")

_stream_formatter = logging.Formatter("[%(levelname)s] > %(message)s")
_stream_handler = logging.StreamHandler()
_stream_handler.setFormatter(_stream_formatter)

logger.addHandler(_stream_handler)
logger.setLevel(logging.DEBUG)
