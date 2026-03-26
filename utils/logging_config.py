import logging
import sys

def configure_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='[%(asctime)s][%(filename)s][%(lineno)d][%(funcName)s] - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=[
            logging.StreamHandler(sys.stdout)
        ]
    )
