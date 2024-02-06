import os
import logging
import sys

log_str_frmt = "[%(asctime)s: %(levelname)s: %(module)s: %(message)s]"

log_dir = 'logs'
log_filepath = os.path.join(log_dir, 'logs.txt')


os.makedirs(log_dir, exist_ok=True)

logging.basicConfig(
    format=log_str_frmt,
    level=logging.INFO,
    handlers=[
        logging.FileHandler(log_filepath),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger('ScaniaLogger')
