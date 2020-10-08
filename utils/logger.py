import os
import logging
import time

def init_logger(save_path, log_file_name):
    head = '(asctime)-15s %(message)s'
    logging.basicConfig(filename=os.path.join(save_path, log_file_name+".log"), format=head)
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    return logger
