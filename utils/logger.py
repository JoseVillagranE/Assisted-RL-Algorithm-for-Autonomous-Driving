import os
import logging
import time

def init_logger(save_path, run_id):

    log_file = f"{run_id}_{time.strftime('%Y-%m-%d-%H-%M')}.log"
    head = '(asctime)-15s %(message)s'
    logging.basicConfig(filename=os.path.join(save_path, log_file), format=head)
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    return logger
