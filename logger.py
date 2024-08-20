import logging
from os import path as osp


def init_logger(log_name, log_path=None):
    logger = logging.getLogger(log_name)
    logger.setLevel(logging.DEBUG)

    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    if log_path is not None:
        fh = logging.FileHandler(osp.join(log_path, f'{log_name}.log'))
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    sh = logging.StreamHandler()
    sh.setLevel(logging.DEBUG)
    sh.setFormatter(formatter)
    logger.addHandler(sh)

    # logging.basicConfig(filename=osp.join(log_path, log_name), level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s',
    #                     datefmt='%d/%m/%Y %H:%M:%S')
    # logging.getLogger().addHandler(logging.StreamHandler())
    logger.info(f'Initialization Success: {log_name}')
    return logger
