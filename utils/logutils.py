import os
import logging
from termcolor import colored
import sys
import time
from datetime import datetime

def create_logger(output_dir, name=''):
    # create logger
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.propagate = False

    # create formatter
    fmt = '[%(asctime)s %(name)s] (%(filename)s %(lineno)d): %(levelname)s %(message)s'
    color_fmt = colored('[%(asctime)s %(name)s]', 'green') + \
                colored('(%(filename)s %(lineno)d)', 'yellow') + ': %(levelname)s %(message)s'

    # create console handlers for master process
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.DEBUG)
    console_handler.setFormatter(
        logging.Formatter(fmt=color_fmt, datefmt='%Y-%m-%d_%H:%M:%S'))
    logger.addHandler(console_handler)

    # 获取当前日期和时间，格式为 YYYY-MM-DD_HH-MM-SS
    current_time = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

    # create file handlers
    file_handler = logging.FileHandler(os.path.join(output_dir, f'log_{current_time}.txt'), mode='a')
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter(fmt=fmt, datefmt='%Y-%m-%d %H:%M:%S'))
    logger.addHandler(file_handler)

    return logger