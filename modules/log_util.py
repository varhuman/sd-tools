import logging
import sys
from logging.handlers import TimedRotatingFileHandler
import os

def setup_logger(name, log_dir, level=logging.INFO):
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')

    log_file = os.path.join(log_dir, "my_log_file.log")
    
    file_handler = TimedRotatingFileHandler(log_file, when="midnight", interval=1, backupCount=7)
    file_handler.suffix = "%Y-%m-%d" # 日志文件名后缀
    file_handler.setFormatter(formatter)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)

    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger

log_directory = "logs"
logger = setup_logger("sd_tools_log", log_directory)