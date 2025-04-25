import logging
from mmengine.hooks import LoggerHook


def setup_logger(log_file='default.log'):
    logger = logging.getLogger('mmdet_logger')
    logger.setLevel(logging.INFO)

    # Create file handler
    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.INFO)

    # Create formatter and add it to the handlers
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)

    # Add the handlers to the logger
    logger.addHandler(fh)

    # Initialize LoggerHook
    logger_hook = LoggerHook(out_dir=log_file)
    return logger_hook


# 使用示例
# logger_hook = setup_logger('my_log.log')
# logger_hook.log('This is a log message.')
