import logging

from fltk.util.config.definitions import LogLevel


def getLogger(module_name, level: LogLevel = LogLevel.INFO):
    logging.basicConfig(
        level=level.value,
        format='%(asctime)s %(levelname)s %(module)s - %(funcName)s: %(message)s',
    )
    return logging.getLogger(module_name)
