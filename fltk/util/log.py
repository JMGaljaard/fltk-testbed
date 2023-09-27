import logging

def getLogger(module_name, level = logging.INFO):
    logging.basicConfig(
        level=level,
        format='%(asctime)s %(levelname)s %(module)s - %(funcName)s: %(message)s',
    )
    return logging.getLogger(module_name)
