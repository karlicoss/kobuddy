import logging

from kobo import get_kobo_data, get_logger

from kython.logging import setup_logzero

logger = get_logger()
setup_logzero(logger, level=logging.INFO)

datas = get_kobo_data()

for d in datas:
    print(d)
