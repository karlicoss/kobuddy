import logging

from kobo import get_datas, get_logger

from kython.logging import setup_logzero

logger = get_logger()
setup_logzero(logger, level=logging.INFO)

datas = get_datas()

import sys, ipdb, traceback; exec("def info(type, value, tb):\n    traceback.print_exception(type, value, tb)\n    ipdb.pm()"); sys.excepthook = info # type: ignore

for d in datas:
    print(d.dt)
    print(d.summary)
