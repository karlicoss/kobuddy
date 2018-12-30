import logging

from kobo import get_events, get_logger, iter_events

from kython.logging import setup_logzero

logger = get_logger()
setup_logzero(logger, level=logging.INFO)

# import sys, ipdb, traceback; exec("def info(type, value, tb):\n    traceback.print_exception(type, value, tb)\n    ipdb.pm()"); sys.excepthook = info # type: ignore

for d in iter_events():
    print(d)
