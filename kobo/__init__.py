import logging
import os

import export_kobo # type: ignore

_PATH = "/L/backups/kobo/"

def get_logger():
    return logging.getLogger('kobo-provider')

def _get_last_backup() -> str:
    import re
    RE = re.compile(r'\d{8}.*.sqlite$')
    last = max([f for f in os.listdir(_PATH) if RE.search(f)])
    return os.path.join(_PATH, last)


def get_kobo_data():
    logger = get_logger()
    bfile = _get_last_backup()

    logger.info(f"Using {bfile}")

    ex = export_kobo.ExportKobo()
    ex.vargs = {
        'db': bfile,
        'bookid': None,
        'book': None,
        'highlights_only': False,
        'annotations_only': False,
    }
    return ex.read_items()
# nn.extraannotationdata
# nn.kind
# nn.kindle_my_clippings
# nn.title
# nn.text
# nn.annotation
