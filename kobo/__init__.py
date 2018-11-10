from datetime import datetime
import logging
import os
from typing import List, NamedTuple


import imp
export_kobo = imp.load_source('ekobo', '/L/Dropbox/repos/export-kobo/export-kobo.py') # type: ignore

_PATH = "/L/backups/kobo/"

def get_logger():
    return logging.getLogger('kobo-provider')

def _get_last_backup() -> str:
    import re
    RE = re.compile(r'\d{8}.*.sqlite$')
    last = max([f for f in os.listdir(_PATH) if RE.search(f)])
    return os.path.join(_PATH, last)

def _parse_date(dts: str) -> datetime:
    for f in (
            "%Y-%m-%dT%H:%M:%S.%f",
            "%Y-%m-%dT%H:%M:%SZ",
    ):
        try:
            return datetime.strptime(dts, f)
        except ValueError:
            pass
    else:
        raise RuntimeError(f"Could not parse {dts}")


class Item(NamedTuple):
    w: export_kobo.Item

    @property
    def dt_created(self) -> datetime:
        return _parse_date(self.w.datecreated)

    @property
    def dt_modified(self) -> datetime:
        return _parse_date(self.w.datemodified)

    @property
    def dt(self) -> datetime:
        return max(self.dt_created, self.dt_modified)

    @property
    def summary(self) -> str:
        return f"{self.w.kind} in {self.w.title}"

    @property
    def annotation(self):
        return self.w.annotation

    @property
    def text(self):
        return self.w.text

    @property
    def iid(self):
        return self.w.datecreated

def get_datas():
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
    return [Item(i) for i in ex.read_items()]
# nn.extraannotationdata
# nn.kind
# nn.kindle_my_clippings
# nn.title
# nn.text
# nn.annotation
