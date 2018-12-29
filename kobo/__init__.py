from datetime import datetime
import logging
import os
from typing import List, NamedTuple
import json
from os.path import basename


import imp
export_kobo = imp.load_source('ekobo', '/L/Dropbox/repos/export-kobo/export-kobo.py') # type: ignore

_PATH = "/L/backups/kobo/"

def get_logger():
    return logging.getLogger('kobo-provider')

def _get_all_dbs() -> List[str]:
    import re
    RE = re.compile(r'\d{8}.*.sqlite$')
    return list(sorted([os.path.join(_PATH, f) for f in os.listdir(_PATH) if RE.search(f)]))


def _get_last_backup() -> str:
    return max(_get_all_dbs())

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
        return f"{self.kind} in {self.title}"

    @property
    def title(self) -> str:
        return self.w.title

    @property
    def kind(self) -> str:
        return self.w.kind

    @property
    def annotation(self):
        return self.w.annotation

    @property
    def text(self):
        return self.w.text

    @property
    def iid(self):
        return self.datecreated

    @property
    def bid(self) -> str:
        return self.w.bookmark_id # TODO use instead of iid?? make sure krill can handle it

    @property
    def datecreated(self):
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
    # TODO eh. item is barely useful, it's just putting sqlite row into an object. just use raw query?
# nn.extraannotationdata
# nn.kind
# nn.kindle_my_clippings
# nn.title
# nn.text
# nn.annotation

# TODO ugh. better name?

# TODO maybe, just query from all annotations?
# basically, I want
# query stuff with certain annotations ('krill')
# query stuff with certain text properties? (one line only)
# comparator to merge them (iid is fine??)
def by_annotation(ann: str):
    datas = get_datas()
    res = []
    for d in datas:
        a = d.annotation
        if a is None:
            continue
        if ann.lower() == a.lower().strip():
            res.append(d)
    return res

class EventTypes:
    START = 'StartReadingBook'
    OPEN = 'OpenContent'
    PROGRESS = 'BookProgress'


class AnalyticsEvents:
    Attributes = 'Attributes'
    Timestamp = 'Timestamp'
    Type = 'Type'


def iter_start_reading():
    import sqlite3

    # TODO wtf?? why didn't dataset work??
    # import dataset # type: ignore
    # TODO dataset??
    # for fname in _get_all_dbs():
        # db = dataset.connect(f'sqlite://{fname}')
        # table = db.load_table('AnalyticsEvents')
        # for x in table.all():
        #     yield x
    for fname in _get_all_dbs():
        query = f'select {AnalyticsEvents.Timestamp},{AnalyticsEvents.Type},{AnalyticsEvents.Attributes} from AnalyticsEvents'
        # TODO with?
        with sqlite3.connect(fname) as conn:
            cursor = conn.cursor()
            cursor.execute(query)
            datas = cursor.fetchall()
            for ts, tp, att in datas:
                att = json.loads(att)
                if tp == EventTypes.START:
                    descr = f"{att.get('title', '')} by {att.get('author', '')}"
                    yield f'{ts}: started reading {descr}'
                elif tp == EventTypes.PROGRESS:
                    prog = att.get('progress', '')
                    vol = att.get('volumeid', '')
                    descr = basename(vol) # TODO retrieve it somehow?..
                    yield f'{ts}: progress on {descr}: {prog}'
            cursor.close()

# TODO mm, could also integrate it with goodreads api?...
def start_reading():
    seen = set()
    res = []
    for x in iter_start_reading():
        if x not in seen:
            seen.add(x)
            res.append(x)
    return res
