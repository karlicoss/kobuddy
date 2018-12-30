# pip3 install dataset
from datetime import datetime
import logging
import os
from typing import List, NamedTuple, Iterator
from typing_extensions import Protocol
import json
from os.path import basename

from kython.pdatetime import parse_mdatetime

import imp
export_kobo = imp.load_source('ekobo', '/L/Dropbox/repos/export-kobo/export-kobo.py') # type: ignore

import dataset # type: ignore

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
    # TODO move to kython


# TODO not sure, is protocol really necessary here?
# TODO maybe what we really want is parsing batch of dates? Then it's easier to guess the format.
class Event(Protocol):
    @property
    def dt(self) -> datetime: # TODO not sure.. optional?
        return self._dt

    # books don't necessarily have title/author, so this is more generic..
    @property
    def book(self) -> str:
        return self._book

    @property
    def eid(self) -> str:
        return self._eid # TODO ugh. properties with fallback??
    # TODO try ellipsis??

    @property
    def summary(self) -> str:
        return f'event in {self.book}'

    def __repr__(self) -> str:
        return f'{self.dt}: {self.summary}'


class Item(NamedTuple):
    w: export_kobo.Item

    @property
    def dt_created(self) -> datetime:
        return _parse_date(self.w.datecreated)

    # modified is either same as created or 0 timestamp. anyway, not very interesting
    @property
    def dt(self) -> datetime:
        return self.dt_created

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

class GenericEvent(Event):
    def __init__(self, dt: datetime, book: str, eid: str):
        self._dt = dt
        self._book = book
        self._eid = eid

class ProgressEvent(GenericEvent):
    def __init__(self, *args, prog: str, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.prog = prog

    @property
    def summary(self) -> str:
        return f'progress on {self.book}: {self.prog}'

class EventTypes:
    START = 'StartReadingBook'
    OPEN = 'OpenContent'
    PROGRESS = 'BookProgress'


class AnalyticsEvents:
    Id = 'Id'
    Attributes = 'Attributes'
    Timestamp = 'Timestamp'
    Type = 'Type'

class StartEvent(GenericEvent):
    @property
    def summary(self) -> str:
        return f'started reading {self.book}'

def _iter_events_aux() -> Iterator[Event]:
    for fname in _get_all_dbs():
        db = dataset.connect(f'sqlite:///{fname}')
        table = db.load_table('AnalyticsEvents')
        for row in table.all():
            AE = AnalyticsEvents
            eid, ts, tp, att = row[AE.Id], row[AE.Timestamp], row[AE.Type], row[AE.Attributes]
            ts = parse_mdatetime(ts) # TODO make dynamic?
            att = json.loads(att)
            if tp == EventTypes.START:
                descr = f"{att.get('title', '')} by {att.get('author', '')}"
                yield StartEvent(
                    dt=ts,
                    book=descr,
                    eid=eid,
                )
            elif tp == EventTypes.PROGRESS:
                prog = att.get('progress', '')
                vol = att.get('volumeid', '')
                descr = basename(vol) # TODO retrieve it somehow?..
                yield ProgressEvent(
                    dt=ts,
                    book=descr,
                    eid=eid,
                    prog=prog,
                )

# TODO mm, could also integrate it with goodreads api?...
def iter_events():
    seen = set()
    for x in _iter_events_aux():
        if x not in seen:
            seen.add(x)
            yield x

def get_events():
    return list(iter_events())
