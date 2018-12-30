# pip3 install dataset
from datetime import datetime
import logging
import os
from typing import List, NamedTuple, Iterator, Optional, Set
from typing_extensions import Protocol
import json
from os.path import basename
import pytz

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

# TODO not sure, is protocol really necessary here?
# TODO maybe what we really want is parsing batch of dates? Then it's easier to guess the format.
class Event(Protocol):
    @property
    def dt(self) -> Optional[datetime]:
        return self._dt # type: ignore

    # books don't necessarily have title/author, so this is more generic..
    @property
    def book(self) -> str:
        return self._book # type: ignore

    @property
    def eid(self) -> str:
        return self._eid # type: ignore
        # TODO ugh. properties with fallback??

    @property
    def summary(self) -> str:
        return f'event in {self.book}'

    def __repr__(self) -> str:
        return f'{self.dt}: {self.summary}'


class Highlight(Event):

    def __init__(self, w):
        self.w = w

    # modified is either same as created or 0 timestamp. anyway, not very interesting
    @property
    def dt(self) -> Optional[datetime]:
        return parse_mdatetime(self.w.datecreated)

    @property
    def book(self) -> str:
        return f'{self.title}' # TODO  by {self.author}'

    @property
    def summary(self) -> str:
        return f"{self.kind} in {self.book}"

    # this is what's actually hightlighted
    @property
    def text(self) -> str:
        return self.w.text

    # TODO optional??
    @property
    def annotation(self):
        return self.w.annotation

    @property
    def iid(self):
        # TODO shit. this is used in kobo provider.. just use eid instead..
        return self.datecreated

    @property
    def eid(self) -> str:
        return self.w.bookmark_id # TODO use instead of iid?? make sure krill can handle it

    @property
    def datecreated(self):
        return self.w.datecreated

    @property
    def kind(self) -> str:
        return self.w.kind

    @property
    def title(self) -> str:
        return self.w.title

class OtherEvent(Event):
    def __init__(self, dt: datetime, book: str, eid: str):
        self._dt = dt
        self._book = book
        self._eid = eid

class ProgressEvent(OtherEvent):
    def __init__(self, *args, prog: str, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.prog = prog

    @property
    def summary(self) -> str:
        return f'progress on {self.book}: {self.prog}'

class StartEvent(OtherEvent):
    @property
    def summary(self) -> str:
        return f'started reading {self.book}'

class EventTypes:
    START = 'StartReadingBook'
    OPEN = 'OpenContent'
    PROGRESS = 'BookProgress'


class AnalyticsEvents:
    Id = 'Id'
    Attributes = 'Attributes'
    Timestamp = 'Timestamp'
    Type = 'Type'

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

def _iter_highlights() -> Iterator[Event]:
    logger = get_logger()
    bfile = _get_last_backup()

    logger.info(f"Using {bfile} for highlights")

    ex = export_kobo.ExportKobo()
    ex.vargs = {
        'db': bfile,
        'bookid': None,
        'book': None,
        'highlights_only': False,
        'annotations_only': False,
    }
    for i in ex.read_items():
        yield Highlight(i)
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

# TODO mm, could also integrate it with goodreads api?...
def iter_events() -> Iterator[Event]:
    yield from _iter_highlights()

    seen = set() # type: Set[Event]
    for x in _iter_events_aux():
        if x not in seen:
            seen.add(x)
            yield x

def get_events() -> List[Event]:
    # TODO shit, dt unaware..
    def kkey(e):
        k = e.dt
        if k.tzinfo is None:
            k = k.replace(tzinfo=pytz.UTC)
        return k
    return list(sorted(iter_events(), key=kkey))


from typing import Callable, Union
# TODO maybe type over T?
_Predicate = Callable[[str], bool]
Predicatish = Union[str, _Predicate]
def from_predicatish(p: Predicatish) -> _Predicate:
    if isinstance(p, str):
        def ff(s):
            return s == p
        return ff
    else:
        return p


def by_annotation(predicatish: Predicatish) -> List[Highlight]:
    pred = from_predicatish(predicatish)

    datas = get_events()
    res: List[Highlight] = []
    for d in datas:
        if not isinstance(d, Highlight):
            continue
        if pred(d.annotation):
            res.append(d)
    return res
