# pip3 install dataset
from datetime import datetime
import logging
import os
from typing import List, NamedTuple, Iterator, Optional, Set, Tuple, Dict
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
    def eid(self) -> str:
        return self.w.bookmark_id # TODO use instead of iid?? make sure krill can handle it

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
        return f'progress on {self.book}: {self.prog}%'

class StartEvent(OtherEvent):
    @property
    def summary(self) -> str:
        return f'started reading {self.book}'

class FinishedEvent(OtherEvent):
    def __init__(self, *args, time_spent_s: int, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.time_spent_s = time_spent_s

    @property
    def time_spent(self) -> str:
        return "undefined" if self.time_spent_s == -1 else str(self.time_spent_s // 60)

    @property
    def summary(self) -> str:
        return f'finished reading {self.book}. total time spent {self.time_spent} minutes'

class LeaveEvent(OtherEvent):
    def __init__(self, *args, prog: str, seconds: int, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.prog = prog
        self.seconds = seconds

    @property
    def summary(self) -> str:
        return f'left {self.book}: {self.prog}%, read for {self.seconds // 60} mins'

class EventTypes:
    START = 'StartReadingBook'
    OPEN = 'OpenContent'
    PROGRESS = 'BookProgress'
    FINISHED = 'FinishedReadingBook'
    LEAVE_CONTENT = 'LeaveContent'


class AnalyticsEvents:
    Id = 'Id'
    Attributes = 'Attributes'
    Metrics = 'Metrics'
    Timestamp = 'Timestamp'
    Type = 'Type'

def _iter_events_aux() -> Iterator[Event]:
    logger = get_logger()
    for fname in _get_all_dbs():
        db = dataset.connect(f'sqlite:///{fname}', reflect_views=False)

        content_table = db.load_table('content')
        # wtf... that fails with some sqlalchemy crap
        # books = content_table.find(ContentType=6)
        # shit, no book id weirdly...
        books = db.query('SELECT * FROM content WHERE ContentType=6')
        title2time: Dict[str, int] = {}
        for b in books:
            title = b['Title']
            reading = b['TimeSpentReading']
            cur = 0
            if title in title2time:
                logger.warning('%s: trying to handle book twice! %s', fname, title)
                cur = title2time[title]
            title2time[title] = max(cur, reading)

        events_table = db.load_table('AnalyticsEvents')
        for row in events_table.all():
            AE = AnalyticsEvents
            eid, ts, tp, att, met = row[AE.Id], row[AE.Timestamp], row[AE.Type], row[AE.Attributes], row[AE.Metrics]
            ts = parse_mdatetime(ts) # TODO make dynamic?
            att = json.loads(att)
            met = json.loads(met)
            if tp == EventTypes.LEAVE_CONTENT:
                # TODO mm. keep only the last in group?...
                descr = f"{att.get('title', '')} by {att.get('author', '')}"
                prog = att.get('progress', '0')
                secs = int(met.get('SecondsRead', 0))
                ev = LeaveEvent(
                    dt=ts,
                    book=descr,
                    eid=eid,
                    prog=prog,
                    seconds=secs,
                )
                if secs >= 60:
                    yield ev
                else:
                    logger.info("skipping %s, it's too short", ev)
            elif tp == EventTypes.START:
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
            elif tp == EventTypes.FINISHED:
                title = att.get('title', '')
                author = att.get('author', '')
                descr = f"{title} by {author}"
                yield FinishedEvent(
                    dt=ts,
                    book=descr,
                    eid=eid,
                    time_spent_s=title2time.get(title, -1),
                )
            elif tp in (
                    'AdobeErrorEncountered',
                    'AppSettings',
                    'AppStart',
                    'BatteryLevelAtSync',
                    'BrightnessAdjusted',
                    '/Home',
                    'HomeWidgetClicked',
                    'MainNavOption',
                    'MarkAsUnreadPrompt',
                    'OpenReadingSettingsMenu',
                    'PluggedIn',
                    'ReadingSettings',
                    'StatusBarOption',
                    'StoreBookClicked',
                    'StoreHome',
                    'Books', # not even clear what's it for
            ):
                pass # just ignore
            elif tp in (
                    # This will be handled later..
                    'CreateBookmark',
                    'CreateHighlight',
            ):
                pass
            elif tp in (
                    # might handle later, but not now..
                    'DictionaryLookup',
                    'OpenContent',
                    'RemoveContent',
                    'Search',
            ):
                pass
            else:
                logger.warning(f'Unhandled entry of type {tp}: {row}')

def _iter_highlights() -> Iterator[Event]:
    logger = get_logger()
    bfile = _get_last_backup()

    logger.info(f"Using {bfile} for highlights")

    ex = export_kobo.ExportKobo() # type: ignore
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

# TODO Activity -- sort of interesting (e.g RecentBook). wonder what is Action (it's always 2)

# TODO mm, could also integrate it with goodreads api?...
def iter_events() -> Iterator[Event]:
    yield from _iter_highlights()

    seen: Set[Tuple[str, str]] = set()
    for x in _iter_events_aux():
        kk = (x.eid, x.summary)
        if kk not in seen:
            seen.add(kk)
            yield x

def get_events() -> List[Event]:
    def kkey(e):
        cls_order = 0
        if isinstance(e, LeaveEvent):
            cls_order = 1
        elif isinstance(e, ProgressEvent):
            cls_order = 2
        elif isinstance(e, FinishedEvent):
            cls_order = 3

        k = e.dt
        if k.tzinfo is None:
            k = k.replace(tzinfo=pytz.UTC)
        return (k, cls_order)
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

def get_todos():
    def with_todo(ann):
        if ann is None:
            ann = ''
        return 'todo' in ann.lower().split()
    return by_annotation(with_todo)

# TODO not sure where to associate it for...
# just extract later... if I ever want some stats
# TODO content database --  Readstatus (2, 1, 0), __PercentRead
# TODO it also contains lots of extra stuff...


def main():
    from kython.klogging import setup_logzero
    logger = get_logger()
    setup_logzero(logger, level=logging.info)
    for d in get_events():
        print(d)


if __name__ == '__main__':
    main()
