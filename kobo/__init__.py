#!/usr/bin/env python3
# pip3 install dataset
from datetime import datetime
import logging
import os
from typing import List, NamedTuple, Iterator, Optional, Set, Tuple, Dict, Sequence
from typing_extensions import Protocol
import json
from os.path import basename
from pathlib import Path
from functools import lru_cache
import pytz

from kython import cproperty, group_by_key, the
from kython.pdatetime import parse_mdatetime

import imp
export_kobo = imp.load_source('ekobo', '/L/zzz_syncthing/repos/export-kobo/export-kobo.py') # type: ignore

import dataset # type: ignore

_PATH = Path("/L/backups/kobo/")

def get_logger():
    return logging.getLogger('kobo-provider')

def _get_all_dbs() -> List[Path]:
    return list(sorted(_PATH.glob('*.sqlite')))


ContentId = str

# some things are only set for book parts: e.g. BookID, BookTitle
class Book(NamedTuple):
    content_id: ContentId
    isbn: str
    title: str
    author: str
    # time_spent: int # TODO not sure..
    # percent: int

    def __repr__(self):
        return f'{self.title} by {self.author}'

    @property
    def bid(self) -> str:
        return self.content_id # not sure but maybe it's fine...


# TODO not sure, is protocol really necessary here?
# TODO maybe what we really want is parsing batch of dates? Then it's easier to guess the format.
class Event(Protocol):
    @property
    def dt(self) -> datetime: # TODO deprecate?
        return self._dt # type: ignore

    @property
    def created(self) -> datetime:
        return self.dt

    # books don't necessarily have title/author, so this is more generic..
    @property
    def book(self) -> Book:
        return self._book # type: ignore

    @property
    def eid(self) -> str:
        return self._eid # type: ignore
        # TODO ugh. properties with fallback??

    @property
    def summary(self) -> str:
        return f'event in {self.book}'

    def __repr__(self) -> str:
        return f'{self.dt.strftime("%Y-%m-%d %H:%M:%S")}: {self.summary}'

def _parse_utcdt(s) -> Optional[datetime]:
    if s is None:
        return None
    res = parse_mdatetime(s)
    if res is None:
        return None
    else:
        if res.tzinfo is None:
            res = pytz.utc.localize(res)
        return res


class Highlight(Event):

    def __init__(self, w, book: Book):
        self.w = w
        self._book = book

    # modified is either same as created or 0 timestamp. anyway, not very interesting
    @property
    def dt(self) -> datetime:
        # I checked and it's definitely UTC
        res = _parse_utcdt(self.w.datecreated)
        assert res is not None
        return res

    # @property
    # def book(self) -> Book: # TODO FIXME should handle it carefully in kobo provider users
    #     return self.book
        # raise RuntimeError
        # return f'{self.title}' # TODO  by {self.author}'

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
    def __init__(self, dt: datetime, book: Book, eid: str):
        self._dt = dt
        self._book = book
        self._eid = eid

class MiscEvent(OtherEvent):
    def __init__(self, dt: datetime, book: Book, payload):
        self._dt = dt
        self._book = book
        self._eid = "TODO FIXME"
        self.payload = payload

    @property
    def summary(self) -> str:
        return str(self.payload)

class BookFinished(MiscEvent):
    def __init__(self, **kwargs):
        super().__init__(**kwargs, payload='finished') # type: ignore

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


# ugh. looks like it's only got one day of retention??
# TODO need to generate 'artificial' progress events from Books?
class AnalyticsEvents:
    Id = 'Id'
    Attributes = 'Attributes'
    Metrics = 'Metrics'
    Timestamp = 'Timestamp'
    Type = 'Type'


from kython import make_dict, group_by_key

class Books:
    def __init__(self) -> None:
        self.cid2books: Dict[str, List[Book]] = {}
        self.isbn2books: Dict[str, List[Book]] = {}
        self.title2books: Dict[str, List[Book]] = {}

    @staticmethod
    def _reg(dct, key, book):
        books = dct.get(key, [])
        present = any((b.title, b.author) == (book.title, book.author) for b in books)
        if not present:
            books.append(book)
            # TODO really need to split out time_spent and percent.....
        dct[key] = books

    @staticmethod
    def _get(dct, key) -> Optional[Book]:
        bb = dct.get(key, [])
        if len(bb) == 0:
            return None
        elif len(bb) == 1:
            return bb[0]
        else:
            raise RuntimeError(key)

    def add(self, book: Book):
        Books._reg(self.cid2books, book.content_id, book)
        Books._reg(self.isbn2books, book.isbn, book)
        Books._reg(self.title2books, book.title, book)

    def by_content_id(self, cid: ContentId) -> Optional[Book]:
        return Books._get(self.cid2books, cid)

    def by_isbn(self, isbn: str) -> Optional[Book]:
        return Books._get(self.isbn2books, isbn)

    def by_title(self, title: str) -> Optional[Book]:
        return Books._get(self.title2books, title)


    # TODO bad name..
    def by_dict(self, d) -> Book:
        vid = d.get('volumeid', None)
        isbn = d.get('isbn', None)
        # 20181021, volumeid and isbn are not present for StartReadingBook
        title = d.get('title', None)
        res = None
        if vid is not None:
            res = self.by_content_id(vid)
        elif isbn is not None:
            res = self.by_isbn(isbn)
        elif title is not None:
            res = self.by_title(title)
        assert res is not None
        return res

class Extra(NamedTuple):
    time_spent: int
    percent: int
    status: int
    last_read: Optional[datetime]


def get_books(db) -> List[Tuple[Book, Extra]]:
    logger = get_logger()
    content_table = db.load_table('content')
    # wtf... that fails with some sqlalchemy crap
    # books = content_table.find(ContentType=6)
    # shit, no book id weirdly...
    books = db.query('SELECT * FROM content WHERE ContentType=6')
    items = []
    for b in books:
        content_id = b['ContentID']
        isbn       = b['ISBN']
        title      = b['Title'].strip() # ugh. not sure about that, but sometimes helped 
        author     = b['Attribution']

        time_spent = b['TimeSpentReading']
        percent    = b['___PercentRead']
        status     = int(b['ReadStatus'])
        last_read  = b['DateLastRead']

        book = Book(
            content_id=content_id,
            isbn=isbn,
            title=title,
            author=author,
        )
        extra = Extra(
            time_spent=time_spent,
            percent=percent,
            status=status,
            last_read=_parse_utcdt(last_read),
        )
        items.append((book, extra))
    return items


def _iter_events_aux(limit=None, **kwargs) -> Iterator[Event]:
    class EventTbl:
        EventType      = 'EventType'
        EventCount     = 'EventCount'
        LastOccurrence = 'LastOccurrence'
        ContentID      = 'ContentID'
        # TODO ExtraData got some interesting blobs..

        class Types:
            BOOK_FINISHED = 5

    # 5 always appears as one of the last events. also seems to be same as DateLastRead. So reasonable to assume it means book finished
    # 80 occurs in pair with 5, but also sometimes repeats for dozens of times. 

    # TODO handle all_ here?
    logger = get_logger()
    dbs = _get_all_dbs()
    if limit is not None:
        # pylint: disable=invalid-unary-operand-type
        dbs = dbs[-limit:]

    books = Books()

    for fname in dbs:
        logger.info('processing %s', fname)
        db = dataset.connect(f'sqlite:///{fname}', reflect_views=False, ensure_schema=False) # TODO ??? 

        for b, extra in get_books(db):
            books.add(b)
            if extra.status == 2:
                dt = extra.last_read
                assert dt is not None
                yield BookFinished(dt=dt, book=b)

        ET = EventTbl
        for row in db.query(f'SELECT {ET.EventType}, {ET.EventCount}, {ET.LastOccurrence}, {ET.ContentID} from Event'): # TODO order by?
            tp, count, last, cid = row[ET.EventType], row[ET.EventCount], row[ET.LastOccurrence], row[ET.ContentID]
            if tp in (
                    46, # ???
                    80,
            ):
                continue

            if count != 1:
                # logger.debug('ignoring %s', row)
                continue

            dt = _parse_utcdt(last)
            assert dt is not None

            book = books.by_content_id(cid)
            # TODO should assert this in 'full' mode when we rebuild from the very start...
            # assert book is not None
            if book is None:
                logger.warning('book not found: %s', cid)
                continue

            if tp == ET.Types.BOOK_FINISHED:
                yield BookFinished(dt=dt, book=book)
            else:
                yield MiscEvent(dt=dt, book=book, payload=row)

        AE = AnalyticsEvents
        # events_table = db.load_table('AnalyticsEvents')
        # TODO ugh. used to be events_table.all(), but started getting some 'Mandatory' field with a wrong schema at some point...
        for row in db.query(f'SELECT {AE.Id}, {AE.Timestamp}, {AE.Type}, {AE.Attributes}, {AE.Metrics} from AnalyticsEvents'): # TODO order by??
            eid, ts, tp, att, met = row[AE.Id], row[AE.Timestamp], row[AE.Type], row[AE.Attributes], row[AE.Metrics]
            ts = parse_mdatetime(ts) # TODO make dynamic?
            att = json.loads(att)
            met = json.loads(met)
            if tp == EventTypes.LEAVE_CONTENT:
                # TODO mm. keep only the last in group?...
                # descr = f"{att.get('title', '')} by {att.get('author', '')}"
                # TODO volumeid

                book = books.by_dict(att)

                prog = att.get('progress', '0')
                # TODO progress is not always present??
                secs = int(met.get('SecondsRead', 0)) # TODO not sure..
                # TODO pages turned?
                ev = LeaveEvent(
                    dt=ts,
                    book=book,
                    eid=eid,
                    prog=prog,
                    seconds=secs,
                )
                if secs >= 60:
                    yield ev
                else:
                    logger.debug("skipping %s, it's too short", ev)
            elif tp == EventTypes.START:
                book = books.by_dict(att)
                yield StartEvent(
                    dt=ts,
                    book=book,
                    eid=eid,
                )
            elif tp == EventTypes.PROGRESS:
                book = books.by_dict(att)
                prog = att['progress']
                yield ProgressEvent(
                    dt=ts,
                    book=book,
                    eid=eid,
                    prog=prog,
                )
            elif tp == EventTypes.FINISHED:
                book = books.by_dict(att)
                # TODO maybe need to emit time_spent separately?..
                yield FinishedEvent(
                    dt=ts,
                    book=book,
                    eid=eid,
                    time_spent_s=-1, # TODO FIXME
                    # time_spent_s=book.time_spent,
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
                    'ChangedSetting',
                    'AmbientLightSensorToggled',
                    'QuickTurnTriggered',
                    'ButtonSwapPreferences',
                    'SearchExecuted',
            ):
                pass # just ignore
            elif tp in (
                    # This will be handled later..
                    'MarkAsFinished',
                    'CreateBookmark',
                    'CreateHighlight',
                    'CreateNote', # TODO??
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


def _get_last_backup() -> Path:
    return max(_get_all_dbs())

def _iter_highlights(**kwargs) -> Iterator[Highlight]:
    logger = get_logger()


    bfile = _get_last_backup() # TODO FIXME really last? or we want to get all??

    books = Books()
    db = dataset.connect(f'sqlite:///{bfile}', reflect_views=False, ensure_schema=False) # TODO ??? 
    for b, _ in get_books(db):
        books.add(b)



    # TODO SHIT! definitely, e.g. if you delete book from device, hightlights/bookmarks just go. can check on mathematician's apology from 20180902

    logger.info(f"Using {bfile} for highlights")

    ex = export_kobo.ExportKobo() # type: ignore
    ex.vargs = {
        'db': str(bfile),
        'bookid': None,
        'book': None,
        'highlights_only': False,
        'annotations_only': False,
    }
    for i in ex.read_items():
        book = books.by_content_id(i.volumeid)
        assert book is not None
        yield Highlight(i, book=book)
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
# TODO which order is that??
def iter_events(**kwargs) -> Iterator[Event]:
    yield from _iter_highlights(**kwargs)

    seen: Set[Tuple] = set()
    for x in _iter_events_aux(**kwargs):
        kk = (x.dt, x.book, x.eid, x.summary)
        if kk not in seen:
            seen.add(kk)
            yield x

def get_events(**kwargs) -> List[Event]:
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
    return list(sorted(iter_events(**kwargs), key=kkey))


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


def get_highlights(**kwargs) -> List[Highlight]:
    return list(sorted(_iter_highlights(**kwargs), key=lambda h: h.created))

def by_annotation(predicatish: Predicatish, **kwargs) -> List[Highlight]:
    pred = from_predicatish(predicatish)

    res: List[Highlight] = []
    for h in get_highlights(**kwargs):
        if pred(h.annotation):
            res.append(h)
    return res

def get_todos():
    def with_todo(ann):
        if ann is None:
            ann = ''
        return 'todo' in ann.lower().split()
    return by_annotation(with_todo)

class Page(NamedTuple):
    highlights: Sequence[Highlight]

    @cproperty
    def book(self) -> Book: # TODO is gonna be used outside..
        return the(h.book for h in self.highlights)

    @cproperty
    def dt(self) -> datetime:
        # makes more sense to move 'last modified' pages to the end
        return max(h.dt for h in self.highlights)


def get_pages(**kwargs) -> List[Page]:
    highlights = get_highlights(**kwargs)
    grouped = group_by_key(highlights, key=lambda e: e.book)
    pages = []
    for book, group in grouped.items():
        sgroup = tuple(sorted(group, key=lambda e: e.created))
        pages.append(Page(highlights=sgroup))
    pages = list(sorted(pages, key=lambda p: p.dt))
    return pages

# TODO content database --  Readstatus (2, 1, 0), __PercentRead
# -- use it to estimate progress over time -- not activity apparently disappears often :(

# TODO Event table? got all sort of things from june 2017
# 1012 looks like starting to read..
# 46 -- opening?
# 1012/1013 -- finishing?

# TODO not sure where to associate it for...
# just extract later... if I ever want some stats
# TODO it also contains lots of extra stuff...

def test_todos():
    todos = get_todos()
    assert len(todos) > 3

def test_get_all():
    events = get_events()
    assert len(events) > 50
    for d in events:
        print(d)

def test_pages():
    pages = get_pages()
    assert len(pages) > 10
    for p in pages:
        print(p)


def main():
    from kython.klogging import setup_logzero
    logger = get_logger()
    setup_logzero(logger, level=logging.DEBUG)

    # TODO also events shouldn't be cumulative?
    evts = iter_events(limit=10)
    # evts = filter(lambda x: not isinstance(x, Highlight), evts)

    from kython import group_by_key
    for book, events in group_by_key(evts, key=lambda e: e.book).items():
        # if book.content_id != 'b09b236c-9a6e-44b7-9727-8187d98d8419':
        #     continue
        print()
        print(type(book), book, book.content_id)
        for e in sorted(events, key=lambda e: e.dt): # TODO not sure when should be sorted
            # TODO shit. offset
            print("-- " + str(e))
    # test_pages()
    # test_get_all()

# import sys; exec("global info\ndef info(type, value, tb):\n    import ipdb, traceback; traceback.print_exception(type, value, tb); ipdb.pm()"); sys.excepthook = info # type: ignore


if __name__ == '__main__':
    main()
