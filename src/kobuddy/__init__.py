#!/usr/bin/env python3
# pip3 install dataset
from pkg_resources import get_distribution, DistributionNotFound

try:
    # Change here if project is renamed and does not equal the package name
    dist_name = __name__
    __version__ = get_distribution(dist_name).version
except DistributionNotFound:
    __version__ = 'unknown'
finally:
    del get_distribution, DistributionNotFound


from datetime import datetime
import logging
import os
from typing import List, NamedTuple, Iterator, Optional, Set, Tuple, Dict, Sequence
from typing_extensions import Protocol
import json
from pathlib import Path
from functools import lru_cache
import struct
import pytz
import warnings

from kython import cproperty, group_by_key, the
from kython.kerror import unwrap
from kython.pdatetime import parse_mdatetime

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
        return f'event in {self.book}' # TODO exclude book from summary? and use it on site instead (need to check in timeline)

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

    # TODO pass books objec?
    def __init__(self, row: Dict[str, str], book: Book):
        self.row = row
        self._book = book

    # modified is either same as created or 0 timestamp? anyway, not very interesting
    @property
    def dt(self) -> datetime:
        # I checked and it's definitely UTC
        return unwrap(_parse_utcdt(self.row['DateCreated']))

    # @property
    # def book(self) -> Book: # TODO FIXME should handle it carefully in kobo provider users
    #     return self.book
        # raise RuntimeError
        # return f'{self.title}' # TODO  by {self.author}'

    # TODO ?? include text?
    @property
    def summary(self) -> str:
        return f"{self.kind}"

    # this is what's actually hightlighted
    @property
    def text(self) -> Optional[str]:
        """
        Highlighted text in the book
        """
        return self.row['Text']

    @property
    def annotation(self) -> str:
        """
        Your comment
        """
        # always non-null judging by db
        return unwrap(self.row['Annotation'])

    @property
    def eid(self) -> str:
        return unwrap(self.row['BookmarkID'])

    @property
    def kind(self) -> str:
        text = self.text
        if text is None:
            return 'bookmark'
        else:
            ann = self.annotation
            if len(ann) > 0:
                return 'annotation'
            else:
                return 'highlight'

            # TODO why title??
    # @property
    # def title(self) -> str:
    #     return self.w.title

class OtherEvent(Event):
    def __init__(self, dt: datetime, book: Book, eid: str):
        self._dt = dt
        self._book = book
        self._eid = eid

class MiscEvent(OtherEvent):
    def __init__(self, dt: datetime, book: Book, payload, eid: str):
        self._dt = dt
        self._book = book
        self._eid = eid
        self.payload = payload

    @property
    def summary(self) -> str:
        return str(self.payload)

class ProgressEvent(OtherEvent):
    def __init__(self, *args, prog: Optional[int]=None, seconds_read: Optional[int]=None, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.prog = prog
        self.seconds_read = seconds_read

    @property
    def verbose(self) -> str:
        progs = '' if self.prog is None else f': {self.prog}%'
        read_for = '' if self.seconds_read is None else f', read for {self.seconds_read // 60} mins'
        return progs + read_for

    @property
    def summary(self) -> str:
        return 'reading' + self.verbose

    # TODO FIXME use progress event instead? 
class StartEvent(OtherEvent):
    @property
    def summary(self) -> str:
        return f'started'

class FinishedEvent(OtherEvent):
    def __init__(self, *args, time_spent_s: Optional[int]=None, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.time_spent_s = time_spent_s

    # @property
    # def time_spent(self) -> str:
    #     return "undefined" if self.time_spent_s == -1 else str(self.time_spent_s // 60)

    @property
    def summary(self) -> str:
        tss = '' if self.time_spent_s is None else f', total time spent {self.time_spent_s // 60} mins'
        return 'finished' + tss

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
            raise RuntimeError(f"Multiple items for {key}: {bb}")

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


def load_books(db) -> List[Tuple[Book, Extra]]:
    logger = get_logger()
    content_table = db.load_table('content')
    # wtf... that fails with some sqlalchemy crap
    # books = content_table.find(ContentType=6)
    # shit, no book id weirdly...
    items: List[Tuple[Book, Extra]] = []
    # for book in MANUAL_BOOKS:
    #     items.append((book, None))

    books = db.query('SELECT * FROM content WHERE ContentType=6')
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
    # TODO Event table? got all sort of things from june 2017
    # 1012 looks like starting to read..
    # 1012/1013 -- finishing?
    class EventTbl:
        EventType      = 'EventType'
        EventCount     = 'EventCount'
        LastOccurrence = 'LastOccurrence'
        ContentID      = 'ContentID'
        Checksum       = 'Checksum'
        ExtraData = 'ExtraData'
        # TODO ExtraData got some interesting blobs..

        class Types:
            # occurs occasionally throughout reading.. first event looks like a good candidate for 'started to read'
            T3 = 3

            # always appears as one of the last events. also seems to be same as DateLastRead. So reasonable to assume it means book finished
            BOOK_FINISHED = 5

            # could be book purchases? although for one book, occurst 4 times so doesn't make sense..
            T7 = 7

            # not sure what are these, doen't have contentid and accumulates throughout history
            T0 = 0
            T1 = 1
            T6 = 6
            T8 = 8
            T79 = 79

            # looks like dictionary lookups (judging to DictionaryName in blob)
            T9 = 9

            # 80 occurs in pair with 5, but also sometimes repeats for dozens of times.
            T80 = 80

            # some random totally unrelated timestamp appearing for some Epubs
            T37 = 37

            # happens very often, sometimes in bursts of 5 over 5 seconds. could be page turns?
            T46 = 46

            PROGRESS_25 = 1012
            # TODO hmm. not sure if progress evennts are true for books which are actually still in progress though..
            PROGRESS_50 = 1013
            PROGRESS_75 = 1014

            # almost always occurs in pairs with T3, but not sure what is it
            T1020 = 1020

            # 1021 seems to coincide with 'progress'
            T1021 = 1021

            # ??? in test database
            T99999 = 99999

            # ??? appeared on 20190701
            T4 = 4
            T68 = 68

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

        for b, extra in load_books(db):
            books.add(b)
            if extra is None:
                continue
            dt = extra.last_read
            if extra.status == 2:
                assert dt is not None
                yield FinishedEvent(dt=dt, book=b, time_spent_s=extra.time_spent, eid=f'{b.content_id}-{fname.name}')

        ET = EventTbl
        ETT = ET.Types
        for row in db.query(f'SELECT {ET.EventType}, {ET.EventCount}, {ET.LastOccurrence}, {ET.ContentID}, {ET.Checksum}, hex({ET.ExtraData}) from Event'): # TODO order by?
            tp, count, last, cid, checksum, extra_data = row[ET.EventType], row[ET.EventCount], row[ET.LastOccurrence], row[ET.ContentID], row[ET.Checksum], row[f'hex({ET.ExtraData})']
            if tp in (
                    ETT.T37,
                    ETT.T7,
                    ETT.T9,
                    ETT.T1020,
                    ETT.T80,
                    ETT.T46,

                    ETT.T0, ETT.T1, ETT.T6, ETT.T8, ETT.T79,

                    ETT.T1021,
                    ETT.T99999,
                    ETT.T4,
                    ETT.T68,
            ):
                continue

            # TODO should assert this in 'full' mode when we rebuild from the very start...
            # assert book is not None
            book = books.by_content_id(cid)
            if book is None:
                # TODO not sure about warnings..  maybe add to books class?
                warnings.warn(f'book not found: {row}')
                continue

            # TODO FIXME need unique uid...
            # TODO def needs tests.. need to run ignored through tests as well
            if tp not in (ETT.T3, ETT.T1021, ETT.PROGRESS_25, ETT.PROGRESS_50, ETT.PROGRESS_75, ETT.BOOK_FINISHED):
                logger.error('unexpected event: %s %s', book, row)
                raise RuntimeError(str(row)) # TODO return kython.Err

            blob = bytearray.fromhex(extra_data)
            if tp == ETT.T46:
                # it's gote some extra stuff before timestamps for we need to locate timestamps first
                found = blob.find(b'\x00e\x00v') # TODO this might be a bit slow, could be good to decypher how to jump straight to start of events
                if found == -1:
                    assert count == 0 # meh
                    continue
                blob = blob[found - 8:]

            data_start = 47

            dts = []
            _, zz, _, cnt = struct.unpack('>8s30s5sI', blob[:data_start])
            assert zz[1::2] == b'eventTimestamps'
            # assert cnt == count # weird mismatches do happen. I guess better off trusting binary data

            data_end = data_start + (5 + 4) * cnt
            for ts,  in struct.iter_unpack('>5xI', blob[data_start: data_end]):
                dts.append(pytz.utc.localize(datetime.utcfromtimestamp(ts)))
            for i, x in enumerate(dts):
                eid = checksum + "_" + str(i)

                if tp == ETT.T3:
                    yield ProgressEvent(dt=x, book=book, eid=eid)
                elif tp == ETT.BOOK_FINISHED:
                    yield FinishedEvent(dt=x, book=book, eid=eid)
                elif tp == ETT.PROGRESS_25:
                    yield ProgressEvent(dt=x, book=book, prog=25, eid=eid)
                elif tp == ETT.PROGRESS_50:
                    yield ProgressEvent(dt=x, book=book, prog=50, eid=eid)
                elif tp == ETT.PROGRESS_75:
                    yield ProgressEvent(dt=x, book=book, prog=75, eid=eid)
                else:
                    yield MiscEvent(dt=x, book=book, payload='EVENT ' + str(tp), eid=eid)

        AE = AnalyticsEvents
        # events_table = db.load_table('AnalyticsEvents')
        # TODO ugh. used to be events_table.all(), but started getting some 'Mandatory' field with a wrong schema at some point...
        for row in db.query(f'SELECT {AE.Id}, {AE.Timestamp}, {AE.Type}, {AE.Attributes}, {AE.Metrics} from AnalyticsEvents'): # TODO order by??
            eid, ts, tp, att, met = row[AE.Id], row[AE.Timestamp], row[AE.Type], row[AE.Attributes], row[AE.Metrics]
            ts = parse_mdatetime(ts) # TODO make dynamic?
            att = json.loads(att)
            met = json.loads(met)
            if tp == EventTypes.LEAVE_CONTENT:
                book = books.by_dict(att)
                prog = att.get('progress', None) # sometimes it doesn't actually have it (e.g. in Oceanic)
                secs = int(met['SecondsRead'])
                # TODO pages turned in met
                ev = ProgressEvent(
                    dt=ts,
                    book=book,
                    prog=prog,
                    seconds_read=secs,
                    eid=eid,
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
                # TODO IsMarkAsFinished?
                yield FinishedEvent(
                    dt=ts,
                    book=book,
                    eid=eid,
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
    # TODO FIXME books should be merged from all available sources

    # TODO dispose?
    db = dataset.connect(f'sqlite:///{bfile}', reflect_views=False, ensure_schema=False) # TODO ??? 
    for b, _ in load_books(db):
        books.add(b)


    # TODO SHIT! definitely, e.g. if you delete book from device, hightlights/bookmarks just go. can check on mathematician's apology from 20180902

    logger.info(f"Using %s for highlights", bfile)

    # TODO returns result?
    for bm in db.query('SELECT * FROM Bookmark'):
        volumeid = bm['VolumeID']
        book = books.by_content_id(volumeid)
        assert book is not None # TODO defensive?
        # TODO rename from Highlight?
        yield Highlight(bm, book=book)

# TODO Activity -- sort of interesting (e.g RecentBook). wonder what is Action (it's always 2)

# TODO mm, could also integrate it with goodreads api?...
# TODO which order is that??

# TODO not sure if need to be exposed
def iter_events(**kwargs) -> Iterator[Event]:
    yield from _iter_highlights(**kwargs)

    seen: Set[Tuple] = set()
    for x in _iter_events_aux(**kwargs):
        kk = (x.dt, x.book, x.summary)
        if kk not in seen:
            seen.add(kk)
            yield x

# TODO is this even used apart from tests??
def get_events(**kwargs) -> List[Event]:
    def kkey(e):
        cls_order = 0
        if isinstance(e, ProgressEvent):
            cls_order = 2
            # TODO might need to get rid of it..
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

# TODO move to private provider..
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


# TODO need to reuse fully assembled highlights, from all backups
# TODO give better names?
def get_pages(**kwargs) -> List[Page]:
    highlights = get_highlights(**kwargs)
    grouped = group_by_key(highlights, key=lambda e: e.book)
    pages = []
    for book, group in grouped.items():
        sgroup = tuple(sorted(group, key=lambda e: e.created))
        pages.append(Page(highlights=sgroup))
    pages = list(sorted(pages, key=lambda p: p.dt))
    return pages


# TODO need to merge 'progress' and 'left'
from kython import group_by_key

def _event_key(evt):
    tie_breaker = 0
    if isinstance(evt, ProgressEvent):
        tie_breaker = 1
    elif isinstance(evt, FinishedEvent):
        tie_breaker = 2
    return (evt.dt, tie_breaker)

class BookEvents:
    def __init__(self, book: Book, events):
        assert all(e.book == book for e in events)
        self.book = book
        self.events = list(sorted(events, key=_event_key))

    @property
    def started(self) -> Optional[datetime]:
        for e in self.events:
            if isinstance(e, ProgressEvent):
                return e.dt
        return None

    @property
    def finished(self) -> Optional[datetime]:
        # TODO go from end?
        for e in self.events:
            if isinstance(e, FinishedEvent):
                return e.dt
        return None

    @property
    def last(self) -> datetime:
        return self.events[-1].dt

def iter_books(**kwargs):
    evts = iter_events(**kwargs)
    for book, events in group_by_key(evts, key=lambda e: e.book).items():
        yield BookEvents(book, events)

def get_books(**kwargs):
    return list(sorted(iter_books(**kwargs), key=lambda be: be.last))

def iter_book_events(**kwargs):
    for b in get_books(**kwargs):
        yield from b.events

def print_history(**kwargs):
    for bevents in get_books(**kwargs):
        print()
        print(bevents.book, bevents.started, bevents.finished)
        for e in bevents.events:
            print("-- " + str(e))


def main():
    from kython.klogging import setup_logzero
    logger = get_logger()
    setup_logzero(logger, level=logging.DEBUG)

    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--limit', type=int)
    p.add_argument('mode', nargs='?')
    args = p.parse_args()
    if args.mode == 'history':
        print_history(limit=args.limit)
    else:
        assert args.mode is None
        raise NotImplementedError


    # TODO also events shouldn't be cumulative?
    # evts = iter_events() # limit=5)
    # evts = filter(lambda x: not isinstance(x, Highlight), evts)

    # for book, events in group_by_key(evts, key=lambda e: e.book).items():
    #     # if book.content_id != 'b09b236c-9a6e-44b7-9727-8187d98d8419':
    #     #     continue
    #     print()
    #     print(type(book), book, book.content_id)
    #     for e in sorted(events, key=lambda e: e.dt): # TODO not sure when should be sorted
    #         # TODO shit. offset
    #         print("-- " + str(e))
    # test_pages()
    # test_get_all()

# import sys; exec("global info\ndef info(type, value, tb):\n    import ipdb, traceback; traceback.print_exception(type, value, tb); ipdb.pm()"); sys.excepthook = info # type: ignore


if __name__ == '__main__':
    main()
