"""
Kobuddy is a tool to backup Kobo Reader sqlite database and extract useful things from it.

It gives you access to books, annotations, progress events and more!

Tested on Kobo Aura One, however database format shouldn't be different on other devices.
I'll happily accept PRs if you find any issues or want to help with reverse engineering more events.
"""
import warnings
from itertools import chain
import json
import shutil
import struct
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from tempfile import NamedTemporaryFile
import typing
from typing import (Dict, Iterator, List, NamedTuple, Optional, Sequence, Set,
                    Tuple, Union, Iterable, Any)

from .common import get_logger, unwrap, cproperty, group_by_key, the, nullcontext, Res, sorted_res, split_res
from .kobo_device import get_kobo_mountpoint
from .sqlite import sqlite_connection


# a bit nasty to have a global variable here... will rewrite later
DATABASES: List[Path] = []

# I use all databases to merge a unified data export since
# e.g. if you delete book from device, hightlights/bookmarks just disappear


def set_databases(dbp: Optional[Union[Path, str]], label='KOBOeReader'):
    if dbp is None:
        mount = get_kobo_mountpoint(label=label)
        if mount is None:
            raise RuntimeError(f"Coulnd't find mounted Kobo device with label '{label}', are you sure it's connected? (perhaps try using different label?)")
        db = mount / '.kobo' / 'KoboReader.sqlite'
        @contextmanager
        def tmp_db():
            # hacky way to use tmp file for database..
            # TODO figure out how to properly open db in read only mode..
            with NamedTemporaryFile() as tf:
                shutil.copy(db, tf.name)
                DATABASES.append(Path(tf.name))
                yield
        return tmp_db()
    else:
        dbp = Path(dbp)
        if dbp.is_dir():
            DATABASES.extend(sorted(dbp.rglob('*.sqlite')))
        else:
            DATABASES.append(dbp)
        return nullcontext()


ContentId = str

# some things are only set for book parts: e.g. BookID, BookTitle
class Book(NamedTuple):
    title: str
    author: str
    content_id: ContentId
    isbn: str

    def __repr__(self):
        return f'{self.title} by {self.author}'

    @property
    def bid(self) -> str:
        return self.content_id # not sure but maybe it's fine...


if typing.TYPE_CHECKING:
    # todo: since 3.8 can use from typing import Protocol everywhere
    from typing_extensions import Protocol
else:
    Protocol = object

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

def _parse_utcdt(s: Optional[str]) -> Optional[datetime]:
    if s is None:
        return None

    res = None
    if s.endswith('Z'):
        s = s[:-1] # python can't handle it...
    for fmt in (
            '%Y-%m-%dT%H:%M:%S',
            '%Y-%m-%dT%H:%M:%S.%f', # contained in older database exports
    ):
        try:
            res = datetime.strptime(s, fmt)
            # starting from certain date, microseconds stopped appearing in Kobo database, so we normalise it
            res = res.replace(microsecond=0)
            break
        except ValueError:
            continue
    assert res is not None

    if res.tzinfo is None:
        res = res.replace(tzinfo=timezone.utc)
    return res


# TODO not so sure about inheriting event..
class Highlight(Event):
    def __init__(self, row: Dict[str, Any], book: Book) -> None:
        self.row = row
        self._book = book

    def _error(self, msg: str) -> Exception:
        return RuntimeError(f'Error while processing {self.row}: {msg}')

    @property
    def dt(self) -> datetime:
        """
        Returns DateCreated.

        On some devices may not be set, so falls back to DateModified (see https://github.com/karlicoss/kobuddy/issues/1 )

        Returns date is in UTC (tested on Kobo Aura One).
        """
        # on Kobo Aura One modified was either same as created or 0 timestamp
        date_attrs = ('DateCreated', 'DateModified')
        for dattr in date_attrs:
            # TODO could warn/log if it's not using DateModified
            res = _parse_utcdt(self.row[dattr])
            if res is not None:
                return res
        raise self._error(f"Couldn't infer date from {date_attrs}")

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
        # sometimes annotation is None, but doesn't seem to be much point to tell the difference with empty annotation
        return self.row['Annotation'] or ''

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

    @property
    def _key(self):
        return (self.dt, self.text, self.annotation)

    def __eq__(self, o):
        return self._key == o._key

    def __hash__(self):
        return hash(self._key)

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
        return 'started'

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


class Books:
    def __init__(self, create_if_missing=False) -> None:
        self.cid2books: Dict[str, List[Book]] = {}
        self.isbn2books: Dict[str, List[Book]] = {}
        self.title2books: Dict[str, List[Book]] = {}
        self.create_if_missing = create_if_missing

    @staticmethod
    def _reg(dct, key, book):
        books = dct.get(key, [])
        present = any((b.title, b.author) == (book.title, book.author) for b in books)
        if not present:
            books.append(book)
            # TODO really need to split out time_spent and percent.....
        dct[key] = books

    @staticmethod
    def _get(dct, key, *, allow_multiple: bool=False) -> Optional[Book]:
        bb = dct.get(key, [])
        if len(bb) == 0:
            return None
        if len(bb) == 1:
            return bb[0]
        if allow_multiple:
            return bb[-1]
        raise RuntimeError(f"Multiple items for {key}: {bb}")

    def all(self) -> List[Book]:
        bset = set()
        for d in [self.cid2books, self.isbn2books, self.title2books]:
            for l in d.values():
                bset.update(l)
        return list(sorted(bset, key=lambda b: b.title))

    def make_orphan_book(self, *, volumeid: str) -> Book:
        # sometimes volumeid might be pretty cryptic, like a random uuid (e.g. for native kobo books)
        # but there isn't much we can do about it -- there isn't any info in Bookmark table
        orphan = Book(
            title=f'<DELETED BOOK {volumeid}>',
            author='<kobuddy>',
            content_id=volumeid,
            isbn='fake_isbn_{volumeid}',
        )
        self.add(orphan)
        return orphan

    def add(self, book: Book) -> None:
        Books._reg(self.cid2books, book.content_id, book)
        Books._reg(self.isbn2books, book.isbn, book)
        Books._reg(self.title2books, book.title, book)

    def by_content_id(self, cid: ContentId) -> Optional[Book]:
        # sometimes title might get updated.. so it's possible to have same contentid with multiple titles
        return Books._get(self.cid2books, cid, allow_multiple=True)

    def by_isbn(self, isbn: str) -> Optional[Book]:
        return Books._get(self.isbn2books, isbn, allow_multiple=True)

    def by_title(self, title: str) -> Optional[Book]:
        return Books._get(self.title2books, title)


    # TODO not a great name?
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

        if res is None and self.create_if_missing:
            book = Book(
                title=title or '???',
                author=title or vid or isbn or '???',
                content_id=vid or '???',
                isbn=isbn or '???',
            )
            self.add(book)
            res = book
        assert res is not None, d
        return res


class Extra(NamedTuple):
    time_spent: int
    percent: int
    status: int
    last_read: Optional[datetime]


def _load_books(db: sqlite3.Connection) -> List[Tuple[Book, Extra]]:
    logger = get_logger()
    items: List[Tuple[Book, Extra]] = []
    books = db.execute('SELECT * FROM content WHERE ContentType=6')
    for b in books:
        content_id = b['ContentID']
        isbn       = b['ISBN']
        title      = b['Title'].strip() # ugh. not sure about that, but sometimes helped
        author     = b['Attribution']

        # TODO not so sure about that; it was the case for KoboShelfes databases
        time_spent = 0 if 'TimeSpentReading' not in b.keys() else b['TimeSpentReading']
        percent    = b['___PercentRead']
        status     = int(b['ReadStatus'])
        last_read  = b['DateLastRead']

        mimetype    = b['MimeType']
        if mimetype == 'application/x-kobo-html+pocket':
            # skip Pocket articles
            continue

        if mimetype == 'image/png':
            # skip images
            continue

        user_id = b['___UserID']
        if user_id == '':
            # that seems to mean that book is an ad, not an actual book loaded on device
            continue

        book = Book(
            title=title,
            author=author,
            content_id=content_id,
            isbn=isbn,
        )
        extra = Extra(
            time_spent=time_spent,
            percent=percent,
            status=status,
            last_read=_parse_utcdt(last_read),
        )
        items.append((book, extra))
    return items


# TODO Event table? got all sort of things from june 2017
# 1012 looks like starting to read..
# 1012/1013 -- finishing?
class EventTbl:
    EventType      = 'EventType'
    EventCount     = 'EventCount'
    LastOccurrence = 'LastOccurrence'
    ContentID      = 'ContentID'
    Checksum       = 'Checksum'
    ExtraData      = 'ExtraData'
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

        # ??? from  KoboShelfes db
        T73 = 73 # 3 of these
        T74 = 74 # 2 of these
        T27 = 27 # 3 of these
        T28 = 28 # 5 of these
        T36 = 36 # 6 of these
        #


# TODO use literal mypy types?
def _iter_events_aux(limit=None, errors='throw') -> Iterator[Res[Event]]:
    # TODO handle all_ here?
    logger = get_logger()
    dbs = DATABASES
    if limit is not None:
        # pylint: disable=invalid-unary-operand-type
        dbs = dbs[-limit:]

    # TODO do it if it's defensive?
    books = Books(create_if_missing=True)


    def connections():
        for fname in dbs:
            logger.info(f'processing {fname}')
            with sqlite_connection(fname, immutable=True, row_factory='row') as db:
                yield fname, db


    for fname, db in connections():
        for b, extra in _load_books(db):
            books.add(b)
            if extra is None:
                continue
            dt = extra.last_read
            if extra.status == 2:
                assert dt is not None
                yield FinishedEvent(dt=dt, book=b, time_spent_s=extra.time_spent, eid=f'{b.content_id}-{fname.name}')

        ET = EventTbl
        for i, row in enumerate(db.execute(f'SELECT {ET.EventType}, {ET.EventCount}, {ET.LastOccurrence}, {ET.ContentID}, {ET.Checksum}, hex({ET.ExtraData}) from Event')): # TODO order by?
            try:
                yield from _iter_events_aux_Event(row=row, books=books, idx=i)
            except Exception as e:
                if   errors == 'throw':
                    raise e
                elif errors == 'return':
                    yield e
                else:
                    raise AssertionError(f'errors={errors}')

        AE = AnalyticsEvents
        # events_table = db.load_table('AnalyticsEvents')
        # TODO ugh. used to be events_table.all(), but started getting some 'Mandatory' field with a wrong schema at some point...
        for row in db.execute(f'SELECT {AE.Id}, {AE.Timestamp}, {AE.Type}, {AE.Attributes}, {AE.Metrics} from AnalyticsEvents'): # TODO order by??
            try:
                yield from _iter_events_aux_AnalyticsEvents(row=row, books=books)
            except Exception as e:
                if   errors == 'throw':
                    raise e
                elif errors == 'return':
                    yield e
                else:
                    raise AssertionError(f'errors={errors}')


# TODO FIXME remove idx
def _iter_events_aux_Event(*, row, books: Books, idx=0) -> Iterator[Event]:
    logger = get_logger()
    ET = EventTbl
    ETT = ET.Types
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
            ETT.T73,
            ETT.T74,
            ETT.T27,
            ETT.T28,
            ETT.T36,
    ):
        return

    # TODO should assert this in 'full' mode when we rebuild from the very start...
    # assert book is not None
    book = books.by_dict({
        'volumeid': cid,
    })
    # TODO FIXME need unique uid...
    # TODO def needs tests.. need to run ignored through tests as well
    if tp not in (ETT.T3, ETT.T1021, ETT.PROGRESS_25, ETT.PROGRESS_50, ETT.PROGRESS_75, ETT.BOOK_FINISHED):
        logger.error('unexpected event: %s %s', book, row)
        raise RuntimeError(str(row))

    blob = bytearray.fromhex(extra_data)

    pos = 0
    parsed = {} # type: Dict[bytes, Any]

    def context():
        return f'row: {row}\nblob: {blob}\n remaining: {blob[pos:]}\n parsed: {parsed}\n xxx {blob[pos:pos+30]}\n idx: {idx}\n parts: {parts}\n pos: {pos}'

    def consume(fmt):
        nonlocal pos
        sz = struct.calcsize(fmt)
        try:
            res = struct.unpack_from(fmt, blob, offset=pos)
        except Exception as e:
            raise RuntimeError(context()) from e
        pos += sz
        return res


    lengths = {
        b'ExtraDataSyncedCount'      : 9,
        b'PagesTurnedThisSession'    : 9,
        b'IsMarkAsFinished'          : 6,
        b'ExtraDataReadingSessions'  : 9,
        b'ExtraDataReadingSeconds'   : 9,
        b'ContentType'               : None,

        # TODO weird, these contain stringified dates
        b'ExtraDataLastModified'     : 49,
        b'ExtraDataDateCreated'      : 49,

        b'ExtraDataSyncedTimeElapsed': None,

        # TODO eh, wordsRead is pretty weird; not sure what's the meaning. some giant blob.
        b'wordsRead'                 : 9,

        b'Monetization'              : None,
        b'ViewType'                  : None,
        b'eventTimestamps'           : None,
        b'wordCounts'                : None,

        # TODO not so sure... these might be part of monetization
        b'Sideloaded'                : 0,
        b'Paid'                      : 0,
        b'Preview'                   : 0,
    }

    parts, = consume('>I')
    # ugh. apparently can't trust parts?
    # for _ in range(parts):
    while pos < len(blob):
        wtf = b'\x000'
        if blob[pos:].startswith(wtf):
            pos += len(wtf)
            continue
        part_name_len, = consume('>I')
        if part_name_len == 0:
            break
        # sanity check...
        assert part_name_len < 1000, context()
        assert part_name_len % 2 == 0, context()
        fmt = f'>{part_name_len}s'
        prename, = consume(fmt)
        # looks like b'\x00P\x00a\x00g\x00e\x00s'
        assert prename[::2].count(0) == part_name_len // 2, context()
        name = prename[1::2]

        if name not in lengths:
            raise RuntimeError(f'Unexpected event kind: {name}\n' + context())
        part_len = lengths[name]

        if part_len is not None:
            part_data = consume(f'>{part_len}s')
        elif name == b'eventTimestamps':
            cnt, = consume('>5xI')
            dts = []
            for _ in range(cnt):
                ts, = consume('>5xI')
                dt = datetime.fromtimestamp(ts, tz=timezone.utc)
                dts.append(dt)
            part_data = dts
        elif name == b'ViewType':
            vt_len, = consume('>5xI')
            vt_body, = consume(f'>{vt_len}s')
            part_data = vt_body
        elif name == b'Monetization':
            qqq, = consume('>5s')
            if qqq != b'\x00\x00\x00\n\x00':
                _, = consume('>4s') # no idea what's that..
            continue
        elif name == b'ExtraDataSyncedTimeElapsed':
            qqq, = consume('>5s4x')
            # TODO wtf???
            if qqq == b'\x00\x00\x00\n\x00':
                consume('>2x')
            continue
        elif name == b'wordCounts':
            vt_cnt, = consume('>5xI')
            vt_len = vt_cnt * 9
            vt_body, = consume(f'>{vt_len}s')
            part_data = vt_body
        elif name == b'ContentType':
            qqq, = consume('>4s')
            if qqq == b'\x00\x00\x00\x00':
                # wtf?
                consume('>5x')
                continue
            vt_len, = consume('>xI')
            vt_body, = consume(f'>{vt_len}s')
            part_data = vt_body
        else:
            raise RuntimeError('Expected fixed length\n' + context())

        parsed[name] = part_data

    assert pos == len(blob), context()

    # assert cnt == count # weird mismatches do happen. I guess better off trusting binary data

    # TODO FIXME handle remaining items
    timestamps = parsed.get(b'eventTimestamps', []) # type: List[datetime]

    for i, x in enumerate(timestamps):
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

def _iter_events_aux_AnalyticsEvents(*, row, books: Books) -> Iterator[Event]:
    logger = get_logger()
    AE = AnalyticsEvents
    eid, ts, tp, att, met = row[AE.Id], row[AE.Timestamp], row[AE.Type], row[AE.Attributes], row[AE.Metrics]
    ts = _parse_utcdt(ts) # TODO make dynamic?
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
            'Sideload',
            'Extras',
            'AutoColorToggled',
            'WifiSettings',
            'WifiToggle',
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
            'AccessLibrary',
            'LibrarySort',
            'TileSelected',
            'UserMetadataUpdate',
    ):
        pass
    else:
        logger.warning(f'Unhandled entry of type {tp}: {row}')


def _get_books() -> Books:
    books = Books()
    for bfile in DATABASES:
        with sqlite_connection(bfile, immutable=True, row_factory='row') as db:
            for b, _ in _load_books(db):
                books.add(b)
    return books


def get_books() -> List[Book]:
    return _get_books().all()


def _iter_highlights(**kwargs) -> Iterator[Highlight]:
    logger = get_logger()
    books = _get_books()

    # todo more_itertools?
    yielded: Set[Highlight] = set()
    for bfile in DATABASES:
        for h in _load_highlights(bfile, books=books):
            if h not in yielded:
                yield h
                yielded.add(h)


def _load_highlights(bfile: Path, books: Books) -> Iterator[Highlight]:
    logger = get_logger()
    logger.info(f"Using {bfile} for highlights")
    with sqlite_connection(bfile, immutable=True, row_factory='row') as db:
        for bm in db.execute('SELECT * FROM Bookmark'):
            volumeid = bm['VolumeID']
            mbook = books.by_content_id(volumeid)
            if mbook is None:
                # sometimes Kobo seems to recycle old books without recycling the corresponding bookmarks
                # so we need to be a bit defensive here
                # see https://github.com/karlicoss/kobuddy/issues/18
                book = books.make_orphan_book(volumeid=volumeid)
            else:
                book = mbook
            # todo maybe in the future it could be a matter of error policy, i.e. throw vs yield exception vs use orphan object vs ignore
            # could be example of useful defensiveness in a provider
            yield Highlight(bm, book=book)


def _load_wordlist(bfile: Path):
    logger = get_logger()
    logger.info(f"Using {bfile} for wordlist")
    with sqlite_connection(bfile, immutable=True, row_factory='row') as db:
        for bm in db.execute('SELECT * FROM WordList'):
            yield bm['Text']


def get_highlights(**kwargs) -> List[Highlight]:
    return list(sorted(_iter_highlights(**kwargs), key=lambda h: h.created))


def get_wordlist() -> Iterator[str]:
    yielded: Set[str] = set()

    for bfile in DATABASES:
        for h in _load_wordlist(bfile):
            if h not in yielded:
                yield h
                yielded.add(h)


# TODO Activity -- sort of interesting (e.g RecentBook). wonder what is Action (it's always 2)

# TODO not sure if need to be exposed
def iter_events(**kwargs) -> Iterator[Res[Event]]:
    yield from _iter_highlights(**kwargs)

    seen: Set[Tuple] = set()
    for x in _iter_events_aux(**kwargs):
        if isinstance(x, Exception):
            yield x
            continue
        kk = (x.dt, x.book, x.summary)
        if kk not in seen:
            seen.add(kk)
            yield x

# TODO is this even used apart from tests??
def get_events(**kwargs) -> List[Res[Event]]:
    def kkey(e):
        cls_order = 0
        if isinstance(e, ProgressEvent):
            cls_order = 2
            # TODO might need to get rid of it..
        elif isinstance(e, FinishedEvent):
            cls_order = 3

        k = e.dt
        if k.tzinfo is None:
            k = k.replace(tzinfo=timezone.utc)
        return (k, cls_order)
    return list(sorted_res(iter_events(**kwargs), key=kkey))


class BookWithHighlights(NamedTuple):
    highlights: Sequence[Highlight]

    @cproperty
    def book(self) -> Book: # TODO is gonna be used outside..
        return the(h.book for h in self.highlights)

    @cproperty
    def dt(self) -> datetime:
        # makes more sense to move 'last modified' pages to the end
        return max(h.dt for h in self.highlights)


def get_books_with_highlights(**kwargs) -> List[BookWithHighlights]:
    highlights = get_highlights(**kwargs)
    grouped = group_by_key(highlights, key=lambda e: e.book)
    res = []
    for book, group in grouped.items():
        sgroup = tuple(sorted(group, key=lambda e: e.created))
        res.append(BookWithHighlights(highlights=sgroup))
    return list(sorted(res, key=lambda p: p.dt))


# TODO need to merge 'progress' and 'left'

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
        for e in reversed(self.events):
            if isinstance(e, FinishedEvent):
                return e.dt
        return None

    @property
    def last(self) -> datetime:
        return self.events[-1].dt


def iter_books_with_events(**kwargs) -> Iterator[Res[BookEvents]]:
    evts = iter_events(**kwargs)
    vit, eit = split_res(evts)
    for book, events in group_by_key(vit, key=lambda e: e.book).items():
        yield BookEvents(book, events)
    yield from eit


def get_books_with_events(**kwargs) -> Sequence[Res[BookEvents]]:
    it = iter_books_with_events(**kwargs)
    vit: Iterable[BookEvents]
    vit, eit = split_res(it)
    vit = sorted(vit, key=lambda be: be.last)
    return list(chain(vit, eit))


def _fmt_dt(dt: datetime) -> str:
    return dt.strftime('%d %b %Y %H:%M')


def print_progress(full=True, **kwargs) -> None:
    logger = get_logger()
    for bevents in get_books_with_events(**kwargs):
        if isinstance(bevents, Exception):
            logger.exception(bevents)
            continue
        print()
        sts = None if bevents.started is None else _fmt_dt(bevents.started) # weird but sometimes it is None..
        fns = '' if bevents.finished  is None else _fmt_dt(bevents.finished)
        print(bevents.book)
        # TODO hmm, f-strings not gonna work in py 3.5
        print(f'Started : {sts}')
        print(f'Finished: {fns}')
        if full:
            for e in bevents.events:
                print(f"-- {_fmt_dt(e.dt)}: {e.summary}")


def print_books() -> None:
    for b in get_books():
        print(b)


def print_annotations() -> None:
    for i in get_highlights():
        h = f"""
{_fmt_dt(i.dt)} {i._book}
    {i.text}
        {i.annotation}
""".strip('\n')
        print(h)
        print("------")


def print_wordlist() -> None:
    for i in get_wordlist():
        print(i)
