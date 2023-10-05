from datetime import datetime, timezone
from pathlib import Path

import kobuddy


def get_test_db() -> Path:
    testdata = Path(__file__).absolute().parent.parent.parent.parent / 'testdata'
    assert testdata.exists(), testdata
    db = testdata / 'kobo_notes' / 'input' / 'KoboReader.sqlite'
    assert db.exists(), db
    return db

# a bit meh, but ok for now
kobuddy.set_databases(get_test_db())

from kobuddy import _iter_events_aux, get_events, get_books_with_highlights, _iter_highlights


def test_events():
    for e in _iter_events_aux():
        print(e)


def test_hls():
    for h in _iter_highlights():
        print(h)


def test_get_all():
    events = get_events()
    assert len(events) > 50
    for d in events:
        print(d)


def test_books_with_highlights():
    pages = get_books_with_highlights()

    g = pages[0]
    assert 'Essentialism' in g.book
    hls = g.highlights
    assert len(hls) == 273

    [b] = [h for h in hls if h.eid == '520b7b13-dbef-4402-9a81-0f4e0c4978de']
    # TODO wonder if there might be any useful info? StartContainerPath, EndContainerPath
    assert b.kind == 'bookmark'

    # TODO move to a more specific test?
    # TODO assert sorted by date or smth?
    assert hls[0].kind == 'highlight'
    # TODO assert highlights got no annotation? not sure if it's even necessary to distinguish..

    [ann] = [h for h in hls if h.annotation is not None and len(h.annotation) > 0]

    assert ann.eid == 'eb264817-9a06-42fd-92ff-7bd38cd9ca79'
    assert ann.kind == 'annotation'
    assert ann.text == 'He does this by finding which machine has the biggest queue of materials waiting behind it and finds a way to increase its efficiency.'
    assert ann.annotation == 'Bottleneck'
    assert ann.dt == datetime(year=2017, month=8, day=12, hour=3, minute=49, second=13, microsecond=0, tzinfo=timezone.utc)
    assert ann.book.author == 'Greg McKeown'

    assert len(pages) == 7


def test_history():
    kobuddy.print_progress()

def test_annotations():
    kobuddy.print_annotations()

def test_books():
    kobuddy.print_books()
