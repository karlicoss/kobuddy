from pathlib import Path

import kobuddy

# TODO ugh, horrible
def get_test_dbs():
    # db = Path(__file__).absolute().parent.parent / 'KoboShelfes' / 'KoboReader.sqlite.0'
    db = Path(__file__).absolute().parent.parent / 'kobo_notes' / 'input' / 'KoboReader.sqlite'
    return [db]
kobuddy._get_all_dbs = get_test_dbs

from kobuddy import _iter_events_aux

def test_events():
    for e in _iter_events_aux():
        print(e)
