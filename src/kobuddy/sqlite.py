# copy from HPI:my/core/sqlite.py
from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, Literal, Union

PathIsh = Union[Path, str]


SqliteRowFactory = Callable[[sqlite3.Cursor, sqlite3.Row], Any]


def dict_factory(cursor: sqlite3.Cursor, row: sqlite3.Row) -> dict[str, Any]:
    fields = [column[0] for column in cursor.description]
    return dict(zip(fields, row))


Factory = Union[SqliteRowFactory, Literal['row', 'dict']]


@contextmanager
def sqlite_connection(db: PathIsh, *, immutable: bool = False, row_factory: Factory | None = None) -> Iterator[sqlite3.Connection]:
    dbp = f'file:{db}'
    # https://www.sqlite.org/draft/uri.html#uriimmutable
    if immutable:
        # assert results in nicer error than sqlite3.OperationalError
        assert Path(db).exists(), db
        dbp = f'{dbp}?immutable=1'
    row_factory_: Any = None
    if row_factory is not None:
        if callable(row_factory):
            row_factory_ = row_factory
        elif row_factory == 'row':
            row_factory_ = sqlite3.Row
        elif row_factory == 'dict':
            row_factory_ = dict_factory
        else:
            raise RuntimeError(f"Can't happen: {row_factory}")

    conn = sqlite3.connect(dbp, uri=True)
    try:
        conn.row_factory = row_factory_
        with conn:
            yield conn
    finally:
        # Connection context manager isn't actually closing the connection, only keeps transaction
        conn.close()
