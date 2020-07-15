#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path
from datetime import datetime
import shutil

from .kobo_device import get_kobo_mountpoint


def run(args):
    target = args.path
    if target.is_dir():
        today = datetime.now().strftime('%Y%m%d')
        # TODO a bit meh...
        target = target / f'KoboReader-{today}.sqlite'

    # TODO maybe use db mtime?
    if target.exists():
        # we don't want to overwrite. maybe configure that with flags?
        # TODO log??
        return

    mount = get_kobo_mountpoint(label=args.label)
    if mount is None:
        return

    db = mount / '.kobo/KoboReader.sqlite'
    assert db.exists()

    print(f'Backing up Kobo database to {target}', file=sys.stderr)
    tmp = target.with_suffix('.tmp')
    shutil.copy(db, tmp)
    tmp.rename(target)


def setup_parser(p):
    p.add_argument('--label', default='KOBOeReader', help="device label (check lsblk if default doesn't work)")
    p.add_argument('path', type=Path, help='target directory or file to dump the database')


def main():
    p = argparse.ArgumentParser()
    setup_parser(p)
    args = p.parse_args()
    run(args)


if __name__ == '__main__':
    main()
