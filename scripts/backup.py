#!/usr/bin/env python3
import argparse
import json
import sys
from subprocess import check_output
from pathlib import Path
from datetime import datetime
import shutil


def get_mountpoint(label: str):
    xxx = check_output(['lsblk', '-f', '--json']).decode('utf8')
    jj = json.loads(xxx)
    kobos = [d for d in jj['blockdevices'] if d.get('label', None) == label]
    if len(kobos) > 1:
        raise RuntimeError(f'Multiple Kobo devices detected: {kobos}')
    elif len(kobos) == 0:
        return None
    else:
        [kobo] = kobos
        return Path(kobo['mountpoint'])


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--label', default='KOBOeReader', help="device label (check lsblk if default doesn't work)")
    p.add_argument('path', type=Path, help='directory or file to dump the database')
    args = p.parse_args()

    target = args.path
    if target.is_dir():
        today = datetime.now().strftime('%Y%m%d')
        target = target / f'KoboReader-{today}.sqlite'

    # TODO maybe use db mtime?
    if target.exists(): # we don't want to overwrite. maybe configure that with flags?
        # TODO log??
        return

    mount = get_mountpoint(label=args.label)
    if mount is None:
        return

    db = mount / '.kobo/KoboReader.sqlite'
    assert db.exists()

    print(f'Backing up Kobo database to {target}', file=sys.stderr)
    # TODO atomic copy in kython?
    tmp = target.with_suffix('.tmp')
    shutil.copy(db, tmp)
    tmp.rename(target)


if __name__ == '__main__':
    main()
