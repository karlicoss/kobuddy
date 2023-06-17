#!/usr/bin/env python3
import logging
import argparse
from pathlib import Path

from kobuddy import set_databases, print_progress, print_books, print_annotations, print_wordlist, get_logger

def setup_logger(logger, level=None, format=None, datefmt=None):
    import logging
    old_root = logging.root
    try:
        logging.root = logger
        logging.basicConfig(
            level=level or logging.DEBUG,
            format=format or '%(name)s %(asctime)s %(levelname)-8s %(filename)s:%(lineno)-4d %(message)s',
            datefmt=datefmt or '%Y-%m-%d %H:%M:%S',
        )
    finally:
        logging.root = old_root


def main():
    # TODO FIXME need to use proper timzone..
    logger = get_logger()
    setup_logger(logger, level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')


    Fmt = lambda prog: argparse.RawTextHelpFormatter(prog, width=100)
    p = argparse.ArgumentParser(
        description="""
Library to parse and provide Python interface for your Kobo reader
""",
        formatter_class=Fmt,
    )
    p.add_argument('--db', type=Path, help='''
By default will try to read the database from your Kobo device.
If you pass a directory, will try to use all Kobo databases it can find.
    ''', required=False)
    p.add_argument('--errors', choices=['throw', 'return'], default='throw', help="throw: raise on errors immediately; return: handle defensively long as possible and reasonable")
    p.add_argument('--label', default='KOBOeReader', help="device label (check lsblk if default doesn't work)")
    sp = p.add_subparsers(dest='mode')
    sp.add_parser('books'      , help='print all books')
    sp.add_parser('progress'   , help='print all book reading progress')
    sp.add_parser('annotations', help='print all annotations (bookmarks/highlights/comments)')
    sp.add_parser('wordlist'   , help='print all words from the wordlist')
    bp = sp.add_parser('backup', help='backup the database from your Kobo device', description='''
You can run it via cron, for example every minute. When you connect your device via USB, the database will be backed up.

: * * * * * kobuddy backup /path/to/backups/kobo/

Alternatively, you can add a udev rule or something similar.
''', formatter_class=Fmt)
    import kobuddy.backup
    kobuddy.backup.setup_parser(bp)

    args = p.parse_args()

    if args.mode == 'backup':
        kobuddy.backup.run(args)
        return

    with set_databases(args.db, label=args.label):
        if args.mode == 'progress':
            print_progress(errors=args.errors)
        elif args.mode == 'books':
            print_books()
        elif args.mode == 'annotations':
            print_annotations()
        elif args.mode == 'wordlist':
            print_wordlist()
        else:
            raise RuntimeError(f'Unexpected mode {args.mode}')


if __name__ == '__main__':
    main()
