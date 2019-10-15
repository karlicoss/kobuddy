#!/usr/bin/env python3
import logging
import argparse
from pathlib import Path

from kobuddy import set_databases, print_progress, print_books, print_annotations, get_logger

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


    p = argparse.ArgumentParser(description="""
Library to parse and provide Python interface for your Kobo reader
""")
    p.add_argument('--db', type=Path, help='''
By default will try to read the database from your Kobo device.
If you pass a directory, will try to use all Kobo databases it can find.
    ''', required=False)
    p.add_argument('--errors', choices=['throw', 'return'], default='throw', help="throw: raise on errors immediately; return: handle defensively long as possible and reasonable")
    sp = p.add_subparsers(dest='mode')
    sp.add_parser('books'      , help='print all books')
    sp.add_parser('progress'   , help='print all book reading progress')
    sp.add_parser('annotations', help='print all annotations (bookmarks/highlights/comments)')
    # TODO FIXME document..

    args = p.parse_args()

    with set_databases(args.db):
        if args.mode == 'progress':
            print_progress(errors=args.errors)
        elif args.mode == 'books':
            print_books()
        elif args.mode == 'annotations':
            print_annotations()
        else:
            raise RuntimeError(f'Unexpected mode {args.mode}')


if __name__ == '__main__':
    main()
