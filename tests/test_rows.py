from collections import OrderedDict

import kobuddy
from kobuddy import Highlight, Book


def test_hl_no_datecreated():
    book = Book(title='test title', author='test author', content_id='fwfwf', isbn='2424')
    row = OrderedDict([
        ('BookmarkID', 'bbbbb'),
        ('VolumeID', 'vvvv'),
        ('ContentID', 'cccc'),
        ('StartContainerPath', 'span#kobo\\.31\\.4'),
        ('StartContainerChildIndex', -99),
        ('StartOffset', 117),
        ('EndContainerPath', 'span#kobo\\.31\\.5'),
        ('EndContainerChildIndex', -99),
        ('EndOffset', 141),
        ('Text', '<blah blah quote>'),
        ('Annotation', None),
        ('ExtraAnnotationData', None),
        ('DateCreated', None),
        ('ChapterProgress', 0.75),
        ('Hidden', 'false'),
        ('Version', ''),
        ('DateModified', '2014-05-17T21:05:35Z'),
        ('Creator', None),
        ('UUID', None),
        ('UserID', '6fc7'),
        ('SyncTime', '2019-05-21T16:46:17Z'),
        ('Published', 'false'),
    ])
    hl = Highlight(row, book)
    # TODO can we automate this?
    for attr in [a for a in dir(Highlight) if not a.startswith('_')]:
        value = getattr(hl, attr)
        assert value is not None


