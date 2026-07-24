import struct

from kobuddy.qdatastream import parse_qvariant_map


def _uint32(value: int) -> bytes:
    return struct.pack('>I', value)


def _int32(value: int) -> bytes:
    return struct.pack('>i', value)


def _byte_array(value: bytes | None) -> bytes:
    if value is None:
        return _uint32(0xFFFFFFFF)
    return _uint32(len(value)) + value


def _string(value: str) -> bytes:
    return _byte_array(value.encode('utf-16-be'))


def _variant(type_id: int, value: bytes, *, is_null: bool = False) -> bytes:
    return _uint32(type_id) + bytes([is_null]) + value


def _map(items: list[tuple[str, bytes]]) -> bytes:
    return _uint32(len(items)) + b''.join(_string(key) + value for key, value in items)


def test_parse_kobo_extra_data() -> None:
    data = bytes.fromhex(
        '0000000400000010005600690065007700540079007000650000000a00000000060054004f00430000003000450078007400720061004400610074006100520065006100640069006e006700530065007300730069006f006e00730000000200000000030000002e00450078007400720061004400610074006100520065006100640069006e0067005300650063006f006e0064007300000002000000000a00000028004500780074007200610044006100740061004400610074006500430072006500610074006500640000000a00000000280032003000310039002d00310031002d00320035005400300031003a00310037003a00310034005a'
    )

    assert parse_qvariant_map(data) == {
        'ViewType': 'TOC',
        'ExtraDataReadingSessions': 3,
        'ExtraDataReadingSeconds': 10,
        'ExtraDataDateCreated': '2019-11-25T01:17:14Z',
    }


def test_parse_all_observed_variant_types() -> None:
    nested = _map([('answer', _variant(3, _uint32(42)))])
    values = _map(
        [
            ('invalid', _variant(0, _byte_array(None), is_null=True)),
            ('bool', _variant(1, b'\x01')),
            ('int', _variant(2, _int32(-1))),
            ('uint', _variant(3, _uint32(2**32 - 1))),
            ('map', _variant(8, nested)),
            ('list', _variant(9, _uint32(2) + _variant(2, _int32(1)) + _variant(10, _string('two')))),
            ('string', _variant(10, _string('hello'))),
            ('bytes', _variant(12, _byte_array(b'\x00\x01'))),
        ]
    )

    assert parse_qvariant_map(values) == {
        'invalid': None,
        'bool': True,
        'int': -1,
        'uint': 2**32 - 1,
        'map': {'answer': 42},
        'list': [1, 'two'],
        'string': 'hello',
        'bytes': b'\x00\x01',
    }
