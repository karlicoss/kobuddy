from __future__ import annotations

import struct
from typing import Any

type QVariant = bool | int | str | bytes | list[QVariant] | dict[str, QVariant] | None
type QVariantMap = dict[str, QVariant]


class _Reader:
    def __init__(self, data: bytes) -> None:
        self.data = data
        self.pos = 0

    def _read(self, fmt: str) -> Any:
        fmt = '>' + fmt
        [value] = struct.unpack_from(fmt, self.data, self.pos)
        self.pos += struct.calcsize(fmt)
        return value

    def _read_byte_array(self) -> bytes | None:
        size = self._read('I')
        if size == 0xFFFFFFFF:
            return None

        end = self.pos + size
        value = self.data[self.pos : end]
        assert len(value) == size, (self.pos, size, len(value))
        self.pos = end
        return value

    def _read_string(self) -> str | None:
        value = self._read_byte_array()
        if value is None:
            return None
        return value.decode('utf-16-be')

    def _read_variant(self) -> QVariant:
        type_id = self._read('I')
        is_null = self._read('B') != 0

        if type_id == 0:
            assert is_null
            # Pre-Qt-5 streams store an empty QString after an invalid QVariant.
            assert self._read_byte_array() is None
            return None
        if is_null:
            return None
        if type_id == 1:
            return self._read('B') != 0
        if type_id == 2:
            return self._read('i')
        if type_id == 3:
            return self._read('I')
        if type_id == 8:
            return self._read_variant_map()
        if type_id == 9:
            return [self._read_variant() for _ in range(self._read('I'))]
        if type_id == 10:
            return self._read_string()
        if type_id == 12:
            return self._read_byte_array()
        raise AssertionError((type_id, self.pos))

    def _read_variant_map(self) -> QVariantMap:
        result: QVariantMap = {}
        for _ in range(self._read('I')):
            key = self._read_string()
            assert key is not None
            result[key] = self._read_variant()
        return result


def parse_qvariant_map(data: bytes) -> QVariantMap:
    reader = _Reader(data)
    result = reader._read_variant_map()
    assert reader.pos == len(data), (reader.pos, len(data))
    return result
