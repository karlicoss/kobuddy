from __future__ import annotations

import logging
from collections.abc import Iterable, Iterator
from itertools import tee
from typing import (
    Any,
    Callable,
    TypeVar,
    Union,
)


def get_logger() -> logging.Logger:
    return logging.getLogger('kobuddy')


T = TypeVar('T')


def unwrap(x: T | None) -> T:
    assert x is not None
    return x


A = TypeVar('A')


# TODO switch to more_itertools?
def the(l: Iterable[A]) -> A:
    it = iter(l)
    try:
        first = next(it)
    except StopIteration as ee:
        raise RuntimeError('Empty iterator?') from ee
    assert all(e == first for e in it)
    return first


K = TypeVar('K')


def group_by_key(l: Iterable[T], key: Callable[[T], K]) -> dict[K, list[T]]:
    res: dict[K, list[T]] = {}
    for i in l:
        kk = key(i)
        lst: list[T] = res.get(kk, [])
        lst.append(i)
        res[kk] = lst
    return res


V = TypeVar('V', covariant=True)  # noqa: PLC0105
Res = Union[V, Exception]


# TODO better name?
def split_res(it: Iterable[Res[V]]) -> tuple[Iterator[V], Iterator[Exception]]:
    vit, eit = tee(it)

    def it_val() -> Iterator[V]:
        for r in vit:
            if not isinstance(r, Exception):
                yield r

    def it_err() -> Iterator[Exception]:
        for r in eit:
            if isinstance(r, Exception):
                yield r

    return it_val(), it_err()


# TODO not sure if should keep it...
def sorted_res(it: Iterable[Res[V]], key: Callable[[V], Any]) -> Iterator[Res[V]]:
    vit, eit = split_res(it)
    yield from sorted(vit, key=key)
    yield from eit
