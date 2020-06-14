import functools
import logging
from typing import Callable, Dict, Iterable, List, Optional, TypeVar, Union, Iterator, Tuple
from itertools import tee, filterfalse


def get_logger():
    return logging.getLogger('kobuddy')


T = TypeVar('T')
def unwrap(x: Optional[T]) -> T:
    assert x is not None
    return x


Cl = TypeVar('Cl')
R = TypeVar('R')
def cproperty(f: Callable[[Cl], R]) -> R:
    return property(functools.lru_cache(maxsize=1)(f)) # type: ignore


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
def group_by_key(l: Iterable[T], key: Callable[[T], K]) -> Dict[K, List[T]]:
    res: Dict[K, List[T]] = {}
    for i in l:
        kk = key(i)
        lst = res.get(kk, [])
        lst.append(i)
        res[kk] = lst
    return res



from contextlib import contextmanager
# unavailable in python < 3.7
@contextmanager
def nullcontext(enter_result=None):
    yield enter_result



V = TypeVar('V', covariant=True)
Res = Union[V, Exception]


# TODO better name?
def split_res(it: Iterable[Res[V]]) -> Tuple[Iterator[V], Iterator[Exception]]:
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
def sorted_res(it: Iterable[Res[V]], key) -> Iterator[Res[V]]:
    vit, eit = split_res(it)
    yield from sorted(vit, key=key)
    yield from eit
