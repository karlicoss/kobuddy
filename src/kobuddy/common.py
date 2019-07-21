import functools
import logging
from typing import Callable, Dict, Iterable, List, Optional, Type, TypeVar


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
