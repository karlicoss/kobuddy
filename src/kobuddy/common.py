import logging
from typing import Optional, TypeVar


def get_logger():
    return logging.getLogger('kobuddy')


T = TypeVar('T')
def unwrap(x: Optional[T]) -> T:
    assert x is not None
    return x
