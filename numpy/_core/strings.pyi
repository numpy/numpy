from typing import overload

import numpy as np
from numpy._typing import (
    NDArray,
    _ArrayLikeStr_co as U_co,
    _ArrayLikeBytes_co as S_co,
    _ArrayLikeInt_co as i_co,
)

@overload
def equal(x1: U_co, x2: U_co) -> NDArray[np.bool]: ...
@overload
def equal(x1: S_co, x2: S_co) -> NDArray[np.bool]: ...

@overload
def not_equal(x1: U_co, x2: U_co) -> NDArray[np.bool]: ...
@overload
def not_equal(x1: S_co, x2: S_co) -> NDArray[np.bool]: ...

@overload
def greater_equal(x1: U_co, x2: U_co) -> NDArray[np.bool]: ...
@overload
def greater_equal(x1: S_co, x2: S_co) -> NDArray[np.bool]: ...

@overload
def less_equal(x1: U_co, x2: U_co) -> NDArray[np.bool]: ...
@overload
def less_equal(x1: S_co, x2: S_co) -> NDArray[np.bool]: ...

@overload
def greater(x1: U_co, x2: U_co) -> NDArray[np.bool]: ...
@overload
def greater(x1: S_co, x2: S_co) -> NDArray[np.bool]: ...

@overload
def less(x1: U_co, x2: U_co) -> NDArray[np.bool]: ...
@overload
def less(x1: S_co, x2: S_co) -> NDArray[np.bool]: ...

@overload
def add(x1: U_co, x2: U_co) -> NDArray[np.str_]: ...
@overload
def add(x1: S_co, x2: S_co) -> NDArray[np.bytes_]: ...

def isalpha(x: U_co | S_co) -> NDArray[np.bool]: ...
def isdigit(x: U_co | S_co) -> NDArray[np.bool]: ...
def isspace(x: U_co | S_co) -> NDArray[np.bool]: ...
def isdecimal(x: U_co) -> NDArray[np.bool]: ...
def isnumeric(x: U_co) -> NDArray[np.bool]: ...

def str_len(x: U_co | S_co) -> NDArray[np.int_]: ...

@overload
def find(
    a: U_co,
    sub: U_co,
    start: i_co = ...,
    end: i_co | None = ...,
) -> NDArray[np.int_]: ...
@overload
def find(
    a: S_co,
    sub: S_co,
    start: i_co = ...,
    end: i_co | None = ...,
) -> NDArray[np.int_]: ...

@overload
def rfind(
    a: U_co,
    sub: U_co,
    start: i_co = ...,
    end: i_co | None = ...,
) -> NDArray[np.int_]: ...
@overload
def rfind(
    a: S_co,
    sub: S_co,
    start: i_co = ...,
    end: i_co | None = ...,
) -> NDArray[np.int_]: ...

@overload
def count(
    a: U_co,
    sub: U_co,
    start: i_co = ...,
    end: i_co | None = ...,
) -> NDArray[np.int_]: ...
@overload
def count(
    a: S_co,
    sub: S_co,
    start: i_co = ...,
    end: i_co | None = ...,
) -> NDArray[np.int_]: ...

@overload
def startswith(
    a: U_co,
    prefix: U_co,
    start: i_co = ...,
    end: i_co | None = ...,
) -> NDArray[np.bool]: ...
@overload
def startswith(
    a: S_co,
    prefix: S_co,
    start: i_co = ...,
    end: i_co | None = ...,
) -> NDArray[np.bool]: ...

@overload
def endswith(
    a: U_co,
    suffix: U_co,
    start: i_co = ...,
    end: i_co | None = ...,
) -> NDArray[np.bool]: ...
@overload
def endswith(
    a: S_co,
    suffix: S_co,
    start: i_co = ...,
    end: i_co | None = ...,
) -> NDArray[np.bool]: ...

@overload
def lstrip(a: U_co, chars: None | U_co = ...) -> NDArray[np.str_]: ...
@overload
def lstrip(a: S_co, chars: None | S_co = ...) -> NDArray[np.bytes_]: ...

@overload
def rstrip(a: U_co, char: None | U_co = ...) -> NDArray[np.str_]: ...
@overload
def rstrip(a: S_co, char: None | S_co = ...) -> NDArray[np.bytes_]: ...

@overload
def strip(a: U_co, chars: None | U_co = ...) -> NDArray[np.str_]: ...
@overload
def strip(a: S_co, chars: None | S_co = ...) -> NDArray[np.bytes_]: ...

@overload
def replace(
    a: U_co,
    old: U_co,
    new: U_co,
    count: i_co = ...,
) -> NDArray[np.str_]: ...
@overload
def replace(
    a: S_co,
    old: S_co,
    new: S_co,
    count: i_co = ...,
) -> NDArray[np.bytes_]: ...

