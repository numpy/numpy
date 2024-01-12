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
    x1: U_co,
    x2: U_co,
    x3: i_co = ...,
    x4: i_co | None = ...,
) -> NDArray[np.int_]: ...
@overload
def find(
    x1: S_co,
    x2: S_co,
    x3: i_co = ...,
    x4: i_co | None = ...,
) -> NDArray[np.int_]: ...

@overload
def rfind(
    x1: U_co,
    x2: U_co,
    x3: i_co = ...,
    x4: i_co | None = ...,
) -> NDArray[np.int_]: ...
@overload
def rfind(
    x1: S_co,
    x2: S_co,
    x3: i_co = ...,
    x4: i_co | None = ...,
) -> NDArray[np.int_]: ...

@overload
def count(
    x1: U_co,
    x2: U_co,
    x3: i_co = ...,
    x4: i_co | None = ...,
) -> NDArray[np.int_]: ...
@overload
def count(
    x1: S_co,
    x2: S_co,
    x3: i_co = ...,
    x4: i_co | None = ...,
) -> NDArray[np.int_]: ...

@overload
def startswith(
    x1: U_co,
    x2: U_co,
    x3: i_co = ...,
    x4: i_co | None = ...,
) -> NDArray[np.bool]: ...
@overload
def startswith(
    x1: S_co,
    x2: S_co,
    x3: i_co = ...,
    x4: i_co | None = ...,
) -> NDArray[np.bool]: ...

@overload
def endswith(
    x1: U_co,
    x2: U_co,
    x3: i_co = ...,
    x4: i_co | None = ...,
) -> NDArray[np.bool]: ...
@overload
def endswith(
    x1: S_co,
    x2: S_co,
    x3: i_co = ...,
    x4: i_co | None = ...,
) -> NDArray[np.bool]: ...

@overload
def lstrip(x1: U_co, x2: None | U_co = ...) -> NDArray[np.str_]: ...
@overload
def lstrip(x1: S_co, x2: None | S_co = ...) -> NDArray[np.bytes_]: ...

@overload
def rstrip(x1: U_co, x2: None | U_co = ...) -> NDArray[np.str_]: ...
@overload
def rstrip(x1: S_co, x2: None | S_co = ...) -> NDArray[np.bytes_]: ...

@overload
def strip(x1: U_co, x2: None | U_co = ...) -> NDArray[np.str_]: ...
@overload
def strip(x1: S_co, x2: None | S_co = ...) -> NDArray[np.bytes_]: ...

@overload
def replace(
    x1: U_co,
    x2: U_co,
    x3: U_co,
    x4: i_co = ...,
) -> NDArray[np.str_]: ...
@overload
def replace(
    a: S_co,
    x2: S_co,
    x3: S_co,
    x4: i_co = ...,
) -> NDArray[np.bytes_]: ...

