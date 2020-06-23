"""Tests for :mod:`numpy.core.fromnumeric`."""

import numpy as np

A = np.array(True, ndmin=2, dtype=bool)
B = np.array(1.0, ndmin=2, dtype=np.float32)
A.setflags(write=False)
B.setflags(write=False)

a = np.bool_(True)
b = np.float32(1.0)
c = 1.0

reveal_type(np.take(a, 0))  # E: numpy.bool_
reveal_type(np.take(b, 0))  # E: numpy.float32
reveal_type(
    np.take(c, 0)  # E: Union[numpy.generic, datetime.datetime, datetime.timedelta]
)
reveal_type(
    np.take(A, 0)  # E: Union[numpy.generic, datetime.datetime, datetime.timedelta]
)
reveal_type(
    np.take(B, 0)  # E: Union[numpy.generic, datetime.datetime, datetime.timedelta]
)
reveal_type(
    np.take(  # E: Union[Union[numpy.generic, datetime.datetime, datetime.timedelta], numpy.ndarray]
        A, [0]
    )
)
reveal_type(
    np.take(  # E: Union[Union[numpy.generic, datetime.datetime, datetime.timedelta], numpy.ndarray]
        B, [0]
    )
)

reveal_type(np.reshape(a, 1))  # E: numpy.ndarray
reveal_type(np.reshape(b, 1))  # E: numpy.ndarray
reveal_type(np.reshape(c, 1))  # E: numpy.ndarray
reveal_type(np.reshape(A, 1))  # E: numpy.ndarray
reveal_type(np.reshape(B, 1))  # E: numpy.ndarray

reveal_type(np.choose(a, [True, True]))  # E: numpy.bool_
reveal_type(np.choose(A, [True, True]))  # E: numpy.ndarray

reveal_type(np.repeat(a, 1))  # E: numpy.ndarray
reveal_type(np.repeat(b, 1))  # E: numpy.ndarray
reveal_type(np.repeat(c, 1))  # E: numpy.ndarray
reveal_type(np.repeat(A, 1))  # E: numpy.ndarray
reveal_type(np.repeat(B, 1))  # E: numpy.ndarray

# TODO: Add tests for np.put()

reveal_type(np.swapaxes(A, 0, 0))  # E: numpy.ndarray
reveal_type(np.swapaxes(B, 0, 0))  # E: numpy.ndarray

reveal_type(np.transpose(a))  # E: numpy.ndarray
reveal_type(np.transpose(b))  # E: numpy.ndarray
reveal_type(np.transpose(c))  # E: numpy.ndarray
reveal_type(np.transpose(A))  # E: numpy.ndarray
reveal_type(np.transpose(B))  # E: numpy.ndarray

reveal_type(np.partition(a, 0, axis=None))  # E: numpy.ndarray
reveal_type(np.partition(b, 0, axis=None))  # E: numpy.ndarray
reveal_type(np.partition(c, 0, axis=None))  # E: numpy.ndarray
reveal_type(np.partition(A, 0))  # E: numpy.ndarray
reveal_type(np.partition(B, 0))  # E: numpy.ndarray

reveal_type(np.argpartition(a, 0))  # E: numpy.integer
reveal_type(np.argpartition(b, 0))  # E: numpy.integer
reveal_type(np.argpartition(c, 0))  # E: numpy.ndarray
reveal_type(np.argpartition(A, 0))  # E: numpy.ndarray
reveal_type(np.argpartition(B, 0))  # E: numpy.ndarray

reveal_type(np.sort(A, 0))  # E: numpy.ndarray
reveal_type(np.sort(B, 0))  # E: numpy.ndarray

reveal_type(np.argsort(A, 0))  # E: numpy.ndarray
reveal_type(np.argsort(B, 0))  # E: numpy.ndarray

reveal_type(np.argmax(A))  # E: numpy.integer
reveal_type(np.argmax(B))  # E: numpy.integer
reveal_type(np.argmax(A, axis=0))  # E: Union[numpy.integer, numpy.ndarray]
reveal_type(np.argmax(B, axis=0))  # E: Union[numpy.integer, numpy.ndarray]

reveal_type(np.argmin(A))  # E: numpy.integer
reveal_type(np.argmin(B))  # E: numpy.integer
reveal_type(np.argmin(A, axis=0))  # E: Union[numpy.integer, numpy.ndarray]
reveal_type(np.argmin(B, axis=0))  # E: Union[numpy.integer, numpy.ndarray]

reveal_type(np.searchsorted(A[0], 0))  # E: numpy.integer
reveal_type(np.searchsorted(B[0], 0))  # E: numpy.integer
reveal_type(np.searchsorted(A[0], [0]))  # E: numpy.ndarray
reveal_type(np.searchsorted(B[0], [0]))  # E: numpy.ndarray

reveal_type(np.resize(a, (5, 5)))  # E: numpy.ndarray
reveal_type(np.resize(b, (5, 5)))  # E: numpy.ndarray
reveal_type(np.resize(c, (5, 5)))  # E: numpy.ndarray
reveal_type(np.resize(A, (5, 5)))  # E: numpy.ndarray
reveal_type(np.resize(B, (5, 5)))  # E: numpy.ndarray

reveal_type(np.squeeze(a))  # E: numpy.bool_
reveal_type(np.squeeze(b))  # E: numpy.float32
reveal_type(np.squeeze(c))  # E: numpy.ndarray
reveal_type(np.squeeze(A))  # E: numpy.ndarray
reveal_type(np.squeeze(B))  # E: numpy.ndarray

reveal_type(np.diagonal(A))  # E: numpy.ndarray
reveal_type(np.diagonal(B))  # E: numpy.ndarray

reveal_type(np.trace(A))  # E: Union[numpy.number, numpy.ndarray]
reveal_type(np.trace(B))  # E: Union[numpy.number, numpy.ndarray]

reveal_type(np.ravel(a))  # E: numpy.ndarray
reveal_type(np.ravel(b))  # E: numpy.ndarray
reveal_type(np.ravel(c))  # E: numpy.ndarray
reveal_type(np.ravel(A))  # E: numpy.ndarray
reveal_type(np.ravel(B))  # E: numpy.ndarray

reveal_type(np.nonzero(a))  # E: tuple[numpy.ndarray]
reveal_type(np.nonzero(b))  # E: tuple[numpy.ndarray]
reveal_type(np.nonzero(c))  # E: tuple[numpy.ndarray]
reveal_type(np.nonzero(A))  # E: tuple[numpy.ndarray]
reveal_type(np.nonzero(B))  # E: tuple[numpy.ndarray]

reveal_type(np.shape(a))  # E: tuple[builtins.int]
reveal_type(np.shape(b))  # E: tuple[builtins.int]
reveal_type(np.shape(c))  # E: tuple[builtins.int]
reveal_type(np.shape(A))  # E: tuple[builtins.int]
reveal_type(np.shape(B))  # E: tuple[builtins.int]

reveal_type(np.compress([True], a))  # E: numpy.ndarray
reveal_type(np.compress([True], b))  # E: numpy.ndarray
reveal_type(np.compress([True], c))  # E: numpy.ndarray
reveal_type(np.compress([True], A))  # E: numpy.ndarray
reveal_type(np.compress([True], B))  # E: numpy.ndarray

reveal_type(np.clip(a, 0, 1.0))  # E: numpy.number
reveal_type(np.clip(b, -1, 1))  # E: numpy.float32
reveal_type(np.clip(c, 0, 1))  # E: numpy.number
reveal_type(np.clip(A, 0, 1))  # E: Union[numpy.number, numpy.ndarray]
reveal_type(np.clip(B, 0, 1))  # E: Union[numpy.number, numpy.ndarray]

reveal_type(np.sum(a))  # E: numpy.number
reveal_type(np.sum(b))  # E: numpy.float32
reveal_type(np.sum(c))  # E: numpy.number
reveal_type(np.sum(A))  # E: numpy.number
reveal_type(np.sum(B))  # E: numpy.number
reveal_type(np.sum(A, axis=0))  # E: Union[numpy.number, numpy.ndarray]
reveal_type(np.sum(B, axis=0))  # E: Union[numpy.number, numpy.ndarray]

reveal_type(np.all(a))  # E: numpy.bool_
reveal_type(np.all(b))  # E: numpy.bool_
reveal_type(np.all(c))  # E: numpy.bool_
reveal_type(np.all(A))  # E: numpy.bool_
reveal_type(np.all(B))  # E: numpy.bool_
reveal_type(np.all(A, axis=0))  # E: Union[numpy.bool_, numpy.ndarray]
reveal_type(np.all(B, axis=0))  # E: Union[numpy.bool_, numpy.ndarray]
reveal_type(np.all(A, keepdims=True))  # E: Union[numpy.bool_, numpy.ndarray]
reveal_type(np.all(B, keepdims=True))  # E: Union[numpy.bool_, numpy.ndarray]

reveal_type(np.any(a))  # E: numpy.bool_
reveal_type(np.any(b))  # E: numpy.bool_
reveal_type(np.any(c))  # E: numpy.bool_
reveal_type(np.any(A))  # E: numpy.bool_
reveal_type(np.any(B))  # E: numpy.bool_
reveal_type(np.any(A, axis=0))  # E: Union[numpy.bool_, numpy.ndarray]
reveal_type(np.any(B, axis=0))  # E: Union[numpy.bool_, numpy.ndarray]
reveal_type(np.any(A, keepdims=True))  # E: Union[numpy.bool_, numpy.ndarray]
reveal_type(np.any(B, keepdims=True))  # E: Union[numpy.bool_, numpy.ndarray]

reveal_type(np.cumsum(a))  # E: numpy.ndarray
reveal_type(np.cumsum(b))  # E: numpy.ndarray
reveal_type(np.cumsum(c))  # E: numpy.ndarray
reveal_type(np.cumsum(A))  # E: numpy.ndarray
reveal_type(np.cumsum(B))  # E: numpy.ndarray

reveal_type(np.ptp(a))  # E: numpy.number
reveal_type(np.ptp(b))  # E: numpy.float32
reveal_type(np.ptp(c))  # E: numpy.number
reveal_type(np.ptp(A))  # E: numpy.number
reveal_type(np.ptp(B))  # E: numpy.number
reveal_type(np.ptp(A, axis=0))  # E: Union[numpy.number, numpy.ndarray]
reveal_type(np.ptp(B, axis=0))  # E: Union[numpy.number, numpy.ndarray]
reveal_type(np.ptp(A, keepdims=True))  # E: Union[numpy.number, numpy.ndarray]
reveal_type(np.ptp(B, keepdims=True))  # E: Union[numpy.number, numpy.ndarray]

reveal_type(np.amax(a))  # E: numpy.number
reveal_type(np.amax(b))  # E: numpy.float32
reveal_type(np.amax(c))  # E: numpy.number
reveal_type(np.amax(A))  # E: numpy.number
reveal_type(np.amax(B))  # E: numpy.number
reveal_type(np.amax(A, axis=0))  # E: Union[numpy.number, numpy.ndarray]
reveal_type(np.amax(B, axis=0))  # E: Union[numpy.number, numpy.ndarray]
reveal_type(np.amax(A, keepdims=True))  # E: Union[numpy.number, numpy.ndarray]
reveal_type(np.amax(B, keepdims=True))  # E: Union[numpy.number, numpy.ndarray]

reveal_type(np.amin(a))  # E: numpy.number
reveal_type(np.amin(b))  # E: numpy.float32
reveal_type(np.amin(c))  # E: numpy.number
reveal_type(np.amin(A))  # E: numpy.number
reveal_type(np.amin(B))  # E: numpy.number
reveal_type(np.amin(A, axis=0))  # E: Union[numpy.number, numpy.ndarray]
reveal_type(np.amin(B, axis=0))  # E: Union[numpy.number, numpy.ndarray]
reveal_type(np.amin(A, keepdims=True))  # E: Union[numpy.number, numpy.ndarray]
reveal_type(np.amin(B, keepdims=True))  # E: Union[numpy.number, numpy.ndarray]
