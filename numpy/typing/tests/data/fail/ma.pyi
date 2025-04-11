from typing import Any

import numpy as np
import numpy.ma
import numpy.typing as npt

m: np.ma.MaskedArray[tuple[int], np.dtype[np.float64]]

AR_b: npt.NDArray[np.bool]

m.shape = (3, 1)  # E: Incompatible types in assignment
m.dtype = np.bool  # E: Incompatible types in assignment

np.ma.min(m, axis=1.0)  # E: No overload variant
np.ma.min(m, keepdims=1.0)  # E: No overload variant
np.ma.min(m, out=1.0)  # E: No overload variant
np.ma.min(m, fill_value=lambda x: 27)  # E: No overload variant

m.min(axis=1.0)  # E: No overload variant
m.min(keepdims=1.0)  # E: No overload variant
m.min(out=1.0)  # E: No overload variant
m.min(fill_value=lambda x: 27)  # E: No overload variant

np.ma.max(m, axis=1.0)  # E: No overload variant
np.ma.max(m, keepdims=1.0)  # E: No overload variant
np.ma.max(m, out=1.0)  # E: No overload variant
np.ma.max(m, fill_value=lambda x: 27)  # E: No overload variant

m.max(axis=1.0)  # E: No overload variant
m.max(keepdims=1.0)  # E: No overload variant
m.max(out=1.0)  # E: No overload variant
m.max(fill_value=lambda x: 27)  # E: No overload variant

np.ma.ptp(m, axis=1.0)  # E: No overload variant
np.ma.ptp(m, keepdims=1.0)  # E: No overload variant
np.ma.ptp(m, out=1.0)  # E: No overload variant
np.ma.ptp(m, fill_value=lambda x: 27)  # E: No overload variant

m.ptp(axis=1.0)  # E: No overload variant
m.ptp(keepdims=1.0)  # E: No overload variant
m.ptp(out=1.0)  # E: No overload variant
m.ptp(fill_value=lambda x: 27)  # E: No overload variant

m.argmin(axis=1.0)  # E: No overload variant
m.argmin(keepdims=1.0)  # E: No overload variant
m.argmin(out=1.0)  # E: No overload variant
m.argmin(fill_value=lambda x: 27)  # E: No overload variant

np.ma.argmin(m, axis=1.0)  # E: No overload variant
np.ma.argmin(m, axis=(1,))  # E: No overload variant
np.ma.argmin(m, keepdims=1.0)  # E: No overload variant
np.ma.argmin(m, out=1.0)  # E: No overload variant
np.ma.argmin(m, fill_value=lambda x: 27)  # E: No overload variant

m.argmax(axis=1.0)  # E: No overload variant
m.argmax(keepdims=1.0)  # E: No overload variant
m.argmax(out=1.0)  # E: No overload variant
m.argmax(fill_value=lambda x: 27)  # E: No overload variant

np.ma.argmax(m, axis=1.0)  # E: No overload variant
np.ma.argmax(m, axis=(0,))  # E: No overload variant
np.ma.argmax(m, keepdims=1.0)  # E: No overload variant
np.ma.argmax(m, out=1.0)  # E: No overload variant
np.ma.argmax(m, fill_value=lambda x: 27)  # E: No overload variant

m.sort(axis=(0,1))  # E: No overload variant
m.sort(axis=None)  # E: No overload variant
m.sort(kind='cabbage')  # E: No overload variant
m.sort(order=lambda: 'cabbage')  # E: No overload variant
m.sort(endwith='cabbage')  # E: No overload variant
m.sort(fill_value=lambda: 'cabbage')  # E: No overload variant
m.sort(stable='cabbage')  # E: No overload variant
m.sort(stable=True)  # E: No overload variant

m.take(axis=1.0)  # E: No overload variant
m.take(out=1)  # E: No overload variant
m.take(mode="bob")  # E: No overload variant

np.ma.take(None)  # E: No overload variant
np.ma.take(axis=1.0)  # E: No overload variant
np.ma.take(out=1)  # E: No overload variant
np.ma.take(mode="bob")  # E: No overload variant

m.partition(['cabbage'])  # E: No overload variant
m.partition(axis=(0,1))  # E: No overload variant
m.partition(kind='cabbage')  # E: No overload variant
m.partition(order=lambda: 'cabbage')  # E: No overload variant
m.partition(AR_b)  # E: No overload variant

m.argpartition(['cabbage'])  # E: No overload variant
m.argpartition(axis=(0,1))  # E: No overload variant
m.argpartition(kind='cabbage')  # E: No overload variant
m.argpartition(order=lambda: 'cabbage')  # E: No overload variant
m.argpartition(AR_b)  # E: No overload variant
