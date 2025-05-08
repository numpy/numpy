from typing import Any

import numpy as np
import numpy.ma
import numpy.typing as npt

MAR_1d_f8: np.ma.MaskedArray[tuple[int], np.dtype[np.float64]]

AR_b: npt.NDArray[np.bool]

MAR_1d_f8.shape = (3, 1)  # E: Incompatible types in assignment
MAR_1d_f8.dtype = np.bool  # E: Incompatible types in assignment

np.ma.min(MAR_1d_f8, axis=1.0)  # E: No overload variant
np.ma.min(MAR_1d_f8, keepdims=1.0)  # E: No overload variant
np.ma.min(MAR_1d_f8, out=1.0)  # E: No overload variant
np.ma.min(MAR_1d_f8, fill_value=lambda x: 27)  # E: No overload variant

MAR_1d_f8.min(axis=1.0)  # E: No overload variant
MAR_1d_f8.min(keepdims=1.0)  # E: No overload variant
MAR_1d_f8.min(out=1.0)  # E: No overload variant
MAR_1d_f8.min(fill_value=lambda x: 27)  # E: No overload variant

np.ma.max(MAR_1d_f8, axis=1.0)  # E: No overload variant
np.ma.max(MAR_1d_f8, keepdims=1.0)  # E: No overload variant
np.ma.max(MAR_1d_f8, out=1.0)  # E: No overload variant
np.ma.max(MAR_1d_f8, fill_value=lambda x: 27)  # E: No overload variant

MAR_1d_f8.max(axis=1.0)  # E: No overload variant
MAR_1d_f8.max(keepdims=1.0)  # E: No overload variant
MAR_1d_f8.max(out=1.0)  # E: No overload variant
MAR_1d_f8.max(fill_value=lambda x: 27)  # E: No overload variant

np.ma.ptp(MAR_1d_f8, axis=1.0)  # E: No overload variant
np.ma.ptp(MAR_1d_f8, keepdims=1.0)  # E: No overload variant
np.ma.ptp(MAR_1d_f8, out=1.0)  # E: No overload variant
np.ma.ptp(MAR_1d_f8, fill_value=lambda x: 27)  # E: No overload variant

MAR_1d_f8.ptp(axis=1.0)  # E: No overload variant
MAR_1d_f8.ptp(keepdims=1.0)  # E: No overload variant
MAR_1d_f8.ptp(out=1.0)  # E: No overload variant
MAR_1d_f8.ptp(fill_value=lambda x: 27)  # E: No overload variant

MAR_1d_f8.argmin(axis=1.0)  # E: No overload variant
MAR_1d_f8.argmin(keepdims=1.0)  # E: No overload variant
MAR_1d_f8.argmin(out=1.0)  # E: No overload variant
MAR_1d_f8.argmin(fill_value=lambda x: 27)  # E: No overload variant

np.ma.argmin(MAR_1d_f8, axis=1.0)  # E: No overload variant
np.ma.argmin(MAR_1d_f8, axis=(1,))  # E: No overload variant
np.ma.argmin(MAR_1d_f8, keepdims=1.0)  # E: No overload variant
np.ma.argmin(MAR_1d_f8, out=1.0)  # E: No overload variant
np.ma.argmin(MAR_1d_f8, fill_value=lambda x: 27)  # E: No overload variant

MAR_1d_f8.argmax(axis=1.0)  # E: No overload variant
MAR_1d_f8.argmax(keepdims=1.0)  # E: No overload variant
MAR_1d_f8.argmax(out=1.0)  # E: No overload variant
MAR_1d_f8.argmax(fill_value=lambda x: 27)  # E: No overload variant

np.ma.argmax(MAR_1d_f8, axis=1.0)  # E: No overload variant
np.ma.argmax(MAR_1d_f8, axis=(0,))  # E: No overload variant
np.ma.argmax(MAR_1d_f8, keepdims=1.0)  # E: No overload variant
np.ma.argmax(MAR_1d_f8, out=1.0)  # E: No overload variant
np.ma.argmax(MAR_1d_f8, fill_value=lambda x: 27)  # E: No overload variant

MAR_1d_f8.all(axis=1.0)  # E: No overload variant
MAR_1d_f8.all(keepdims=1.0)  # E: No overload variant
MAR_1d_f8.all(out=1.0)  # E: No overload variant

MAR_1d_f8.any(axis=1.0)  # E: No overload variant
MAR_1d_f8.any(keepdims=1.0)  # E: No overload variant
MAR_1d_f8.any(out=1.0)  # E: No overload variant

MAR_1d_f8.sort(axis=(0,1))  # E: No overload variant
MAR_1d_f8.sort(axis=None)  # E: No overload variant
MAR_1d_f8.sort(kind='cabbage')  # E: No overload variant
MAR_1d_f8.sort(order=lambda: 'cabbage')  # E: No overload variant
MAR_1d_f8.sort(endwith='cabbage')  # E: No overload variant
MAR_1d_f8.sort(fill_value=lambda: 'cabbage')  # E: No overload variant
MAR_1d_f8.sort(stable='cabbage')  # E: No overload variant
MAR_1d_f8.sort(stable=True)  # E: No overload variant

MAR_1d_f8.take(axis=1.0)  # E: No overload variant
MAR_1d_f8.take(out=1)  # E: No overload variant
MAR_1d_f8.take(mode="bob")  # E: No overload variant

np.ma.take(None)  # E: No overload variant
np.ma.take(axis=1.0)  # E: No overload variant
np.ma.take(out=1)  # E: No overload variant
np.ma.take(mode="bob")  # E: No overload variant

MAR_1d_f8.partition(['cabbage'])  # E: No overload variant
MAR_1d_f8.partition(axis=(0,1))  # E: No overload variant
MAR_1d_f8.partition(kind='cabbage')  # E: No overload variant
MAR_1d_f8.partition(order=lambda: 'cabbage')  # E: No overload variant
MAR_1d_f8.partition(AR_b)  # E: No overload variant

MAR_1d_f8.argpartition(['cabbage'])  # E: No overload variant
MAR_1d_f8.argpartition(axis=(0,1))  # E: No overload variant
MAR_1d_f8.argpartition(kind='cabbage')  # E: No overload variant
MAR_1d_f8.argpartition(order=lambda: 'cabbage')  # E: No overload variant
MAR_1d_f8.argpartition(AR_b)  # E: No overload variant

np.ma.ndim(lambda: 'lambda')  # E: No overload variant

np.ma.size(AR_b, axis='0')  # E: No overload variant

MAR_1d_f8 >= (lambda x: 'mango') # E: No overload variant

MAR_1d_f8 > (lambda x: 'mango') # E: No overload variant

MAR_1d_f8 <= (lambda x: 'mango') # E: No overload variant

MAR_1d_f8 < (lambda x: 'mango') # E: No overload variant

MAR_1d_f8.count(axis=0.)  # E: No overload variant

np.ma.count(MAR_1d_f8, axis=0.)  # E: No overload variant

MAR_1d_f8.put(4, 999, mode='flip')  # E: No overload variant

np.ma.put(MAR_1d_f8, 4, 999, mode='flip')  # E: No overload variant

np.ma.put([1,1,3], 0, 999)  # E: No overload variant

np.ma.compressed(lambda: 'compress me')  # E: No overload variant

np.ma.allequal(MAR_1d_f8, [1,2,3], fill_value=1.5)  # E: No overload variant

np.ma.allclose(MAR_1d_f8, [1,2,3], masked_equal=4.5)  # E: No overload variant
np.ma.allclose(MAR_1d_f8, [1,2,3], rtol='.4')  # E: No overload variant
np.ma.allclose(MAR_1d_f8, [1,2,3], atol='.5')  # E: No overload variant

MAR_1d_f8.__setmask__('mask')  # E: No overload variant

MAR_1d_f8.swapaxes(axis1=1, axis2=0)  # E: No overload variant
