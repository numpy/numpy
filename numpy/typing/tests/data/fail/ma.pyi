from typing import Any

import numpy as np
import numpy.ma
import numpy.typing as npt

m: np.ma.MaskedArray[tuple[int], np.dtype[np.float64]]

AR_b: npt.NDArray[np.bool]

m.shape = (3, 1)  # type: ignore[assignment]
m.dtype = np.bool  # type: ignore[assignment]

np.ma.min(m, axis=1.0)  # type: ignore[call-overload]
np.ma.min(m, keepdims=1.0)  # type: ignore[call-overload]
np.ma.min(m, out=1.0)  # type: ignore[call-overload]
np.ma.min(m, fill_value=lambda x: 27)  # type: ignore[call-overload]

m.min(axis=1.0)  # type: ignore[call-overload]
m.min(keepdims=1.0)  # type: ignore[call-overload]
m.min(out=1.0)  # type: ignore[call-overload]
m.min(fill_value=lambda x: 27)  # type: ignore[call-overload]

np.ma.max(m, axis=1.0)  # type: ignore[call-overload]
np.ma.max(m, keepdims=1.0)  # type: ignore[call-overload]
np.ma.max(m, out=1.0)  # type: ignore[call-overload]
np.ma.max(m, fill_value=lambda x: 27)  # type: ignore[call-overload]

m.max(axis=1.0)  # type: ignore[call-overload]
m.max(keepdims=1.0)  # type: ignore[call-overload]
m.max(out=1.0)  # type: ignore[call-overload]
m.max(fill_value=lambda x: 27)  # type: ignore[call-overload]

np.ma.ptp(m, axis=1.0)  # type: ignore[call-overload]
np.ma.ptp(m, keepdims=1.0)  # type: ignore[call-overload]
np.ma.ptp(m, out=1.0)  # type: ignore[call-overload]
np.ma.ptp(m, fill_value=lambda x: 27)  # type: ignore[call-overload]

m.ptp(axis=1.0)  # type: ignore[call-overload]
m.ptp(keepdims=1.0)  # type: ignore[call-overload]
m.ptp(out=1.0)  # type: ignore[call-overload]
m.ptp(fill_value=lambda x: 27)  # type: ignore[call-overload]

m.argmin(axis=1.0)  # type: ignore[call-overload]
m.argmin(keepdims=1.0)  # type: ignore[call-overload]
m.argmin(out=1.0)  # type: ignore[call-overload]
m.argmin(fill_value=lambda x: 27)  # type: ignore[call-overload]

np.ma.argmin(m, axis=1.0)  # type: ignore[call-overload]
np.ma.argmin(m, axis=(1,))  # type: ignore[call-overload]
np.ma.argmin(m, keepdims=1.0)  # type: ignore[call-overload]
np.ma.argmin(m, out=1.0)  # type: ignore[call-overload]
np.ma.argmin(m, fill_value=lambda x: 27)  # type: ignore[call-overload]

m.argmax(axis=1.0)  # type: ignore[call-overload]
m.argmax(keepdims=1.0)  # type: ignore[call-overload]
m.argmax(out=1.0)  # type: ignore[call-overload]
m.argmax(fill_value=lambda x: 27)  # type: ignore[call-overload]

np.ma.argmax(m, axis=1.0)  # type: ignore[call-overload]
np.ma.argmax(m, axis=(0,))  # type: ignore[call-overload]
np.ma.argmax(m, keepdims=1.0)  # type: ignore[call-overload]
np.ma.argmax(m, out=1.0)  # type: ignore[call-overload]
np.ma.argmax(m, fill_value=lambda x: 27)  # type: ignore[call-overload]

m.all(axis=1.0)  # type: ignore[call-overload]
m.all(keepdims=1.0)  # type: ignore[call-overload]
m.all(out=1.0)  # type: ignore[call-overload]

m.any(axis=1.0)  # type: ignore[call-overload]
m.any(keepdims=1.0)  # type: ignore[call-overload]
m.any(out=1.0)  # type: ignore[call-overload]

m.sort(axis=(0,1))  # type: ignore[arg-type]
m.sort(axis=None)  # type: ignore[arg-type]
m.sort(kind='cabbage')  # type: ignore[arg-type]
m.sort(order=lambda: 'cabbage')  # type: ignore[arg-type]
m.sort(endwith='cabbage')  # type: ignore[arg-type]
m.sort(fill_value=lambda: 'cabbage')  # type: ignore[arg-type]
m.sort(stable='cabbage')  # type: ignore[arg-type]
m.sort(stable=True)  # type: ignore[arg-type]

m.take(axis=1.0)  # type: ignore[call-overload]
m.take(out=1)  # type: ignore[call-overload]
m.take(mode="bob")  # type: ignore[call-overload]

np.ma.take(None)  # type: ignore[call-overload]
np.ma.take(axis=1.0)  # type: ignore[call-overload]
np.ma.take(out=1)  # type: ignore[call-overload]
np.ma.take(mode="bob")  # type: ignore[call-overload]

m.partition(['cabbage'])  # type: ignore[arg-type]
m.partition(axis=(0,1))  # type: ignore[arg-type, call-arg]
m.partition(kind='cabbage')  # type: ignore[arg-type, call-arg]
m.partition(order=lambda: 'cabbage')  # type: ignore[arg-type, call-arg]
m.partition(AR_b)  # type: ignore[arg-type]

m.argpartition(['cabbage'])  # type: ignore[arg-type]
m.argpartition(axis=(0,1))  # type: ignore[arg-type, call-arg]
m.argpartition(kind='cabbage')  # type: ignore[arg-type, call-arg]
m.argpartition(order=lambda: 'cabbage')  # type: ignore[arg-type, call-arg]
m.argpartition(AR_b)  # type: ignore[arg-type]

np.ma.ndim(lambda: 'lambda')  # type: ignore[arg-type]

np.ma.size(AR_b, axis='0')  # type: ignore[arg-type]

m >= (lambda x: 'mango') # type: ignore[operator]
m > (lambda x: 'mango') # type: ignore[operator]
m <= (lambda x: 'mango') # type: ignore[operator]
m < (lambda x: 'mango') # type: ignore[operator]

m.count(axis=0.)  # type: ignore[call-overload]

np.ma.count(m, axis=0.)  # type: ignore[call-overload]

m.put(4, 999, mode='flip')  # type: ignore[arg-type]

np.ma.put(m, 4, 999, mode='flip')  # type: ignore[arg-type]

np.ma.put([1,1,3], 0, 999)  # type: ignore[arg-type]

np.ma.compressed(lambda: 'compress me')  # type: ignore[call-overload]

np.ma.allequal(m, [1,2,3], fill_value=1.5)  # type: ignore[arg-type]

np.ma.allclose(m, [1,2,3], masked_equal=4.5)  # type: ignore[arg-type]
np.ma.allclose(m, [1,2,3], rtol='.4')  # type: ignore[arg-type]
np.ma.allclose(m, [1,2,3], atol='.5')  # type: ignore[arg-type]

m.__setmask__('mask')  # type: ignore[arg-type]

m.swapaxes(axis1=1, axis2=0)  # type: ignore[call-arg]
