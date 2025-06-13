from __future__ import annotations

from typing import Any, Dict, Iterable, Mapping, Sequence, Tuple, Union

import numpy as np

__all__ = ["array_info"]

Number = Union[int, float, complex, bool]
ArrayLike = Union[np.ndarray, Sequence[Number], Number]


def _safe_stat(fn, arr: np.ndarray, *args, **kwargs):
    try:
        return fn(arr, *args, **kwargs) if arr.size else None
    except Exception:  
        return None


def _to_tuple(x: Union[int, Iterable[int]]) -> Tuple[int, ...]:
    return (x,) if isinstance(x, (int, np.integer)) else tuple(x)


def array_info(
    arr: ArrayLike,
    *,
    axis: Union[int, Sequence[int], None] = None,
    percentiles: Sequence[int] | None = (25, 50, 75),
    sample_unique_threshold: int = 1_000_000,
    example_values: int = 3,
) -> Dict[str, Any]:
    """
   Return a dictionary (or nested dict) that summarises the **structure** and
    **contents** of a NumPy array.

    The function combines three layers of insight:

    1. **Flat statistics** - min/max/mean/std, NaN/Inf/None counts, percentiles
        and inter-quartile range (IQR) over the whole array.
    2. **Axis-wise statistics** - the same metrics computed along a user-selected
        axis or tuple of axes (optional, via ``axis=``), placed under
        ``info["axis_stats"]``.
    3. **Quick metadata** - shape, dtype, itemsize, contiguity flags, unique-value
        count (sampled for very large arrays) and a small sample of actual values.

    Parameters
    ----------
    arr : array-like
        Data to analyse.  It is converted to an ``ndarray`` with ``np.asarray``.
    axis : int or tuple of int, optional
        If provided, compute statistics **along** the given axis/axes instead of
        over the flattened array, and store them in the nested ``"axis_stats"`` dictionary.
    percentiles : sequence of int or None, optional
        Percentile values (0-100) to compute in addition to the core stats.
        The default ``(25, 50, 75)`` returns Q1, median, and Q3.  Pass ``None``
        to skip percentile and IQR output.
    sample_unique_threshold : int, optional
        Exact unique-element counting becomes expensive on very large arrays.  If
        ``arr.size`` exceeds this threshold, a random subsample of that size is
        used instead and the result is prefixed with ``">="``.
    example_values : int, optional
        Number of head elements (flattened order) to include under ``"example_values"``.

    Returns
    -------
    dict
        A mapping of summary keys to values.  Always present keys::
            shape, dtype, size, ndim, itemsize, contiguous, aligned, min, max, mean, std,
            nan_count, inf_count, none_count, num_unique, example_values
        + ``pXX`` percentile keys and ``iqr`` if *percentiles* is not ``None``.
        + ``axis_stats`` nested dictionary if *axis* is supplied.

    Examples
    --------
    >>> import numpy as np
    >>> from numpy.lib import array_info
    >>> a = np.array([[1, 2, np.nan], [4, 5, 6]])
    >>> array_info(a)
    {'shape': (2, 3), 'dtype': 'float64', 'size': 6, ...}

    >>> array_info(a, axis=0)["axis_stats"]["mean"]
    array([2.5, 3.5, 4.5])
    """

    arr = np.asarray(arr)

    info: Dict[str, Any] = {
        "shape": arr.shape,
        "dtype": str(arr.dtype),
        "size": arr.size,
        "ndim": arr.ndim,
        "itemsize": arr.itemsize,
        "contiguous": bool(arr.flags.c_contiguous),
        "aligned": bool(arr.flags.aligned),
    }


    is_bool = arr.dtype == np.bool_
    is_numeric = (
        np.issubdtype(arr.dtype, np.number)
        or np.issubdtype(arr.dtype, np.complexfloating)
        or is_bool
    )

    def _numeric_stats(a: np.ndarray) -> Mapping[str, Any]:
        if not is_numeric or a.size == 0:
            return {k: None for k in ("min", "max", "mean", "std")}

        a_float = a.astype(float, copy=False) if is_bool else a
        finite = a_float[np.isfinite(a_float)]
        if finite.size == 0:
            return {k: None for k in ("min", "max", "mean", "std")}

        out: Dict[str, Any] = {
            "min": _safe_stat(np.nanmin, finite),
            "max": _safe_stat(np.nanmax, finite),
            "mean": _safe_stat(np.nanmean, finite),
            "std": _safe_stat(np.nanstd, finite),
        }


        if percentiles:
            pct_vals = _safe_stat(np.nanpercentile, finite, q=percentiles)
            if pct_vals is not None:
                out.update({f"p{p}": v for p, v in zip(percentiles, pct_vals)})
                if 75 in percentiles and 25 in percentiles:
                    out["iqr"] = out.get("p75") - out.get("p25")
        return out


    info.update(_numeric_stats(arr))

    info["nan_count"] = int(np.isnan(arr).sum()) if arr.dtype.kind == "f" else 0
    info["inf_count"] = int(np.isinf(arr).sum()) if arr.dtype.kind == "f" else 0
    info["none_count"] = int(np.sum(arr == None)) if arr.dtype == object else 0  


    if arr.size == 0:
        info["num_unique"] = 0
    elif arr.dtype == object:
        info["num_unique"] = "n/a"
    elif arr.size <= sample_unique_threshold:
        info["num_unique"] = int(len(np.unique(arr)))
    else:
        rng = np.random.default_rng(0)
        idx = rng.choice(arr.size, size=sample_unique_threshold, replace=False)
        sample = arr.reshape(-1)[idx]
        info["num_unique"] = f">= {len(np.unique(sample))} (sample)"


    flat = arr.reshape(-1)
    info["example_values"] = flat[:example_values].tolist() if flat.size else []


    if axis is not None:
        axis_tuple = _to_tuple(axis)
        axis_stats: Dict[str, Any] = {}

        axis_stats.update({
            "min": _safe_stat(np.nanmin, arr, axis=axis_tuple),
            "max": _safe_stat(np.nanmax, arr, axis=axis_tuple),
            "mean": _safe_stat(np.nanmean, arr, axis=axis_tuple),
            "std": _safe_stat(np.nanstd, arr, axis=axis_tuple),
        })

        if arr.dtype.kind == "f":
            axis_stats["nan_count"] = np.isnan(arr).sum(axis=axis_tuple)
            axis_stats["inf_count"] = np.isinf(arr).sum(axis=axis_tuple)
        else:
            axis_stats["nan_count"] = axis_stats["inf_count"] = None

        if percentiles and is_numeric:
            pcts = _safe_stat(np.nanpercentile, arr, axis=axis_tuple, q=percentiles)
            if pcts is not None:
                for p, arr_pct in zip(percentiles, pcts):
                    axis_stats[f"p{p}"] = arr_pct
                if 75 in percentiles and 25 in percentiles:
                    axis_stats["iqr"] = axis_stats["p75"] - axis_stats["p25"]

        info["axis_stats"] = axis_stats

    return info