"""
Create the numpy.core.multiarray namespace for backward compatibility. In v1.16
the multiarray and umath c-extension modules were merged into a single
_multiarray_umath extension module. So we replicate the old namespace
by importing from the extension module.
"""

from . import _multiarray_umath
from numpy.core._multiarray_umath import *
from numpy.core._multiarray_umath import (_fastCopyAndTranspose, _flagdict, _insert,
     _reconstruct, _vec_string, _ARRAY_API, _monotonicity)

__all__ = ['_ARRAY_API', 'ALLOW_THREADS', 'BUFSIZE', 'CLIP', 'DATETIMEUNITS',
    'ITEM_HASOBJECT', 'ITEM_IS_POINTER', 'LIST_PICKLE', 'MAXDIMS',
    'MAY_SHARE_BOUNDS', 'MAY_SHARE_EXACT', 'NEEDS_INIT', 'NEEDS_PYAPI',
    'RAISE', 'USE_GETITEM', 'USE_SETITEM', 'WRAP',
    '_fastCopyAndTranspose', '_flagdict', '_insert', '_reconstruct',
    '_vec_string', '_monotonicity',
    'add_docstring', 'arange', 'array', 'bincount', 'broadcast', 'busday_count',
    'busday_offset', 'busdaycalendar', 'can_cast', 'compare_chararrays',
    'concatenate', 'copyto', 'correlate', 'correlate2', 'count_nonzero',
    'c_einsum', 'datetime_as_string', 'datetime_data', 'digitize', 'dot',
    'dragon4_positional', 'dragon4_scientific', 'dtype', 'empty', 'empty_like',
    'error', 'flagsobj', 'flatiter', 'format_longfloat', 'frombuffer',
    'fromfile', 'fromiter', 'fromstring', 'getbuffer', 'inner', 'int_asbuffer',
    'interp', 'interp_complex', 'is_busday', 'lexsort', 'matmul',
    'may_share_memory', 'min_scalar_type', 'ndarray', 'nditer', 'nested_iters',
    'newbuffer', 'normalize_axis_index', 'packbits', 'promote_types',
    'putmask', 'ravel_multi_index', 'result_type', 'scalar',
    'set_datetimeparse_function', 'set_legacy_print_mode', 'set_numeric_ops',
    'set_string_function', 'set_typeDict', 'shares_memory', 'test_interrupt',
    'tracemalloc_domain', 'typeinfo', 'unpackbits', 'unravel_index', 'vdot',
    'where', 'zeros']

