import math as math
from typing import Any

from numpy._pytesttester import PytestTester

from numpy import (
    ndenumerate as ndenumerate,
    ndindex as ndindex,
)

from numpy.version import version

from numpy.lib import (
    format as format,
    mixins as mixins,
    scimath as scimath,
    stride_tricks as stride_tricks,
)

from numpy.lib._version import (
    NumpyVersion as NumpyVersion,
)

from numpy.lib.arraypad import (
    pad as pad,
)

from numpy.lib.arraysetops import (
    ediff1d as ediff1d,
    intersect1d as intersect1d,
    setxor1d as setxor1d,
    union1d as union1d,
    setdiff1d as setdiff1d,
    unique as unique,
    in1d as in1d,
    isin as isin,
)

from numpy.lib.arrayterator import (
    Arrayterator as Arrayterator,
)

from numpy.lib.index_tricks import (
    ravel_multi_index as ravel_multi_index,
    unravel_index as unravel_index,
    mgrid as mgrid,
    ogrid as ogrid,
    r_ as r_,
    c_ as c_,
    s_ as s_,
    index_exp as index_exp,
    ix_ as ix_,
    fill_diagonal as fill_diagonal,
    diag_indices as diag_indices,
    diag_indices_from as diag_indices_from,
)

from numpy.lib.npyio import (
    savetxt as savetxt,
    loadtxt as loadtxt,
    genfromtxt as genfromtxt,
    recfromtxt as recfromtxt,
    recfromcsv as recfromcsv,
    load as load,
    save as save,
    savez as savez,
    savez_compressed as savez_compressed,
    packbits as packbits,
    unpackbits as unpackbits,
    fromregex as fromregex,
    DataSource as DataSource,
)

from numpy.lib.polynomial import (
    poly as poly,
    roots as roots,
    polyint as polyint,
    polyder as polyder,
    polyadd as polyadd,
    polysub as polysub,
    polymul as polymul,
    polydiv as polydiv,
    polyval as polyval,
    polyfit as polyfit,
    poly1d as poly1d,
)

from numpy.lib.shape_base import (
    column_stack as column_stack,
    row_stack as row_stack,
    dstack as dstack,
    array_split as array_split,
    split as split,
    hsplit as hsplit,
    vsplit as vsplit,
    dsplit as dsplit,
    apply_over_axes as apply_over_axes,
    expand_dims as expand_dims,
    apply_along_axis as apply_along_axis,
    kron as kron,
    tile as tile,
    get_array_wrap as get_array_wrap,
    take_along_axis as take_along_axis,
    put_along_axis as put_along_axis,
)

from numpy.lib.stride_tricks import (
    broadcast_to as broadcast_to,
    broadcast_arrays as broadcast_arrays,
    broadcast_shapes as broadcast_shapes,
)

from numpy.lib.ufunclike import (
    fix as fix,
    isposinf as isposinf,
    isneginf as isneginf,
)

from numpy.lib.utils import (
    get_include as get_include,
    info as info,
    byte_bounds as byte_bounds,
    show_runtime as show_runtime,
)

from numpy.core.multiarray import (
    add_docstring as add_docstring,
    tracemalloc_domain as tracemalloc_domain,
)

from numpy.core.function_base import (
    add_newdoc as add_newdoc,
)

__all__: list[str]
__path__: list[str]
test: PytestTester

__version__ = version
emath = scimath
