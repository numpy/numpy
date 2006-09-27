"""Array printing function

$Id: arrayprint.py,v 1.9 2005/09/13 13:58:44 teoliphant Exp $
"""
__all__ = ["array2string", "set_printoptions", "get_printoptions"]

#
# Written by Konrad Hinsen <hinsenk@ere.umontreal.ca>
# last revision: 1996-3-13
# modified by Jim Hugunin 1997-3-3 for repr's and str's (and other details)
# and by Perry Greenfield 2000-4-1 for numarray
# and by Travis Oliphant  2005-8-22 for numpy

import sys
import numeric      as _gen
import numerictypes as _nt
import umath        as _uf
from multiarray import format_longfloat
from fromnumeric import ravel
_nc = _gen

# The following functions are emergency substitutes for numeric functions
# which sometimes get broken during development.

def product(x, y): return x*y

def _maximum_reduce(arr):
    maximum = arr[0]
    for i in xrange(1, arr.nelements()):
        if arr[i] > maximum: maximum = arr[i]
    return maximum

def _minimum_reduce(arr):
    minimum = arr[0]
    for i in xrange(1, arr.nelements()):
        if arr[i] < minimum: minimum = arr[i]
    return minimum

def _numeric_compress(arr):
    nonzero = 0
    for i in xrange(arr.nelements()):
        if arr[i] != 0: nonzero += 1
    retarr = _nc.zeros((nonzero,))
    nonzero = 0
    for i in xrange(arr.nelements()):
        if arr[i] != 0:
            retarr[nonzero] = abs(arr[i])
            nonzero += 1
    return retarr

_failsafe = 0
if _failsafe:
    max_reduce = _maximum_reduce
    min_reduce = _minimum_reduce
else:
    max_reduce = _uf.maximum.reduce
    min_reduce = _uf.minimum.reduce

_summaryEdgeItems = 3     # repr N leading and trailing items of each dimension
_summaryThreshold = 1000 # total items > triggers array summarization

_float_output_precision = 8
_float_output_suppress_small = False
_line_width = 75


def set_printoptions(precision=None, threshold=None, edgeitems=None,
                     linewidth=None, suppress=None):
    """Set options associated with printing.

    precision  the default number of digits of precision for floating
                   point output
                   (default 8)
    threshold  total number of array elements which trigger summarization
                   rather than full repr.
                   (default 1000)
    edgeitems  number of array items in summary at beginning and end of
                   each dimension.
                   (default 3)
    linewidth  the number of characters per line for the purpose of inserting
                   line breaks.
                   (default 75)
    suppress    Boolean value indicating whether or not suppress printing
                   of small floating point values using scientific notation
                   (default False)
    """

    global _summaryThreshold, _summaryEdgeItems, _float_output_precision, \
           _line_width, _float_output_suppress_small
    if (linewidth is not None):
        _line_width = linewidth
    if (threshold is not None):
        _summaryThreshold = threshold
    if (edgeitems is not None):
        _summaryEdgeItems = edgeitems
    if (precision is not None):
        _float_output_precision = precision
    if (suppress is not None):
        _float_output_suppress_small = not not suppress
    return

def get_printoptions():
    return _float_output_precision, _summaryThreshold, _summaryEdgeItems, \
           _line_width, _float_output_suppress_small


def _leading_trailing(a):
    if a.ndim == 1:
        if len(a) > 2*_summaryEdgeItems:
            b = _gen.concatenate((a[:_summaryEdgeItems],
                                     a[-_summaryEdgeItems:]))
        else:
            b = a
    else:
        if len(a) > 2*_summaryEdgeItems:
            l = [_leading_trailing(a[i]) for i in range(
                min(len(a), _summaryEdgeItems))]
            l.extend([_leading_trailing(a[-i]) for i in range(
                min(len(a), _summaryEdgeItems),0,-1)])
        else:
            l = [_leading_trailing(a[i]) for i in range(0, len(a))]
        b = _gen.concatenate(tuple(l))
    return b

def _array2string(a, max_line_width, precision, suppress_small, separator=' ',
                  prefix=""):

    if max_line_width is None:
        max_line_width = _line_width

    if precision is None:
        precision = _float_output_precision

    if suppress_small is None:
        suppress_small = _float_output_suppress_small

    if a.size > _summaryThreshold:
        summary_insert = "..., "
        data = _leading_trailing(a)
    else:
        summary_insert = ""
        data = ravel(a)

    try:
        format_function = a._format
    except AttributeError:
        dtype = a.dtype.type
        if issubclass(dtype, _nt.bool):
            format = "%s"
            format_function = lambda x: format % x
        if issubclass(dtype, _nt.integer):
            max_str_len = max(len(str(max_reduce(data))),
                              len(str(min_reduce(data))))
            format = '%' + str(max_str_len) + 'd'
            format_function = lambda x: _formatInteger(x, format)
        elif issubclass(dtype, _nt.floating):
            if issubclass(dtype, _nt.longfloat):
                format_function = _longfloatFormatter(precision)
            else:
                format = _floatFormat(data, precision, suppress_small)
                format_function = lambda x: _formatFloat(x, format)
        elif issubclass(dtype, _nt.complexfloating):
            if issubclass(dtype, _nt.clongfloat):
                format_function = _clongfloatFormatter(precision)
            else:
                real_format = _floatFormat(
                    data.real, precision, suppress_small, sign=0)
                imag_format = _floatFormat(
                    data.imag, precision, suppress_small, sign=1)
                format_function = lambda x: \
                              _formatComplex(x, real_format, imag_format)
        elif issubclass(dtype, _nt.unicode_) or \
             issubclass(dtype, _nt.string_):
            format = "%s"
            format_function = lambda x: repr(x)
        else:
            format = '%s'
            format_function = lambda x: format % str(x)

    next_line_prefix = " " # skip over "["
    next_line_prefix += " "*len(prefix)                  # skip over array(


    lst = _formatArray(a, format_function, len(a.shape), max_line_width,
                       next_line_prefix, separator,
                       _summaryEdgeItems, summary_insert)[:-1]

    return lst

def _convert_arrays(obj):
    newtup = []
    for k in obj:
        if isinstance(k, _gen.ndarray):
            k = k.tolist()
        elif isinstance(k, tuple):
            k = _convert_arrays(k)
        newtup.append(k)
    return tuple(newtup)


def array2string(a, max_line_width = None, precision = None,
                 suppress_small = None, separator=' ', prefix="",
                 style=repr):

    if a.shape == ():
        x = a.item()
        try:
            lst = a._format(x)
        except AttributeError:
            if isinstance(x, tuple):
                x = _convert_arrays(x)
            lst = style(x)
    elif reduce(product, a.shape) == 0:
        # treat as a null array if any of shape elements == 0
        lst = "[]"
    else:
        lst = _array2string(a, max_line_width, precision, suppress_small,
                            separator, prefix)
    return lst

def _extendLine(s, line, word, max_line_len, next_line_prefix):
    if len(line.rstrip()) + len(word.rstrip()) >= max_line_len:
        s += line.rstrip() + "\n"
        line = next_line_prefix
    line += word
    return s, line


def _formatArray(a, format_function, rank, max_line_len,
                 next_line_prefix, separator, edge_items, summary_insert):
    """formatArray is designed for two modes of operation:

    1. Full output

    2. Summarized output

    """
    if rank == 0:
        obj = a.item()
        if isinstance(obj, tuple):
            obj = _convert_arrays(obj)
        return str(obj)

    if summary_insert and 2*edge_items < len(a):
        leading_items, trailing_items, summary_insert1 = \
                       edge_items, edge_items, summary_insert
    else:
        leading_items, trailing_items, summary_insert1 = 0, len(a), ""

    if rank == 1:

        s = ""
        line = next_line_prefix
        for i in xrange(leading_items):
            word = format_function(a[i]) + separator
            s, line = _extendLine(s, line, word, max_line_len, next_line_prefix)

        if summary_insert1:
            s, line = _extendLine(s, line, summary_insert1, max_line_len, next_line_prefix)

        for i in xrange(trailing_items, 1, -1):
            word = format_function(a[-i]) + separator
            s, line = _extendLine(s, line, word, max_line_len, next_line_prefix)

        word = format_function(a[-1])
        s, line = _extendLine(s, line, word, max_line_len, next_line_prefix)
        s += line + "]\n"
        s = '[' + s[len(next_line_prefix):]
    else:
        s = '['
        sep = separator.rstrip()
        for i in xrange(leading_items):
            if i > 0:
                s += next_line_prefix
            s += _formatArray(a[i], format_function, rank-1, max_line_len,
                              " " + next_line_prefix, separator, edge_items,
                              summary_insert)
            s = s.rstrip() + sep.rstrip() + '\n'*max(rank-1,1)

        if summary_insert1:
            s += next_line_prefix + summary_insert1 + "\n"

        for i in xrange(trailing_items, 1, -1):
            if leading_items or i != trailing_items:
                s += next_line_prefix
            s += _formatArray(a[-i], format_function, rank-1, max_line_len,
                              " " + next_line_prefix, separator, edge_items,
                              summary_insert)
            s = s.rstrip() + sep.rstrip() + '\n'*max(rank-1,1)
        if leading_items or trailing_items > 1:
            s += next_line_prefix
        s += _formatArray(a[-1], format_function, rank-1, max_line_len,
                          " " + next_line_prefix, separator, edge_items,
                          summary_insert).rstrip()+']\n'
    return s

def _floatFormat(data, precision, suppress_small, sign = 0):
    exp_format = 0
    non_zero = _uf.absolute(data.compress(_uf.not_equal(data, 0)))
    ##non_zero = _numeric_compress(data) ##
    if len(non_zero) == 0:
        max_val = 0.
        min_val = 0.
    else:
        max_val = max_reduce(non_zero)
        min_val = min_reduce(non_zero)
        if max_val >= 1.e8:
            exp_format = 1
        if not suppress_small and (min_val < 0.0001
                                   or max_val/min_val > 1000.):
            exp_format = 1
    if exp_format:
        large_exponent = 0 < min_val < 1e-99 or max_val >= 1e100
        max_str_len = 8 + precision + large_exponent
        if sign: format = '%+'
        else: format = '%'
        format = format + str(max_str_len) + '.' + str(precision) + 'e'
        if large_exponent: format = format + '3'
    else:
        format = '%.' + str(precision) + 'f'
        precision = min(precision, max([_digits(x, precision, format)
                                        for x in data]))
        max_str_len = len(str(int(max_val))) + precision + 2
        if sign: format = '%#+'
        else: format = '%#'
        format = format + str(max_str_len) + '.' + str(precision) + 'f'
    return format

def _digits(x, precision, format):
    s = format % x
    zeros = len(s)
    while s[zeros-1] == '0': zeros = zeros-1
    return precision-len(s)+zeros


_MAXINT = sys.maxint
_MININT = -sys.maxint-1
def _formatInteger(x, format):
    if (x < _MAXINT) and (x > _MININT):
        return format % x
    else:
        return "%s" % x

def _longfloatFormatter(precision):
    def formatter(x):
        return format_longfloat(x, precision)
    return formatter

def _formatFloat(x, format, strip_zeros = 1):
    if format[-1] == '3':
        # 3-digit exponent
        format = format[:-1]
        s = format % x
        third = s[-3]
        if third == '+' or third == '-':
            s = s[1:-2] + '0' + s[-2:]
    elif format[-1] == 'e':
        # 2-digit exponent
        s = format % x
        if s[-3] == '0':
            s = ' ' + s[:-3] + s[-2:]
    elif format[-1] == 'f':
        s = format % x
        if strip_zeros:
            zeros = len(s)
            while s[zeros-1] == '0': zeros = zeros-1
            s = s[:zeros] + (len(s)-zeros)*' '
    else:
        s = format % x
    return s

def _clongfloatFormatter(precision):
    def formatter(x):
        r = format_longfloat(x.real, precision)
        i = format_longfloat(x.imag, precision)
        if x.imag < 0:
            i = '-' + i
        else:
            i = '+' + i
        return r + i + 'j'
    return formatter

def _formatComplex(x, real_format, imag_format):
    r = _formatFloat(x.real, real_format)
    i = _formatFloat(x.imag, imag_format, 0)
    if imag_format[-1] == 'f':
        zeros = len(i)
        while zeros > 2 and i[zeros-1] == '0': zeros = zeros-1
        i = i[:zeros] + 'j' + (len(i)-zeros)*' '
    else:
        i = i + 'j'
    return r + i

def _formatGeneral(x):
    return str(x) + ' '

if __name__ == '__main__':
    a = _nc.arange(10)
    print array2string(a)
    print array2string(_nc.array([[],[]]))
