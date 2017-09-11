from __future__ import division, print_function

import os
import re
import struct
import sys
import textwrap

sys.path.insert(0, os.path.dirname(__file__))
import ufunc_docstrings as docstrings
sys.path.pop(0)

Zero = "PyUFunc_Zero"
One = "PyUFunc_One"
None_ = "PyUFunc_None"
AllOnes = "PyUFunc_MinusOne"
ReorderableNone = "PyUFunc_ReorderableNone"

# Sentinel value to specify using the full type description in the
# function name
class FullTypeDescr(object):
    pass

class FuncNameSuffix(object):
    """Stores the suffix to append when generating functions names.
    """
    def __init__(self, suffix):
        self.suffix = suffix

class TypeDescription(object):
    """Type signature for a ufunc.

    Attributes
    ----------
    type : str
        Character representing the nominal type.
    func_data : str or None or FullTypeDescr or FuncNameSuffix, optional
        The string representing the expression to insert into the data
        array, if any.
    in_ : str or None, optional
        The typecode(s) of the inputs.
    out : str or None, optional
        The typecode(s) of the outputs.
    astype : dict or None, optional
        If astype['x'] is 'y', uses PyUFunc_x_x_As_y_y/PyUFunc_xx_x_As_yy_y
        instead of PyUFunc_x_x/PyUFunc_xx_x.
    simd: list
        Available SIMD ufunc loops, dispatched at runtime in specified order
        Currently only supported for simples types (see make_arrays)
    """
    def __init__(self, type, f=None, in_=None, out=None, astype=None, simd=None):
        self.type = type
        self.func_data = f
        if astype is None:
            astype = {}
        self.astype_dict = astype
        if in_ is not None:
            in_ = in_.replace('P', type)
        self.in_ = in_
        if out is not None:
            out = out.replace('P', type)
        self.out = out
        self.simd = simd

    def finish_signature(self, nin, nout):
        if self.in_ is None:
            self.in_ = self.type * nin
        assert len(self.in_) == nin
        if self.out is None:
            self.out = self.type * nout
        assert len(self.out) == nout
        self.astype = self.astype_dict.get(self.type, None)

_fdata_map = dict(e='npy_%sf', f='npy_%sf', d='npy_%s', g='npy_%sl',
                  F='nc_%sf', D='nc_%s', G='nc_%sl')
def build_func_data(types, f):
    func_data = []
    for t in types:
        d = _fdata_map.get(t, '%s') % (f,)
        func_data.append(d)
    return func_data

def TD(types, f=None, astype=None, in_=None, out=None, simd=None):
    if f is not None:
        if isinstance(f, str):
            func_data = build_func_data(types, f)
        else:
            assert len(f) == len(types)
            func_data = f
    else:
        func_data = (None,) * len(types)
    if isinstance(in_, str):
        in_ = (in_,) * len(types)
    elif in_ is None:
        in_ = (None,) * len(types)
    if isinstance(out, str):
        out = (out,) * len(types)
    elif out is None:
        out = (None,) * len(types)
    tds = []
    for t, fd, i, o in zip(types, func_data, in_, out):
        # [(simd-name, list of types)]
        if simd is not None:
            simdt = [k for k, v in simd if t in v]
        else:
            simdt = []
        tds.append(TypeDescription(t, f=fd, in_=i, out=o, astype=astype, simd=simdt))
    return tds

class Ufunc(object):
    """Description of a ufunc.

    Attributes
    ----------
    nin : number of input arguments
    nout : number of output arguments
    identity : identity element for a two-argument function
    docstring : docstring for the ufunc
    type_descriptions : list of TypeDescription objects
    """
    def __init__(self, nin, nout, identity, docstring, typereso,
                 *type_descriptions):
        self.nin = nin
        self.nout = nout
        if identity is None:
            identity = None_
        self.identity = identity
        self.docstring = docstring
        self.typereso = typereso
        self.type_descriptions = []
        for td in type_descriptions:
            self.type_descriptions.extend(td)
        for td in self.type_descriptions:
            td.finish_signature(self.nin, self.nout)

# String-handling utilities to avoid locale-dependence.

import string
if sys.version_info[0] < 3:
    UPPER_TABLE = string.maketrans(string.ascii_lowercase,
                                   string.ascii_uppercase)
else:
    UPPER_TABLE = bytes.maketrans(bytes(string.ascii_lowercase, "ascii"),
                                  bytes(string.ascii_uppercase, "ascii"))

def english_upper(s):
    """ Apply English case rules to convert ASCII strings to all upper case.

    This is an internal utility function to replace calls to str.upper() such
    that we can avoid changing behavior with changing locales. In particular,
    Turkish has distinct dotted and dotless variants of the Latin letter "I" in
    both lowercase and uppercase. Thus, "i".upper() != "I" in a "tr" locale.

    Parameters
    ----------
    s : str

    Returns
    -------
    uppered : str

    Examples
    --------
    >>> from numpy.lib.utils import english_upper
    >>> s = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789_'
    >>> english_upper(s)
    'ABCDEFGHIJKLMNOPQRSTUVWXYZABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_'
    >>> english_upper('')
    ''
    """
    uppered = s.translate(UPPER_TABLE)
    return uppered


#each entry in defdict is a Ufunc object.

#name: [string of chars for which it is defined,
#       string of characters using func interface,
#       tuple of strings giving funcs for data,
#       (in, out), or (instr, outstr) giving the signature as character codes,
#       identity,
#       docstring,
#       output specification (optional)
#       ]

chartoname = {'?': 'bool',
              'b': 'byte',
              'B': 'ubyte',
              'h': 'short',
              'H': 'ushort',
              'i': 'int',
              'I': 'uint',
              'l': 'long',
              'L': 'ulong',
              'q': 'longlong',
              'Q': 'ulonglong',
              'e': 'half',
              'f': 'float',
              'd': 'double',
              'g': 'longdouble',
              'F': 'cfloat',
              'D': 'cdouble',
              'G': 'clongdouble',
              'M': 'datetime',
              'm': 'timedelta',
              'O': 'OBJECT',
              # '.' is like 'O', but calls a method of the object instead
              # of a function
              'P': 'OBJECT',
              }

all = '?bBhHiIlLqQefdgFDGOMm'
O = 'O'
P = 'P'
ints = 'bBhHiIlLqQ'
times = 'Mm'
timedeltaonly = 'm'
intsO = ints + O
bints = '?' + ints
bintsO = bints + O
flts = 'efdg'
fltsO = flts + O
fltsP = flts + P
cmplx = 'FDG'
cmplxO = cmplx + O
cmplxP = cmplx + P
inexact = flts + cmplx
inexactvec = 'fd'
noint = inexact+O
nointP = inexact+P
allP = bints+times+flts+cmplxP
nobool = all[1:]
noobj = all[:-3]+all[-2:]
nobool_or_obj = all[1:-3]+all[-2:]
nobool_or_datetime = all[1:-2]+all[-1:]
intflt = ints+flts
intfltcmplx = ints+flts+cmplx
nocmplx = bints+times+flts
nocmplxO = nocmplx+O
nocmplxP = nocmplx+P
notimes_or_obj = bints + inexact
nodatetime_or_obj = bints + inexact

# Find which code corresponds to int64.
int64 = ''
uint64 = ''
for code in 'bhilq':
    if struct.calcsize(code) == 8:
        int64 = code
        uint64 = english_upper(code)
        break

# This dictionary describes all the ufunc implementations, generating
# all the function names and their corresponding ufunc signatures.  TD is
# an object which expands a list of character codes into an array of
# TypeDescriptions.
defdict = {
'add':
    Ufunc(2, 1, Zero,
          docstrings.get('numpy.core.umath.add'),
          'PyUFunc_AdditionTypeResolver',
          TD(notimes_or_obj, simd=[('avx2', ints)]),
          [TypeDescription('M', FullTypeDescr, 'Mm', 'M'),
           TypeDescription('m', FullTypeDescr, 'mm', 'm'),
           TypeDescription('M', FullTypeDescr, 'mM', 'M'),
          ],
          TD(O, f='PyNumber_Add'),
          ),
'subtract':
    Ufunc(2, 1, None, # Zero is only a unit to the right, not the left
          docstrings.get('numpy.core.umath.subtract'),
          'PyUFunc_SubtractionTypeResolver',
          TD(notimes_or_obj, simd=[('avx2', ints)]),
          [TypeDescription('M', FullTypeDescr, 'Mm', 'M'),
           TypeDescription('m', FullTypeDescr, 'mm', 'm'),
           TypeDescription('M', FullTypeDescr, 'MM', 'm'),
          ],
          TD(O, f='PyNumber_Subtract'),
          ),
'multiply':
    Ufunc(2, 1, One,
          docstrings.get('numpy.core.umath.multiply'),
          'PyUFunc_MultiplicationTypeResolver',
          TD(notimes_or_obj, simd=[('avx2', ints)]),
          [TypeDescription('m', FullTypeDescr, 'mq', 'm'),
           TypeDescription('m', FullTypeDescr, 'qm', 'm'),
           TypeDescription('m', FullTypeDescr, 'md', 'm'),
           TypeDescription('m', FullTypeDescr, 'dm', 'm'),
          ],
          TD(O, f='PyNumber_Multiply'),
          ),
'divide':
    Ufunc(2, 1, None, # One is only a unit to the right, not the left
          docstrings.get('numpy.core.umath.divide'),
          'PyUFunc_MixedDivisionTypeResolver',
          TD(intfltcmplx),
          [TypeDescription('m', FullTypeDescr, 'mq', 'm'),
           TypeDescription('m', FullTypeDescr, 'md', 'm'),
           TypeDescription('m', FullTypeDescr, 'mm', 'd'),
          ],
          TD(O, f='PyNumber_Divide'),
          ),
'floor_divide':
    Ufunc(2, 1, None, # One is only a unit to the right, not the left
          docstrings.get('numpy.core.umath.floor_divide'),
          'PyUFunc_DivisionTypeResolver',
          TD(intfltcmplx),
          [TypeDescription('m', FullTypeDescr, 'mq', 'm'),
           TypeDescription('m', FullTypeDescr, 'md', 'm'),
           #TypeDescription('m', FullTypeDescr, 'mm', 'd'),
          ],
          TD(O, f='PyNumber_FloorDivide'),
          ),
'true_divide':
    Ufunc(2, 1, None, # One is only a unit to the right, not the left
          docstrings.get('numpy.core.umath.true_divide'),
          'PyUFunc_TrueDivisionTypeResolver',
          TD(flts+cmplx),
          [TypeDescription('m', FullTypeDescr, 'mq', 'm'),
           TypeDescription('m', FullTypeDescr, 'md', 'm'),
           TypeDescription('m', FullTypeDescr, 'mm', 'd'),
          ],
          TD(O, f='PyNumber_TrueDivide'),
          ),
'conjugate':
    Ufunc(1, 1, None,
          docstrings.get('numpy.core.umath.conjugate'),
          None,
          TD(ints+flts+cmplx, simd=[('avx2', ints)]),
          TD(P, f='conjugate'),
          ),
'fmod':
    Ufunc(2, 1, None,
          docstrings.get('numpy.core.umath.fmod'),
          None,
          TD(ints),
          TD(flts, f='fmod', astype={'e':'f'}),
          TD(P, f='fmod'),
          ),
'square':
    Ufunc(1, 1, None,
          docstrings.get('numpy.core.umath.square'),
          None,
          TD(ints+inexact, simd=[('avx2', ints)]),
          TD(O, f='Py_square'),
          ),
'reciprocal':
    Ufunc(1, 1, None,
          docstrings.get('numpy.core.umath.reciprocal'),
          None,
          TD(ints+inexact, simd=[('avx2', ints)]),
          TD(O, f='Py_reciprocal'),
          ),
# This is no longer used as numpy.ones_like, however it is
# still used by some internal calls.
'_ones_like':
    Ufunc(1, 1, None,
          docstrings.get('numpy.core.umath._ones_like'),
          'PyUFunc_OnesLikeTypeResolver',
          TD(noobj),
          TD(O, f='Py_get_one'),
          ),
'power':
    Ufunc(2, 1, None,
          docstrings.get('numpy.core.umath.power'),
          None,
          TD(ints),
          TD(inexact, f='pow', astype={'e':'f'}),
          TD(O, f='npy_ObjectPower'),
          ),
'float_power':
    Ufunc(2, 1, None,
          docstrings.get('numpy.core.umath.float_power'),
          None,
          TD('dgDG', f='pow'),
          ),
'absolute':
    Ufunc(1, 1, None,
          docstrings.get('numpy.core.umath.absolute'),
          'PyUFunc_AbsoluteTypeResolver',
          TD(bints+flts+timedeltaonly),
          TD(cmplx, out=('f', 'd', 'g')),
          TD(O, f='PyNumber_Absolute'),
          ),
'_arg':
    Ufunc(1, 1, None,
          docstrings.get('numpy.core.umath._arg'),
          None,
          TD(cmplx, out=('f', 'd', 'g')),
          ),
'negative':
    Ufunc(1, 1, None,
          docstrings.get('numpy.core.umath.negative'),
          'PyUFunc_NegativeTypeResolver',
          TD(bints+flts+timedeltaonly, simd=[('avx2', ints)]),
          TD(cmplx, f='neg'),
          TD(O, f='PyNumber_Negative'),
          ),
'positive':
    Ufunc(1, 1, None,
          docstrings.get('numpy.core.umath.positive'),
          'PyUFunc_SimpleUnaryOperationTypeResolver',
          TD(ints+flts+timedeltaonly),
          TD(cmplx, f='pos'),
          TD(O, f='PyNumber_Positive'),
          ),
'sign':
    Ufunc(1, 1, None,
          docstrings.get('numpy.core.umath.sign'),
          'PyUFunc_SimpleUnaryOperationTypeResolver',
          TD(nobool_or_datetime),
          ),
'greater':
    Ufunc(2, 1, None,
          docstrings.get('numpy.core.umath.greater'),
          'PyUFunc_SimpleBinaryComparisonTypeResolver',
          TD(all, out='?', simd=[('avx2', ints)]),
          ),
'greater_equal':
    Ufunc(2, 1, None,
          docstrings.get('numpy.core.umath.greater_equal'),
          'PyUFunc_SimpleBinaryComparisonTypeResolver',
          TD(all, out='?', simd=[('avx2', ints)]),
          ),
'less':
    Ufunc(2, 1, None,
          docstrings.get('numpy.core.umath.less'),
          'PyUFunc_SimpleBinaryComparisonTypeResolver',
          TD(all, out='?', simd=[('avx2', ints)]),
          ),
'less_equal':
    Ufunc(2, 1, None,
          docstrings.get('numpy.core.umath.less_equal'),
          'PyUFunc_SimpleBinaryComparisonTypeResolver',
          TD(all, out='?', simd=[('avx2', ints)]),
          ),
'equal':
    Ufunc(2, 1, None,
          docstrings.get('numpy.core.umath.equal'),
          'PyUFunc_SimpleBinaryComparisonTypeResolver',
          TD(all, out='?', simd=[('avx2', ints)]),
          ),
'not_equal':
    Ufunc(2, 1, None,
          docstrings.get('numpy.core.umath.not_equal'),
          'PyUFunc_SimpleBinaryComparisonTypeResolver',
          TD(all, out='?', simd=[('avx2', ints)]),
          ),
'logical_and':
    Ufunc(2, 1, One,
          docstrings.get('numpy.core.umath.logical_and'),
          'PyUFunc_SimpleBinaryComparisonTypeResolver',
          TD(nodatetime_or_obj, out='?', simd=[('avx2', ints)]),
          TD(O, f='npy_ObjectLogicalAnd'),
          ),
'logical_not':
    Ufunc(1, 1, None,
          docstrings.get('numpy.core.umath.logical_not'),
          None,
          TD(nodatetime_or_obj, out='?', simd=[('avx2', ints)]),
          TD(O, f='npy_ObjectLogicalNot'),
          ),
'logical_or':
    Ufunc(2, 1, Zero,
          docstrings.get('numpy.core.umath.logical_or'),
          'PyUFunc_SimpleBinaryComparisonTypeResolver',
          TD(nodatetime_or_obj, out='?', simd=[('avx2', ints)]),
          TD(O, f='npy_ObjectLogicalOr'),
          ),
'logical_xor':
    Ufunc(2, 1, Zero,
          docstrings.get('numpy.core.umath.logical_xor'),
          'PyUFunc_SimpleBinaryComparisonTypeResolver',
          TD(nodatetime_or_obj, out='?'),
          TD(P, f='logical_xor'),
          ),
'maximum':
    Ufunc(2, 1, ReorderableNone,
          docstrings.get('numpy.core.umath.maximum'),
          'PyUFunc_SimpleBinaryOperationTypeResolver',
          TD(noobj),
          TD(O, f='npy_ObjectMax')
          ),
'minimum':
    Ufunc(2, 1, ReorderableNone,
          docstrings.get('numpy.core.umath.minimum'),
          'PyUFunc_SimpleBinaryOperationTypeResolver',
          TD(noobj),
          TD(O, f='npy_ObjectMin')
          ),
'fmax':
    Ufunc(2, 1, ReorderableNone,
          docstrings.get('numpy.core.umath.fmax'),
          'PyUFunc_SimpleBinaryOperationTypeResolver',
          TD(noobj),
          TD(O, f='npy_ObjectMax')
          ),
'fmin':
    Ufunc(2, 1, ReorderableNone,
          docstrings.get('numpy.core.umath.fmin'),
          'PyUFunc_SimpleBinaryOperationTypeResolver',
          TD(noobj),
          TD(O, f='npy_ObjectMin')
          ),
'logaddexp':
    Ufunc(2, 1, None,
          docstrings.get('numpy.core.umath.logaddexp'),
          None,
          TD(flts, f="logaddexp", astype={'e':'f'})
          ),
'logaddexp2':
    Ufunc(2, 1, None,
          docstrings.get('numpy.core.umath.logaddexp2'),
          None,
          TD(flts, f="logaddexp2", astype={'e':'f'})
          ),
'bitwise_and':
    Ufunc(2, 1, AllOnes,
          docstrings.get('numpy.core.umath.bitwise_and'),
          None,
          TD(bints, simd=[('avx2', ints)]),
          TD(O, f='PyNumber_And'),
          ),
'bitwise_or':
    Ufunc(2, 1, Zero,
          docstrings.get('numpy.core.umath.bitwise_or'),
          None,
          TD(bints, simd=[('avx2', ints)]),
          TD(O, f='PyNumber_Or'),
          ),
'bitwise_xor':
    Ufunc(2, 1, Zero,
          docstrings.get('numpy.core.umath.bitwise_xor'),
          None,
          TD(bints, simd=[('avx2', ints)]),
          TD(O, f='PyNumber_Xor'),
          ),
'invert':
    Ufunc(1, 1, None,
          docstrings.get('numpy.core.umath.invert'),
          None,
          TD(bints, simd=[('avx2', ints)]),
          TD(O, f='PyNumber_Invert'),
          ),
'left_shift':
    Ufunc(2, 1, None,
          docstrings.get('numpy.core.umath.left_shift'),
          None,
          TD(ints, simd=[('avx2', ints)]),
          TD(O, f='PyNumber_Lshift'),
          ),
'right_shift':
    Ufunc(2, 1, None,
          docstrings.get('numpy.core.umath.right_shift'),
          None,
          TD(ints, simd=[('avx2', ints)]),
          TD(O, f='PyNumber_Rshift'),
          ),
'heaviside':
    Ufunc(2, 1, None,
          docstrings.get('numpy.core.umath.heaviside'),
          None,
          TD(flts, f='heaviside', astype={'e':'f'}),
          ),
'degrees':
    Ufunc(1, 1, None,
          docstrings.get('numpy.core.umath.degrees'),
          None,
          TD(fltsP, f='degrees', astype={'e':'f'}),
          ),
'rad2deg':
    Ufunc(1, 1, None,
          docstrings.get('numpy.core.umath.rad2deg'),
          None,
          TD(fltsP, f='rad2deg', astype={'e':'f'}),
          ),
'radians':
    Ufunc(1, 1, None,
          docstrings.get('numpy.core.umath.radians'),
          None,
          TD(fltsP, f='radians', astype={'e':'f'}),
          ),
'deg2rad':
    Ufunc(1, 1, None,
          docstrings.get('numpy.core.umath.deg2rad'),
          None,
          TD(fltsP, f='deg2rad', astype={'e':'f'}),
          ),
'arccos':
    Ufunc(1, 1, None,
          docstrings.get('numpy.core.umath.arccos'),
          None,
          TD(inexact, f='acos', astype={'e':'f'}),
          TD(P, f='arccos'),
          ),
'arccosh':
    Ufunc(1, 1, None,
          docstrings.get('numpy.core.umath.arccosh'),
          None,
          TD(inexact, f='acosh', astype={'e':'f'}),
          TD(P, f='arccosh'),
          ),
'arcsin':
    Ufunc(1, 1, None,
          docstrings.get('numpy.core.umath.arcsin'),
          None,
          TD(inexact, f='asin', astype={'e':'f'}),
          TD(P, f='arcsin'),
          ),
'arcsinh':
    Ufunc(1, 1, None,
          docstrings.get('numpy.core.umath.arcsinh'),
          None,
          TD(inexact, f='asinh', astype={'e':'f'}),
          TD(P, f='arcsinh'),
          ),
'arctan':
    Ufunc(1, 1, None,
          docstrings.get('numpy.core.umath.arctan'),
          None,
          TD(inexact, f='atan', astype={'e':'f'}),
          TD(P, f='arctan'),
          ),
'arctanh':
    Ufunc(1, 1, None,
          docstrings.get('numpy.core.umath.arctanh'),
          None,
          TD(inexact, f='atanh', astype={'e':'f'}),
          TD(P, f='arctanh'),
          ),
'cos':
    Ufunc(1, 1, None,
          docstrings.get('numpy.core.umath.cos'),
          None,
          TD(inexact, f='cos', astype={'e':'f'}),
          TD(P, f='cos'),
          ),
'sin':
    Ufunc(1, 1, None,
          docstrings.get('numpy.core.umath.sin'),
          None,
          TD(inexact, f='sin', astype={'e':'f'}),
          TD(P, f='sin'),
          ),
'tan':
    Ufunc(1, 1, None,
          docstrings.get('numpy.core.umath.tan'),
          None,
          TD(inexact, f='tan', astype={'e':'f'}),
          TD(P, f='tan'),
          ),
'cosh':
    Ufunc(1, 1, None,
          docstrings.get('numpy.core.umath.cosh'),
          None,
          TD(inexact, f='cosh', astype={'e':'f'}),
          TD(P, f='cosh'),
          ),
'sinh':
    Ufunc(1, 1, None,
          docstrings.get('numpy.core.umath.sinh'),
          None,
          TD(inexact, f='sinh', astype={'e':'f'}),
          TD(P, f='sinh'),
          ),
'tanh':
    Ufunc(1, 1, None,
          docstrings.get('numpy.core.umath.tanh'),
          None,
          TD(inexact, f='tanh', astype={'e':'f'}),
          TD(P, f='tanh'),
          ),
'exp':
    Ufunc(1, 1, None,
          docstrings.get('numpy.core.umath.exp'),
          None,
          TD(inexact, f='exp', astype={'e':'f'}),
          TD(P, f='exp'),
          ),
'exp2':
    Ufunc(1, 1, None,
          docstrings.get('numpy.core.umath.exp2'),
          None,
          TD(inexact, f='exp2', astype={'e':'f'}),
          TD(P, f='exp2'),
          ),
'expm1':
    Ufunc(1, 1, None,
          docstrings.get('numpy.core.umath.expm1'),
          None,
          TD(inexact, f='expm1', astype={'e':'f'}),
          TD(P, f='expm1'),
          ),
'log':
    Ufunc(1, 1, None,
          docstrings.get('numpy.core.umath.log'),
          None,
          TD(inexact, f='log', astype={'e':'f'}),
          TD(P, f='log'),
          ),
'log2':
    Ufunc(1, 1, None,
          docstrings.get('numpy.core.umath.log2'),
          None,
          TD(inexact, f='log2', astype={'e':'f'}),
          TD(P, f='log2'),
          ),
'log10':
    Ufunc(1, 1, None,
          docstrings.get('numpy.core.umath.log10'),
          None,
          TD(inexact, f='log10', astype={'e':'f'}),
          TD(P, f='log10'),
          ),
'log1p':
    Ufunc(1, 1, None,
          docstrings.get('numpy.core.umath.log1p'),
          None,
          TD(inexact, f='log1p', astype={'e':'f'}),
          TD(P, f='log1p'),
          ),
'sqrt':
    Ufunc(1, 1, None,
          docstrings.get('numpy.core.umath.sqrt'),
          None,
          TD('e', f='sqrt', astype={'e':'f'}),
          TD(inexactvec),
          TD(inexact, f='sqrt', astype={'e':'f'}),
          TD(P, f='sqrt'),
          ),
'cbrt':
    Ufunc(1, 1, None,
          docstrings.get('numpy.core.umath.cbrt'),
          None,
          TD(flts, f='cbrt', astype={'e':'f'}),
          TD(P, f='cbrt'),
          ),
'ceil':
    Ufunc(1, 1, None,
          docstrings.get('numpy.core.umath.ceil'),
          None,
          TD(flts, f='ceil', astype={'e':'f'}),
          TD(P, f='ceil'),
          ),
'trunc':
    Ufunc(1, 1, None,
          docstrings.get('numpy.core.umath.trunc'),
          None,
          TD(flts, f='trunc', astype={'e':'f'}),
          TD(P, f='trunc'),
          ),
'fabs':
    Ufunc(1, 1, None,
          docstrings.get('numpy.core.umath.fabs'),
          None,
          TD(flts, f='fabs', astype={'e':'f'}),
          TD(P, f='fabs'),
       ),
'floor':
    Ufunc(1, 1, None,
          docstrings.get('numpy.core.umath.floor'),
          None,
          TD(flts, f='floor', astype={'e':'f'}),
          TD(P, f='floor'),
          ),
'rint':
    Ufunc(1, 1, None,
          docstrings.get('numpy.core.umath.rint'),
          None,
          TD(inexact, f='rint', astype={'e':'f'}),
          TD(P, f='rint'),
          ),
'arctan2':
    Ufunc(2, 1, None,
          docstrings.get('numpy.core.umath.arctan2'),
          None,
          TD(flts, f='atan2', astype={'e':'f'}),
          TD(P, f='arctan2'),
          ),
'remainder':
    Ufunc(2, 1, None,
          docstrings.get('numpy.core.umath.remainder'),
          None,
          TD(intflt),
          TD(O, f='PyNumber_Remainder'),
          ),
'divmod':
    Ufunc(2, 2, None,
          docstrings.get('numpy.core.umath.divmod'),
          None,
          TD(intflt),
          TD(O, f='PyNumber_Divmod'),
          ),
'hypot':
    Ufunc(2, 1, Zero,
          docstrings.get('numpy.core.umath.hypot'),
          None,
          TD(flts, f='hypot', astype={'e':'f'}),
          TD(P, f='hypot'),
          ),
'isnan':
    Ufunc(1, 1, None,
          docstrings.get('numpy.core.umath.isnan'),
          None,
          TD(inexact, out='?'),
          ),
'isnat':
    Ufunc(1, 1, None,
          docstrings.get('numpy.core.umath.isnat'),
          'PyUFunc_IsNaTTypeResolver',
          TD(times, out='?'),
          ),
'isinf':
    Ufunc(1, 1, None,
          docstrings.get('numpy.core.umath.isinf'),
          None,
          TD(inexact, out='?'),
          ),
'isfinite':
    Ufunc(1, 1, None,
          docstrings.get('numpy.core.umath.isfinite'),
          None,
          TD(inexact, out='?'),
          ),
'signbit':
    Ufunc(1, 1, None,
          docstrings.get('numpy.core.umath.signbit'),
          None,
          TD(flts, out='?'),
          ),
'copysign':
    Ufunc(2, 1, None,
          docstrings.get('numpy.core.umath.copysign'),
          None,
          TD(flts),
          ),
'nextafter':
    Ufunc(2, 1, None,
          docstrings.get('numpy.core.umath.nextafter'),
          None,
          TD(flts),
          ),
'spacing':
    Ufunc(1, 1, None,
          docstrings.get('numpy.core.umath.spacing'),
          None,
          TD(flts),
          ),
'modf':
    Ufunc(1, 2, None,
          docstrings.get('numpy.core.umath.modf'),
          None,
          TD(flts),
          ),
'ldexp' :
    Ufunc(2, 1, None,
          docstrings.get('numpy.core.umath.ldexp'),
          None,
          [TypeDescription('e', None, 'ei', 'e'),
          TypeDescription('f', None, 'fi', 'f'),
          TypeDescription('e', FuncNameSuffix('long'), 'el', 'e'),
          TypeDescription('f', FuncNameSuffix('long'), 'fl', 'f'),
          TypeDescription('d', None, 'di', 'd'),
          TypeDescription('d', FuncNameSuffix('long'), 'dl', 'd'),
          TypeDescription('g', None, 'gi', 'g'),
          TypeDescription('g', FuncNameSuffix('long'), 'gl', 'g'),
          ],
          ),
'frexp' :
    Ufunc(1, 2, None,
          docstrings.get('numpy.core.umath.frexp'),
          None,
          [TypeDescription('e', None, 'e', 'ei'),
          TypeDescription('f', None, 'f', 'fi'),
          TypeDescription('d', None, 'd', 'di'),
          TypeDescription('g', None, 'g', 'gi'),
          ],
          )
}

if sys.version_info[0] >= 3:
    # Will be aliased to true_divide in umathmodule.c.src:InitOtherOperators
    del defdict['divide']

def indent(st, spaces):
    indention = ' '*spaces
    indented = indention + st.replace('\n', '\n'+indention)
    # trim off any trailing spaces
    indented = re.sub(r' +$', r'', indented)
    return indented

chartotype1 = {'e': 'e_e',
               'f': 'f_f',
               'd': 'd_d',
               'g': 'g_g',
               'F': 'F_F',
               'D': 'D_D',
               'G': 'G_G',
               'O': 'O_O',
               'P': 'O_O_method'}

chartotype2 = {'e': 'ee_e',
               'f': 'ff_f',
               'd': 'dd_d',
               'g': 'gg_g',
               'F': 'FF_F',
               'D': 'DD_D',
               'G': 'GG_G',
               'O': 'OO_O',
               'P': 'OO_O_method'}
#for each name
# 1) create functions, data, and signature
# 2) fill in functions and data in InitOperators
# 3) add function.

def make_arrays(funcdict):
    # functions array contains an entry for every type implemented NULL
    # should be placed where PyUfunc_ style function will be filled in
    # later
    code1list = []
    code2list = []
    names = sorted(funcdict.keys())
    for name in names:
        uf = funcdict[name]
        funclist = []
        datalist = []
        siglist = []
        k = 0
        sub = 0

        if uf.nin > 1:
            assert uf.nin == 2
            thedict = chartotype2  # two inputs and one output
        else:
            thedict = chartotype1  # one input and one output

        for t in uf.type_descriptions:
            if (t.func_data not in (None, FullTypeDescr) and
                    not isinstance(t.func_data, FuncNameSuffix)):
                funclist.append('NULL')
                astype = ''
                if not t.astype is None:
                    astype = '_As_%s' % thedict[t.astype]
                astr = ('%s_functions[%d] = PyUFunc_%s%s;' %
                           (name, k, thedict[t.type], astype))
                code2list.append(astr)
                if t.type == 'O':
                    astr = ('%s_data[%d] = (void *) %s;' %
                               (name, k, t.func_data))
                    code2list.append(astr)
                    datalist.append('(void *)NULL')
                elif t.type == 'P':
                    datalist.append('(void *)"%s"' % t.func_data)
                else:
                    astr = ('%s_data[%d] = (void *) %s;' %
                               (name, k, t.func_data))
                    code2list.append(astr)
                    datalist.append('(void *)NULL')
                    #datalist.append('(void *)%s' % t.func_data)
                sub += 1
            elif t.func_data is FullTypeDescr:
                tname = english_upper(chartoname[t.type])
                datalist.append('(void *)NULL')
                funclist.append(
                        '%s_%s_%s_%s' % (tname, t.in_, t.out, name))
            elif isinstance(t.func_data, FuncNameSuffix):
                datalist.append('(void *)NULL')
                tname = english_upper(chartoname[t.type])
                funclist.append(
                        '%s_%s_%s' % (tname, name, t.func_data.suffix))
            else:
                datalist.append('(void *)NULL')
                tname = english_upper(chartoname[t.type])
                funclist.append('%s_%s' % (tname, name))
                if t.simd is not None:
                    for vt in t.simd:
                        code2list.append("""\
#ifdef HAVE_ATTRIBUTE_TARGET_{ISA}
if (NPY_CPU_SUPPORTS_{ISA}) {{
    {fname}_functions[{idx}] = {type}_{fname}_{isa};
}}
#endif
""".format(ISA=vt.upper(), isa=vt, fname=name, type=tname, idx=k))

            for x in t.in_ + t.out:
                siglist.append('NPY_%s' % (english_upper(chartoname[x]),))

            k += 1

        funcnames = ', '.join(funclist)
        signames = ', '.join(siglist)
        datanames = ', '.join(datalist)
        code1list.append("static PyUFuncGenericFunction %s_functions[] = {%s};"
                         % (name, funcnames))
        code1list.append("static void * %s_data[] = {%s};"
                         % (name, datanames))
        code1list.append("static char %s_signatures[] = {%s};"
                         % (name, signames))
    return "\n".join(code1list), "\n".join(code2list)

def make_ufuncs(funcdict):
    code3list = []
    names = sorted(funcdict.keys())
    for name in names:
        uf = funcdict[name]
        mlist = []
        docstring = textwrap.dedent(uf.docstring).strip()
        if sys.version_info[0] < 3:
            docstring = docstring.encode('string-escape')
            docstring = docstring.replace(r'"', r'\"')
        else:
            docstring = docstring.encode('unicode-escape').decode('ascii')
            docstring = docstring.replace(r'"', r'\"')
            # XXX: I don't understand why the following replace is not
            # necessary in the python 2 case.
            docstring = docstring.replace(r"'", r"\'")
        # Split the docstring because some compilers (like MS) do not like big
        # string literal in C code. We split at endlines because textwrap.wrap
        # do not play well with \n
        docstring = '\\n\"\"'.join(docstring.split(r"\n"))
        mlist.append(\
r"""f = PyUFunc_FromFuncAndData(%s_functions, %s_data, %s_signatures, %d,
                                %d, %d, %s, "%s",
                                "%s", 0);""" % (name, name, name,
                                                len(uf.type_descriptions),
                                                uf.nin, uf.nout,
                                                uf.identity,
                                                name, docstring))
        if uf.typereso is not None:
            mlist.append(
                r"((PyUFuncObject *)f)->type_resolver = &%s;" % uf.typereso)
        mlist.append(r"""PyDict_SetItemString(dictionary, "%s", f);""" % name)
        mlist.append(r"""Py_DECREF(f);""")
        code3list.append('\n'.join(mlist))
    return '\n'.join(code3list)


def make_code(funcdict, filename):
    code1, code2 = make_arrays(funcdict)
    code3 = make_ufuncs(funcdict)
    code2 = indent(code2, 4)
    code3 = indent(code3, 4)
    code = r"""

/** Warning this file is autogenerated!!!

    Please make changes to the code generator program (%s)
**/

%s

static void
InitOperators(PyObject *dictionary) {
    PyObject *f;

%s
%s
}
""" % (filename, code1, code2, code3)
    return code


if __name__ == "__main__":
    filename = __file__
    fid = open('__umath_generated.c', 'w')
    code = make_code(defdict, filename)
    fid.write(code)
    fid.close()
