
# missing Numarray defined names (in from numarray import *)
##__all__ = ['ArrayType', 'CLIP', 'ClassicUnpickler', 'Complex32_fromtype',
##           'Complex64_fromtype', 'ComplexArray', 'EarlyEOFError', 'Error',
##           'FileSeekWarning', 'MAX_ALIGN', 'MAX_INT_SIZE', 'MAX_LINE_WIDTH',
##           'MathDomainError', 'NDArray', 'NewArray', 'NewAxis', 'NumArray',
##           'NumError', 'NumOverflowError', 'PRECISION', 'Py2NumType',
##           'PyINT_TYPES', 'PyLevel2Type', 'PyNUMERIC_TYPES', 'PyREAL_TYPES',
##           'RAISE', 'SLOPPY', 'STRICT', 'SUPPRESS_SMALL', 'SizeMismatchError',
##           'SizeMismatchWarning', 'SuitableBuffer', 'USING_BLAS',
##           'UnderflowError', 'UsesOpPriority', 'WARN', 'WRAP', 'all',
##           'allclose', 'alltrue', 'and_', 'any', 'arange', 'argmax',
##           'argmin', 'argsort', 'around', 'array2list', 'array_equal',
##           'array_equiv', 'array_repr', 'array_str', 'arrayprint',
##           'arrayrange', 'average', 'choose', 'clip',
##           'codegenerator', 'compress', 'concatenate', 'conjugate',
##           'copy', 'copy_reg', 'diagonal', 'divide_remainder',
##           'dotblas', 'e', 'explicit_type', 'flush_caches', 'fromfile',
##           'fromfunction', 'fromlist', 'fromstring', 'generic',
##           'genericCoercions', 'genericPromotionExclusions', 'genericTypeRank',
##           'getShape', 'getTypeObject', 'handleError', 'identity', 'indices',
##           'info', 'innerproduct', 'inputarray', 'isBigEndian',
##           'kroneckerproduct', 'lexsort', 'libnumarray', 'libnumeric',
##           'load', 'make_ufuncs', 'math', 'memory',
##           'numarrayall', 'numarraycore', 'numerictypes', 'numinclude',
##           'operator', 'os', 'outerproduct', 'pi', 'put', 'putmask',
##           'pythonTypeMap', 'pythonTypeRank', 'rank', 'repeat',
##           'reshape', 'resize', 'round', 'safethread', 'save', 'scalarTypeMap',
##           'scalarTypes', 'searchsorted', 'session', 'shape', 'sign', 'size',
##           'sometrue', 'sort', 'swapaxes', 'sys', 'take', 'tcode',
##           'tensormultiply', 'tname', 'trace', 'transpose', 'typeDict',
##           'typecode', 'typecodes', 'typeconv', 'types', 'ufunc',
##           'ufuncFactory', 'value', ]


__all__ = ['asarray', 'ones', 'zeros', 'array', 'where']
__all__ += ['vdot', 'dot', 'matrixmultiply', 'ravel', 'indices',
            'arange', 'concatenate']

from numpy import dot as matrixmultiply, dot, vdot, ravel

def array(sequence=None, typecode=None, copy=1, savespace=0,
          type=None, shape=None, dtype=None):
    dtype = type2dtype(typecode, type, dtype)
    if sequence is None:
        if shape is None:
            return None
        if dtype is None:
            dtype = 'l'
        return N.empty(shape, dtype)
    arr = N.array(sequence, dtype, copy=copy)
    if shape is not None:
        arr.shape = shape
    return arr

def asarray(seq, type=None, typecode=None, dtype=None):
    if seq is None:
        return None
    dtype = type2dtype(typecode, type, dtype)
    return N.array(seq, dtype, copy=0)

def ones(shape, type=None, typecode=None, dtype=None):
    dtype = type2dtype(typecode, type, dtype)
    return N.ones(shape, dtype)

def zeros(shape, type=None, typecode=None, dtype=None):
    dtype = type2dtype(typecode, type, dtype)
    return N.zeros(shape, dtype)

def where(condition, x=None, y=None, out=None):
    if x is None and y is None:
        arr = N.where(condition)
    else:
        arr = N.where(condition, x, y)
    if out is not None:
        out[...] = arr
        return out
    return arr
    
def indices(shape, type=None):
    return N.indices(shape, type)

def arange(a1, a2=None, stride=1, type=None, shape=None,
           typecode=None, dtype=None):
    dtype = type2dtype(typecode, type, dtype)
    return N.arange(a1, a2, stride, dtype)
