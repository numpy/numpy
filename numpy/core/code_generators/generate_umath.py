import re, textwrap
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
import ufunc_docstrings as docstrings
sys.path.pop(0)

Zero = "PyUFunc_Zero"
One = "PyUFunc_One"
None_ = "PyUFunc_None"

class TypeDescription(object):
    """Type signature for a ufunc

    Attributes
    ----------

    type: character representing the type
    func_data:
    in_:
    out:
    """
    def __init__(self, type, f=None, in_=None, out=None):
        self.type = type
        self.func_data = f
        if in_ is not None:
            in_ = in_.replace('.', type)
        self.in_ = in_
        if out is not None:
            out = out.replace('.', type)
        self.out = out

    def finish_signature(self, nin, nout):
        if self.in_ is None:
            self.in_ = self.type * nin
        assert len(self.in_) == nin
        if self.out is None:
            self.out = self.type * nout
        assert len(self.out) == nout

_fdata_map = dict(f='npy_%sf', d='npy_%s', g='npy_%sl',
                  F='nc_%sf', D='nc_%s', G='nc_%sl')
def build_func_data(types, f):
    func_data = []
    for t in types:
        d = _fdata_map.get(t, '%s') % (f,)
        func_data.append(d)
    return func_data

def TD(types, f=None, in_=None, out=None):
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
        tds.append(TypeDescription(t, f=fd, in_=i, out=o))
    return tds

class Ufunc(object):
    """Description of a ufunc.

    Attributes
    ----------

    nin: number of input arguments
    nout: number of output arguments
    identity: identity element for a two-argument function
    docstring: docstring for the ufunc
    type_descriptions: list of TypeDescription objects
    """
    def __init__(self, nin, nout, identity, docstring,
                 *type_descriptions):
        self.nin = nin
        self.nout = nout
        if identity is None:
            identity = None_
        self.identity = identity
        self.docstring = docstring
        self.type_descriptions = []
        for td in type_descriptions:
            self.type_descriptions.extend(td)
        for td in self.type_descriptions:
            td.finish_signature(self.nin, self.nout)

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
              'f': 'float',
              'd': 'double',
              'g': 'longdouble',
              'F': 'cfloat',
              'D': 'cdouble',
              'G': 'clongdouble',
              'O': 'OBJECT',
              # M is like O, but calls a method of the object instead
              # of a function
              'M': 'OBJECT',
              }

all = '?bBhHiIlLqQfdgFDGO'
O = 'O'
M = 'M'
ints = 'bBhHiIlLqQ'
intsO = ints + O
bints = '?' + ints
bintsO = bints + O
flts = 'fdg'
fltsO = flts + O
fltsM = flts + M
cmplx = 'FDG'
cmplxO = cmplx + O
cmplxM = cmplx + M
inexact = flts + cmplx
noint = inexact+O
nointM = inexact+M
allM = bints+flts+cmplxM
nobool = all[1:]
nobool_or_obj = all[1:-1]
intflt = ints+flts
intfltcmplx = nobool_or_obj
nocmplx = bints+flts
nocmplxO = nocmplx+O
nocmplxM = nocmplx+M
noobj = all[:-1]

defdict = {
'add' :
    Ufunc(2, 1, Zero,
          docstrings.get('numpy.core.umath.add'),
          TD(noobj),
          TD(O, f='PyNumber_Add'),
          ),
'subtract' :
    Ufunc(2, 1, Zero,
          docstrings.get('numpy.core.umath.subtract'),
          TD(noobj),
          TD(O, f='PyNumber_Subtract'),
          ),
'multiply' :
    Ufunc(2, 1, One,
          docstrings.get('numpy.core.umath.multiply'),
          TD(noobj),
          TD(O, f='PyNumber_Multiply'),
          ),
'divide' :
    Ufunc(2, 1, One,
          docstrings.get('numpy.core.umath.divide'),
          TD(intfltcmplx),
          TD(O, f='PyNumber_Divide'),
          ),
'floor_divide' :
    Ufunc(2, 1, One,
          docstrings.get('numpy.core.umath.floor_divide'),
          TD(intfltcmplx),
          TD(O, f='PyNumber_FloorDivide'),
          ),
'true_divide' :
    Ufunc(2, 1, One,
          docstrings.get('numpy.core.umath.true_divide'),
          TD('bBhH', out='f'),
          TD('iIlLqQ', out='d'),
          TD(flts+cmplx),
          TD(O, f='PyNumber_TrueDivide'),
          ),
'conjugate' :
    Ufunc(1, 1, None,
          docstrings.get('numpy.core.umath.conjugate'),
          TD(nobool_or_obj),
          TD(M, f='conjugate'),
          ),
'fmod' :
    Ufunc(2, 1, Zero,
          docstrings.get('numpy.core.umath.fmod'),
          TD(ints),
          TD(flts, f='fmod'),
          TD(M, f='fmod'),
          ),
'square' :
    Ufunc(1, 1, None,
          docstrings.get('numpy.core.umath.square'),
          TD(nobool_or_obj),
          TD(O, f='Py_square'),
          ),
'reciprocal' :
    Ufunc(1, 1, None,
          docstrings.get('numpy.core.umath.reciprocal'),
          TD(nobool_or_obj),
          TD(O, f='Py_reciprocal'),
          ),
'ones_like' :
    Ufunc(1, 1, None,
          docstrings.get('numpy.core.umath.ones_like'),
          TD(noobj),
          TD(O, f='Py_get_one'),
          ),
'power' :
    Ufunc(2, 1, One,
          docstrings.get('numpy.core.umath.power'),
          TD(ints),
          TD(inexact, f='pow'),
          TD(O, f='npy_ObjectPower'),
          ),
'absolute' :
    Ufunc(1, 1, None,
          docstrings.get('numpy.core.umath.absolute'),
          TD(nocmplx),
          TD(cmplx, out=('f', 'd', 'g')),
          TD(O, f='PyNumber_Absolute'),
          ),
'negative' :
    Ufunc(1, 1, None,
          docstrings.get('numpy.core.umath.negative'),
          TD(nocmplx),
          TD(cmplx, f='neg'),
          TD(O, f='PyNumber_Negative'),
          ),
'sign' :
    Ufunc(1, 1, None,
          docstrings.get('numpy.core.umath.sign'),
          TD(nobool),
          ),
'greater' :
    Ufunc(2, 1, None,
          docstrings.get('numpy.core.umath.greater'),
          TD(all, out='?'),
          ),
'greater_equal' :
    Ufunc(2, 1, None,
          docstrings.get('numpy.core.umath.greater_equal'),
          TD(all, out='?'),
          ),
'less' :
    Ufunc(2, 1, None,
          docstrings.get('numpy.core.umath.less'),
          TD(all, out='?'),
          ),
'less_equal' :
    Ufunc(2, 1, None,
          docstrings.get('numpy.core.umath.less_equal'),
          TD(all, out='?'),
          ),
'equal' :
    Ufunc(2, 1, None,
          docstrings.get('numpy.core.umath.equal'),
          TD(all, out='?'),
          ),
'not_equal' :
    Ufunc(2, 1, None,
          docstrings.get('numpy.core.umath.not_equal'),
          TD(all, out='?'),
          ),
'logical_and' :
    Ufunc(2, 1, One,
          docstrings.get('numpy.core.umath.logical_and'),
          TD(noobj, out='?'),
          TD(M, f='logical_and'),
          ),
'logical_not' :
    Ufunc(1, 1, None,
          docstrings.get('numpy.core.umath.logical_not'),
          TD(noobj, out='?'),
          TD(M, f='logical_not'),
          ),
'logical_or' :
    Ufunc(2, 1, Zero,
          docstrings.get('numpy.core.umath.logical_or'),
          TD(noobj, out='?'),
          TD(M, f='logical_or'),
          ),
'logical_xor' :
    Ufunc(2, 1, None,
          docstrings.get('numpy.core.umath.logical_xor'),
          TD(noobj, out='?'),
          TD(M, f='logical_xor'),
          ),
'maximum' :
    Ufunc(2, 1, None,
          docstrings.get('numpy.core.umath.maximum'),
          TD(noobj),
          TD(O, f='npy_ObjectMax')
          ),
'minimum' :
    Ufunc(2, 1, None,
          docstrings.get('numpy.core.umath.minimum'),
          TD(noobj),
          TD(O, f='npy_ObjectMin')
          ),
'fmax' :
    Ufunc(2, 1, None,
          "",
          TD(noobj),
          TD(O, f='npy_ObjectMax')
          ),
'fmin' :
    Ufunc(2, 1, None,
          "",
          TD(noobj),
          TD(O, f='npy_ObjectMin')
          ),
'logaddexp' :
    Ufunc(2, 1, None,
          docstrings.get('numpy.core.umath.logaddexp'),
          TD(flts, f="logaddexp")
          ),
'logaddexp2' :
    Ufunc(2, 1, None,
          docstrings.get('numpy.core.umath.logaddexp2'),
          TD(flts, f="logaddexp2")
          ),
'bitwise_and' :
    Ufunc(2, 1, One,
          docstrings.get('numpy.core.umath.bitwise_and'),
          TD(bints),
          TD(O, f='PyNumber_And'),
          ),
'bitwise_or' :
    Ufunc(2, 1, Zero,
          docstrings.get('numpy.core.umath.bitwise_or'),
          TD(bints),
          TD(O, f='PyNumber_Or'),
          ),
'bitwise_xor' :
    Ufunc(2, 1, None,
          docstrings.get('numpy.core.umath.bitwise_xor'),
          TD(bints),
          TD(O, f='PyNumber_Xor'),
          ),
'invert' :
    Ufunc(1, 1, None,
          docstrings.get('numpy.core.umath.invert'),
          TD(bints),
          TD(O, f='PyNumber_Invert'),
          ),
'left_shift' :
    Ufunc(2, 1, None,
          docstrings.get('numpy.core.umath.left_shift'),
          TD(ints),
          TD(O, f='PyNumber_Lshift'),
          ),
'right_shift' :
    Ufunc(2, 1, None,
          docstrings.get('numpy.core.umath.right_shift'),
          TD(ints),
          TD(O, f='PyNumber_Rshift'),
          ),
'degrees' :
    Ufunc(1, 1, None,
          docstrings.get('numpy.core.umath.degrees'),
          TD(fltsM, f='degrees'),
          ),
'rad2deg' :
    Ufunc(1, 1, None,
          docstrings.get('numpy.core.umath.rad2deg'),
          TD(fltsM, f='rad2deg'),
          ),
'radians' :
    Ufunc(1, 1, None,
          docstrings.get('numpy.core.umath.radians'),
          TD(fltsM, f='radians'),
          ),
'deg2rad' :
    Ufunc(1, 1, None,
          docstrings.get('numpy.core.umath.deg2rad'),
          TD(fltsM, f='deg2rad'),
          ),
'arccos' :
    Ufunc(1, 1, None,
          docstrings.get('numpy.core.umath.arccos'),
          TD(inexact, f='acos'),
          TD(M, f='arccos'),
          ),
'arccosh' :
    Ufunc(1, 1, None,
          docstrings.get('numpy.core.umath.arccosh'),
          TD(inexact, f='acosh'),
          TD(M, f='arccosh'),
          ),
'arcsin' :
    Ufunc(1, 1, None,
          docstrings.get('numpy.core.umath.arcsin'),
          TD(inexact, f='asin'),
          TD(M, f='arcsin'),
          ),
'arcsinh' :
    Ufunc(1, 1, None,
          docstrings.get('numpy.core.umath.arcsinh'),
          TD(inexact, f='asinh'),
          TD(M, f='arcsinh'),
          ),
'arctan' :
    Ufunc(1, 1, None,
          docstrings.get('numpy.core.umath.arctan'),
          TD(inexact, f='atan'),
          TD(M, f='arctan'),
          ),
'arctanh' :
    Ufunc(1, 1, None,
          docstrings.get('numpy.core.umath.arctanh'),
          TD(inexact, f='atanh'),
          TD(M, f='arctanh'),
          ),
'cos' :
    Ufunc(1, 1, None,
          docstrings.get('numpy.core.umath.cos'),
          TD(inexact, f='cos'),
          TD(M, f='cos'),
          ),
'sin' :
    Ufunc(1, 1, None,
          docstrings.get('numpy.core.umath.sin'),
          TD(inexact, f='sin'),
          TD(M, f='sin'),
          ),
'tan' :
    Ufunc(1, 1, None,
          docstrings.get('numpy.core.umath.tan'),
          TD(inexact, f='tan'),
          TD(M, f='tan'),
          ),
'cosh' :
    Ufunc(1, 1, None,
          docstrings.get('numpy.core.umath.cosh'),
          TD(inexact, f='cosh'),
          TD(M, f='cosh'),
          ),
'sinh' :
    Ufunc(1, 1, None,
          docstrings.get('numpy.core.umath.sinh'),
          TD(inexact, f='sinh'),
          TD(M, f='sinh'),
          ),
'tanh' :
    Ufunc(1, 1, None,
          docstrings.get('numpy.core.umath.tanh'),
          TD(inexact, f='tanh'),
          TD(M, f='tanh'),
          ),
'exp' :
    Ufunc(1, 1, None,
          docstrings.get('numpy.core.umath.exp'),
          TD(inexact, f='exp'),
          TD(M, f='exp'),
          ),
'exp2' :
    Ufunc(1, 1, None,
          docstrings.get('numpy.core.umath.exp2'),
          TD(inexact, f='exp2'),
          TD(M, f='exp2'),
          ),
'expm1' :
    Ufunc(1, 1, None,
          docstrings.get('numpy.core.umath.expm1'),
          TD(inexact, f='expm1'),
          TD(M, f='expm1'),
          ),
'log' :
    Ufunc(1, 1, None,
          docstrings.get('numpy.core.umath.log'),
          TD(inexact, f='log'),
          TD(M, f='log'),
          ),
'log2' :
    Ufunc(1, 1, None,
          docstrings.get('numpy.core.umath.log2'),
          TD(inexact, f='log2'),
          TD(M, f='log2'),
          ),
'log10' :
    Ufunc(1, 1, None,
          docstrings.get('numpy.core.umath.log10'),
          TD(inexact, f='log10'),
          TD(M, f='log10'),
          ),
'log1p' :
    Ufunc(1, 1, None,
          docstrings.get('numpy.core.umath.log1p'),
          TD(inexact, f='log1p'),
          TD(M, f='log1p'),
          ),
'sqrt' :
    Ufunc(1, 1, None,
          docstrings.get('numpy.core.umath.sqrt'),
          TD(inexact, f='sqrt'),
          TD(M, f='sqrt'),
          ),
'ceil' :
    Ufunc(1, 1, None,
          docstrings.get('numpy.core.umath.ceil'),
          TD(flts, f='ceil'),
          TD(M, f='ceil'),
          ),
'trunc' :
    Ufunc(1, 1, None,
          docstrings.get('numpy.core.umath.trunc'),
          TD(flts, f='trunc'),
          TD(M, f='trunc'),
          ),
'fabs' :
    Ufunc(1, 1, None,
          docstrings.get('numpy.core.umath.fabs'),
          TD(flts, f='fabs'),
          TD(M, f='fabs'),
       ),
'floor' :
    Ufunc(1, 1, None,
          docstrings.get('numpy.core.umath.floor'),
          TD(flts, f='floor'),
          TD(M, f='floor'),
          ),
'rint' :
    Ufunc(1, 1, None,
          docstrings.get('numpy.core.umath.rint'),
          TD(inexact, f='rint'),
          TD(M, f='rint'),
          ),
'arctan2' :
    Ufunc(2, 1, None,
          docstrings.get('numpy.core.umath.arctan2'),
          TD(flts, f='atan2'),
          TD(M, f='arctan2'),
          ),
'remainder' :
    Ufunc(2, 1, None,
          docstrings.get('numpy.core.umath.remainder'),
          TD(intflt),
          TD(O, f='PyNumber_Remainder'),
          ),
'hypot' :
    Ufunc(2, 1, None,
          docstrings.get('numpy.core.umath.hypot'),
          TD(flts, f='hypot'),
          TD(M, f='hypot'),
          ),
'isnan' :
    Ufunc(1, 1, None,
          docstrings.get('numpy.core.umath.isnan'),
          TD(inexact, out='?'),
          ),
'isinf' :
    Ufunc(1, 1, None,
          docstrings.get('numpy.core.umath.isinf'),
          TD(inexact, out='?'),
          ),
'isfinite' :
    Ufunc(1, 1, None,
          docstrings.get('numpy.core.umath.isfinite'),
          TD(inexact, out='?'),
          ),
'signbit' :
    Ufunc(1, 1, None,
          docstrings.get('numpy.core.umath.signbit'),
          TD(flts, out='?'),
          ),
'modf' :
    Ufunc(1, 2, None,
          docstrings.get('numpy.core.umath.modf'),
          TD(flts),
          ),
}

def indent(st,spaces):
    indention = ' '*spaces
    indented = indention + st.replace('\n','\n'+indention)
    # trim off any trailing spaces
    indented = re.sub(r' +$',r'',indented)
    return indented

chartotype1 = {'f': 'f_f',
               'd': 'd_d',
               'g': 'g_g',
               'F': 'F_F',
               'D': 'D_D',
               'G': 'G_G',
               'O': 'O_O',
               'M': 'O_O_method'}

chartotype2 = {'f': 'ff_f',
               'd': 'dd_d',
               'g': 'gg_g',
               'F': 'FF_F',
               'D': 'DD_D',
               'G': 'GG_G',
               'O': 'OO_O',
               'M': 'OO_O_method'}
#for each name
# 1) create functions, data, and signature
# 2) fill in functions and data in InitOperators
# 3) add function.

# String-handling utilities to avoid locale-dependence.

import string
UPPER_TABLE = string.maketrans(string.ascii_lowercase, string.ascii_uppercase)

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
    >>> english_upper('ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789_')
    'ABCDEFGHIJKLMNOPQRSTUVWXYZABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_'
    >>> english_upper('')
    ''
    """
    uppered = s.translate(UPPER_TABLE)
    return uppered


def make_arrays(funcdict):
    # functions array contains an entry for every type implemented
    #   NULL should be placed where PyUfunc_ style function will be filled in later
    #
    code1list = []
    code2list = []
    names = funcdict.keys()
    names.sort()
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
            if t.func_data is not None:
                funclist.append('NULL')
                astr = '%s_functions[%d] = PyUFunc_%s;' % \
                       (name, k, thedict[t.type])
                code2list.append(astr)
                if t.type == 'O':
                    astr = '%s_data[%d] = (void *) %s;' % \
                           (name, k, t.func_data)
                    code2list.append(astr)
                    datalist.append('(void *)NULL')
                elif t.type == 'M':
                    datalist.append('(void *)"%s"' % t.func_data)
                else:
                    astr = '%s_data[%d] = (void *) %s;' % \
                           (name, k, t.func_data)
                    code2list.append(astr)
                    datalist.append('(void *)NULL')
                    #datalist.append('(void *)%s' % t.func_data)
                sub += 1
            else:
                datalist.append('(void *)NULL');
                tname = english_upper(chartoname[t.type])
                funclist.append('%s_%s' % (tname, name))

            for x in t.in_ + t.out:
                siglist.append('PyArray_%s' % (english_upper(chartoname[x]),))

            k += 1

        funcnames = ', '.join(funclist)
        signames = ', '.join(siglist)
        datanames = ', '.join(datalist)
        code1list.append("static PyUFuncGenericFunction %s_functions[] = { %s };" \
                         % (name, funcnames))
        code1list.append("static void * %s_data[] = { %s };" \
                         % (name, datanames))
        code1list.append("static char %s_signatures[] = { %s };" \
                         % (name, signames))
    return "\n".join(code1list),"\n".join(code2list)

def make_ufuncs(funcdict):
    code3list = []
    names = funcdict.keys()
    names.sort()
    for name in names:
        uf = funcdict[name]
        mlist = []
        docstring = textwrap.dedent(uf.docstring).strip()
        docstring = docstring.encode('string-escape').replace(r'"', r'\"')
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
        mlist.append(r"""PyDict_SetItemString(dictionary, "%s", f);""" % name)
        mlist.append(r"""Py_DECREF(f);""")
        code3list.append('\n'.join(mlist))
    return '\n'.join(code3list)


def make_code(funcdict,filename):
    code1, code2 = make_arrays(funcdict)
    code3 = make_ufuncs(funcdict)
    code2 = indent(code2,4)
    code3 = indent(code3,4)
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
    return code;


if __name__ == "__main__":
    filename = __file__
    fid = open('__umath_generated.c','w')
    code = make_code(defdict, filename)
    fid.write(code)
    fid.close()
