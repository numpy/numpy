# Last Change: Tue Aug 05 12:00 PM 2008 J
# vim:syntax=python
import os
import sys
from os.path import join as pjoin, basename as pbasename, dirname as pdirname
from copy import deepcopy

from numscons import get_pythonlib_dir
from numscons import GetNumpyEnvironment
from numscons import CheckCBLAS
from numscons import write_info

from scons_support import CheckBrokenMathlib, define_no_smp, \
    check_mlib, check_mlibs, is_npy_no_signal
from scons_support import array_api_gen_bld, ufunc_api_gen_bld, template_bld, \
                          umath_bld


env = GetNumpyEnvironment(ARGUMENTS)
env.Append(CPPPATH = env["PYEXTCPPPATH"])
if os.name == 'nt':
    # NT needs the pythonlib to run any code importing Python.h, including
    # simple code using only typedef and so on, so we need it for configuration
    # checks
    env.AppendUnique(LIBPATH = [get_pythonlib_dir()])

#=======================
# Starting Configuration
#=======================
config = env.NumpyConfigure(custom_tests = {'CheckBrokenMathlib' : CheckBrokenMathlib,
    'CheckCBLAS' : CheckCBLAS}, config_h = pjoin('config.h'))

# numpyconfig_sym will keep the values of some configuration variables, the one
# needed for the public numpy API.

# Convention: list of tuples (definition, value). value:
# - 0: #undef definition
# - 1: #define definition
# - string: #define definition value
numpyconfig_sym = []

#---------------
# Checking Types
#---------------
if not config.CheckHeader("Python.h"):
    errmsg = []
    for line in config.GetLastError():
        errmsg.append("%s " % line)
    print """
Error: Python.h header cannot be compiled (or cannot be found).
On linux, check that you have python-dev/python-devel packages. On windows,
check that you have he platform SDK. You may also use unsupported cflags.
Configuration error log says: \n\n%s""" % ''.join(errmsg)
    Exit(-1)

def check_type(type, include = None):
    st = config.CheckTypeSize(type, includes = include)
    type = type.replace(' ', '_')
    if st:
        numpyconfig_sym.append(('SIZEOF_%s' % type.upper(), '%d' % st))
    else:
        numpyconfig_sym.append(('SIZEOF_%s' % type.upper(), 0))

for type in ('short', 'int', 'long', 'float', 'double', 'long double'):
    check_type(type)

for type in ('Py_intptr_t',):
    check_type(type, include = "#include <Python.h>\n")

# We check declaration AND type because that's how distutils does it.
if config.CheckDeclaration('PY_LONG_LONG', includes = '#include <Python.h>\n'):
    st = config.CheckTypeSize('PY_LONG_LONG',
                              includes = '#include <Python.h>\n')
    assert not st == 0
    numpyconfig_sym.append(('DEFINE_NPY_SIZEOF_LONGLONG',
                            '#define NPY_SIZEOF_LONGLONG %d' % st))
    numpyconfig_sym.append(('DEFINE_NPY_SIZEOF_PY_LONG_LONG',
                            '#define NPY_SIZEOF_PY_LONG_LONG %d' % st))
else:
    numpyconfig_sym.append(('DEFINE_NPY_SIZEOF_LONGLONG', ''))
    numpyconfig_sym.append(('DEFINE_NPY_SIZEOF_PY_LONG_LONG', ''))

if not config.CheckDeclaration('CHAR_BIT', includes= '#include <Python.h>\n'):
    raise RuntimeError(\
"""Config wo CHAR_BIT is not supported with scons: please contact the
maintainer (cdavid)""")

#----------------------
# Checking signal stuff
#----------------------
if is_npy_no_signal():
    numpyconfig_sym.append(('DEFINE_NPY_NO_SIGNAL', '#define NPY_NO_SIGNAL\n'))
    config.Define('__NPY_PRIVATE_NO_SIGNAL',
                  comment = "define to 1 to disable SMP support ")
else:
    numpyconfig_sym.append(('DEFINE_NPY_NO_SIGNAL', ''))

#---------------------
# Checking SMP option
#---------------------
if define_no_smp():
    nosmp = 1
else:
    nosmp = 0
numpyconfig_sym.append(('NPY_NO_SMP', nosmp))

#----------------------------------------------
# Check whether we can use C99 printing formats
#----------------------------------------------
if config.CheckDeclaration(('PRIdPTR'), includes  = '#include <inttypes.h>'):
    usec99 = 1
else:
    usec99 = 0
numpyconfig_sym.append(('USE_C99_FORMATS', usec99))
    
#----------------------
# Checking the mathlib
#----------------------
mlibs = [[], ['m'], ['cpml']]
mathlib = os.environ.get('MATHLIB')
if mathlib:
    mlibs.insert(0, mathlib)

mlib = check_mlibs(config, mlibs)

# XXX: this is ugly: mathlib has nothing to do in a public header file
numpyconfig_sym.append(('MATHLIB', ','.join(mlib)))

#----------------------------------
# Checking the math funcs available
#----------------------------------
# Function to check:
mfuncs = ('expl', 'expf', 'log1p', 'expm1', 'asinh', 'atanhf', 'atanhl',
          'isnan', 'isinf', 'rint')

# Set value to 1 for each defined function (in math lib)
mfuncs_defined = dict([(f, 0) for f in mfuncs])

# TODO: checklib vs checkfunc ?
def check_func(f):
    """Check that f is available in mlib, and add the symbol appropriately.  """
    st = config.CheckDeclaration(f, language = 'C', includes = "#include <math.h>")
    if st:
        st = config.CheckFunc(f, language = 'C')
    if st:
        mfuncs_defined[f] = 1
    else:
        mfuncs_defined[f] = 0

for f in mfuncs:
    check_func(f)

if mfuncs_defined['expl'] == 1:
    config.Define('HAVE_LONGDOUBLE_FUNCS',
                  comment = 'Define to 1 if long double funcs are available')
if mfuncs_defined['expf'] == 1:
    config.Define('HAVE_FLOAT_FUNCS',
                  comment = 'Define to 1 if long double funcs are available')
if mfuncs_defined['asinh'] == 1:
    config.Define('HAVE_INVERSE_HYPERBOLIC',
                  comment = 'Define to 1 if inverse hyperbolic funcs are '\
                            'available')
if mfuncs_defined['atanhf'] == 1:
    config.Define('HAVE_INVERSE_HYPERBOLIC_FLOAT',
                  comment = 'Define to 1 if inverse hyperbolic float funcs '\
                            'are available')
if mfuncs_defined['atanhl'] == 1:
    config.Define('HAVE_INVERSE_HYPERBOLIC_LONGDOUBLE',
                  comment = 'Define to 1 if inverse hyperbolic long double '\
                            'funcs are available')

#-------------------------------------------------------
# Define the function PyOS_ascii_strod if not available
#-------------------------------------------------------
if not config.CheckDeclaration('PyOS_ascii_strtod',
                               includes = "#include <Python.h>"):
    if config.CheckFunc('strtod'):
        config.Define('PyOS_ascii_strtod', 'strtod',
                      "Define to a function to use as a replacement for "\
                      "PyOS_ascii_strtod if not available in python header")

#------------------------------------
# DISTUTILS Hack on AMD64 on windows
#------------------------------------
# XXX: this is ugly
if sys.platform=='win32' or os.name=='nt':
    from distutils.msvccompiler import get_build_architecture
    a = get_build_architecture()
    print 'BUILD_ARCHITECTURE: %r, os.name=%r, sys.platform=%r' % \
          (a, os.name, sys.platform)
    if a == 'AMD64':
        distutils_use_sdk = 1
        config.Define('DISTUTILS_USE_SDK', distutils_use_sdk,
                      "define to 1 to disable SMP support ")

#--------------
# Checking Blas
#--------------
if config.CheckCBLAS():
    build_blasdot = 1
else:
    build_blasdot = 0

config.Finish()
write_info(env)

#==========
#  Build
#==========

#---------------------------------------
# Generate the public configuration file
#---------------------------------------
config_dict = {}
# XXX: this is ugly, make the API for config.h and numpyconfig.h similar
for key, value in numpyconfig_sym:
    config_dict['@%s@' % key] = str(value)
env['SUBST_DICT'] = config_dict

include_dir = 'include/numpy'
env.SubstInFile(pjoin(include_dir, 'numpyconfig.h'), pjoin(include_dir, 'numpyconfig.h.in'))

env['CONFIG_H_GEN'] = numpyconfig_sym

#---------------------------
# Builder for generated code
#---------------------------
env.Append(BUILDERS = {'GenerateMultiarrayApi' : array_api_gen_bld,
                       'GenerateUfuncApi' : ufunc_api_gen_bld,
                       'GenerateFromTemplate' : template_bld,
                       'GenerateUmath' : umath_bld})

#------------------------
# Generate generated code
#------------------------
scalartypes_src = env.GenerateFromTemplate(pjoin('src', 'scalartypes.inc.src'))
arraytypes_src = env.GenerateFromTemplate(pjoin('src', 'arraytypes.inc.src'))
sortmodule_src = env.GenerateFromTemplate(pjoin('src', '_sortmodule.c.src'))
umathmodule_src = env.GenerateFromTemplate(pjoin('src', 'umathmodule.c.src'))
scalarmathmodule_src = env.GenerateFromTemplate(
                            pjoin('src', 'scalarmathmodule.c.src'))

umath = env.GenerateUmath('__umath_generated',
                          pjoin('code_generators', 'generate_umath.py'))

multiarray_api = env.GenerateMultiarrayApi('multiarray_api',
                        [ pjoin('code_generators', 'numpy_api_order.txt')])

ufunc_api = env.GenerateUfuncApi('ufunc_api',
                    pjoin('code_generators', 'ufunc_api_order.txt'))

env.Prepend(CPPPATH = ['include', '.'])

#-----------------
# Build multiarray
#-----------------
multiarray_src = [pjoin('src', 'multiarraymodule.c')]
multiarray = env.DistutilsPythonExtension('multiarray', source = multiarray_src)

#------------------
# Build sort module
#------------------
sort = env.DistutilsPythonExtension('_sort', source = sortmodule_src)

#-------------------
# Build umath module
#-------------------
umathmodule = env.DistutilsPythonExtension('umath', source = umathmodule_src)

#------------------------
# Build scalarmath module
#------------------------
scalarmathmodule = env.DistutilsPythonExtension('scalarmath',
                                            source = scalarmathmodule_src)

#----------------------
# Build _dotblas module
#----------------------
if build_blasdot:
    dotblas_src = [pjoin('blasdot', i) for i in ['_dotblas.c']]
    # because _dotblas does #include CBLAS_HEADER instead of #include
    # "cblas.h", scons does not detect the dependency
    # XXX: PythonExtension builder does not take the Depends on extension into
    # account for some reason, so we first build the object, with forced
    # dependency, and then builds the extension. This is more likely a bug in
    # our PythonExtension builder, but I cannot see how to solve it.
    dotblas_o = env.PythonObject('_dotblas', source = dotblas_src)
    env.Depends(dotblas_o, pjoin("blasdot", "cblas.h"))
    dotblas = env.DistutilsPythonExtension('_dotblas', dotblas_o)
