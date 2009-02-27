import imp
import os
import sys
from os.path import join
from numpy.distutils import log
from distutils.dep_util import newer
from distutils.sysconfig import get_config_var

# XXX: ugly, we use a class to avoid calling twice some expensive functions in
# config.h/numpyconfig.h. I don't see a better way because distutils force
# config.h generation inside an Extension class, and as such sharing
# configuration informations between extensions is not easy.
# Using a pickled-based memoize does not work because config_cmd is an instance
# method, which cPickle does not like.
try:
    import cPickle as _pik
except ImportError:
    import pickle as _pik
import copy

class CallOnceOnly(object):
    def __init__(self):
        self._check_types = None
        self._check_ieee_macros = None

    def check_types(self, *a, **kw):
        if self._check_types is None:
            out = check_types(*a, **kw)
            self._check_types = _pik.dumps(out)
        else:
            out = copy.deepcopy(_pik.loads(self._check_types))
        return out

    def check_ieee_macros(self, *a, **kw):
        if self._check_ieee_macros is None:
            out = check_ieee_macros(*a, **kw)
            self._check_ieee_macros = _pik.dumps(out)
        else:
            out = copy.deepcopy(_pik.loads(self._check_ieee_macros))
        return out

def pythonlib_dir():
    """return path where libpython* is."""
    if sys.platform == 'win32':
        return os.path.join(sys.prefix, "libs")
    else:
        return get_config_var('LIBDIR')

def is_npy_no_signal():
    """Return True if the NPY_NO_SIGNAL symbol must be defined in configuration
    header."""
    return sys.platform == 'win32'

def is_npy_no_smp():
    """Return True if the NPY_NO_SMP symbol must be defined in public
    header (when SMP support cannot be reliably enabled)."""
    # Python 2.3 causes a segfault when
    #  trying to re-acquire the thread-state
    #  which is done in error-handling
    #  ufunc code.  NPY_ALLOW_C_API and friends
    #  cause the segfault. So, we disable threading
    #  for now.
    if sys.version[:5] < '2.4.2':
        nosmp = 1
    else:
        # Perhaps a fancier check is in order here.
        #  so that threads are only enabled if there
        #  are actually multiple CPUS? -- but
        #  threaded code can be nice even on a single
        #  CPU so that long-calculating code doesn't
        #  block.
        try:
            nosmp = os.environ['NPY_NOSMP']
            nosmp = 1
        except KeyError:
            nosmp = 0
    return nosmp == 1

def win32_checks(deflist):
    from numpy.distutils.misc_util import get_build_architecture
    a = get_build_architecture()

    # Distutils hack on AMD64 on windows
    print 'BUILD_ARCHITECTURE: %r, os.name=%r, sys.platform=%r' % \
          (a, os.name, sys.platform)
    if a == 'AMD64':
        deflist.append('DISTUTILS_USE_SDK')

    # On win32, force long double format string to be 'g', not
    # 'Lg', since the MS runtime does not support long double whose
    # size is > sizeof(double)
    if a == "Intel" or a == "AMD64":
        deflist.append('FORCE_NO_LONG_DOUBLE_FORMATTING')

def check_math_capabilities(config, moredefs, mathlibs):
    def check_func(func_name):
        return config.check_func(func_name, libraries=mathlibs,
                                 decl=True, call=True)

    def check_funcs_once(funcs_name):
        decl = dict([(f, True) for f in funcs_name])
        st = config.check_funcs_once(funcs_name, libraries=mathlibs,
                                     decl=decl, call=decl)
        if st:
            moredefs.extend([fname2def(f) for f in funcs_name])
        return st

    def check_funcs(funcs_name):
        # Use check_funcs_once first, and if it does not work, test func per
        # func. Return success only if all the functions are available
        if not check_funcs_once(funcs_name):
            # Global check failed, check func per func
            for f in funcs_name:
                if check_func(f):
                    moredefs.append(fname2def(f))
            return 0
        else:
            return 1

    #use_msvc = config.check_decl("_MSC_VER")

    # Mandatory functions: if not found, fail the build
    mandatory_funcs = ["sin", "cos", "tan", "sinh", "cosh", "tanh", "fabs",
                "floor", "ceil", "sqrt", "log10", "log", "exp", "asin",
                "acos", "atan", "fmod", 'modf', 'frexp', 'ldexp']

    if not check_funcs_once(mandatory_funcs):
        raise SystemError("One of the required function to build numpy is not"
                " available (the list is %s)." % str(mandatory_funcs))

    # Standard functions which may not be available and for which we have a
    # replacement implementation. Note that some of these are C99 functions.
    # XXX: we do not test for hypot because python checks for it (HAVE_HYPOT in
    # python.h... I wish they would clean their public headers someday)
    optional_stdfuncs = ["expm1", "log1p", "acosh", "asinh", "atanh",
                         "rint", "trunc", "exp2", "log2"]

    # XXX: hack to circumvent cpp pollution from python: python put its
    # config.h in the public namespace, so we have a clash for the common
    # functions we test. We remove every function tested by python's autoconf,
    # hoping their own test are correct
    if sys.version_info[0] == 2 and sys.version_info[1] >= 6:
        for f in ["expm1", "log1p", "acosh", "atanh", "asinh"]:
	    if config.check_decl(fname2def(f), headers = ["Python.h", "math.h"]):
                optional_stdfuncs.remove(f)

    check_funcs(optional_stdfuncs)

    # C99 functions: float and long double versions
    c99_funcs = ["sin", "cos", "tan", "sinh", "cosh", "tanh", "fabs", "floor",
                 "ceil", "rint", "trunc", "sqrt", "log10", "log", "log1p", "exp",
                 "expm1", "asin", "acos", "atan", "asinh", "acosh", "atanh",
                 "hypot", "atan2", "pow", "fmod", "modf", 'frexp', 'ldexp',
                 "exp2", "log2"]

    for prec in ['l', 'f']:
        fns = [f + prec for f in c99_funcs]
        check_funcs(fns)

def fname2def(name):
    return "HAVE_%s" % name.upper()

def check_ieee_macros(config):
    priv = []
    pub = []

    # Normally, isnan and isinf are macro (C99), but some platforms only have
    # func, or both func and macro version. Check for macro only, and define
    # replacement ones if not found.
    # Note: including Python.h is necessary because it modifies some math.h
    # definitions
    for f in ["isnan", "isinf", "signbit", "isfinite"]:
        st = config.check_decl(f, headers = ["Python.h", "math.h"])
        if st:
            priv.append(fname2def("decl_%s" % f))
            pub.append('NPY_%s' % fname2def("decl_%s" % f))

    return priv, pub

def check_types(config_cmd, ext, build_dir):
    private_defines = []
    public_defines = []

    # Expected size (in number of bytes) for each type. This is an
    # optimization: those are only hints, and an exhaustive search for the size
    # is done if the hints are wrong.
    expected = {}
    expected['short'] = [2]
    expected['int'] = [4]
    expected['long'] = [8, 4]
    expected['float'] = [4]
    expected['double'] = [8]
    expected['long double'] = [8, 12, 16]
    expected['Py_intptr_t'] = [4, 8]
    expected['PY_LONG_LONG'] = [8]
    expected['long long'] = [8]

    # Check we have the python header (-dev* packages on Linux)
    result = config_cmd.check_header('Python.h')
    if not result:
        raise SystemError(
                "Cannot compiler 'Python.h'. Perhaps you need to "\
                "install python-dev|python-devel.")

    # Check basic types sizes
    for type in ('short', 'int', 'long', 'float', 'double', 'long double'):
        res = config_cmd.check_type_size(type, expected=expected[type])
        if res >= 0:
            if not type == 'long double':
                private_defines.append(('SIZEOF_%s' % sym2def(type), '%d' % res))
            public_defines.append(('NPY_SIZEOF_%s' % sym2def(type), '%d' % res))
        else:
            raise SystemError("Checking sizeof (%s) failed !" % type)

    for type in ('Py_intptr_t',):
        res = config_cmd.check_type_size(type, headers=["Python.h"],
                library_dirs=[pythonlib_dir()],
                expected=expected[type])

        if res >= 0:
            private_defines.append(('SIZEOF_%s' % sym2def(type), '%d' % res))
            public_defines.append(('NPY_SIZEOF_%s' % sym2def(type), '%d' % res))
        else:
            raise SystemError("Checking sizeof (%s) failed !" % type)

    # We check declaration AND type because that's how distutils does it.
    if config_cmd.check_decl('PY_LONG_LONG', headers=['Python.h']):
        res = config_cmd.check_type_size('PY_LONG_LONG',  headers=['Python.h'],
                library_dirs=[pythonlib_dir()],
                expected=expected['PY_LONG_LONG'])
        if res >= 0:
            private_defines.append(('SIZEOF_%s' % sym2def('PY_LONG_LONG'), '%d' % res))
            public_defines.append(('NPY_SIZEOF_%s' % sym2def('PY_LONG_LONG'), '%d' % res))
        else:
            raise SystemError("Checking sizeof (%s) failed !" % 'PY_LONG_LONG')

        res = config_cmd.check_type_size('long long',
                expected=expected['long long'])
        if res >= 0:
            #private_defines.append(('SIZEOF_%s' % sym2def('long long'), '%d' % res))
            public_defines.append(('NPY_SIZEOF_%s' % sym2def('long long'), '%d' % res))
        else:
            raise SystemError("Checking sizeof (%s) failed !" % 'long long')

    if not config_cmd.check_decl('CHAR_BIT', headers=['Python.h']):
        raise RuntimeError(
            "Config wo CHAR_BIT is not supported"\
            ", please contact the maintainers")

    return private_defines, public_defines

def sym2def(symbol):
    define = symbol.replace(' ', '')
    return define.upper()

def check_mathlib(config_cmd):
    # Testing the C math library
    mathlibs = []
    tc = testcode_mathlib()
    mathlibs_choices = [[],['m'],['cpml']]
    mathlib = os.environ.get('MATHLIB')
    if mathlib:
        mathlibs_choices.insert(0,mathlib.split(','))
    for libs in mathlibs_choices:
        if config_cmd.try_run(tc,libraries=libs):
            mathlibs = libs
            break
    else:
        raise EnvironmentError("math library missing; rerun "
                               "setup.py after setting the "
                               "MATHLIB env variable")
    return mathlibs

def configuration(parent_package='',top_path=None):
    from numpy.distutils.misc_util import Configuration,dot_join
    from numpy.distutils.system_info import get_info, default_lib_dirs

    config = Configuration('core',parent_package,top_path)
    local_dir = config.local_path
    codegen_dir = join(local_dir,'code_generators')

    generate_umath_py = join(codegen_dir,'generate_umath.py')
    n = dot_join(config.name,'generate_umath')
    generate_umath = imp.load_module('_'.join(n.split('.')),
                                     open(generate_umath_py,'U'),generate_umath_py,
                                     ('.py','U',1))

    header_dir = 'include/numpy' # this is relative to config.path_in_package

    cocache = CallOnceOnly()

    def generate_config_h(ext, build_dir):
        target = join(build_dir,header_dir,'config.h')
        d = os.path.dirname(target)
        if not os.path.exists(d):
            os.makedirs(d)
        if newer(__file__,target):
            config_cmd = config.get_config_cmd()
            log.info('Generating %s',target)

            # Check sizeof
            moredefs, ignored = cocache.check_types(config_cmd, ext, build_dir)

            # Check math library and C99 math funcs availability
            mathlibs = check_mathlib(config_cmd)
            moredefs.append(('MATHLIB',','.join(mathlibs)))

            check_math_capabilities(config_cmd, moredefs, mathlibs)
            moredefs.extend(cocache.check_ieee_macros(config_cmd)[0])

            # Signal check
            if is_npy_no_signal():
                moredefs.append('__NPY_PRIVATE_NO_SIGNAL')

            # Windows checks
            if sys.platform=='win32' or os.name=='nt':
                win32_checks(moredefs)

            # Generate the config.h file from moredefs
            target_f = open(target,'a')
            for d in moredefs:
                if isinstance(d,str):
                    target_f.write('#define %s\n' % (d))
                else:
                    target_f.write('#define %s %s\n' % (d[0],d[1]))

            target_f.close()
            print 'File:',target
            target_f = open(target)
            print target_f.read()
            target_f.close()
            print 'EOF'
        else:
            mathlibs = []
            target_f = open(target)
            for line in target_f.readlines():
                s = '#define MATHLIB'
                if line.startswith(s):
                    value = line[len(s):].strip()
                    if value:
                        mathlibs.extend(value.split(','))
            target_f.close()

        # Ugly: this can be called within a library and not an extension,
        # in which case there is no libraries attributes (and none is
        # needed).
        if hasattr(ext, 'libraries'):
            ext.libraries.extend(mathlibs)

        incl_dir = os.path.dirname(target)
        if incl_dir not in config.numpy_include_dirs:
            config.numpy_include_dirs.append(incl_dir)

        return target

    def generate_numpyconfig_h(ext, build_dir):
        """Depends on config.h: generate_config_h has to be called before !"""
        target = join(build_dir,header_dir,'numpyconfig.h')
        d = os.path.dirname(target)
        if not os.path.exists(d):
            os.makedirs(d)
        if newer(__file__,target):
            config_cmd = config.get_config_cmd()
            log.info('Generating %s',target)

            # Check sizeof
            ignored, moredefs = cocache.check_types(config_cmd, ext, build_dir)

            if is_npy_no_signal():
                moredefs.append(('NPY_NO_SIGNAL', 1))

            if is_npy_no_smp():
                moredefs.append(('NPY_NO_SMP', 1))
            else:
                moredefs.append(('NPY_NO_SMP', 0))

            moredefs.extend(cocache.check_ieee_macros(config_cmd)[1])

            # Check wether we can use inttypes (C99) formats
            if config_cmd.check_decl('PRIdPTR', headers = ['inttypes.h']):
                moredefs.append(('NPY_USE_C99_FORMATS', 1))
            else:
                moredefs.append(('NPY_USE_C99_FORMATS', 0))

            # Add moredefs to header
            target_f = open(target,'a')
            for d in moredefs:
                if isinstance(d,str):
                    target_f.write('#define %s\n' % (d))
                else:
                    target_f.write('#define %s %s\n' % (d[0],d[1]))

            # Define __STDC_FORMAT_MACROS
            target_f.write("""
#ifndef __STDC_FORMAT_MACROS
#define __STDC_FORMAT_MACROS 1
#endif
""")
            target_f.close()

            # Dump the numpyconfig.h header to stdout
            print 'File: %s' % target
            target_f = open(target)
            print target_f.read()
            target_f.close()
            print 'EOF'
        config.add_data_files((header_dir, target))
        return target

    def generate_api_func(module_name):
        def generate_api(ext, build_dir):
            script = join(codegen_dir, module_name + '.py')
            sys.path.insert(0, codegen_dir)
            try:
                m = __import__(module_name)
                log.info('executing %s', script)
                h_file, c_file, doc_file = m.generate_api(os.path.join(build_dir, header_dir))
            finally:
                del sys.path[0]
            config.add_data_files((header_dir, h_file),
                                  (header_dir, doc_file))
            return (h_file,)
        return generate_api

    generate_numpy_api = generate_api_func('generate_numpy_api')
    generate_ufunc_api = generate_api_func('generate_ufunc_api')

    def generate_umath_c(ext,build_dir):
        target = join(build_dir,header_dir,'__umath_generated.c')
        dir = os.path.dirname(target)
        if not os.path.exists(dir):
            os.makedirs(dir)
        script = generate_umath_py
        if newer(script,target):
            f = open(target,'w')
            f.write(generate_umath.make_code(generate_umath.defdict,
                                             generate_umath.__file__))
            f.close()
        return []

    config.add_data_files('include/numpy/*.h')
    config.add_include_dirs('src')

    config.numpy_include_dirs.extend(config.paths('include'))

    deps = [join('src','arrayobject.c'),
            join('src','arraymethods.c'),
            join('src','scalartypes.inc.src'),
            join('src','numpyos.c'),
            join('src','arraytypes.inc.src'),
            join('src','_signbit.c'),
            join('src','ucsnarrow.c'),
            join('include','numpy','*object.h'),
            'include/numpy/fenv/fenv.c',
            'include/numpy/fenv/fenv.h',
            join(codegen_dir,'genapi.py'),
            join(codegen_dir,'*.txt')
            ]

    # Don't install fenv unless we need them.
    if sys.platform == 'cygwin':
        config.add_data_dir('include/numpy/fenv')

    config.add_extension('_sort',
                         sources=[join('src','_sortmodule.c.src'),
                                  generate_config_h,
                                  generate_numpyconfig_h,
                                  generate_numpy_api,
                                  ],
                         )

    # npymath needs the config.h and numpyconfig.h files to be generated, but
    # build_clib cannot handle generate_config_h and generate_numpyconfig_h
    # (don't ask). Because clib are generated before extensions, we have to
    # explicitly add an extension which has generate_config_h and
    # generate_numpyconfig_h as sources *before* adding npymath.
    config.add_library('npymath', 
            sources=[join('src', 'npy_math.c.src')],
            depends=[])

    config.add_extension('multiarray',
                         sources = [join('src','multiarraymodule.c'),
                                    generate_config_h,
                                    generate_numpyconfig_h,
                                    generate_numpy_api,
                                    join('src','scalartypes.inc.src'),
                                    join('src','arraytypes.inc.src'),
                                    join(codegen_dir,'generate_numpy_api.py'),
                                    join('*.py')
                                    ],
                         depends = deps,
                         libraries=['npymath'],
                         )

    config.add_extension('umath',
                         sources = [generate_config_h,
                                    generate_numpyconfig_h,
                                    join('src','umathmodule.c.src'),
                                    generate_umath_c,
                                    generate_ufunc_api,
                                    join('src','scalartypes.inc.src'),
                                    join('src','arraytypes.inc.src'),
                                    join('src','umath_funcs.inc.src'),
                                    join('src','umath_loops.inc.src'),
                                    ],
                         depends = [join('src','umath_ufunc_object.inc'),
                                    generate_umath_py,
                                    join(codegen_dir,'generate_ufunc_api.py'),
                                    ]+deps,
                         libraries=['npymath'],
                         )

    config.add_extension('scalarmath',
                         sources=[join('src','scalarmathmodule.c.src'),
                                  generate_config_h,
                                  generate_numpyconfig_h,
                                  generate_numpy_api,
                                  generate_ufunc_api],
                         )

    # Configure blasdot
    blas_info = get_info('blas_opt',0)
    #blas_info = {}
    def get_dotblas_sources(ext, build_dir):
        if blas_info:
            if ('NO_ATLAS_INFO',1) in blas_info.get('define_macros',[]):
                return None # dotblas needs ATLAS, Fortran compiled blas will not be sufficient.
            return ext.depends[:1]
        return None # no extension module will be built

    config.add_extension('_dotblas',
                         sources = [get_dotblas_sources],
                         depends=[join('blasdot','_dotblas.c'),
                                  join('blasdot','cblas.h'),
                                  ],
                         include_dirs = ['blasdot'],
                         extra_info = blas_info
                         )

#    config.add_extension('umath_tests',
#                         sources = [join('src','umath_tests.c.src'),
#                                    ],
#                         depends = [join('blasdot','cblas.h'),] + deps,
#                         include_dirs = ['blasdot'],
#                         extra_info = blas_info
#                         )


    config.add_data_dir('tests')
    config.add_data_dir('tests/data')

    config.make_svn_version_py()

    return config

def testcode_mathlib():
    return """\
/* check whether libm is broken */
#include <math.h>
int main(int argc, char *argv[])
{
  return exp(-720.) > 1.0;  /* typically an IEEE denormal */
}
"""

if __name__=='__main__':
    from numpy.distutils.core import setup
    setup(configuration=configuration)
