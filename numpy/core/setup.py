import imp
import os
import sys
from os.path import join
from numpy.distutils import log
from distutils.dep_util import newer

FUNCTIONS_TO_CHECK = [
    ('expl', 'HAVE_LONGDOUBLE_FUNCS'),
    ('expf', 'HAVE_FLOAT_FUNCS'),
    ('log1p', 'HAVE_LOG1P'),
    ('expm1', 'HAVE_EXPM1'),
    ('asinh', 'HAVE_INVERSE_HYPERBOLIC'),
    ('atanhf', 'HAVE_INVERSE_HYPERBOLIC_FLOAT'),
    ('atanhl', 'HAVE_INVERSE_HYPERBOLIC_LONGDOUBLE'),
    ('isnan', 'HAVE_ISNAN'),
    ('isinf', 'HAVE_ISINF'),
    ('rint', 'HAVE_RINT'),
    ]

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

    def generate_config_h(ext, build_dir):
        target = join(build_dir,'config.h')
        if newer(__file__,target):
            config_cmd = config.get_config_cmd()
            log.info('Generating %s',target)
            tc = generate_testcode(target)
            from distutils import sysconfig
            python_include = sysconfig.get_python_inc()
            python_h = join(python_include, 'Python.h')
            if not os.path.isfile(python_h):
                raise SystemError,\
                      "Non-existing %s. Perhaps you need to install"\
                      " python-dev|python-devel." % (python_h)
            result = config_cmd.try_run(tc,include_dirs=[python_include],
                                        library_dirs = default_lib_dirs)
            if not result:
                raise SystemError,"Failed to test configuration. "\
                      "See previous error messages for more information."

            moredefs = []
            #
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
            ext.libraries.extend(mathlibs)
            moredefs.append(('MATHLIB',','.join(mathlibs)))

            def check_func(func_name):
                return config_cmd.check_func(func_name,
                                             libraries=mathlibs, decl=False,
                                             headers=['math.h'])

            for func_name, defsymbol in FUNCTIONS_TO_CHECK:
                if check_func(func_name):
                    moredefs.append(defsymbol)

            if is_npy_no_signal():
                moredefs.append('__NPY_PRIVATE_NO_SIGNAL')

            if sys.platform=='win32' or os.name=='nt':
                from distutils.msvccompiler import get_build_architecture
                a = get_build_architecture()
                print 'BUILD_ARCHITECTURE: %r, os.name=%r, sys.platform=%r' % (a, os.name, sys.platform)
                if a == 'AMD64':
                    moredefs.append('DISTUTILS_USE_SDK')

            if sys.version[:3] < '2.4':
                if config_cmd.check_func('strtod', decl=False,
                                         headers=['stdlib.h']):
                    moredefs.append(('PyOS_ascii_strtod', 'strtod'))

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

        ext.libraries.extend(mathlibs)

        incl_dir = os.path.dirname(target)
        if incl_dir not in config.numpy_include_dirs:
            config.numpy_include_dirs.append(incl_dir)

        config.add_data_files((header_dir,target))
        return target

    def generate_numpyconfig_h(ext, build_dir):
        """Depends on config.h: generate_config_h has to be called before !"""
        target = join(build_dir,'numpyconfig.h')
        if newer(__file__,target):
            config_cmd = config.get_config_cmd()
            log.info('Generating %s',target)
            testcode = generate_numpyconfig_code(target)

            from distutils import sysconfig
            python_include = sysconfig.get_python_inc()
            python_h = join(python_include, 'Python.h')
            if not os.path.isfile(python_h):
                raise SystemError,\
                      "Non-existing %s. Perhaps you need to install"\
                      " python-dev|python-devel." % (python_h)

            config.numpy_include_dirs
            result = config_cmd.try_run(testcode,
                                include_dirs = [python_include] + \
                                                       config.numpy_include_dirs,
                                        library_dirs = default_lib_dirs)

            if not result:
                raise SystemError,"Failed to generate numpy configuration. "\
                      "See previous error messages for more information."

            print 'File: %s' % target
            target_f = open(target)
            print target_f.read()
            target_f.close()
            print 'EOF'
        return target

    def generate_api_func(module_name):
        def generate_api(ext, build_dir):
            script = join(codegen_dir, module_name + '.py')
            sys.path.insert(0, codegen_dir)
            try:
                m = __import__(module_name)
                log.info('executing %s', script)
                h_file, c_file, doc_file = m.generate_api(build_dir)
            finally:
                del sys.path[0]
            config.add_data_files((header_dir, h_file),
                                  (header_dir, doc_file))
            return (h_file,)
        return generate_api

    generate_array_api = generate_api_func('generate_array_api')
    generate_ufunc_api = generate_api_func('generate_ufunc_api')

    def generate_umath_c(ext,build_dir):
        target = join(build_dir,'__umath_generated.c')
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
            join('src','arraytypes.inc.src'),
            join('src','_signbit.c'),
            join('src','_isnan.c'),
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

    config.add_extension('multiarray',
                         sources = [join('src','multiarraymodule.c'),
                                    generate_config_h,
                                    generate_numpyconfig_h,
                                    generate_array_api,
                                    join('src','scalartypes.inc.src'),
                                    join('src','arraytypes.inc.src'),
                                    join(codegen_dir,'generate_array_api.py'),
                                    join('*.py')
                                    ],
                         depends = deps,
                         )

    config.add_extension('umath',
                         sources = [generate_config_h,
                                    generate_numpyconfig_h,
                                    join('src','umathmodule.c.src'),
                                    generate_umath_c,
                                    generate_ufunc_api,
                                    join('src','scalartypes.inc.src'),
                                    join('src','arraytypes.inc.src'),
                                    ],
                         depends = [join('src','ufuncobject.c'),
                                    generate_umath_py,
                                    join(codegen_dir,'generate_ufunc_api.py'),
                                    ]+deps,
                         )

    config.add_extension('_sort',
                         sources=[join('src','_sortmodule.c.src'),
                                  generate_config_h,
                                  generate_numpyconfig_h,
                                  generate_array_api,
                                  ],
                         )

    config.add_extension('scalarmath',
                         sources=[join('src','scalarmathmodule.c.src'),
                                  generate_config_h,
                                  generate_numpyconfig_h,
                                  generate_array_api,
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


    config.add_data_dir('tests')
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

import sys
def generate_testcode(target):
    if sys.platform == 'win32':
        target = target.replace('\\','\\\\')
    testcode = [r'''
#include <Python.h>
#include <limits.h>
#include <stdio.h>

int main(int argc, char **argv)
{

        FILE *fp;

        fp = fopen("'''+target+'''","w");
        ''']

    c_size_test = r'''
#ifndef %(sz)s
          fprintf(fp,"#define %(sz)s %%d\n", sizeof(%(type)s));
#else
          fprintf(fp,"/* #define %(sz)s %%d */\n", %(sz)s);
#endif
'''
    for sz, t in [('SIZEOF_SHORT', 'short'),
                  ('SIZEOF_INT', 'int'),
                  ('SIZEOF_LONG', 'long'),
                  ('SIZEOF_FLOAT', 'float'),
                  ('SIZEOF_DOUBLE', 'double'),
                  ('SIZEOF_LONG_DOUBLE', 'long double'),
                  ('SIZEOF_PY_INTPTR_T', 'Py_intptr_t'),
                  ]:
        testcode.append(c_size_test % {'sz' : sz, 'type' : t})

    testcode.append('#ifdef PY_LONG_LONG')
    testcode.append(c_size_test % {'sz' : 'SIZEOF_LONG_LONG',
                                   'type' : 'PY_LONG_LONG'})
    testcode.append(c_size_test % {'sz' : 'SIZEOF_PY_LONG_LONG',
                                   'type' : 'PY_LONG_LONG'})


    testcode.append(r'''
#else
        fprintf(fp, "/* PY_LONG_LONG not defined */\n");
#endif
#ifndef CHAR_BIT
          {
             unsigned char var = 2;
             int i=0;
             while (var >= 2) {
                     var = var << 1;
                     i++;
             }
             fprintf(fp,"#define CHAR_BIT %d\n", i+1);
          }
#else
          fprintf(fp, "/* #define CHAR_BIT %d */\n", CHAR_BIT);
#endif
          fclose(fp);
          return 0;
}
''')
    testcode = '\n'.join(testcode)
    return testcode

def generate_numpyconfig_code(target):
    """Return the source code as a string of the code to generate the
    numpyconfig header file."""
    if sys.platform == 'win32':
        target = target.replace('\\','\\\\')
    # Config symbols to prepend
    prepends = [('NPY_SIZEOF_SHORT', 'SIZEOF_SHORT'),
            ('NPY_SIZEOF_INT', 'SIZEOF_INT'),
            ('NPY_SIZEOF_LONG', 'SIZEOF_LONG'),
            ('NPY_SIZEOF_FLOAT', 'SIZEOF_FLOAT'),
            ('NPY_SIZEOF_DOUBLE', 'SIZEOF_DOUBLE'),
            ('NPY_SIZEOF_LONGDOUBLE', 'SIZEOF_LONG_DOUBLE'),
            ('NPY_SIZEOF_PY_INTPTR_T', 'SIZEOF_PY_INTPTR_T')]

    testcode = ["""
#include <Python.h>
#include "config.h"

int main()
{
    FILE* f;

    f = fopen("%s", "w");
    if (f == NULL) {
        return -1;
    }
""" % target]

    testcode.append(r"""
    fprintf(f, "/*\n * This file is generated by %s. DO NOT EDIT \n */\n");
""" % __file__)

    # Prepend NPY_ to any SIZEOF defines
    testcode.extend([r'    fprintf(f, "#define ' + i + r' %%d \n", %s);' % j for i, j in prepends])

    # Conditionally define NPY_NO_SIGNAL
    if is_npy_no_signal():
        testcode.append(r'    fprintf(f, "\n#define NPY_NO_SIGNAL\n");')

    # Define NPY_NOSMP to 1 if explicitely requested, or if we cannot
    # support thread support reliably
    if is_npy_no_smp():
        testcode.append(r'    fprintf(f, "#define NPY_NO_SMP 1\n");')
    else:
        testcode.append(r'    fprintf(f, "#define NPY_NO_SMP 0\n");')

    tmpcode = r"""
    #ifdef PY_LONG_LONG
        fprintf(f, "\n#define %s %%d \n", %s);
        fprintf(f, "#define %s %%d \n", %s);
    #else
        fprintf(f, "/* PY_LONG_LONG not defined  */ \n");
    #endif"""
    testcode.append(tmpcode % ('NPY_SIZEOF_LONGLONG', 'SIZEOF_LONG_LONG',
                               'NPY_SIZEOF_PY_LONG_LONG', 'SIZEOF_PY_LONG_LONG'))

    testcode.append(r"""
#ifndef CHAR_BIT
          {
             unsigned char var = 2;
             int i = 0;
             while (var >= 2) {
                     var = var << 1;
                     i++;
             }
             fprintf(f,"#define CHAR_BIT %d\n", i+1);
          }
#else
          fprintf(f, "/* #define CHAR_BIT %d */\n", CHAR_BIT);
#endif""")

    testcode.append("""
    fclose(f);

    return 0;
}
""")
    return "\n".join(testcode)

if __name__=='__main__':
    from numpy.distutils.core import setup
    setup(configuration=configuration)
