import imp
import os
import sys
from os.path import join
from numpy.distutils import log
from distutils.dep_util import newer

def configuration(parent_package='',top_path=None):
    from numpy.distutils.misc_util import Configuration,dot_join
    from numpy.distutils.system_info import get_info, default_lib_dirs

    config = Configuration('core',parent_package,top_path)
    local_dir = config.local_path

    header_dir = 'include/numpy' # this is relative to config.path_in_package

    # Add generated files to distutils...
    def add_config_header():
        scons_build_dir = config.get_scons_build_dir()
        # XXX: I really have to think about how to communicate path info
        # between scons and distutils, and set the options at one single
        # location.
        target = join(scons_build_dir, local_dir, 'config.h')
        incl_dir = os.path.dirname(target)
        if incl_dir not in config.numpy_include_dirs:
            config.numpy_include_dirs.append(incl_dir)
        config.add_data_files((header_dir, target)) 

    def add_array_api():
        scons_build_dir = config.get_scons_build_dir()
        # XXX: I really have to think about how to communicate path info
        # between scons and distutils, and set the options at one single
        # location.
        h_file = join(scons_build_dir, local_dir, '__multiarray_api.h')
        t_file = join(scons_build_dir, local_dir, '__multiarray_api.txt')
        config.add_data_files((header_dir, h_file),
                              (header_dir, t_file))

    def add_ufunc_api():
        scons_build_dir = config.get_scons_build_dir()
        # XXX: I really have to think about how to communicate path info
        # between scons and distutils, and set the options at one single
        # location.
        h_file = join(scons_build_dir, local_dir, '__ufunc_api.h')
        t_file = join(scons_build_dir, local_dir, '__ufunc_api.txt')
        config.add_data_files((header_dir, h_file),
                              (header_dir, t_file))

    def add_generated_files():
        add_config_header()
        add_array_api()
        add_ufunc_api()

    config.add_sconscript('SConstruct', post_hook = add_generated_files)

    config.add_data_files('include/numpy/*.h')
    config.add_include_dirs('src')

    config.numpy_include_dirs.extend(config.paths('include'))

    # Don't install fenv unless we need them.
    if sys.platform == 'cygwin':
        config.add_data_dir('include/numpy/fenv')

    # # Configure blasdot
    # blas_info = get_info('blas_opt',0)
    # #blas_info = {}
    # def get_dotblas_sources(ext, build_dir):
    #     if blas_info:
    #         if ('NO_ATLAS_INFO',1) in blas_info.get('define_macros',[]):
    #             return None # dotblas needs ATLAS, Fortran compiled blas will not be sufficient.
    #         return ext.depends[:1]
    #     return None # no extension module will be built

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

if __name__=='__main__':
    from numpy.distutils.core import setup
    setup(configuration=configuration)
