import os
import sys
import glob
from os.path import join, basename

from numpy.distutils import log

from numscons import get_scons_build_dir

def configuration(parent_package='',top_path=None):
    from numpy.distutils.misc_util import Configuration,dot_join
    from numpy.distutils.command.scons import get_scons_pkg_build_dir
    from numpy.distutils.system_info import get_info, default_lib_dirs

    config = Configuration('core',parent_package,top_path)
    local_dir = config.local_path

    header_dir = 'include/numpy' # this is relative to config.path_in_package

    config.add_subpackage('code_generators')

    # List of files to register to numpy.distutils
    dot_blas_src = [join('blasdot', '_dotblas.c'),
                    join('blasdot', 'cblas.h')]
    api_definition = [join('code_generators', 'numpy_api_order.txt'),
                      join('code_generators', 'ufunc_api_order.txt')]
    core_src = [join('src', basename(i)) for i in glob.glob(join(local_dir,
                                                                'src',
                                                                '*.c'))]
    core_src += [join('src', basename(i)) for i in glob.glob(join(local_dir,
                                                                 'src',
                                                                 '*.src'))]

    source_files = dot_blas_src + api_definition + core_src + \
                   [join(header_dir, 'numpyconfig.h.in')]

    # Add generated files to distutils...
    def add_config_header():
        scons_build_dir = get_scons_build_dir()
        # XXX: I really have to think about how to communicate path info
        # between scons and distutils, and set the options at one single
        # location.
        target = join(get_scons_pkg_build_dir(config.name), 'config.h')
        incl_dir = os.path.dirname(target)
        if incl_dir not in config.numpy_include_dirs:
            config.numpy_include_dirs.append(incl_dir)

    def add_numpyconfig_header():
        scons_build_dir = get_scons_build_dir()
        # XXX: I really have to think about how to communicate path info
        # between scons and distutils, and set the options at one single
        # location.
        target = join(get_scons_pkg_build_dir(config.name),
                      'include/numpy/numpyconfig.h')
        incl_dir = os.path.dirname(target)
        if incl_dir not in config.numpy_include_dirs:
            config.numpy_include_dirs.append(incl_dir)
        config.add_data_files((header_dir, target))

    def add_array_api():
        scons_build_dir = get_scons_build_dir()
        # XXX: I really have to think about how to communicate path info
        # between scons and distutils, and set the options at one single
        # location.
        h_file = join(get_scons_pkg_build_dir(config.name),
                'include/numpy/__multiarray_api.h')
        t_file = join(get_scons_pkg_build_dir(config.name),
                'include/numpy/multiarray_api.txt')
        config.add_data_files((header_dir, h_file),
                              (header_dir, t_file))

    def add_ufunc_api():
        scons_build_dir = get_scons_build_dir()
        # XXX: I really have to think about how to communicate path info
        # between scons and distutils, and set the options at one single
        # location.
        h_file = join(get_scons_pkg_build_dir(config.name),
                'include/numpy/__ufunc_api.h')
        t_file = join(get_scons_pkg_build_dir(config.name),
                'include/numpy/ufunc_api.txt')
        config.add_data_files((header_dir, h_file),
                              (header_dir, t_file))

    def add_generated_files(*args, **kw):
        add_config_header()
        add_numpyconfig_header()
        add_array_api()
        add_ufunc_api()

    config.add_sconscript('SConstruct',
                          post_hook = add_generated_files,
                          source_files = source_files)
    config.add_scons_installed_library('npymath', 'lib')

    config.add_data_files('include/numpy/*.h')
    config.add_include_dirs('src')

    config.numpy_include_dirs.extend(config.paths('include'))

    # Don't install fenv unless we need them.
    if sys.platform == 'cygwin':
        config.add_data_dir('include/numpy/fenv')

    config.add_data_dir('tests')
    config.make_svn_version_py()

    return config

if __name__=='__main__':
    from numpy.distutils.core import setup
    setup(configuration=configuration)
