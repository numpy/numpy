
import types
from distutils.core import *
try:
    from setuptools import setup as old_setup
    have_setuptools = 1
except ImportError:
    from distutils.core import setup as old_setup
    have_setuptools = 0

from scipy.distutils.extension import Extension
from scipy.distutils.command import config
from scipy.distutils.command import build
from scipy.distutils.command import build_py
from scipy.distutils.command import config_compiler
from scipy.distutils.command import build_ext
from scipy.distutils.command import build_clib
from scipy.distutils.command import build_src
from scipy.distutils.command import build_scripts
from scipy.distutils.command import sdist
from scipy.distutils.command import install_data
from scipy.distutils.command import install_headers
from scipy.distutils.command import install
from scipy.distutils.command import bdist_rpm
from scipy.distutils.misc_util import get_data_files

scipy_cmdclass = {'build':            build.build,
                  'build_src':        build_src.build_src,
                  'build_scripts':    build_scripts.build_scripts,
                  'config_fc':        config_compiler.config_fc,
                  'config':           config.config,
                  'build_ext':        build_ext.build_ext,
                  'build_py':         build_py.build_py,
                  'build_clib':       build_clib.build_clib,
                  'sdist':            sdist.sdist,
                  'install_data':     install_data.install_data,
                  'install_headers':  install_headers.install_headers,
                  'install':          install.install,
                  'bdist_rpm':        bdist_rpm.bdist_rpm,
                  }
if have_setuptools:
    from setuptools.command import bdist_egg, develop, easy_install, egg_info
    scipy_cmdclass['bdist_egg'] = bdist_egg.bdist_egg
    scipy_cmdclass['develop'] = develop.develop
    scipy_cmdclass['easy_install'] = easy_install.easy_install
    scipy_cmdclass['egg_info'] = egg_info.egg_info

def setup(**attr):

    cmdclass = scipy_cmdclass.copy()

    new_attr = attr.copy()
    if new_attr.has_key('cmdclass'):
        cmdclass.update(new_attr['cmdclass'])
    new_attr['cmdclass'] = cmdclass

    # Move extension source libraries to libraries
    libraries = []
    for ext in new_attr.get('ext_modules',[]):
        new_libraries = []
        for item in ext.libraries:
            if type(item) is type(()):
                lib_name,build_info = item
                _check_append_ext_library(libraries, item)
                new_libraries.append(lib_name)
            else:
                assert type(item) is type(''),`item`
                new_libraries.append(item)
        ext.libraries = new_libraries
    if libraries:
        if not new_attr.has_key('libraries'):
            new_attr['libraries'] = []
        for item in libraries:
            _check_append_library(new_attr['libraries'], item)

    # sources in ext_modules or libraries may contain header files
    if (new_attr.has_key('ext_modules') or new_attr.has_key('libraries')) \
       and not new_attr.has_key('headers'):
        new_attr['headers'] = []

    # Expand directories in data_files to files
    if new_attr.has_key('data_files'):
        new_data_files = []
        for data in new_attr['data_files']:
            if type(data) is types.StringType:
                new_data_files.append(get_data_files(data)[0])
            else:
                new_data_files.append((data[0],get_data_files(data)))
        new_attr['data_files'] = new_data_files

    return old_setup(**new_attr)

def _check_append_library(libraries, item):
    import warnings
    for libitem in libraries:
        if type(libitem) is type(()):
            if type(item) is type(()):
                if item[0]==libitem[0]:
                    if item[1] is libitem[1]:
                        return
                    warnings.warn("[0] libraries list contains '%s' with"\
                                  " different build_info" % (item[0]))
                    break
            else:
                if item==libitem[0]:
                    warnings.warn("[1] libraries list contains '%s' with"\
                                  " no build_info" % (item[0]))
                    break
        else:
            if type(item) is type(()):
                if item[0]==libitem:
                    warnings.warn("[2] libraries list contains '%s' with"\
                                  " no build_info" % (item[0]))
                    break
            else:
                if item==libitem:
                    return
    libraries.append(item)
    return

def _check_append_ext_library(libraries, (lib_name,build_info)):
    import warnings
    for item in libraries:
        if type(item) is type(()):
            if item[0]==lib_name:
                if item[1] is build_info:
                    return
                warnings.warn("[3] libraries list contains '%s' with"\
                              " different build_info" % (lib_name))
                break
        elif item==lib_name:
            warnings.warn("[4] libraries list contains '%s' with"\
                          " no build_info" % (lib_name))
            break
    libraries.append((lib_name,build_info))
    return
