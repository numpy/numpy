
from distutils.core import *
from distutils.core import setup as old_setup

from scipy_distutils.dist import Distribution
from scipy_distutils.extension import Extension
from scipy_distutils.command import build
from scipy_distutils.command import build_py
from scipy_distutils.command import config_compiler
from scipy_distutils.command import build_ext
from scipy_distutils.command import build_clib
from scipy_distutils.command import build_src
from scipy_distutils.command import sdist
from scipy_distutils.command import install_data
from scipy_distutils.command import install
from scipy_distutils.command import install_headers
from scipy_distutils.command import bdist_rpm

def setup(**attr):

    distclass = Distribution
    cmdclass = {'build':            build.build,
                'build_src':        build_src.build_src,
                'config_fc':        config_compiler.config_fc,
                'build_ext':        build_ext.build_ext,
                'build_py':         build_py.build_py,                
                'build_clib':       build_clib.build_clib,
                'sdist':            sdist.sdist,
                'install_data':     install_data.install_data,
                'install':          install.install,
                'install_headers':  install_headers.install_headers,
                'bdist_rpm':        bdist_rpm.bdist_rpm,
                }
    new_attr = attr.copy()
    if new_attr.has_key('cmdclass'):
        cmdclass.update(new_attr['cmdclass'])        
    new_attr['cmdclass'] = cmdclass
    
    if not new_attr.has_key('distclass'):
        new_attr['distclass'] = distclass    

    fortran_libraries = new_attr.get('fortran_libraries',None)

    if fortran_libraries is not None:
        print 64*'*'+"""
    Using fortran_libraries setup option is depreciated
    ---------------------------------------------------
    Use libraries option instead. Yes, scipy_distutils
    now supports Fortran sources in libraries.
"""+64*'*'
        new_attr['libraries'].extend(fortran_libraries)
        del new_attr['fortran_libraries']

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
