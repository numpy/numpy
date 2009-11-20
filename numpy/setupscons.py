#!/usr/bin/env python
from os.path import join as pjoin

def configuration(parent_package='', top_path=None):
    from numpy.distutils.misc_util import Configuration
    from numpy.distutils.misc_util import scons_generate_config_py

    pkgname = 'numpy'
    config = Configuration(pkgname, parent_package, top_path,
                           setup_name = 'setupscons.py')
    config.add_subpackage('distutils')
    config.add_subpackage('testing')
    config.add_subpackage('f2py')
    config.add_subpackage('core')
    config.add_subpackage('lib')
    config.add_subpackage('oldnumeric')
    config.add_subpackage('numarray')
    config.add_subpackage('fft')
    config.add_subpackage('linalg')
    config.add_subpackage('random')
    config.add_subpackage('ma')
    config.add_subpackage('matrixlib')
    config.add_subpackage('compat')
    config.add_subpackage('polynomial')
    config.add_subpackage('doc')
    config.add_data_dir('doc')
    config.add_data_dir('tests')

    def add_config(*args, **kw):
        # Generate __config__, handle inplace issues.
        if kw['scons_cmd'].inplace:
            target = pjoin(kw['pkg_name'], '__config__.py')
        else:
            target = pjoin(kw['scons_cmd'].build_lib, kw['pkg_name'],
                           '__config__.py')
        scons_generate_config_py(target)
    config.add_sconscript(None, post_hook = add_config)

    return config

if __name__ == '__main__':
    print 'This is the wrong setup.py file to run'
