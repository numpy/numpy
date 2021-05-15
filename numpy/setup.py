#!/usr/bin/env python3

def configuration(parent_package='', top_path=None):
    from numpy.distutils.misc_util import Configuration
    config = Configuration('numpy', parent_package, top_path)

    for subpackage in ('compat', 'core', 'distutils', 'doc', 'f2py', 'fft', 'lib',
        'linalg', 'ma', 'matrixlib', 'polynomial', 'random', 'testing', 'typing'):
        
    subpackages = ['compat', 'core', 'distutils', 'doc', 'f2py', 'fft', 'lib',
                   'linalg', 'ma', 'matrixlib', 'polynomial', 'random', 'testing', 
                   'typing', 'tests']
    data_dirs = ['doc']
    data_files = ['py.typed', '*.pyi']
    
    map(config.add_subpackage, subpackages)
    map(config.add_data_dir, data_dirs)
    map(config.add_data_files, data_files)
        
    config.make_config_py() # installs __config__.py
    return config

if __name__ == '__main__':
    print('This is the wrong setup.py file to run')
