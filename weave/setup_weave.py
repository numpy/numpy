#!/usr/bin/env python

import os
from glob import glob
from scipy_distutils.misc_util import get_path, default_config_dict, dot_join

def configuration(parent_package='',parent_path=None):
    parent_path2 = parent_path
    parent_path = parent_package
    local_path = get_path(__name__,parent_path2)
    config = default_config_dict('weave',parent_package)
    config['packages'].append(dot_join(parent_package,'weave.tests'))
    test_path = os.path.join(local_path,'tests')
    config['package_dir']['weave.tests'] = test_path
    
    scxx_files = glob(os.path.join(local_path,'scxx','*.*'))
    install_path = os.path.join(parent_path,'weave','scxx')
    config['data_files'].extend( [(install_path,scxx_files)])
    
    blitz_files = glob(os.path.join(local_path,'blitz-20001213','blitz','*.*'))
    install_path = os.path.join(parent_path,'weave','blitz-20001213',
                                'blitz')
    config['data_files'].extend( [(install_path,blitz_files)])
    
    array_files = glob(os.path.join(local_path,'blitz-20001213','blitz',
                                    'array','*.*'))
    install_path = os.path.join(parent_path,'weave','blitz-20001213',
                                'blitz','array')
    config['data_files'].extend( [(install_path,array_files)])
    
    meta_files = glob(os.path.join(local_path,'blitz-20001213','blitz',
                                    'meta','*.*'))
    install_path = os.path.join(parent_path,'weave','blitz-20001213',
                                'blitz','meta')
    config['data_files'].extend( [(install_path,meta_files)])

    swig_files = glob(os.path.join(local_path,'swig','*.c'))
    install_path = os.path.join(parent_path,'weave','swig')
    config['data_files'].extend( [(install_path,swig_files)])

    doc_files = glob(os.path.join(local_path,'doc','*.html'))
    install_path = os.path.join(parent_path,'weave','doc')
    config['data_files'].extend( [(install_path,doc_files)])

    example_files = glob(os.path.join(local_path,'examples','*.py'))
    install_path = os.path.join(parent_path,'weave','examples')
    config['data_files'].extend( [(install_path,example_files)])
    
    return config

if __name__ == '__main__':    
    from scipy_distutils.core import setup
    setup(version = "0.3.0",
          description = "Tools for inlining C/C++ in Python",
          author = "Eric Jones",
          author_email = "eric@enthought.com",
          licence = "SciPy License (BSD Style)",
          url = 'http://www.scipy.org',
          **configuration(parent_path=''))
