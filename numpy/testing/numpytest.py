from __future__ import division, absolute_import, print_function

import os
import warnings

__all__ = ['importall']


def importall(package):
    """
    `importall` is DEPRECATED and will be removed in numpy 1.9.0

    Try recursively to import all subpackages under package.
    """
    warnings.warn("`importall is deprecated, and will be remobed in numpy 1.9.0",
                  DeprecationWarning)

    if isinstance(package,str):
        package = __import__(package)

    package_name = package.__name__
    package_dir = os.path.dirname(package.__file__)
    for subpackage_name in os.listdir(package_dir):
        subdir = os.path.join(package_dir, subpackage_name)
        if not os.path.isdir(subdir):
            continue
        if not os.path.isfile(os.path.join(subdir,'__init__.py')):
            continue
        name = package_name+'.'+subpackage_name
        try:
            exec('import %s as m' % (name))
        except Exception as msg:
            print('Failed importing %s: %s' %(name, msg))
            continue
        importall(m)
    return
