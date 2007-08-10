
__all__ = ['SetupPy']

import os
import sys
from numpy.distutils.exec_command import exec_command
from base import Component
from utils import FileSource

def write_files(container):
    s = ['creating files and directories:']
    for filename, i in container.label_map.items():
        content = container.list[i]
        d,f = os.path.split(filename)
        if d and not os.path.isdir(d):
            s.append('  %s/' % (d))
            if not Component._generate_dry_run:
                os.makedirs(d)
        s.append('  %s' % (filename))
        if not Component._generate_dry_run:
            f = file(filename,'w')
            f.write(content)
            f.close()
    return '\n'.join(s)


class SetupPy(Component):

    """
    >>> from __init__ import *
    >>> s = SetupPy('SetupPy_doctest')
    >>> s += PyCModule('foo')
    >>> s,o = s.execute('build_ext', '--inplace')
    >>> assert s==0,`s`
    >>> import SetupPy_doctest as mypackage
    >>> print mypackage.foo.__doc__ #doctest: +ELLIPSIS
    This module 'foo' is generated with ExtGen from NumPy version...
    
    """
    template_setup_py_start = '''\
def configuration(parent_package='', top_path = ''):
    from numpy.distutils.misc_util import Configuration
    config = Configuration('',parent_package,top_path)'''
    template_setup_py_end = '''\
    return config
if __name__ == "__main__":
    from numpy.distutils.core import setup
    setup(configuration=configuration)
'''
    template = '%(SourceWriter)s'

    container_options = dict(
      SourceWriter = dict(user_defined_str = write_files),
      TMP = dict()
    )

    component_container_map = dict(
        FileSource = 'SourceWriter',
        ExtensionModule = 'TMP',
    )

    def initialize(self, build_dir, *components, **options):
        self.name = self.path = build_dir
        if not self.path:
            self.setup_py = setup_py = Component.PySource('extgen_setup.py')
            self.init_py = init_py = Component.PySource('extgen__init__.py')
        else:
            self.setup_py = setup_py = Component.PySource('setup.py')
            self.init_py = init_py = Component.PySource('__init__.py')

        setup_py += self.template_setup_py_start

        self += init_py
        self += setup_py
        
        map(self.add, components)

        return self

    def finalize(self):
        self.setup_py += self.template_setup_py_end

    def execute(self, *args):
        """
        Run generated setup.py file with given arguments.
        """
        if not args:
            raise ValueError('need setup.py arguments')
        self.info(self.generate(dry_run=False))
        cmd = [sys.executable,'setup.py'] + list(args)
        self.info('entering %r directory' % (self.path))
        self.info('executing command %r' % (' '.join(cmd)))
        r = exec_command(cmd, execute_in=self.path, use_tee=False)
        self.info('leaving %r directory' % (self.path))
        return r

    
def _test():
    import doctest
    doctest.testmod()
    
if __name__ == "__main__":
    _test()
