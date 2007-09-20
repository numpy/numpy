"""distutils.command

Package containing implementation of all the standard Distutils
commands."""

__revision__ = "$Id: __init__.py,v 1.3 2005/05/16 11:08:49 pearu Exp $"

distutils_all = [  'build_py',
                   'clean',
                   'install_lib',
                   'install_scripts',
                   'bdist',
                   'bdist_dumb',
                   'bdist_wininst',
                ]

__import__('distutils.command',globals(),locals(),distutils_all)

__all__ = ['build',
           'config_compiler',
           'config',
           'build_src',
           'build_ext',
           'build_clib',
           'build_scripts',
           'install',
           'install_data',
           'install_headers',
           'bdist_rpm',
           'scons',
           'sdist',
          ] + distutils_all
