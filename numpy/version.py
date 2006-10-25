version='1.0'
release=True

if not release:
    import os
    svn_version_file = os.path.join(os.path.dirname(__file__),
                                   'core','__svn_version__.py')
    if os.path.isfile(svn_version_file):
        import imp
        svn = imp.load_module('numpy.core.__svn_version__',
                              open(svn_version_file),
                              svn_version_file,
                              ('.py','U',1))
        version += '.dev'+svn.version
