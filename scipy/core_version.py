version='0.4.2'

try:
    import __svn_version__  as svn
    version += '.'+svn.version
except ImportError:
    pass

