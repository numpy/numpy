version='0.4.2'

try:
    import scipy.base.__svn_version__  as svn
    version += '.'+svn.version
except ImportError:
    pass

