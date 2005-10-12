version='0.4.3'

try:
    import base.__svn_version__  as svn
    version += '.'+svn.version
except ImportError:
    pass

