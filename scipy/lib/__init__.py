try:
    __import__('pkg_resources').declare_namespace(__name__)
    print 'scipy_core.lib.__init__'
except ImportError:
    pass
