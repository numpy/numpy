major = 0
minor = 2
micro = 2
release_level = 'alpha'

try:
    from __cvs_version__ import cvs_version
    cvs_minor = cvs_version[-3]
    cvs_serial = cvs_version[-1]
except ImportError,msg:
    print msg
    cvs_minor = 0
    cvs_serial = 0

scipy_distutils_version = '%(major)d.%(minor)d.%(micro)d_%(release_level)s'\
                          '_%(cvs_minor)d.%(cvs_serial)d' % (locals ())
