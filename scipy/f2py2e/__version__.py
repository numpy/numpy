major = 2

try:
    from __cvs_version__ import cvs_version
    version_info = (major,)+cvs_version[1:]
    version = '%s.%s.%s_%s' % version_info
except:
    version = '%s_tarball' % (major)

