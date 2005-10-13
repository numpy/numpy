version='0.4.3'

import os
svn_version_file = os.path.join(os.path.dirname(__file__),
                                'base','__svn_version__.py')
if os.path.isfile(svn_version_file):
    import base.__svn_version as svn
    version += '.'+svn.version
