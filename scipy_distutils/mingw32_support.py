
__revision__ = "$Id:"
import warnings
warnings.warn("""
*******************************************************************
The content of mingw32_support.py (this file) has been moved to
mingw32ccompiler.py. If you see this message then it may be because
f2py2e version older than 2.39.235 was importing this file. You can
either ignore this message or upgrade to the latest f2py2e version.
The last action is recommended as in future this file might be removed
and then for older f2py2e releases removing 'import mingw32_support'
statement manually is required.
*******************************************************************""")
