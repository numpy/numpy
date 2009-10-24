from numpy.testing import *
import numpy.lib.utils as utils
from StringIO import StringIO

def test_lookfor():
    out = StringIO()
    utils.lookfor('eigenvalue', module='numpy', output=out,
                  import_modules=False)
    out = out.getvalue()
    assert 'numpy.linalg.eig' in out
