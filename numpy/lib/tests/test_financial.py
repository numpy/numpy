"""
>>> np.round(np.rate(10,0,-3500,10000),4)==0.1107
True

>>> np.round(np.irr([-150000, 15000, 25000, 35000, 45000, 60000]),4)==0.0524
True

>>> np.round(np.pv(0.07,20,12000,0),2) == -127128.17
True

>>> np.round(np.fv(0.075, 20, -2000,0,0),2) == 86609.36
True

>>> np.round(np.pmt(0.08/12,5*12,15000),3) == -304.146
True

>>> np.round(np.nper(0.075,-2000,0,100000.),2) == 21.54
True

>>> np.round(np.npv(0.05,[-15000,1500,2500,3500,4500,6000]),2) == 117.04
True

>>> np.round(np.mirr([-4500,-800,800,800,600,600,800,800,700,3000],0.08,0.055),4) == 0.0665
True

>>> np.round(np.mirr([-120000,39000,30000,21000,37000,46000],0.10,0.12),4)==0.1344
True
"""

from numpy.testing import *
import numpy as np

def test():
    import doctest
    doctest.testmod()

if __name__ == "__main__":
    run_module_suite()
