from __future__ import division, absolute_import, print_function

from decimal import Decimal

import numpy as np
from numpy.testing import (
    run_module_suite,assert_, assert_almost_equal, assert_allclose, assert_equal, assert_raises
    )

class TestFinancialRate(object):
    
    def test_rate(self):
        assert_almost_equal(np.rate(10, 0, -3500, 10000),0.1107, 4)   
        
    def test_rate_decimal(self):
        """Test that decimals are supported"""
        rate = np.rate(Decimal('10'), Decimal('0'), Decimal('-3500'), Decimal('10000'))
        assert_equal(Decimal('0.1106908537142689284704528100'), rate)
        
        # nan
        rate = np.rate(Decimal(12), Decimal('400'), Decimal('10000'), Decimal(0))
        assert_equal(np.nan, float(rate))     
        
        rate = np.rate(Decimal(12), Decimal('-400'), Decimal('10000'), Decimal(20000))
        assert_equal(np.nan, float(rate))        
        
    def test_when(self):
        # begin
        assert_equal(np.rate(10, 20, -3500, 10000, 1),
                     np.rate(10, 20, -3500, 10000, 'begin'))        
        # end
        assert_equal(np.rate(10, 20, -3500, 10000),
                     np.rate(10, 20, -3500, 10000, 'end'))
        assert_equal(np.rate(10, 20, -3500, 10000, 0),
                     np.rate(10, 20, -3500, 10000, 'end'))
        
      
        
    def test_decimal_with_when(self):
        """Test that decimals are still supported if the when argument is passed"""
        # begin
        assert_equal(np.rate(Decimal('10'), Decimal('20'), Decimal('-3500'), Decimal('10000'), Decimal('1')),
                     np.rate(Decimal('10'), Decimal('20'), Decimal('-3500'), Decimal('10000'), 'begin'))
        
        # nan
        rate = np.rate(Decimal(12), Decimal('400'), Decimal('10000'), Decimal(20000),1)
        assert_equal(np.nan, float(rate))  
    
        rate = np.rate(Decimal(12), Decimal('400'), Decimal('10000'), Decimal(20000),'begin')
        assert_equal(np.nan, float(rate)) 
        
        # end
        assert_equal(np.rate(Decimal('10'), Decimal('20'), Decimal('-3500'), Decimal('10000')),
                     np.rate(Decimal('10'), Decimal('20'), Decimal('-3500'), Decimal('10000'), 'end'))
        assert_equal(np.rate(Decimal('10'), Decimal('20'), Decimal('-3500'), Decimal('10000'), Decimal('0')),
                     np.rate(Decimal('10'), Decimal('20'), Decimal('-3500'), Decimal('10000'), 'end'))
        
        #nan
        rate = np.rate(Decimal(12), Decimal('400'), Decimal('10000'), Decimal(0))
        assert_equal(np.nan, float(rate))  
    
        rate = np.rate(Decimal(12), Decimal('400'), Decimal('10000'), Decimal(0),'end')
        assert_equal(np.nan, float(rate))  
        
if __name__ == "__main__":
    run_module_suite()