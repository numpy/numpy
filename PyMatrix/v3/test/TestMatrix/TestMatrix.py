# -*- coding: utf-8 -*-
'''
Created on 21 Jan, 2015

PyMatrix implementation based on pure python oop charisma

Description:

@author: WANG LEI / YI, Research Associate @ NTU

@emial: L.WANG@ntu.edu.sg, Nanyang Technologcial University

@licence: licence
'''

import unittest

from core.matrixArray import *

class const(object):
    # three ties nested matrix
    a = _TEST_MATRIX_MULTI = matrixArrayLists([
                             [['000', '001', '002'], ['010', '011', '012'], ['020', '021', '022']],
                             [['100', '101', '102'], ['110', '111', '112'], ['120', '121', '122']],
                             [['200', '201', '202'], ['210', '211', '212'], ['220', '221', '222']],
                             [['300', '301', '302'], ['310', '311', '312'], ['320', '321', '322']]
                             ])
    
    # empty matrix
    b = _TEST_EMPTY = matrixArrayLists()    

def get_variabels(name, type=0):
    from copy import deepcopy
    return {key:value for key, value in name.__dict__.items() if not key.startswith('__') and not key.startswith('_') and not callable(key)}
    
    
# temp variables
test_case_list = get_variabels(const)

class TestMatrix(unittest.TestCase):
    
    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_getitem(self):
        
        doc = """
get one element, x[0], for multi-demension matrix, for example 2 demension matrix it should return
a rwo vector; for vector itself, it should return a value along the axis. If the element not exists, 
return none.
        """
        print(doc)
        
        a = test_case_list['a']
    
        print("a[:,1,1]", a[:,1,1])
        print("a[[1,2],1,1]",a[[1,2],1,1])
        print("a[[1,2],0,[1,2]]",a[[1,2],0,[1,2]])
        print("a[0]",a[0])
        print("a[1,2,1]",a[1,2,1])
        print("a[1,:,2]",a[1,:,2])
            
    def test_fundamental_transportation(self):
        
        a = matrixArrayLists([[1,2],[3,4]])
        
        print("original:", a)
        
        temp   = a[0,:]
        a[0,:] = a[1,:]
        a[1,:] = temp
        
        print("row transformation-0", a)
        
        temp   = a[:,0]
        a[:,0] = a[:,1]
        a[:,1] = temp
        
        print("col transformation-1", a)

        a = test_case_list['a']
        temp     = a[0,:,:]
        a[0,:,:] = a[1,:,:]
        a[1,:,:] = temp
        
        print("row transformation-2", a)
        

if __name__ == "__main__":
    unittest.main()