#! /usr/local/Python/bin/python

from Numeric import *
from cabase import *

o1 = zeros((2,3))
o2 = ones((4,5),Float16)
cabase(1,o1,o2,'error')
print 'OK'
