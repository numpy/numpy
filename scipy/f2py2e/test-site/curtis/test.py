#! /usr/local/Python/bin/python

from Numeric import *
from cabase import *

a = zeros((2,3))
cabase(1,a,'wrong')
print a
cabase(0,a,'verry_wrong')
