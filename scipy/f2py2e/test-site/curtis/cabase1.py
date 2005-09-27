#! /usr/local/Python/bin/python

from Numeric import *
from cabase import *

def main():
  o1 = ones([20,10])
  o2 = ones([20,5])

  o2[0][1] = 2
  o2[0][2] = 3
  o2[0][3] = 4
  o2[0][4] = 5
  print o2
  print o2.shape
  
  cabase(1,o2,5,20)
  cabase(0,o2,5,20)
#  print "switching x and y"
#  cabase(1,o2,20,5)
#end def main()

if __name__== '__main__':
  main()
#end if
    
