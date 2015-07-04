'''
Created on 31 Mar, 2015

@author: wangyi
'''

class C:
    def __init__(self):
        self._x_ = None

    @property
    def x(self):
        """I'm the 'x' property."""
        return self._x_

    @x.setter
    def x(self, value):
        self._x_ = value

    @x.deleter
    def x(self):
        del self._x_
        
if __name__ == "__main__":
    c = C()    
    
    c.x
    c.x = 100
    print(c.x)  
        
