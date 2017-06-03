'''
Created on 1 Apr, 2015

@author: wangyi
@famework: all rights reserved
'''
#===========================================================================
# matrix input output middleware, this part will check all the input values on behalf of matrix
#===========================================================================
# wraps all the dirty parts here
class meta(type):
    
    def __init__(cls, name, bases, nmspc):
        super(meta, cls).__init__(name, bases, nmspc)
        
        # binding methods to cls
        # modify the cls here
        # ...

# with parameter
class matrixClsMiddleWare(object):
    #@ author Lei Wang
    
    def __init__(self, cls, *callbacks, position=0):
        self.pre_callbacks, self.post_callbacks = \
                        list(callbacks[position:]), list(callbacks[:position])      
        self.cls = cls
    ## instance
    def fire(self, cls=None):
        if cls is None:
            cls = self.cls
        
        return Handlers(cls, self.pre_callbacks, self.post_callbacks)
    __call__ = fire
 
class Handlers(object):
    
    def __init__(self, cls, preHdl=[], pstHdl=[]):
               
        self.cls    = cls
        self.preHdl = preHdl
        self.pstHdl = pstHdl  
           
## iterator hook
    def __iter__(self):
        return self.cls.__iter__() 
        
    def attach__pre(self, *callbacks):
        self.preHdl.extend(callbacks)
        return self
        
    def attach_post(self, *callbacks):
        self.pstHdl.extend(callbacks)
        return self
        
    def fire(self, *args, **keywords):
        self.inst = self.cls(*args, **keywords)        
    __call__ = fire
    
    def addh(self, callback, attr=None, type=''):
        raise Exception("not implementated ")
    
# remaining code has been deprecated