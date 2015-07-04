'''
Created on 7 Feb, 2015

@author: wangyi
'''
import logging

__all__ = ['logFactory']

class logFactory(object):
    
    def __init__(self, formmat, message, 
                     name=None, 
                     streamHandlers=None, 
                     LOG_LEVEL=None, 
                     path=None
                     ):
        self.formmat = formmat
        self.message = message

        self._instantiate(name, streamHandlers, LOG_LEVEL, path)
    
    def _instantiate(self, name=__name__, 
                     streamHandlers=[logging.StreamHandler(), logging.FileHandler()], 
                     LOG_LEVEL     =[logging.INFO,            logging.WARNING], 
                     path='../../logs/' + __name__ + '.log'
                     ):
        if  self.logger == None:
            self.logger  = logging(name)
        else:
            return self.logger
        # keep a local reference
        logger   = self.logger
        formmat  = self.formmat
        
        logger.setLevel(logging.DEBUG)
        
        for i, handler in enumerate(streamHandlers):
            handler.setLevel(LOG_LEVEL[i])
            handler.setFormatter(formmat)
            
            logger.addHandler(handler)  
                      
        return \
            logger
            
    def _fire(self, Func):
        
        def wrapper(*args, **keywords):
            
            # set up
            logger  = self.logger
            message = self.message
            
            logger.info('start'  + Func.__name__ + '-->\n', message)        
            result  = Func(*args, **keywords)
            logger.info('finish' + Func.__name__ + '-->\n')
            
            return result
        
        return wrapper
            
    __call__ = _fire    