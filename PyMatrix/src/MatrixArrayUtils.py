'''
Created on 6 Jun, 2015

@author: wangyi
'''

from time import time as Time

DEBUG_TIME_ELAPSE = False
# this function for elapse time measurement

# from core.framework.middleware import *

def timmer(func):
    
    def wrapper(*args, **keywords):
        # start time
        start  = Time()
        # original call
        result = func(*args, **keywords)

        elapse = Time() - start
        
        # errors reporting control
        # later will change it to log version 
        if  DEBUG_TIME_ELAPSE:
            print(func.__name__, ':\n\tconsumed ', '{0:<2.6f}'.format(elapse), ' seconds')
        return result
    return wrapper