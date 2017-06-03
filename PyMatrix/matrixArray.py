from __future__ import division
# -*- coding: utf-8 -*-
'''
Created on 18 Sep, 2014

PyMatrix implementation based on pure python oop charisma

Description:

@author : WANG LEI / YI, Research Associate @ NTU

@emial: L.WANG@ntu.edu.sg, Nanyang Technologcial University

@licence: licence
'''
__all__ = ["matrixArrayLists", "matrixArrayNum", "matrixArray"] 

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


# helper class for iteration over empty elements
class Null():
    
    def __init__(self, *args, **hints):
        pass
    
    def  __len__(self):
        return 1
    
    def  __str__(self):
        return '*'
    
    def __repr__(self):
        return 'nullObect'
     
    def  __add__(self, other):
        return Null()
    __radd__ = __add__
     
    def  __sub__(self, other):
        return Null()
    __rsub__ = __sub__
     
    def  __mul__(self, other):
        return Null()
    __rmul__ = __mul__
    
#===========================================================================
# n-d matrix size discriptor: when it is 2-d or 1-d, it reduces to {row, col} form
#===========================================================================
class Size(object):
    
    def __init__(self, data=[]):        
        self.data = data
    
    def __iter__(self):
        return \
        self.data.__iter__() 
    
    def __get__( self, caller, callerType,):
        if  caller == None:#caller == None:
            return \
                 self
        else:
            return \
                 self.__class__(caller._shape_array)
    
    def __getitem__(self, key,):
        try:
            return self.data[key]
        except:
            return 0
        
    def __setitem__(self, key, val):

        self.data[key] = val
    
    def __getattribute__(self, name,):
        try:
            return object.__getattribute__(self, name)
        except:
            if  name in ['row', 'col']:
                try:
                    return self.data[name]
                except:
                    return self.data[{'row':0, 'col':1}[name]]
            raise Exception("no such attribute")
        
    def __len__(self):
        if  not self.data:
            return 0
        return \
            len(self.data)
    
    def __str__(self):
        return str( len(self.data) ) + ':' + str(self.data) + '\n'
    
    def   count(self, item):
        self.data.count(item)
        return self.data
    
    def  append(self, item):
        self.data.append(item)
        return self.data
    
    def assert_equal(self, size):
        # do implementation here
        return self, size, True
    
    def assert_tolerate(self, size):
        # do implementation here
        return self, size, True

# for matrix values, we just have two types, numeric and non-numeric classes
#===========================================================================
# n-d matrix formatter discriptor
#===========================================================================
from math import floor
class Formatter(object):
    
    def __init__(self, size=None, data = {'width':[2], 'float':2}, description= ['{0:<{width}.{float}f} ', '{0:<{width}s} ', '{0:<{width}s} ']):
        self.templates = []
        self.templates.extend(description)
        self.data = data
        self.size = size

    def __get__(self, caller, callerType):
        if  caller == None:
            return \
                self
        else:
            return Formatter(caller.size, caller._init_formatter())
    
    def __getitem__(self, _id):
        return self.templates[_id]
    
    def __getattribute__(self, name):
        try:
            return object.__getattribute__(self, name)
        except:
            if  name == 'width':
                # get stored position
                a, b  = self.position
                width = self.data[name]
                
                # get width for the element in column b
                try:
                    return -self.float if a == self.size.row - 1 and b == self.size.col - 1 else width[b]# len(c), len(c[a])
                except Exception:
                    #import traceback
                    #traceback.print_exc()
                    return -self.float
            if  name == 'float':
                return self.data[name]
    
    def register(self, template):
        self.templates.append(template)
    
    # used to decorate element
    # format(element) -> new string
    def fire(self, element, a, b):
        # store the value
        self.position = (a, b)
        
        element = str(element) if not isinstance(element, float) or isinstance(element, int) \
        else element
        # return processed string
        try:
            return self[0].format(element,
                                  width=self.width + self.float + 1, 
                                  float=self.float)
        except ValueError:
            return self[1].format(element.__str__(), 
                                  width=self.width + self.float + 1)
        except TypeError:
            return self[2].format(element.__str__(), 
                                  width=self.width + self.float + 1)
    __call__ = fire    
    
from copy import copy, deepcopy
# helper function
def index_len(key, l):
    if  isinstance(key, int):#1
        return (1,)
    
    if  isinstance(key, list):#2
        return (len(key),)
    
    if  isinstance(key, slice):#3
        start, stop, step = key.indices(l if isinstance(l, int) 
                                        else l[0])
        return (len(range(start, stop, step)),) 
    #   entrance
    if  isinstance(key, tuple): 
        l = \
        copy(l)#deepcopy(l)
        return \
          tuple([index_len(i, l.pop(0) if len(l) > 0 else 0)[0]
                  for i in key]) 
    else:
        return (1,)

# helper function
def index_val(key, l):
    
    if  isinstance(key, int):
        return ([key],)
    
    if  isinstance(key, list):
        return ( key ,)
    
    if  isinstance(key, slice):
        start, stop, step = key.indices(l if isinstance(l, int) 
                                        else l[0])
        return (list(range(start, stop, step)),)  
    #   entrance
    if  isinstance(key, tuple):
        l = \
        copy(l)#deepcopy(l)        
        return \
          tuple([index_val(i, l.pop(0) if len(l) > 0 else 0)[0]
                  for i in key])
    else:
        return ([key], )

def edit(level, position, _list, Array):
    if level == 0 and Array.x_label_enabled:
        pass#_list[level][position] = None
    if level == 1 and Array.y_label_enabled:
        pass#_list[level][position] = None
# this implementation has been deprecated,
# but later used to enhance core function ba
def index_map(func):
    
    def wrpper(self, _id, l):
        
        _list = []
        
        list( map(lambda i: _list.append(i) \
               if i < len(l) else None, _id) )
        
        return func(self, _list, l)
    return wrpper
# this function might be changed in the future
# (1) if it found in column index, 
#     if key is of single value, replace it with '[:, map_kay(column_ref)]', and 'reference_name' -> headers
# (2) if it found in row index,
#     if key is of single value, replace it with '[map_key(row_ref), :]', and 'row_ref' -> indice
# (3) store indice and headers in children selected
# keywords mapping   
def map_key(func):
    
    def wrpper(self, key):
             
        size = self._shape_array
        # middleware preprocessing       
        hint = index_len(key, size)
        # middleware preprocessing 
        key  = index_val(key, size)
        # get inner representation of the query object
        root = self.inner_rep
        # ...
        if self.R is False:
            return func(self, key)
        
        # do R like preprocessing
        _list = [id for id in key]
        
        headers = []; indice = []
        
        for a, id in enumerate(_list[0:2]):
                        
            for b, idx in enumerate(id ):# id is not str
                result, axis = \
                 self.isIndexHeader(idx)   
                if axis == 1:  \
                     headers.append(idx)
                if axis == 0:  \
                      indice.append(idx)
                                   
                if result is not None:
                    id[b] = result# legal  
                if axis is None:
                    edit(a, b, _list, self)
                    continue
            
            _list[a] = list(filter(lambda x: x is not None,id))
        # you have to do as what I show you.
        
        child = func(self, tuple(_list))# PROCESSING
        # post processing:
        # child may not be matrixArray object anymore
        # need verification
        try:
            child.setIndice( indice)
            child.setheader(headers)
        except:
            pass
        return child
    return wrpper

# loop process memo        
def loopprocess(func):
    # memo = {}
    def wrapper(self): 

        memos = self.memos

        if self.flag is False and memos[0] is not None:
            return memos[0]
        
        memos[0] = func(self)
        return \
           memos[0]
           
    return wrapper
            
    
def checkKey(name, default, _dict):
    try:
        value = _dict[name]
    except KeyError:
        return default
    return \
        value

class matrixArrayLists(list):
    '''
    Created on 17 Nov, 2014
    
    @author: wang yi/Lei, Researcher Associate @ EIRAN, Nanyang Technological University
    
    @email: L.WANG@ntu.edu.sg
    
    @copyright: 2014 www.yiak.co. All rights reserved.
    
    @license: license
    
    @decription: N-Matrix container for objects of any type. It then could be 2 or demensions numeric matrix for computation
    
    @param:
    '''
    # discriptors initialzation
    size                = Size()
    formatter           = Formatter()
    
    # options, configuration context
    Init_hint_options   = { 'r': None,  'c': None, 'debug' : False, 'ori_mem': None, 'R':False, 'row_indexList':[], 'col_indexList':[],
                            'Indice' : [], 'Headers': [], 'flag': False, 'memos':[None]}
    Init_matrix_options = { 'width' : 0,  'float'  : 2,} 
    
    def __init__(self, *args, **hint):
        # initialization fo hint data 
        # because repesentation in 2 demension, r(row), c(col) are needed. It should be modifed by descriptor which will compute r, c whenver an instance call it.   
        
        self._init_hint(hint)
      
        numberOfargs = len(args)
        
        # no inputting arguments
        if   numberOfargs == 0:
            if   hint == {}:
                super(matrixArrayLists, self).__init__([])
                # no hints              
            elif hint != {}:
                # set up empty matrix
                # To do:
                
                # initialization
                super(matrixArrayLists, self).__init__([])
            self.append([])
            
        elif numberOfargs == 1:
            # create a square null matrix. 2-D version
            if   isinstance(args[0], int):
                super(matrixArrayLists, self).__init__()
                # specify n * n null matrix, done
                self.nil(args[0], args[0], Null())
            
            # create a matrix based on one inputting list    
            elif isinstance(args[0], list):
                # copy or convert
                super(matrixArrayLists, self).__init__()
                # this works for matrix
                self.setUp( args[0], self.r, self.c, self.ori_mem )    
                 
        elif numberOfargs == 2:
                # two integers are specified
            if   isinstance(args[0], int) and isinstance(args[1], int ):
                super(matrixArrayLists, self).__init__()    
                # specify m * n null matrix
                self.nil(args[0], args[1], Null())
                # speed up techniques
                self.r = args[0]
                self.c = args[1]
                 
                # combination of integer and list inputtings 
            elif isinstance(args[0], int) and isinstance(args[1], list):
                super(matrixArrayLists, self).__init__()
                # To do: specify m * n null matrix
                # self.nil( args[0], args[0], Null())
                
                self.setUp( args[1], args[0], self.c, self.ori_mem) 
                              
        elif numberOfargs  > 2:
            for i in range( 0,len(args) ):
                if not isinstance(args[i], int):
                    break
 
            if  i == 0 and isinstance(args[ 0 ], list):
                # To do: matrix cantenation
                super(matrixArrayLists, self).__init__()
                
                # To do: union
                                  
            if  i != 0 and isinstance(args[ i ], list):
                # To do: specify, filling missing data by other iteratables
                super(matrixArrayLists, self).__init__()
                
            if  i == 2 and len(args[1:]) == 1:
                # faster
                self.setUp( args[2], args[0], args[1], self.ori_mem)
            
            else:
                self.nil_multi(*args[0:i])
                # fillup
                self.fillUp(*args[i:])

   
    def _init_hint(self, hint):
        for name, default in self.Init_hint_options.items():
            # set local variables
            exec("self.%s = %s" % (name, checkKey(name, default, hint)))
   
    def _init_matrix_formatter(self, _float=2, _width=2, formatter=None):
        for name, default in self.Init_matrix_options.items():
            # set local variables
            exec("self.%s = %s" % (name, default))
         
        size = self.size
        
        col_length = 0
        width_list = []
        inner_list = self.inner_rep
        
        if self.R is True:
            inner_list.extend(self.Headers)
        
        for j in range(size[1]):      
            col_length = self.width
            for i in range(size[0]):
                try:            
                    element = inner_list[i][j]
                except:
                    break
                
                if  isinstance(element, float):
                    element = floor(element)     
                            
                if  col_length < len(str(element)):
                    col_length = len(str(element))
            # indice part
            for i in range(len(self.Headers)):
                try:
                    element = inner_list[i+size[0]][j]
                except:
                    break
                if  isinstance(element, float):
                    element = floor(element)     
                            
                if  col_length < len(str(element)):
                    col_length = len(str(element))
                
            width_list.append(col_length)         
               
        return {'float':_float, 'width':width_list}
    
    
    def _init_array_formatter(self, indice=None, _float=0, _width=2):
        
        col_length = _width
        
        for j in range(len(indice)):
            element = indice[j]
            
            if  isinstance(element, float):
                element = floor(element)
                
            if  col_length < len(str(element)):
                col_length = len(str(element))
                
        return {'float': _float, 'width':[col_length]}
    
    init_array_formatter = _init_array_formatter
    
    # interface function
    def _init_formatter(self, _float=2, _width=2, formatter=None):   
        return self._init_matrix_formatter(_float, _width, formatter)
    # matrix STL iterators
    class matrixIterator(object):
        def __init__(self, mat):
            self.matrixArray = mat
            self.counter = self.__counter__()
            
        def __iter__(self):
            return   self

        ## ! just for two dimensions for the moment
        def __counter__(self):
            
            def routine(iter, size, curr):
                try:
                    iter[curr] += 1
                except Exception:
                    pass
                # check whether it is flow out         
                if  iter[curr] >= size[curr]:
                    
                    # last positon
                    if  curr == 0:
                        return True
                    else:
                        # clear the current bit
                        iter[curr] = 0
                        # go into higher bit
                        return routine(iter, size, curr - 1)              
                
                return  False
            # commment the following lines when debug, other wise comment out
            # when apply matrix 2 list this will be call            
            size = self.matrixArray._shape_array

            # initialization
            tier = len(size)
            
            # iteration indice
            iter = tier * [0]
            
            while True:
                yield iter
                # update              
                signal = routine(iter, size, tier - 1)
                # exit processing
                if  signal:
                    break
     
        def __next__(self):
            try:
                index = next(self.counter)
                return self.matrixArray[tuple(index)]
            except StopIteration:
                raise StopIteration()   
    
        def nextIndex(self):
            try:
                index = next(self.counter)
                return tuple(index)
            except StopIteration:
                raise StopIteration()
            
        def __str__(self):
            return "matrix iterator"

    def setup_r_mode(self):
        if  self.R != True:
            self.R  = True
    
    def clear_r_mode(self):
        if  self.R == True:
            self.R = False
            
            self.clear_headers().clear_indice()
            
    
    def clear_headers(self):
        self.col_indexList.clear()
        self.Headers.clear()
        return self
        
    def clear_indice(self):
        self.row_indexList.clear()
        self.Indice.clear()
        return self
    
    def setIndice(self, l):
        if not isinstance(l, list):
            l = self._popCol(l)
        if len(l) == 0:# pythonic way to check empty list
            return -1
            
        from itertools import zip_longest
        self.row_indexList.append( dict(zip_longest(l, 
                                                    range(0, self.size.row) 
                                                    )) )
        self.setup_r_mode(); self.Indice.append(l)
    
    def isIndex(self, val):
        for item in self.row_indexList:  
            if val in item:
                return item[val]
    
    def setheader(self, l):
        if not isinstance(l, list):
            l = self._popRol(l)
        if len(l) == 0:
            return -1
        
        from itertools import zip_longest
        self.col_indexList.append( dict(zip_longest(l, 
                                                    range(0, self.size.col)
                                                    )) )
        self.setup_r_mode(); self.Headers.append(l)
    
    def isheader(self, val):
        for item in self.col_indexList:
            if val in item:
                return item[val]

    
    def isIndexHeader(self, val):

        result = \
               self.isIndex(val)
        if result is not None: 
            return (result, 0)
        
        result = \
              self.isheader(val)
        if result is not None:
            return (result, 1)
        
        else:
            return (None , None)
    
    @property
    def x_label_enabled(self):
        return True if  len(self.Indice) > 0 and  len(self.Indice[0]) > 0 else False
    
    @property    
    def y_label_enabled(self):
        return True if len(self.Headers) > 0 and len(self.Headers[0]) > 0 else False
    
    def _pop(self, *index):
        # I will consider to extend its functionalities later
        # basic logics is 'del' the object the matrix container referred
        root = self.inner_rep
        
        return self.setitem(index, None, root, "-")
    
    
    def _popCol(self, col):
        # pythonic way to give data simutaneously
        size, list = self.size, [] 
        # collect data returned   
        for i in range(size.row):
            list.append( self._pop(i, col) )  
        return list
    
    def _popRol(self, row):
        # pythonic way to give data simutaneously
        size, list = self.size, []
        # collect data returned
        for i in range(size.col):
            list.append( self._pop(row, i) )
        return list
        
               
        
    def web2list(self, web, header):
        h = header; l = [[record[key] for key in h] for record in web]
        # load data
        self.setUp(l)
        # set header
        self.setheader(h)
        
    #===========================================================================
    # n-d matrix helper functions
    #===========================================================================           
    def name(self):
        return "matrixArrayLists:"
    
    #===========================================================================
    # elementary setup funciton from a iterable
    #===========================================================================
    def _toRow(self, r, c, l):
        # row vector
        if  not c and (r and r == 1):
            if  not isinstance(l[0], list):
                self.extend([l])
                self.modifed_to_row_col = True
            else:
                self.extend(l)
            return True

        # row vector:[[0,1,2...]] 
        if  c and c != 1 and (r and r == 1):
            if  not isinstance(l[0], list):
                self.extend([l])
                self.modifed_to_row_col = True
            else:
                self.extend(l)
            return True
            
        if  c and c != 1 and not r:
            if  not isinstance(l[0], list):
                self.extend([l])
                self.modifed_to_row_col = True
            else:
                self.extend(l)
            return True
    
    def _toCol(self, r, c, l):
        # col vector processing, simple mode
        if  r and r != 1 and not c:
            for i in l:
                if  isinstance(  i,  list ):
                    self.append( i )
                else:
                    self.append([i])
                    self.modifed_to_row_col = True
            return True

        # col vector processing, simple mode
        if  r and r != 1 and (c and c == 1):
            # row vector:[[0],[1],[2],,,]
            for i in l:
                if isinstance(   i,  list ):
                    self.append( i )
                else:
                    self.append([i])
                    self.modifed_to_row_col = True
            return True
        # col vector
        if not r and (c and c == 1):
            for i in l:
                if  isinstance(  i,  list ):
                    self.append( i )
                else:
                    self.append([i])
                    self.modifed_to_row_col = True
            return True
    
    def clear(self):
        # for python2 ...
        try:
            super(matrixArrayLists, self).clear()
        except Exception:
            root = self.inner_rep
            del root[:]
        # clear up all related marks
        # clear modified tag
        self.clear_modify_tag()
        # clear up all associated indice
        self.clear_r_mode()
    

    def clear_modify_tag(self):
        pass


    def setUp(self, l=None, r=None, c=None, o=None):
        # clearn up
        self.clear()
        # set up container values
        if  str(type(l)) == "<class 'list'>":
            # calling from inside
            if  o == 'ori': 
                self.extend(l)
                return
            
            # for default situation, from outside
            if  r == None and c == None:
                flag = True
                
                if  len(l) == 0:
                    self.append(l)
                    return
                
                for i in l:
                    if  isinstance(i, list):
                        pass
                    else:
                        flag = False
                        break
            
                if  flag:
                    self.extend(l)
                    return
                else:
                    # 1 columns vector
                    c = 1    
            #===================================================================
            # col - vector processing
            #===================================================================            
            # col vector processing, simple mode
            if  r and r != 1 and not c:
                for i in l:
                    self.append(i if isinstance(i, list) \
                                else [i])
                return
            # col vector processing, simple mode
            if  r and r != 1 and (c and c == 1):
                # row vector:[[0],[1],[2],,,]
                for i in l:
                    self.append(i if isinstance(i, list) \
                                else [i])
                return
            # col vector
            if not r and (c and c == 1):
                
                for i in l:
                    self.append(i if isinstance(i, list) \
                                else [i])
                return  

            #===================================================================
            # row vector processing
            #===================================================================
            # row vector
            if  not c and (r and r == 1):
                self.extend([l] if not isinstance(l[0], list) else l)
                return

            # row vector:[[0,1,2...]] 
            if  c and c != 1 and (r and r == 1):
                self.extend([l] if not isinstance(l[0], list) else l)
                return
                
            if  c and c != 1 and not r:
                self.extend([l] if not isinstance(l[0], list) else l)
                return 
            #===================================================================
            # special situation
            #===================================================================
            if  r and r == 1 and c and c == 1:
                self.extend([l] if not isinstance(l[0], list) else l)
                return
            #===================================================================
            # by default
            #===================================================================
            self.extend(l)       
        else:
            
            it_index, it_value = l.__iter__(), l.__iter__()
            while True:
                try:
                    index, value = it_index.nextIndex(), it_value.__next__()
                    # use redefined method
                    # use customised magic expression "mat[:,[1,2,3],0,2:4] = another_matrix"
                    self[index] = value
                    # this is actually an  expression
                except StopIteration:
                    break

        # modify shape accordingly
    #===============================================================================
    # basic matrix filling function, m = matrixArray(list1, list2, list3 ...)
    #===============================================================================
    def fillUp(self, *iterators):    
        obj = self
        for itx in iterators:
            itl, itr = ( obj.__iter__(), itx.__iter__())
            while True:
                try:
                    p, q = (itl.nextIndex(), itr.__next__())
                    # use redefined method
                    obj[p] = q                  
                except StopIteration:
                    break 
        
        return self
    
    def nil_multi(self, *args):
        from copy import deepcopy
        # To do
        # check r, c, when it is used by user
        
      # super(matrixArrayLists, self).clear()
        self.clear()

        deep  = len(args)
        # set root node
        root  = [None for i in range(args[0])]
        # initialize a queue
        queue = []
        queue.append((root,0))
        # broad first searching
        while len(queue) > 0:
            child, i = queue.pop(0)
            # modify children           
            for j in range(args[i]):
                child[j] = deepcopy([None] * args[i+1])
                if  i+1  < deep-1:
                    queue.append( (child[j], i+1) )
                else:
                    gchild = child[j]
                    for j in range(0, args[i+1]):
                        gchild[j] = Null()
        # reset the empty matrix
        self.setUp(root)
                    
    # this help funciton is exclusively for 2-demension case. I consider it seriously. 
    def nil(self, r, c, value=None):
        from copy import deepcopy
        
        # To do
        # check r, c, when it is used by user
#############################################        
#       super(matrixArrayLists, self).clear() 
#############################################
        self.clear()

        self.setUp([deepcopy([deepcopy(value) for _ in range(c)]) for _ in range(r)])
    
    # further extension form nil funciton            
    def Zeors(self, r, c=None):
        if  c == None:
            self.nil(r, r, 0)
        else:
            self.nil(r, c, 0) 
            
            
    def __call__(self, key=None, value=None):
        if  key == None:
            return self
        elif \
            key != None:
            if  value == None:
                return super(matrixArrayLists, self).__getitem__(key)
            elif \
                value != None:
                super(matrixArrayLists, self).__setitem__(key, value)
                return self
            
            
    @timmer    
    def tolist(self):
        
        l = self.inner_rep
        
        # if it is a vector we need to leave it as list format
        # vector processing
        if len(l) == 1:
            if isinstance(l[0],list):
                l = l[0] 
        else:
            for i, v in enumerate(l):
                if isinstance(v, list):
                    if  len(v) == 1 :
                        l[i] = v[0]
        
        return  l
    
    @property
    def inner_rep(self):
        return self(slice(0, len(self)))
        
        
    def __setattr__(self, name, value):
        if   isinstance(name, tuple):
            pass
        elif True:
            self.__dict__[name] = value   
    
    def __iter__(self):
        return self.matrixIterator(self)
     
    def setitem(self, id, element, l, type="+"):
        # see idex processing from int, slice, list and tuple, .
        # [[1],[1,2,3,4],[5]...] --> [1] or [1,2,3,4] or [5] --> [1,1,5], [1,2,5], [1,3,5],[1,4,5]..., depth first searching is applied to do such indexing
        # this assuming that value will automaticaly fill the part what index indicates
        # e.g.: 'matrix[[1,2,3],0] = value' means that find elements in 'value' to fill in a[1,0], a[2,0], a[3,0]
        curr = 0
        hook = l
        flag = False
        
        while curr < len(id) -1:
            try:
                l = l[id[curr]]
                
                if  isinstance(l, list):
                    hook = l
                else:
                    # make the element wrapped in a list to poin to a new dimension
                    l = [l]
                    # redirect the list element reference
                    hook[id[curr]] = l
                    hook = l
                curr += 1
            except:      
                steps = id[curr]  - len(l) + 1
                # l.extend([Null() for _ in range(steps)])
                l.extend([Null()] * steps); flag = True
                
        while True:
            try:
                if   type == "+":
                    # element casting for matrix list member
                    l[id[curr]] = element; break #if not isinstance(element, self.__class__) \
                    #else element.matrix2list(); break
                elif type == "-":
                    # available for retrieve
                    return l.pop(id[curr]); flag = True
                else:
                    raise Exception("wrong type!")
                
            except:
                steps = id[curr]  - len(l) + 1
                # l.extend([Null() for _ in range(steps)])
                l.extend([Null()] * steps); flag = True
        
        self.flag = flag                
            
    # @timmer
    @index_map
    def getitem(self, _id, l):
        # see idex processing from int, slice, list and tuple
        # [[1],[1,2,3,4],[5]...] --> [1] or [1,2,3,4] or [5] ..., 
        try:
            return list(map(lambda idx: l[idx], _id))#l.__getitem__(idx)
        except:
            # this will be enhanced for some senarios
            return [Null()]       
    
    def setitem_multi(self, ids, root, it, type="+"):
        # deduce user behavior
        # get all possible id for setting
        def element_generator():      
            yield next(it) if hasattr(it,'__next__') else it
        
        # no values return
        def routines(curr):
            #   exitance
            if  curr == depth:
                for i in ids[curr-1]:
                    index[curr-1] = i 
                    try: 
                        self.setitem(index, next(element_generator()), root, type)                    
                    except StopIteration:
                        return 
            else:
                for i in ids[curr-1]:
                    index[curr-1] = i
                    # push into functional stack    
                    routines(curr+1)
        
        # return a value
        def routines2(curr):
            
            list = []
            #   exitance
            if  curr == depth:
                for i in ids[curr-1]:
                    index[curr-1] = i 
                    try: 
                        list + [self.setitem(index, next(element_generator()), root, type)]           
                    except StopIteration:
                        return list
            else:
                for i in ids[curr-1]:
                    index[curr-1] = i
                    # push into functional stack    
                    list + routines(curr+1)            
            
        
        depth = len(ids)
        # this initialization will reduce exception handling
        index = depth * [0]
        # running
        if  type == '+':
            routines(1)
        if  type == '-':
            return routines2(1)
        # raise an error here
         
    def getitem_multi(self, ids, root):
        # convert tuple to list(queue)  to obtain built-in methods
        # breadth first strategy
        l     = len(ids)
        stack = [(root, 0)]
        final = []
        
        while len(stack) != 0:
            try:
                child, axis = stack.pop(0)
                # processing
                for grdchild in self.getitem(ids[axis], child): 
                     
                    if   axis <  l - 1:
                        stack.append((grdchild, axis+1))
                    else:
                        # item = self.getitem(ids[axis], grdchild)
                        final.append( grdchild )

            except IndexError as e:
                print(e);break
        self.flag = False    
        return  final    
    

    def __setitem__(self, key, value): 
        
        size = self._shape_array
        # middleware preprocessing       
        hint = index_len(key, size)
        # middleware preprocessing 
        key  = index_val(key, size)
        # get inner representation
        root = self.inner_rep
        
        #print('first part of __setitme__ interface:', '{0:<2.6f}'.format(e), ' seconds\n')
        # infer user 's attention
        # enhanced functionality for better customers experience
        if  len(hint) == 1 or (size.__len__() > len(hint) and 1 in size): # should use >= provided that i cannot be exec if matrix is empty
            # entry point
            """
            (1) if hint == 1 we deduce the user want to pass one index to get values,
            basically for one dimensional data like vector, will do this
            so if 1 exists in size object, we deduce that user that use this to retrive values
            (2) if hint != 1 but size.__len__() > len(hint), the user think some data should be filled automatically for the index,
            which is of course the same thing.
            
            help func get_offset():
            
            result = [xxx]
            
            return tuple(generator_result)
            """
            def get_offset():
                
                result = iter([v for v in key])
                
                offset = map(lambda i: [0] if i == 0 else next(result), size)
                
                # the following implementation has been deprecated
                # this will slow the whole eco-system
                # 22-May-2015, Lei Wang
####################################################################
#                 for i in range(size.__len__()):
#                     if size[i] != 1:
#                         break
#                 offset = [[0]] * i# errors might occur
#                 offset.extend(result)
#                 result = offset
#                 for i in range(size.__len__()):
#                     if size[len(size) - i - 1] != 1:
#                         break
#                 offset = [[0]] * i# errors
#                 result.extend(offset)
#                 return  tuple(result)
####################################################################
                # it is better to return an generator
                return  tuple(offset) 
            # redefine
            key = get_offset()
        
        #print('second part of __setitme__ interface:', '{0:<2.6f}'.format(e), ' seconds\n')
        if  max(hint) <= 1:
            """
            user want to get a value or object stored:
            
            will need to convert key from 'standard mode' to 'single mode'
            """
            self.setitem([item[0] for item in key], value, root) 
        else:
            """
            user want to set multi values if 'hint' indicates multi - indice 
            
            if the value is not iterable, we will use the value as default
            
            warning! if you want to use 'iterable' as default value, you might very likely get wrong answer
            """
            # the value might not has iterator, hence need error handler here
            self.setitem_multi(key, root, value) if not hasattr(value, '__iter__') else \
            self.setitem_multi(key, root, value.__iter__())
        # make changes to the whole matrixArray
        self.setUp(root)
        return \
            self
    
    @timmer
    @map_key
    def __getitem__(self, key):  
             
        size = self._shape_array
        # middleware preprocessing       
        hint = index_len(key, size)
        # middleware preprocessing 
        key  = index_val(key, size)
        # get inner representation of the query object
        root = self.inner_rep
        
        try:
            if  max(hint) <= 1:#? max performance?
                """
                user want to get one element
                (1) user complete index input
                (2) user need assistance if some dimensions is of 1 data
                """             
                # return the value wrapped in the list
                # judge this by user inputting and matrix demensions
                if len(hint) == len(size):
                    slot = self.getitem_multi(key, root)
                    return slot[0]# 01
                # user might use a simple way to get list data
                if len(hint) + size.count(1) \
                             == len(size):
                    def realidx():
                        it = key.__iter__(); id = []
                        for i in size:
                            id.append([0] if i == 1 else next(it))
                        return id
                    
                    real = realidx()
                    slot = self.getitem_multi(real, root)
                    return slot[0]# 02
                else:
                    slot = self.getitem_multi( key, root)
                    return self.__class__(slot,r=hint[0])# 03
#                number = slot[0][0] if isinstance(slot[0], list) else slot[0]
#                return number if len(hint) == len(size) or 1 in size \
#                                   else self.__class__(slot, r=hint[0])
            # later I will wrap this method in middleWare postprocessing
            # some additional adjugement to make sure it is safe and stable
            # get inner representation of the query result
            slot = self.getitem_multi(key, root); array = []
            if  len(hint) >= 3:
                for i in range(len(hint) - 1):
                    if  hint[len(hint) - 1 - i] != 1:  break
                hint =  hint[0:len(hint) - i]
            self.setitem_multi( tuple(map(list, map(range, hint))), array, iter(slot) ) 
            # go           
            if  len(hint) == 1:
                return self.__class__(array, r=hint[0]) #04
            else:
                return self.__class__(array, r=hint[0], c=hint[1]) #05
        except:
            return Null()

    
    def _str(self):
        
        size = self.size
        
        if  len(size)  > 2:
            pass#return self.name() + '\n' + super(matrixArrayLists, self).__str__()
        
        formatter = self.formatter  
        # string representation
        out  = []#""
        pre  = ' '
        succ = '\n'
        c    = self.inner_rep
        
        if len(c[0]) == 0:
            c[0].append(Null())
        # set title
        out.extend([self.name() + '\n\n', "["]); lenc = len(c)   # 0, 1 out += self.name() + "\n["#  position 0
        
        for a in range(lenc):
            
            out.append( pre)
            # get sub len for matrixarray
            lena = len(c[a])                                     # out += pre # position 1 + self.col * (a + 1) 
            
            for b in range(lena):
                out.append(self._element2str(a, b, c, formatter))# out += self._element2str(a, b, c, formatter)
            
            if lena == 0:
                b = -1
            
            for d in range(b + 1, b + size.col - len(c[a]) + 1):
                out.append(self._element2str(a, d, c, formatter))# out += self._element2str(a, d, t, formatter)                    
            # handling for special cases
            if  a < len(c) - 1:
                out.extend([succ, pre])#out += succ   
                    
        out.append("]\n")#out += "]\n"# position 1 + size.row * size.col 
        
        return out,  \
               size, \
               formatter

    def _header(self, axis=None,
                      formatter=None, l=None):
        # string representation
        out  = []#""
        pre  = ' ' + ' '
        succ = '\n'
        # this is random, very dangerous
        c    = l
        # add pre
        out.append(pre)
        
        for b in range(len(c)):
            out.append(self._element2str_sim(0, b, c[b]  , formatter))
        for d in range(b + 1, b + axis - len(c) + 1):
            out.append(self._element2str_sim(0, d, Null(), formatter))
            
        out.append(succ)
        
        return out       
# helper function for rendering    

    def _indice(self, axis=None,
                      formatter=None, l=None):
        # string representation
        out  = []#""
        pre  = ' ' + ' '
        succ = '\n'
        # this is random, very dangerous
        c    = l
        # add pre
        out.append(pre)
        
        for b in range(len(c)):
            out.append(self._element2str_sim(b, 0, c[b]  , formatter))
        for d in range(b + 1, b + axis - len(c) + 1):
            out.append(self._element2str_sim(b, 0, Null(), formatter))
            
        out.append(succ)
        
        return out
        
    
    def __str__(self):
        out, size, formatter = self._str()
        
        if self.R == True:
        # offset controlling, a, b parameters
        # add header
            a = 0
            for item in self.Headers:
                header = self._header(size.col, formatter, item)
                out.insert(1, ''.join(header)); a += 1
        # add index
            for item in self.Indice:
                # call this help function to dynamically change formatter
                formatter.data = self.init_array_formatter(item)
                
                index  = self._indice(size.row, formatter, item)
                # modify indice
                self._modify_line(out, index[1:], a,
                                  size)
                # b = 0
                # for line in out[a+1:]:
                    # pass#line.insert(0, index[b]); b += 1
        
        return ''.join(out)
    
    # out must be list so that it is mutable
    def _modify_line(self, out, index, offset, size):
        # ...
        row = size.row
        col = size.col
        
        # modify header
        for i in range(offset):
            out[i + 1] = ' ' * len(index[0]) + out[i + 1]
            
        # modify body
        # empirical equation for this matrix
        for i in range(row):
            out[i*(col + 3) + offset + 1] = index[i] + out[i*(col + 3) + offset + 1]
    
    # by index
    def _element2str(self, a, b, c, 
                     formatter):
        # values errors logics wrapped here!     
        try:
            return formatter(c[a][b], a, b)
        except TypeError:
            return formatter(c[a], a, b)
        except IndexError:
            return formatter('-'    , a, b)
    
    # by value
    def _element2str_sim(self, a, b, c, formatter):
        return formatter(c, a, b)

    @staticmethod
    def routines(obj, shape, axis, queue):
        
        while len(queue) > 0:
            
            child, axis = queue.pop(0)   
            # temporary storage
            _max = -1
                           
            if   len(child) == 0: _max = 0#array.append( 0)
            elif len(child) >= 1:
                # broadth first searching
                # easier to ask for forgiveness than permission
                # according to my test this approach is much faster than 'isinstance'
                # 'isinstance': 3.848855972290039
                # 'try-except': 0.36542391777038574
                for i in range(len(child)):
                    try:
                        if len(child[i]) > _max:
                            _max = len(child[i])
                        queue.append((child[i], axis+1))
                    except:
                        if  _max == 0:
                            _max = 1
######################################################################
#                     if  isinstance(child[i], list):    
#                         _max = len(child[i]) if len(child[i]) > _max else _max#array.append(len(child[i]))
#                         queue.append(\
#                                   (child[i], axis+1))
#                     elif True:
#                         _max = 1 if _max == 0 else 0#array.append(1)
######################################################################
                    
            # try to update shape  
            try:
                if  shape[axis] < _max:
                    shape[axis] = _max
            except:
                shape.append(_max)
    @property # 1 
    #@loopprocess # 2          
    def _shape_array(self):
        queue = []
        shape = []
        
        axis  = 0
        # updating axis 
        axis += 2
        
        root  = self.inner_rep
        # updating current axis
        shape.extend([len(root), max(map(lambda i:len(i), root))])
        # start processing
        # queue.append((root,axis))
        queue.extend([(item, axis) for item in root])
        # compute next demensions  
        # do not use inner method definition 
################################################           
#         def routines(obj, shape, axis, queue):
#             
#             while len(queue) > 0:
#                 
#                 child, axis = queue.pop(0)   
#                 # temporary storage
#                 _max = -1
#                                
#                 if   len(child) == 0: _max = 0#array.append( 0)
#                 elif len(child) >= 1:
#                     # broadth first searching
#                     for i in range(len(child)):
#                         if  isinstance(child[i], list):  
#                             _max = len(child[i]) if len(child[i]) > _max else _max#array.append(len(child[i]))
#                             queue.append(\
#                                       (child[i], axis+1))
#                         elif True:
#                             _max = 1 if _max == 0 else 0#array.append(1)
#                         
#                 # try to update shape  
#                 try:
#                     if  shape[axis] < _max:
#                         shape[axis] = _max
#                 except:
#                     shape.append(_max) 
##############################################                                                 
        self.routines(root, shape, axis, queue)  
        return shape[0:-1]      
    
    def _get_shape_array2(self):
        queue = []
        shape = []
        
        axis  = 0
        # updating axis 
        axis += 1
        
        root  = self.inner_rep
        # updating curr axis
        shape.append(len(root))
        # start processing
        queue.append((root,axis))
        # compute next demensions   
           
        def routines(obj, shape, axis, queue):
            
            while len(queue) > 0:
                
                child, axis = queue.pop(0)   
                # temporary storage
                array = []
                               
                if   len(child) == 0: array.append( 0)
                elif len(child) >= 1:
                    # broadth first searching
                    for i in range(len(child)):
                        if  isinstance(child[i], list):    
                            array.append(len(child[i]))
                            queue.append((child[i], axis+1))
                        elif True:
                            array.append(1)
                     
                # updating current axis - maximu lenth
                # axis control the looping layer  
                _max = max(array)    
                # try to update shape  
                try:
                    if  shape[axis] < _max:
                        shape[axis] = _max
                except:
                    shape.append(_max) 
                                                 
        routines(root, shape, axis, queue)  
        return shape[0:-1]
     
    def trp(self):
        root = self.inner_rep
        
        size = self.size
        
        if len(root[0]) == 0:
            root[0].append(Null())
        
        mat  = self.__class__([[row[i] for row in root] for i in range(size[1])])

        # the following codes have been deprecated because they are too slow
###################################################
#         size = self.size
#         mat  = self.__class__(size.col, size.row) 
#         
#         for i in range(size.col):
#             for j in range(size.row):
#                 mat[i,j] = self[j,i] 
###################################################  
        return  mat
    
    def is_equal(self, obj):
        return self.size.assert_equal(obj.size) if isinstance(obj, self.__class__) else (self.size, None, False); raise(Exception('wrong types: should be matrix'))
        
    def is_tolerate(self, obj):
        return self.size.assert_tolerate(obj.size) if isinstance(obj, self.__class__) else (self.size, None, False); raise(Exception('wrong types: should be matrix'))        
#===============================================================================
# operations between matrix
#===============================================================================
def union(*c, direction='l2r'):
    '''
    Created on 10 Dec, 2014
    
    @author: wangyi, Researcher Associate @ EIRAN, Nanyang Technological University
    
    @email: L.WANG@ntu.edu.sg
    
    @copyright: 2014 www.yiak.co. All rights reserved.
    
    @license: license
    
    @param: 
    
    @decription:
    
    @param: union
    '''

    def routine(left, right, direction):
        
        if   direction == 'l2r':
            if  isinstance(left, matrixArrayLists) and isinstance(right, matrixArrayLists):
                for i in range(0, max(left.size[0], right.size[0])):
                    r = left[i]
                    if   r != None:
                        # see documentation for difference between () and []
                        left(i).extend(right[i])        
                    elif r == None:
                        # do assignment
                        left[i]=right[i]
        elif direction == 'u2d':
            if  isinstance(left, matrixArrayLists) and isinstance(right, matrixArrayLists):
                for i in range(0, max(left.size[1], right.size[1])):
                    # see documentation for difference between () and []
                    left.append(right(i)) 
            
    # create an empty matrix            
    a = matrixArrayLists()
    
    # mian loop
    for b in c:
        routine(a, b, direction)
    # print a for test
    return a

def row(m,i,j):
    temp   = m[i,:]
    m[i,:] = m[j,:]
    m[j,:] = temp

def col(m,i,j):
    temp   = m[:,i]
    m[:,i] = m[:,j]
    m[:,j] = temp                 






import math

from operator import *
# this is one of key features provided by Python3
from functools import reduce


# pypy
# need check wether it is installed
try:
    from statistics import *
except:
    pass
# TO DO PYCUDA IMPLEMENTATION 
    
    
    
    
    
                   
class matrixArrayNum(matrixArrayLists):

    def __init__(self, *args, **hints):
        super(matrixArrayNum, self).__init__(*args, **hints)

    def match(self):
        """
        match will extract the insection of indice and and headers of self and all iteralbles. 
        """
        pass

    def name(self):
        return "matrixArrayNum:"
    
    def map(self, Func, *iterables):
        # do some preprocessing
        
        # match self with *iterables
        map_object = map(Func, self, *iterables)
        try:
            args = self.size.append([m for m in map_object])
        except:
            pass
        # post processing
        return self.__class__(*args)#self.__class__([m for m in map_object])
    
    def add_matrix(self, obj):
        return self.map(add, obj) if all(self.is_equal(obj)) \
            else self.map(lambda v: v + obj)# error will be raised inside
    __add__ = add_matrix
    __radd__ = __add__
    
    def sub_matrix(self, obj):
        return self.map(sub, obj) if all(self.is_equal(obj)) \
            else self.map(lambda v: v - obj)# error will be raised inside
    __sub__ = sub_matrix
    __rsub__ = __sub__
    
    def neg_matrix(self):
        return self.map(neg)
    __neg__ = neg_matrix
    
    # self add module
    # single operant operator
    def iad_matrix(self, other):
        return self.add_matrix(other)
    __iadd__ \
    = iad_matrix

    def dot_in(self, obj):    
        sizel, sizer , flag = self.is_tolerate(obj)

        if  flag == False:
            return self.map(lambda v: v * obj)

        # return numeric value
        if  sizel.row == 1 and sizer.col == 1:
            return reduce(lambda x, y: x + y, 
                      map(lambda k: self[0,k] * obj[k, 0], range(sizel.col)))
#############################################            
#             sum = 0.0
#             for k in range(sizel.col):
#                 sum += self[0,k] * obj[k,0]
#             return sum
#############################################        
        # return matrixArray-series object
        mat = []
        
        for i in range(sizel.row):
            for j in range(sizer.col):
                _sum = reduce(lambda x, y: x + y, map(lambda k: self[i,k] * obj[k,j], range(sizel.col)))
                mat.append(_sum)
        
        return self.__class__(sizel.row, sizer.col, mat)
#################################################
#         mat = self.__class__(sizel.row, sizer.col)        
#         for i in range(sizel.row):
#             for j in range(sizer.col):
#                 sum = 0.0
#                 for k in range(sizel.col):
#                     sum += self[i,k] * obj[k,j]
#                 mat[i,j] = sum                
#         return  mat
#################################################
    
    def dot_in2(self, obj):    
        sizel, sizer , flag = self.is_tolerate(obj)

        if  flag == False:
            return self.map(lambda v: v * obj)

        # return numeric value
        if  sizel.row == 1 and sizer.col == 1:
            sum = 0.0
            for k in range(sizel.col):
                sum += self[0,k] * obj[k,0]
            return sum
        
        # return matrixArray-series object
        mat = self.__class__(sizel.row, sizer.col)
        for i in range(sizel.row):
            for j in range(sizer.col):
                sum = 0.0
                for k in range(sizel.col):
                    sum += self[i,k] * obj[k,j]
                mat[i,j] = sum
                
        return  mat
    # idealy if the two vectors are tolerated to each other we call dot_in. Otherwise, if they have the same size we call dot_out
    __mul__ = dot_in
    __rmul__ = __mul__
    
    def dot_out(self, obj):
        pass
    
    def dot_mix(self, obj):
        pass
    
    def div_matrix(self, obj):
        sizel, sizer , flag = self.is_tolerate(obj)
        
        if flag == False:
            return self.map(lambda v: v / obj)
    __truediv__ = div_matrix
 
    # one way to implement mean of matrix: matrix algebra, this is an example, not recommended
    # because if the matrix a row vector, .mean_vt cannot summerise a result
    def mean_vt(self, index=None):
        result = self[0,:]; size = self.size
        # add rows
        for i in range( 1, size.row ):
            result += self[i,:]
        return result / size.row
  
    def ubds_vt(self, index=None):
        # older method to implement it has been deprecated
        # check size now
        if  index != None:
            return matrixArrayNum.ubds(self[:, index])
        return matrixArrayNum.ubds(self)
        

## These funtions deal with relationship between matrices 
    @staticmethod 
    def  sum(*c, offset=0):
        """
        matrixArrayNum sub over matrice
        @param c: matrice
        """
        return sum(c, offset) if len(c) > 1 else  sum(c[0])

    @staticmethod
    def mean(*c):
        """
        This function will cacualted vector mean alone any axes. Currently it just supports nd matrix or 1d vector. 
        @param c: matrice
        """
        return sum(c)/ len(c) if len(c) > 1 else mean(c[0])

    @staticmethod 
    def ubds(*c):
        """
        
        @param c: matrice
        """
        def ubd(v):
            new_v = [(i-m) **2 for i in v]
            # in case of row or col vectors
            return math.sqrt( sum(new_v) ) / len(new_v)
            
        
        if  len(c) > 1:
            return  [matrixArrayNum.ubds(item) for item in c]
        else:
            # vector, ubds: 1/n * sqrt( sum ( [(v - mean) ** 2 for v in vector] ) )
            # matrix, ubds: [ubds(col) for col in matrix]
            size = c[0].size
            
            if  size.col == 1 or size.row == 1:
                vector = c[0]; m = mean(vector)# hint
                return ubd(vector)
            else:
                matrix = c[0]
                # matrix api does not provides col selector
                return [matrixArrayNum.ubds(matrix[:,i]) for i in range(size.col)]
 
         
class matrixArray(matrixArrayNum):
    
    def __init__(self, *args, **hints):
        super(self.__class__, self).__init__(*args, **hints)
        
    def name(self):
        return "matrixArray:"

# for easy testing purpose            
# a = _TEST_MATRIX_MULTI = matrixArrayNum([
#                          [['000', '001', '002'], ['010', '011', '012'], ['020', '021', '022']],
#                          [['100', '101', '102'], ['110', '111', '112'], ['120', '121', '122']],
#                          [['200', '201', '202'], ['210', '211', '212'], ['220', '221', '222']],
#                          [['300', '301', '302'], ['310', '311', '312'], ['320', '321', '322']]
#                          ])

# b = _TEST_COMPUT = matrixArrayNum(5, 5)
########################## 
# from numpy import  array
# e = _TEST_array  = array([
#                          [['000', '001', '002'], ['010', '011', '012'], ['020', '021', '022']],
#                          [['100', '101', '102'], ['110', '111', '112'], ['120', '121', '122']],
#                          [['200', '201', '202'], ['210', '211', '212'], ['220', '221', '222']],
#                          [['300', '301', '302'], ['310', '311', '312'], ['320', '321', '322']]
#                          ])
###########################
if __name__ == "__main__":
# 2015 5：
#     print(a[:])
#     b = a[:,:,0]
#     print(b)
#     b = matrixArrayNum([[1,2,3,],[4,5,6]])
#     b[0, 5] = 100
#     print(b)
#     b[5] = 100
#     print(b)
#     a = matrixArrayLists()
#      
#     a[5] = 100
#      
#     print(a)
#      
#     print(a.trp())
#     
#     
#     b.setheader(['header_1', 'header_2'])
#     print(b)
#     b[[0,5],[0,5]]
#     print(b)
#     b.setIndice(0)#'header_1'
#     c = b.trp() * b
#     b[0,10] = -1
#     b.clear()
#     b[5] = 100
#     print(b)
#     var = input("please input...\n")
#     print(var)
#     pass
# 2015 4:
#     print(matrixArrayNum([[1,2],[3,4],[5,6]]).mean_vt())
#     print(matrixArrayNum([[1,2],[3,4],[5,6]]).ubds_vt())
    import  random
    import  time
    b = matrixArrayLists([1,[[1]]])
    b[1]
    #print(b)
    d = b.inner_rep
     
    start = time.time()   
     
    for i in range(100):
        for j in range(50):
            #c = b[i,j]
            b[i,j] = random.randrange(1, 1000)
            #d[i][j] = random.randrange(1, 1000)
            # print( b )
    elpse = time.time() - start
    print(elpse)
     
    start = time.time()
     
    for i in range(100):
        for j in range(50):
            b[i,j]#b._shape_array
    elpse = time.time() - start
    print(elpse)
     
    start = time.time()
     
    for i in range(100):
        for j in range(50):
            size = b._shape_array#size = [100, 50]# size = b._shape_array
            # middleware preprocessing       
            hint = index_len((i,j), size)
            # middleware preprocessing 
            key  = index_val((i,j), size)
            # get inner representation of the query object
            root = b.inner_rep#b[i,j]#b._shape_array
             
            slot = b.getitem_multi(key, root)
             
            max(hint) <= 1; len(hint) == len(size);
    elpse = time.time() - start
    print(elpse)
    
    l = []
    
    start = time.time()
    
    for i in range(100):
        m = []
        for j in range(50):
            m.append(None)
        l.append(m)
        
    elpse = time.time() - start
    print(elpse)
    
    start = time.time()
    
    for i in range(100):
        for j in range(50):
            l[i][j]
        
    elpse = time.time() - start
    print(elpse)
# 2015 3:
#     a = matrixArrayNum([1,1]);print(a)
#     b = matrixArrayNum([1,1])
#     c = matrixArrayNum([1,1])
#     
#     b + 0 
#     print(isum(a,b,c))
#     ubds(a)
#     
#     a = a / 2.0
#     print(a)
    
# 2015 3, middleware has been removed into another package as an independent work:
#
#     b = matrixArrayNum([[1,2],[3,4]])
#     b.setHeader(['time','power'])
#     b.setIndice(['1','2'])
#     
#     c = matrixArray([[1,2],[3,4]])
#     c.trp()
#     print(c[0])
# 
#     c.addh(checkRIndex, matrixArrayLists.__getitem__)
#     c.setHeader(['time','power'])
#     c.setIndice(['1','2'])
#     
#     print(a[[1,2],[1,2],[1,2]])
#     print(a[[1,2],[1,2],0])
     
