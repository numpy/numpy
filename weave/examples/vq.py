""" 
"""
# C:\home\ej\wrk\scipy\weave\examples>python vq.py
# vq with 1000 observation, 10 features and 30 codes fo 100 iterations
#  speed in python: 0.150119999647
# [25 29] [ 2.49147266  3.83021032]
#  speed in standard c: 0.00710999965668
# [25 29] [ 2.49147266  3.83021032]
#  speed up: 21.11
#  speed inline/blitz: 0.0186300003529
# [25 29] [ 2.49147272  3.83021021]
#  speed up: 8.06
#  speed inline/blitz2: 0.00461000084877
# [25 29] [ 2.49147272  3.83021021]
#  speed up: 32.56
 
import Numeric
from Numeric import *
import sys
sys.path.insert(0,'..')
import inline_tools
import converters
blitz_type_converters = converters.blitz
import c_spec

def vq(obs,code_book):
    # make sure we're looking at arrays.
    obs = asarray(obs)
    code_book = asarray(code_book)
    # check for 2d arrays and compatible sizes.
    obs_sh = shape(obs)
    code_book_sh = shape(code_book)
    assert(len(obs_sh) == 2 and len(code_book_sh) == 2)   
    assert(obs_sh[1] == code_book_sh[1])   
    type = c_spec.num_to_c_types[obs.typecode()]
    # band aid for now.
    ar_type = 'PyArray_FLOAT'
    code =  """
            #line 37 "vq.py"
            // Use tensor notation.            
            blitz::Array<%(type)s,2> dist_sq(Ncode_book[0],Nobs[0]);
 	        blitz::firstIndex i;    
            blitz::secondIndex j;   
            blitz::thirdIndex k;
            dist_sq = sum(pow2(obs(j,k) - code_book(i,k)),k);
            // Surely there is a better way to do this...
            PyArrayObject* py_code = (PyArrayObject*) PyArray_FromDims(1,&Nobs[0],PyArray_LONG);
 	        blitz::Array<int,1> code((int*)(py_code->data),
                                     blitz::shape(Nobs[0]), blitz::neverDeleteData);
 	        code = minIndex(dist_sq(j,i),j);
 	        
 	        PyArrayObject* py_min_dist = (PyArrayObject*) PyArray_FromDims(1,&Nobs[0],PyArray_FLOAT);
 	        blitz::Array<float,1> min_dist((float*)(py_min_dist->data),
 	                                       blitz::shape(Nobs[0]), blitz::neverDeleteData);
 	        min_dist = sqrt(min(dist_sq(j,i),j));
 	        py::tuple results(2);
 	        results[0] = py_code;
 	        results[1] = py_min_dist;
 	        return_val = results; 	        
            """ % locals()
    code, distortion = inline_tools.inline(code,['obs','code_book'],
                                           type_converters = blitz_type_converters,
                                           compiler = 'gcc',
                                           verbose = 1)
    return code, distortion

def vq2(obs,code_book):
    """ doesn't use blitz (except in conversion)
        ALSO DOES NOT HANDLE STRIDED ARRAYS CORRECTLY
    """
    # make sure we're looking at arrays.
    obs = asarray(obs)
    code_book = asarray(code_book)
    # check for 2d arrays and compatible sizes.
    obs_sh = shape(obs)
    code_book_sh = shape(code_book)
    assert(len(obs_sh) == 2 and len(code_book_sh) == 2)   
    assert(obs_sh[1] == code_book_sh[1])   
    assert(obs.typecode() == code_book.typecode())   
    type = c_spec.num_to_c_types[obs.typecode()]
    # band aid for now.
    ar_type = 'PyArray_FLOAT'
    code =  """
            #line 83 "vq.py"
            // THIS DOES NOT HANDLE STRIDED ARRAYS CORRECTLY
            // Surely there is a better way to do this...
            PyArrayObject* py_code = (PyArrayObject*) PyArray_FromDims(1,&Nobs[0],PyArray_LONG);	        
 	        PyArrayObject* py_min_dist = (PyArrayObject*) PyArray_FromDims(1,&Nobs[0],PyArray_FLOAT);
 	        
            int* raw_code = (int*)(py_code->data);
            float* raw_min_dist = (float*)(py_min_dist->data);
            %(type)s* raw_obs = obs.data();
            %(type)s* raw_code_book = code_book.data(); 
            %(type)s* this_obs = NULL;
            %(type)s* this_code = NULL; 
            int Nfeatures = Nobs[1];
            float diff,dist;
            for(int i=0; i < Nobs[0]; i++)
            {
                this_obs = &raw_obs[i*Nfeatures];
                raw_min_dist[i] = (%(type)s)10000000.; // big number
                for(int j=0; j < Ncode_book[0]; j++)
                {
                    this_code = &raw_code_book[j*Nfeatures];
                    dist = 0;
                    for(int k=0; k < Nfeatures; k++)
                    {
                        diff = this_obs[k] - this_code[k];
                        dist +=  diff*diff;
                    }
                    dist = dist;
                    if (dist < raw_min_dist[i])
                    {
                        raw_code[i] = j;
                        raw_min_dist[i] = dist;                           
                    }    
                }
                raw_min_dist[i] = sqrt(raw_min_dist[i]);
 	        }
 	        py::tuple results(2);
 	        results[0] = py_code;
 	        results[1] = py_min_dist;
 	        return_val = results; 	        
            """ % locals()
    code, distortion = inline_tools.inline(code,['obs','code_book'],
                                         type_converters = blitz_type_converters,
                                         compiler = 'gcc',
                                         verbose = 1)
    return code, distortion


def vq3(obs,code_book):
    """ Uses standard array conversion completely bi-passing blitz.
        THIS DOES NOT HANDLE STRIDED ARRAYS CORRECTLY
    """
    # make sure we're looking at arrays.
    obs = asarray(obs)
    code_book = asarray(code_book)
    # check for 2d arrays and compatible sizes.
    obs_sh = shape(obs)
    code_book_sh = shape(code_book)
    assert(len(obs_sh) == 2 and len(code_book_sh) == 2)   
    assert(obs_sh[1] == code_book_sh[1])   
    assert(obs.typecode() == code_book.typecode())   
    type = c_spec.num_to_c_types[obs.typecode()]
    code =  """
            #line 139 "vq.py"
            // Surely there is a better way to do this...
            PyArrayObject* py_code = (PyArrayObject*) PyArray_FromDims(1,&Nobs[0],PyArray_LONG);	        
 	        PyArrayObject* py_min_dist = (PyArrayObject*) PyArray_FromDims(1,&Nobs[0],PyArray_FLOAT);
 	        
            int* code_data = (int*)(py_code->data);
            float* min_dist_data = (float*)(py_min_dist->data);
            %(type)s* this_obs = NULL;
            %(type)s* this_code = NULL; 
            int Nfeatures = Nobs[1];
            float diff,dist;

            for(int i=0; i < Nobs[0]; i++)
            {
                this_obs = &obs_data[i*Nfeatures];
                min_dist_data[i] = (float)10000000.; // big number
                for(int j=0; j < Ncode_book[0]; j++)
                {
                    this_code = &code_book_data[j*Nfeatures];
                    dist = 0;
                    for(int k=0; k < Nfeatures; k++)
                    {
                        diff = this_obs[k] - this_code[k];
                        dist +=  diff*diff;
                    }
                    if (dist < min_dist_data[i])
                    {
                        code_data[i] = j;
                        min_dist_data[i] = dist;                           
                    }    
                }
                min_dist_data[i] = sqrt(min_dist_data[i]);
 	        }
 	        py::tuple results(2);
 	        results[0] = py_code;
 	        results[1] = py_min_dist;
 	        return_val = results; 	        
            """ % locals()
    # this is an unpleasant way to specify type factories -- work on it.
    import ext_tools
    code, distortion = inline_tools.inline(code,['obs','code_book'])
    return code, distortion

import time
import RandomArray
def compare(m,Nobs,Ncodes,Nfeatures):
    obs = RandomArray.normal(0.,1.,(Nobs,Nfeatures))
    codes = RandomArray.normal(0.,1.,(Ncodes,Nfeatures))
    import scipy.cluster.vq
    scipy.cluster.vq
    print 'vq with %d observation, %d features and %d codes for %d iterations' % \
           (Nobs,Nfeatures,Ncodes,m)
    t1 = time.time()
    for i in range(m):
        code,dist = scipy.cluster.vq.py_vq(obs,codes)
    t2 = time.time()
    py = (t2-t1)
    print ' speed in python:', (t2 - t1)/m
    print code[:2],dist[:2]
    
    t1 = time.time()
    for i in range(m):
        code,dist = scipy.cluster.vq.vq(obs,codes)
    t2 = time.time()
    print ' speed in standard c:', (t2 - t1)/m
    print code[:2],dist[:2]
    print ' speed up: %3.2f' % (py/(t2-t1))
    
    # load into cache    
    b = vq(obs,codes)
    t1 = time.time()
    for i in range(m):
        code,dist = vq(obs,codes)
    t2 = time.time()
    print ' speed inline/blitz:',(t2 - t1)/ m    
    print code[:2],dist[:2]
    print ' speed up: %3.2f' % (py/(t2-t1))

    # load into cache    
    b = vq2(obs,codes)
    t1 = time.time()
    for i in range(m):
        code,dist = vq2(obs,codes)
    t2 = time.time()
    print ' speed inline/blitz2:',(t2 - t1)/ m    
    print code[:2],dist[:2]
    print ' speed up: %3.2f' % (py/(t2-t1))

    # load into cache    
    b = vq3(obs,codes)
    t1 = time.time()
    for i in range(m):
        code,dist = vq3(obs,codes)
    t2 = time.time()
    print ' speed using C arrays:',(t2 - t1)/ m    
    print code[:2],dist[:2]
    print ' speed up: %3.2f' % (py/(t2-t1))
    
if __name__ == "__main__":
    compare(100,1000,30,10)    
    #compare(1,10,2,10)    
