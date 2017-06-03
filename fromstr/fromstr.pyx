from libc.stdlib cimport malloc, free

cimport numpy as np
import numpy as np

cdef extern from 'stdlib.h':
    double atof(char*)


cdef extern from 'string.h':
    char* strtok(char*, char*)
    char* strcpy(char*, char*)
    void* memcpy(void*, void*, size_t)


def fromstring(s):
    data = np.fromstring(s, sep=" ", dtype=float)
    return data


def split_and_array(s):
    rawdata = s.split()
    data = np.array(rawdata, dtype=float)
    return data


def comp_and_array(s):
    rawdata = [float(d) for d in s.split()]
    data = np.array(rawdata, dtype=float)
    return data


def split_atof(s):
    # As per the following thread: 
    # http://comments.gmane.org/gmane.comp.python.numeric.general/41504
    cdef char* cstring = ""
    cdef int i, I
    cdef np.ndarray[np.float64_t, ndim=1] cdata

    rawdata = s.split()
    I = len(rawdata)
    data = np.empty(I, dtype=np.float64)
    cdata = data

    for i from 0 <= i < I:
        cstring = rawdata[i]
        cdata[i] = atof(cstring)

    return data        


def token_atof(s):
    cdef char* cstring
    cdef char* cs 
    cdef int i, I
    cdef np.ndarray[np.float64_t, ndim=1] cdata

    I = len(s)
    cs = <char *> malloc(I * sizeof(char))
    strcpy(cs, s)

    I = (I / 2) + 1
    data = np.empty(I, dtype=np.float64)
    cdata = data

    i = 0
    cstring = strtok(cs, " ")
    while cstring != NULL:
        cdata[i] = atof(cstring)
        cstring = strtok(NULL, " ")
        i += 1

    free(cs)

    data = data[:i].copy()
    return data        
