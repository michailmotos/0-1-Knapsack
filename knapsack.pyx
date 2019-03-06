import numpy as np
cimport numpy as np
#import math
#from libc.math cimport floor
cimport cython
#from libc.math cimport fmaxf


# We now need to fix a datatype for our arrays. I've used the variable
# DTYPE for this, which is assigned to the usual NumPy runtime
# type info object.
DTYPE = np.long

# "ctypedef" assigns a corresponding compile-time type to DTYPE_t. For
# every type in the numpy module there's a corresponding compile-time
# type with a _t-suffix.
ctypedef np.long_t DTYPE_t

#cdef extern from "math.h":
#    float fmaxf(float x,float y)
#cdef int floorFunc(float target):
#    cdef int temp;
#    temp = floor(target);
#    return temp;
#@cython.boundscheck(False) # turn off bounds-checking for entire function
#@cython.wraparound(False)  # turn off negative index wrapping for entire function
def solve_cython(int capacity,int itemSize,np.ndarray[DTYPE_t, ndim=1] weightDist,np.ndarray[DTYPE_t, ndim=1] utilityDist):
    assert weightDist.dtype == DTYPE and utilityDist.dtype == DTYPE
    #result = [[float(0) for i in range(capacity+1)] for w in range(itemSize+1)]
    #result = np.array(result).astype(float)
    cdef np.ndarray[DTYPE_t, ndim=2] result = np.zeros([itemSize+1, capacity+1], dtype=DTYPE)
    cdef int item,k,idx
    for item in range(1,itemSize+1):
        idx = weightDist[item]
        result[item][:idx] = result[item-1][:idx]
        for k in range(idx,capacity + 1):
            if weightDist[item] <= k:
                #idx = <int>weightDist[item]
                result[item][k] = max(result[item-1][k], result[item-1][k - idx] + utilityDist[item]) #result[item][k] = max(result[item-1][k], result[item-1][k - idx] + utilityDist[item])
            else:
                result[item][k] = result[item-1][k]
    #print(result)
    return result[itemSize, capacity]
