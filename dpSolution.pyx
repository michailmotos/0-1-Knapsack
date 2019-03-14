import numpy as np
cimport numpy as np
cimport cython


# We now need to fix a datatype for our arrays. I've used the variable
# DTYPE for this, which is assigned to the usual NumPy runtime
# type info object.
DTYPE = np.float

# "ctypedef" assigns a corresponding compile-time type to DTYPE_t. For
# every type in the numpy module there's a corresponding compile-time
# type with a _t-suffix.
ctypedef np.float_t DTYPE_t


#Uncommenting these improves performance but turns off important things such as bounds checking and negative index wrapping
#@cython.boundscheck(False) # turn off bounds-checking
#@cython.wraparound(False)  # turn off negative index wrapping 
def solve_cython(int distSize,int itemSize,np.ndarray[DTYPE_t, ndim=1] weightDist,np.ndarray[DTYPE_t, ndim=1] utilityDist):
    assert weightDist.dtype == DTYPE and utilityDist.dtype == DTYPE
    cdef np.ndarray[DTYPE_t, ndim=2] result = np.zeros([itemSize+1, distSize+1], dtype=DTYPE)
    cdef int item,k,idx
    for item in range(1,itemSize + 1):
        idx = <int>weightDist[item]
        result[item][:idx] = result[item-1][:idx]
        for k in range(idx,distSize + 1):
            if weightDist[item] <= k:
                result[item][k] = max(result[item-1][k], result[item-1][k - idx] + utilityDist[item]) #result[item][k] = max(result[item-1][k], result[item-1][k - idx] + utilityDist[item])
            else:
                result[item][k] = result[item-1][k]
    return result[itemSize][distSize]
