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
@cython.cdivision(True)

def greedyCHR(int distSize,int itemSize,np.ndarray[DTYPE_t, ndim=1] weightDist,np.ndarray[DTYPE_t, ndim=1] utilityDist):
    assert weightDist.dtype == DTYPE and utilityDist.dtype == DTYPE
    heuristic = []
    cdef int x
    cdef float CHR,temp
    temp = itemSize
    for x in range(distSize-1):
        heuristic.append(utilityDist[x]/weightDist[x])

    heuristic = list(enumerate(heuristic))
    heuristic = sorted(heuristic,key=lambda x: x[-1])

    for x in reversed(range(distSize-1)):
        if temp >= weightDist[<int>heuristic[x][0]]:
            temp = temp - weightDist[<int>heuristic[x][0]]
            CHR = CHR + utilityDist[<int>heuristic[x][0]]
    return CHR
