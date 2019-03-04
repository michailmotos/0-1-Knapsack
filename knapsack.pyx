import numpy as np
cimport numpy as np
import math

#ctypedef np.float_t DTYPE_t


def solve_cython(int capacity,int items,weights,values):
    #cdef np.ndarray[long, ndim=2] grid = np.empty((items + 1, capacity + 1), dtype=long)
    grid = [[float(0) for i in range(capacity+1)] for w in range(items+1)]
    grid = np.array(grid).astype(float)

    for item in range(items+1):
        this_weight = math.floor(weights[item])
        this_value = values[item]
        #grid[item + 1, :this_weight] = grid[item, :this_weight]

        for k in range(0, capacity + 1):
            if weights[item] <= k:
                grid[item][k] = max(grid[item-1][k], grid[item-1][k - this_weight] + this_value)
            else:
                grid[item][k] = grid[item-1][k]
    #print(grid)
    return grid[items, capacity]
