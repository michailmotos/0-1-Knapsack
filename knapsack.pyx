import numpy as np
cimport numpy as np
import math



def solve_cython(int capacity,int itemSize,weightDist,utilityDist):
    result = [[float(0) for i in range(capacity+1)] for w in range(itemSize+1)]
    result = np.array(result).astype(float)

    for item in range(itemSize+1):
        currWeight = math.floor(weights[item])
        currValue = utilityDist[item]

        for k in range(0, capacity + 1):
            if weightDist[item] <= k:
                result[item][k] = max(result[item-1][k], result[item-1][k - currWeight] + currValue)
            else:
                result[item][k] = result[item-1][k]
    #print(grid)
    return result[itemSize, capacity]
