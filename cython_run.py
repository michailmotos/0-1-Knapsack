import pyximport; pyximport.install()
import dpSolution
import numpy as np
import math
import random
from math import e
import os
import time
import greedy

#distSize = # of elements
#cacheSize = max Knapsack weight
distSize = 4
cacheSize = 5


def geometricDist(distSize,p):
	geo_dist = [0 for i in range(distSize)]
	geo_dist = np.array(geo_dist).astype(int)
	uniform = np.random.uniform(0, 1, distSize)

	geo_dist = np.ceil(np.log(1.0-uniform)/np.log(1.0-p))
	return geo_dist

def zipfDist(distSize,s):
    tmp = 0
    for i in range(1,distSize+1):	#(1,V+1)
        tmp = tmp + 1/pow(i,1-s)
    zipf_probability = [0 for i in range(distSize)]
    zipf_probability = np.empty(distSize,dtype=float)
    c = 1/tmp
    for i in range(0,distSize):
        zipf_probability[i] = c/pow(i+1,1-s)
    #print(zipf_probability)
    return zipf_probability

def paretoDist(distSize,b):
    uniform = np.random.uniform(0, 1, distSize)
    #cdf = [0 for i in range(distSize)]
    cdf = np.empty(distSize,dtype=float)
    #cdf = np.array(cdf).astype(float)
    for i in range(distSize):
        cdf[i] = 1/pow(e,np.log(uniform[i])/b)

    return cdf

def expandArray(array):
    temp = np.concatenate(([0], array))
    return temp



#weightDist = geometricDist(distSize,0.5)
#print(weightDist)
#print(weightDist)
#utilDist = zipfDist(distSize,0.2)

#print(utilDist)
weightDist = [3,2,4,1]
utilDist = [100,20,60,40]
weightDist = np.array(weightDist,dtype = float)
utilDist = np.array(utilDist,dtype = float)
dynamicStart = time.time()
greed = greedy.greedyCHR(cacheSize,distSize,weightDist,utilDist)
a = final.dynamicProgCHR(cacheSize,distSize,expandArray(weightDist),expandArray(utilDist))
finish = time.time() - dynamicStart
print(greed,a,finish)
