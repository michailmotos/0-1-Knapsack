import numpy as np
#import matplotlib.pyplot as plt
#from scipy.stats import zipf
#from scipy.stats import geom
#from scipy.stats import pareto
import math
import random
from math import e
import os
import time
import multiprocessing as mp
#import pyximport; pyximport.install()


#import winsound
def heuristic(weight_dist,probability_dist,dist_size):
	''' Simple heuristic function
	using a utility to weight ratio
	'''
	output = [0 for i in range(dist_size)]
	for i in range(dist_size):
		#print("prob: "+ str(probability_dist[i])+ "weight: "+str(weight_dist[i]))
		output[i] = probability_dist[i]/float(weight_dist[i])
	return output
'''
def knapsack(weight_dist,probability_dist,distSize,cache_size):
	 Function that uses the Dynamic Programming approach, to solve the 0-1 Knapsack Problem

	pinakas = [[0 for i in range(cache_size+1)] for w in range(distSize+1)]
	pinakas = np.array(pinakas).astype(float)
	for i in range(0,distSize+1):
		for w in range(0,cache_size+1):
			if weight_dist[i] <= w:
				#print("weight_dist[i]: "+str(weight_dist[i])+" w is: "+str(w)+" ")
				#print("pinakas[i-1][w]: "+str(pinakas[i-1][w])+" probability_dist[i] is: "+str(probability_dist[i])+" ")
				pinakas[i][w] = max(pinakas[i-1][w], probability_dist[i] + pinakas[i-1][(w-math.floor(weight_dist[i]))])
			else:
				pinakas[i][w] = pinakas[i-1][w]
	return pinakas[distSize][cache_size]
'''

def knapsack(weight_dist,probability_dist,distSize,cacheSize):
	''' Function that uses the Dynamic Programming approach, to solve the 0-1 Knapsack Problem
	'''
	#pinakas = [[0 for i in range(cacheSize+1)] for w in range(distSize+1)]
	#pinakas = np.array(pinakas).astype(float)
	pinakas = np.zeros((distSize + 1, cacheSize + 1), dtype=float)
	for i in range(0,distSize):
		this_weight = math.ceil(weight_dist[i])
		floor_weight = math.floor(weight_dist[i])
		this_value = probability_dist[i]
			#if weight_dist[i] <= w:
				#print("weight_dist[i]: "+str(weight_dist[i])+" w is: "+str(w)+" ")
				#print("pinakas[i-1][w]: "+str(pinakas[i-1][w])+" probability_dist[i] is: "+str(probability_dist[i])+" ")
			#pinakas[i][w] = max(pinakas[i-1][w], probability_dist[i] + pinakas[i-1][(w-math.floor(weight_dist[i]))])
		#pinakas[i+1][math.ceil(weight_dist[i]):] = list(map(lambda k: max(pinakas[i][k], pinakas[i][k - this_weight] + this_value), range(this_weight, cacheSize+1)))
		#pinakas[i+1][this_weight:] = [max(pinakas[i][k], pinakas[i][k - this_weight] + this_value) for k in range(this_weight, cacheSize+1)]
		pinakas[i+1, :this_weight] = pinakas[i, :this_weight]
		print("pinakas[i+1][:floor_weight]")
		print(pinakas[i+1][:this_weight])
		temp = pinakas[i,:-this_weight] + this_value
		print("this_weight")
		print(this_weight)
		print("temp")
		print(temp)
		print("pinakas[i,:-this_weight]")
		print(pinakas[i,:i-this_weight])
		pinakas[i+1,this_weight:] = np.where(temp > pinakas[i,this_weight:],temp,pinakas[i,this_weight:])
		print("pinakas[i,this_weight:]")
		print(pinakas[i,this_weight:])
		print(pinakas)
			#else:
			#	pinakas[i][w] = pinakas[i-1][w]
	return pinakas[distSize][cacheSize]

def oldKnapsack(weight_dist,probability_dist,distSize,cache_size):
	''' Function that uses the Dynamic Programming approach, to solve the 0-1 Knapsack Problem
	'''
	pinakas = [[0 for i in range(cache_size+1)] for w in range(distSize+1)]
	pinakas = np.array(pinakas).astype(float)
	for i in range(0,distSize+1):
		for w in range(0,cache_size+1):
			if weight_dist[i] <= w:
				#print("weight_dist[i]: "+str(weight_dist[i])+" w is: "+str(w)+" ")
				#print("pinakas[i-1][w]: "+str(pinakas[i-1][w])+" probability_dist[i] is: "+str(probability_dist[i])+" ")
				pinakas[i][w] = max(pinakas[i-1][w], probability_dist[i] + pinakas[i-1][(w-math.floor(weight_dist[i]))])
			else:
				pinakas[i][w] = pinakas[i-1][w]
	print(pinakas)
	return pinakas[distSize][cache_size]


def min_knapsack(weight_dist,probability_dist,distSize,cacheSize):
	''' Function that uses the Dynamic Programming approach, to solve the 0-1 Knapsack Problem
	'''
	pinakas = np.zeros((2, cacheSize + 1), dtype=float)
	#pinakas = np.array(pinakas).astype(float)
	for i in range(0,distSize):
		#for w in range(0,cache_size+1):
		this_weight = int(np.ceil(weight_dist[i]))
		this_value = probability_dist[i]
		#	if weight_dist[i] <= w:
				#print("weight_dist[i]: "+str(weight_dist[i])+" w is: "+str(w)+" ")
				#print("pinakas[i-1][w]: "+str(pinakas[i-1][w])+" probability_dist[i] is: "+str(probability_dist[i])+" ")
				#pinakas[i%2][w] = max(pinakas[(i-1)%2][w], probability_dist[i] + pinakas[(i-1)%2][(w-math.floor(weight_dist[i]))])
		pinakas[(i+1)%2, :this_weight] = pinakas[i%2, :this_weight]
		temp = pinakas[i%2,:-this_weight] + this_value
		pinakas[(i+1)%2,this_weight:] = np.where(temp > pinakas[i%2,this_weight:],temp,pinakas[i%2,this_weight:])
		#	else:
		#		pinakas[i%2][w] = pinakas[(i-1)%2][w]
		#if w == cache_size:
		#	print(pinakas[i%2])
	#print("Knapsack CHR:" +str(pinakas[i%2][cache_size]))
	return pinakas[i%2][cacheSize]

def traceKnapsack(pinakas,weight_dist,probability_dist,dist_size,cache_size):
	''' Backtracking on the Knapsack table to find the items on the list
	'''
	comparator = pinakas[dist_size][cache_size]
	included = [-1 for i in range(dist_size)]
	included = np.array(included).astype(float)
	#compatator_i = dist_size
	w = cache_size
	for i in range(dist_size,0,-1):
		#print("Comparator: "+str(comparator)+ " pinakas[i][w]: "+str(pinakas[i][w]))
		if comparator == pinakas[i-1][w]:
			continue
		else:
			included[i] = i
			print("Item: "+str(i)+ " is included, with weight: "+str(weight_dist[i])+" and probability: " +str(probability_dist[i]))
			comparator = comparator - probability_dist[i]
			w = w - weight_dist[i]
	#return i
	#for line in pinakas:
	#	print(line)

def expandArray(array):
	'''
	Array shift right
	'''
	temp = [0 for i in range(len(array)+1)]
	temp = np.array(temp).astype(float)
	for i in range(1,len(array)+1):
		temp[i] = array[i-1]
	return temp

def calcHitRatio(array,cache_size,weight_dist,probability_dist,distSize):
	'''
	Calculates the CHR using as input the utility to weight heuristic results
	'''
	#included = [-1 for i in range(distSize)]
	temp = cache_size
	CHR = 0
	for i in reversed(range(distSize)):
		if temp >= weight_dist[array[i][0]]:
			#print("array 0:"+str(array[i][0]))
			#print("array 1:"+str(array[i][1]))
			#included[i] = array[i][0]+1
			#print("Item: "+str(array[i][0]+1)+" is included,with weight: "+str(weight_dist[array[i][0]])+" and probability: "+str(probability_dist[array[i][0]]))
			temp = temp - weight_dist[array[i][0]]
			CHR = CHR + float(probability_dist[array[i][0]])

	#print("Greedy Approximation: "+str(CHR))
	#return included
	return CHR

def geometricDist(distSize,p):
	'''
	Constructs a geometric Distribution sample using Inverse Transform Sampling method, size = distSize & p = geometric distribution parameter
	'''
	geo_dist = [0 for i in range(distSize)]
	geo_dist = np.array(geo_dist).astype(float)
	#uniform = np.random.uniform(low = 0.0, high = 1.0, size = distSize)
	uniform = np.random.uniform(0, 1, distSize)
	#print(uniform)
	#os.system("pause")

	geo_dist = np.ceil(np.log(1.0-uniform)/np.log(1.0-p))
	#print("mean:"+str(geo_dist.mean()))
	#print("var:"+str(geo_dist.var()))
	#plt.plot(z,geo_dist)
	#plt.show()
	#geo_var = (1/p)*((1/p) - 1)
	#geo_std = math.sqrt(geo_var)
	#mean = 1/p
	#print("mean: "+str(mean)+ " variance: " +str(geo_var)+ " standard deviation: " +str(geo_std))
	return geo_dist


def zipfDist(distSize,s):
	'''
	Constructs a Zipfian Distribution sample
	'''
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
	'''
	Constructs a Pareto Distribution
	'''
	uniform = np.random.uniform(0, 1, distSize)
	cdf = [0 for i in range(distSize)]
	cdf = np.empty(distSize,dtype=float)
	#cdf = np.array(cdf).astype(float)
	for i in range(distSize):
		cdf[i] = 1/pow(e,np.log(uniform[i])/b)

	return cdf




def fullrun(distSize,cacheSize):
	loops = 1
	knapLoops = 1
	knapLoops2 = knapLoops
	knapCHR = [0 for i in range(knapLoops)]
	greedCHR = [0 for i in range(loops)]
	greedTime = [0 for i in range(loops)]
	knapTime = [0 for i in range(knapLoops)]
	start3 = time.time()
	for i in range(0,loops):
		#utilDist = np.empty(distSize,dtype = float)
		#utilDist = zipfDist(distSize,0.2)
		utilDist = [100,20,60,40]
		#weightDist = geometricDist(distSize,0.8)
		#weightDist = np.empty(distSize,dtype = float)
		#weightDist = paretoDist(distSize,1.16)
		weightDist = [3.5,2.5,3.5,1.5]
		#utilDist = np.array(utilDist).astype(float)
		#weightDist = np.array(weightDist).astype(float)

		start1 = time.time()
		heur_output = [0 for i in range(distSize)]
		heur_output = np.array(heur_output).astype(float)
		heur_output = heuristic(weightDist,utilDist,distSize)
		heur_output = list(enumerate(heur_output))
		heur_output = sorted(heur_output,key=lambda x: x[-1])
		greedCHR[i] = calcHitRatio(heur_output,cacheSize,weightDist,utilDist,distSize)
		greedTime[i] = time.time() - start1
		start2 = time.time()
		knapCHR[i] = oldKnapsack(expandArray(weightDist),expandArray(utilDist),distSize,cacheSize)
		knapTime[i] = time.time() - start2

	#strs = ["" for x in range(8)]
	#strs[0] = str(np.mean(greedCHR))
	#strs[1] = str(round(sum(greedTime)/loops,4))
	print("cache size:"+str(cacheSize)+"\n")
	print("Greedy CHR: "+str(np.mean(greedCHR)))
	print("Greedy Algorithm Average Execution Time: "+str(round(sum(greedTime)/loops,4)))
	print("Knapsack CHR: "+str(np.mean(knapCHR)))
	#strs[2] = str(np.mean(knapCHR))
	#strs[3] = str(round(sum(knapTime)/knapLoops2,4))
	#strs[4] = str(distSize)
	#strs[5] = str(cacheSize)
	#strs[6] = str(loops)
	#strs[7] = str(knapLoops)
	print("Knapsack Algorithm Average Execution Time: "+str(round(sum(knapTime)/knapLoops2,4)))
	print("[Finished in "+str(time.time()-start3)+"s]")
	print("\n")
	#print(strs)
	final = "\nDistribution size: " +str(distSize)+ " , cache size = " +str(cacheSize)+"\nGr. Loops: "+str(loops)+" Kn. Loops: "+str(knapLoops2)+"\nGreedy CHR: "+str(np.mean(greedCHR))+" Gr. Avg. Runtime: "+str(round(sum(greedTime)/loops,4))+"\nKnapsack CHR: "+str(np.mean(knapCHR))+" Knap Avg. Runtime: "+str(round(sum(knapTime)/knapLoops2,4))+"\nFinished in: "+str(time.time()-start3)+"\n"
	#print(final)
	text_file = open(str(distSize)+"par116_1k.txt", "a")
	text_file.write(final)
	text_file.close()
	#os.system("scp mmotos@ui.grid.tuc.gr:1kp08.txt c:\\Users\laptop\Desktop\results")

#duration = 1000  # millisecond
#freq = 440  # Hz
#winsound.Beep(freq, duration)


distSize = 4
cacheSize = [5]
if __name__ == '__main__':
	for x in range(len(cacheSize)):
		p = mp.Process(target=fullrun, args=(distSize,cacheSize[x]))
		p.start()
