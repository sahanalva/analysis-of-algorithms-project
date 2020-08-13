#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 26 19:05:30 2019

@author: sahanalva
"""

from functions_util import *
from time import process_time
import numpy as np


avg_degree_for_sparse = 6
adjacency_fraction_for_dense = 0.2 
num_vertices = 5000
max_weight = 1000



results = {
    "Dijkstra without Heap" : {"Sparse" : [], "Dense" : []}, 
    "Dijkstra with Heap" : {"Sparse" : [], "Dense" : []}, 
    "Kruskals" : {"Sparse" : [], "Dense" : []},
    "Kruskals (using python sort)" : {"Sparse" : [], "Dense" : []}
    }

tests = []

for i in range(5):
    tests.append(("\nExperiment : {}".format(i + 1),""))
    
    g = Graph(num_vertices)
    
    g.createSparseGraph(avg_degree_for_sparse, max_weight)
    for j in range(5):
        s = np.random.randint(0,num_vertices)
        t = s
        while( t == s):

            t = np.random.randint(0,num_vertices)

        tests.append((s,t))


        start_time = process_time() 
        max_bandwidth = g.dijkstrasMaxBandwidthPath(s, t, False)
        elapsed_time = process_time() - start_time
        results["Dijkstra without Heap"]["Sparse"].append((max_bandwidth,elapsed_time))

        start_time = process_time() 
        max_bandwidth = g.dijkstrasMaxBandwidthPath(s, t, True)
        elapsed_time = process_time() - start_time
        results["Dijkstra with Heap"]["Sparse"].append((max_bandwidth,elapsed_time))

        start_time = process_time() 
        max_bandwidth = g.kruskalsMaxBandwidthPath(s, t)
        elapsed_time = process_time() - start_time
        results["Kruskals"]["Sparse"].append((max_bandwidth,elapsed_time))

        start_time = process_time() 
        max_bandwidth = g.kruskalsMaxBandwidthPathwithoutHeap(s, t)
        elapsed_time = process_time() - start_time
        results["Kruskals (using python sort)"]["Sparse"].append((max_bandwidth,elapsed_time))

      
    g.createDenseGraph(adjacency_fraction_for_dense,max_weight)
    for j in range(5):
        s = np.random.randint(0,num_vertices)
        t = s
        while( t == s):

            t = np.random.randint(0,num_vertices)

        tests.append((s,t))


        start_time = process_time() 
        max_bandwidth = g.dijkstrasMaxBandwidthPath(s, t, False)
        elapsed_time = process_time() - start_time
        results["Dijkstra without Heap"]["Dense"].append((max_bandwidth,elapsed_time))

        start_time = process_time() 
        max_bandwidth = g.dijkstrasMaxBandwidthPath(s, t, True)
        elapsed_time = process_time() - start_time
        results["Dijkstra with Heap"]["Dense"].append((max_bandwidth,elapsed_time))

        start_time = process_time() 
        max_bandwidth = g.kruskalsMaxBandwidthPath(s, t)
        elapsed_time = process_time() - start_time
        results["Kruskals"]["Dense"].append((max_bandwidth,elapsed_time))

        start_time = process_time() 
        max_bandwidth = g.kruskalsMaxBandwidthPathwithoutHeap(s, t)
        elapsed_time = process_time() - start_time
        results["Kruskals (using python sort)"]["Dense"].append((max_bandwidth,elapsed_time))


with open('results.txt', 'w+') as f:
    for k in results.keys():
        f.write("\n" + k + "\n")
        f.write("Sparse\t\tDense\n")
        f.write("Max Bandwidth\tTime in Seconds\tMax Bandwidth\tTime in Seconds\n")
        for i in range(len(results[k]["Sparse"])):
            f.write("{}\t{}\t{}\t{}\n".format(results[k]["Sparse"][i][0],results[k]["Sparse"][i][1], results[k]["Dense"][i][0], results[k]["Dense"][i][1]))


