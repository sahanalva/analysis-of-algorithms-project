# -*- coding: utf-8 -*-
"""
Spyder Editor

"""

#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Nov  9 13:17:17 2019

@author: sahanalva
"""
import random 
import math
from operator import itemgetter


class AdjNode:
    def __init__(self, weight,vertex):
        self.next = None
        self.vertex = vertex
        self.weight = weight
    
    
class Graph:
    
    def __init__(self, num_vertices):
        self.num_vertices = num_vertices
        self.graph = [None] * self.num_vertices
        
    def addEdge(self, start_vertex, end_vertex, weight):
        new_node = AdjNode(weight,end_vertex)
        new_node.next = self.graph[start_vertex]
        self.graph[start_vertex] = new_node
        
        new_node = AdjNode(weight,start_vertex)
        new_node.next = self.graph[end_vertex]
        self.graph[end_vertex] = new_node
        
    def printGraph(self):
        for i in range(self.num_vertices):
            print("Adj list of vertex " + str(i) +" is")
            pointer = self.graph[i]
            while(pointer != None):
                print(pointer.weight, pointer.vertex)
                pointer = pointer.next

            
    def printVertexDegree(self):
        for i in range(self.num_vertices):
            print("Degree of vertex " + str(i) +" is")
            pointer = self.graph[i]
            count = 0
            while(pointer != None):
                pointer = pointer.next
                count += 1
            print(count)
            
         
            
    def removeAllEdges(self):
        for i in range(self.num_vertices):
            self.graph[i] = None
                
    def createAllVertexLoop(self, max_weight = 1000):
        self.removeAllEdges()
        for i in range(self.num_vertices-1):
            weight = random.randint(1, max_weight)
            self.addEdge(i,i+1,weight)
        self.addEdge(self.num_vertices-1,0,1)
            
    
    def pair_generator(self, numbers):
      used_pairs = set()
     
      while True:
        pair = random.sample(numbers, 2)
        pair = tuple(sorted(pair))
        diff = pair[1]-pair[0]
        if((pair not in used_pairs) and (diff != 1) and (diff != self.num_vertices - 1)):
          used_pairs.add(pair)
          yield pair
      
    def createSparseGraph(self, avg_degree, max_weight = 1000):
        self.removeAllEdges()
        list_of_vertices = range(self.num_vertices)
        self.createAllVertexLoop(max_weight)
        num_edges = int(math.ceil((avg_degree - 2)*self.num_vertices*0.5))
        
        gen = self.pair_generator(list_of_vertices)
        
        for i in range(num_edges):
            weight = random.randint(1, max_weight)
            vertex_pair = next(gen)
            self.addEdge(vertex_pair[0], vertex_pair[1], weight)
    
    def createDenseGraph(self, adjacency_fraction, max_weight = 1000):
        self.removeAllEdges()
        list_of_vertices = range(self.num_vertices)
        self.createAllVertexLoop(max_weight)
        num_edges = int(math.ceil(self.num_vertices* (self.num_vertices - 1)* 0.5 *adjacency_fraction)) - self.num_vertices
        
        gen = self.pair_generator(list_of_vertices)
        
        for i in range(num_edges):
            weight = random.randint(1, max_weight)
            vertex_pair = next(gen)
            self.addEdge(vertex_pair[0], vertex_pair[1], weight)
    
    def dijkstrasMaxBandwidthPath(self, source, destination, use_heap):
        
        status = ['unseen'] * self.num_vertices
        bandwidth = [1001] * self.num_vertices
        dad = [None] * self.num_vertices
        if(use_heap == True):
            maxHeap = Heap()
            index_list = [-1] * self.num_vertices
        else:
            max_array = [-2] * self.num_vertices
        
        status[source] = 'intree'
        
        temp = self.graph[source]
        while(temp != None):
            status[temp.vertex] =  'fringe'
            bandwidth[temp.vertex] = temp.weight
            dad[temp.vertex] = source
            
            if(use_heap == True):
                maxHeap.insertHeapNode(name = temp.vertex, value =temp.weight, index_list= index_list)
            else:
                max_array[temp.vertex] = temp.weight
                
            temp = temp.next
        
        while('fringe' in status):
            if use_heap == True:
                max_node = maxHeap.maxNode()
                max_vertex = max_node.name
                index_list[max_vertex] = -1
                maxHeap.deleteHeapNode(index = 0, index_list= index_list)
            else:
                max_bandwidth = -2
                for i,val in enumerate(max_array):
                    if(val > max_bandwidth):
                        max_bandwidth = val
                        max_vertex = i
                max_array[max_vertex] = -2    
                #max_vertex = max(zip(max_array, range(len(max_array))))[1]
                #max_array[max_vertex] = -1
            
            status[max_vertex] = 'intree'
            
            temp = self.graph[max_vertex]
            while(temp != None):
                if(status[temp.vertex] == "unseen"):
                    status[temp.vertex] =  'fringe'
                    bandwidth[temp.vertex] = min(temp.weight, bandwidth[max_vertex])
                    dad[temp.vertex] = max_vertex
                    if(use_heap == True):
                        maxHeap.insertHeapNode(name = temp.vertex, value = bandwidth[temp.vertex], index_list = index_list)
                    else:
                        max_array[temp.vertex] = bandwidth[temp.vertex]
                        
                    
                elif(status[temp.vertex] == "fringe" and 
                   min(temp.weight, bandwidth[max_vertex]) > bandwidth[temp.vertex]):
                    bandwidth[temp.vertex] = min(temp.weight, bandwidth[max_vertex])
                    dad[temp.vertex] = max_vertex
                    if(use_heap == True):
                        
                        delete_index = index_list[temp.vertex]
                        index_list[temp.vertex] = -1
                        maxHeap.deleteHeapNode(index = delete_index, index_list=index_list)
                        maxHeap.insertHeapNode(name = temp.vertex, value = bandwidth[temp.vertex], index_list= index_list)
                    else:
                        max_array[temp.vertex] = bandwidth[temp.vertex]
                temp = temp.next
        return(bandwidth[destination])
        
    def kruskalsMaxBandwidthPath(self, source, destination):
            

        maxHeap = BinHeap()

        for vertex in range(self.num_vertices):
            temp = self.graph[vertex]
            
            while(temp != None):
                if(temp.vertex > vertex):
                    maxHeap.insert((vertex,temp.vertex, temp.weight))
                temp = temp.next

        

        sorted_edge_node_array = maxHeap.heapSort()
        
        max_span_tree = Graph(self.num_vertices)
        self.spantree_rank = [0] * self.num_vertices
        self.spantree_dad = [-1] * self.num_vertices
        

        for edge_node in sorted_edge_node_array:
            r1 = self.find(edge_node[0])
            r2 = self.find(edge_node[1])
            
            if(r1 != r2):
                self.union(r1,r2)
                max_span_tree.addEdge(edge_node[0],edge_node[1],edge_node[2])

        

        max_span_tree.treeTrace(source, destination)

        return(max_span_tree.kruskal_max_bandwidth[destination])
                
    def kruskalsMaxBandwidthPathwithoutHeap(self, source, destination):
            

        val_list = []
        for vertex in range(self.num_vertices):
            temp = self.graph[vertex]
            
            while(temp != None):
                if(temp.vertex > vertex):
                    val_list.append((vertex,temp.vertex, temp.weight))
                temp = temp.next

        

        sorted_edge_node_array = sorted(val_list,key=itemgetter(2), reverse = True)

        
        max_span_tree = Graph(self.num_vertices)
        self.spantree_rank = [0] * self.num_vertices
        self.spantree_dad = [-1] * self.num_vertices
        
        for edge_node in sorted_edge_node_array:
            r1 = self.find(edge_node[0])
            r2 = self.find(edge_node[1])
            
            if(r1 != r2):
                self.union(r1,r2)
                max_span_tree.addEdge(edge_node[0],edge_node[1],edge_node[2])

        max_span_tree.treeTrace(source, destination)

        return(max_span_tree.kruskal_max_bandwidth[destination])
                
        
    def find(self, vertex):
        w = vertex
        stack = []
        while(self.spantree_dad[w] != -1):
            stack.append(w)
            w = self.spantree_dad[w]
        while(len(stack) != 0):
            v = stack.pop()
            self.spantree_dad[v] = w
        return w
    
    def union(self,r1,r2):
        if(self.spantree_rank[r1] > self.spantree_rank[r2]):
            self.spantree_dad[r2] = r1
        elif(self.spantree_rank[r1] < self.spantree_rank[r2]):
            self.spantree_dad[r1] = r2
        else:
            self.spantree_dad[r2] = r1;
            self.spantree_rank[r1] += 1
    
    def treeTrace(self, source, destination):
        
        self.colour = ['white'] * self.num_vertices
        self.dfs_dad = [None] *self.num_vertices
        self.kruskal_max_bandwidth = [None] * self.num_vertices
        
        self.kruskal_max_bandwidth[source] = 1001
        self.depthFirstSearch(source, destination)
        
            
            
        
    def depthFirstSearch(self, vertex, destination):
        
        self.colour[vertex] = 'grey'
        
        temp = self.graph[vertex]
        while(temp != None):
            
            if(self.colour[temp.vertex] == 'white'):
                self.kruskal_max_bandwidth[temp.vertex] = min(self.kruskal_max_bandwidth[vertex], temp.weight)
                self.dfs_dad[temp.vertex] = vertex
                

                self.depthFirstSearch(temp.vertex, destination)
            
            temp = temp.next
            
        self.colour[vertex] = 'black'

        return         
        
class HeapNode:
    def __init__(self, name,value):
        self.name = name
        self.value = value        
        
            
class Heap:
    
    def __init__(self):
        self.heap = []
        

    def parentIndex(self,index):
        parent_index = int(math.ceil(index*0.5 - 1))
        return parent_index
    
    def childrenIndices(self, index):
        child_1_index = 2*index +1 
        child_2_index = 2*index +2
        
        return child_1_index, child_2_index

    def maxNode(self):
        return self.heap[0]
    

    
    def insertHeapNode(self, name, value, index_list = None):
        
             
        heap_node = HeapNode(name,value)
        self.heap.append(heap_node)
        
        index = len(self.heap)-1
        
        if(index_list != None):
            index_list[name] = index
            
        while( index > 0):
            parent_index = self.parentIndex(index)
            if(self.heap[index].value > self.heap[parent_index].value):
                if(index_list != None):
                    index_list[self.heap[parent_index].name] = index
                    index_list[self.heap[index].name] = parent_index
                    
                self.heap[parent_index],self.heap[index] = self.heap[index],self.heap[parent_index]
                index = parent_index
            else:
                break
        
            
    
    def deleteHeapNode(self, index, index_list = None):

        
        if(index > len(self.heap)-1):
            
            raise ValueError("Index greater than the length of heap")
            
        elif(index == len(self.heap)-1):
                    
            self.heap.pop()
            
            
        else:
            self.heap[index] = self.heap.pop()
            
            if(index_list != None):
                index_list[self.heap[index].name] = index        
                
            n = len(self.heap)-1
            
            if(index > 0 and self.heap[index].value > self.heap[self.parentIndex(index)].value):
                while( index > 0):
                    parent_index = self.parentIndex(index)
                    if(self.heap[index].value > self.heap[parent_index].value):
                        
                        if(index_list != None):
                            index_list[self.heap[parent_index].name] = index
                            index_list[self.heap[index].name] = parent_index
                            
                        self.heap[parent_index],self.heap[index] = self.heap[index],self.heap[parent_index]
                        index = parent_index
                    else:
                        break
                    
            else:
                
                while(self.childrenIndices(index)[0] <= n):
                    child_indices = self.childrenIndices(index)
    
                    if(self.heap[child_indices[0]].value > self.heap[index].value):
                        if(child_indices[0] == n):
                            
                            if(index_list != None):
                                index_list[self.heap[child_indices[0]].name] = index
                                index_list[self.heap[index].name] = child_indices[0]
                                
                            
                            self.heap[index],self.heap[child_indices[0]] = self.heap[child_indices[0]],self.heap[index]
                            index = child_indices[0]
                            
                        elif(self.heap[child_indices[0]].value > self.heap[child_indices[1]].value):
                            if(index_list != None):
                                index_list[self.heap[child_indices[0]].name] = index
                                index_list[self.heap[index].name] = child_indices[0]
                                
                            
                            self.heap[index],self.heap[child_indices[0]] = self.heap[child_indices[0]],self.heap[index]
                            index = child_indices[0]
                        else:
                            if(index_list != None):
                                index_list[self.heap[child_indices[1]].name] = index
                                index_list[self.heap[index].name] = child_indices[1]
                                
                            
                            self.heap[index],self.heap[child_indices[1]] = self.heap[child_indices[1]],self.heap[index]
                            index = child_indices[1]
                    elif(child_indices[0] != n):
                        if(self.heap[child_indices[1]].value > self.heap[index].value):
                            if(index_list != None):
                                index_list[self.heap[child_indices[1]].name] = index
                                index_list[self.heap[index].name] = child_indices[1]
                                
                            
                            self.heap[index],self.heap[child_indices[1]] = self.heap[child_indices[1]],self.heap[index]
                            index = child_indices[1]
                        else:
                            break
                    else:
                        break
    
    def heapSort(self):
        sorted_array = []
        while(len(self.heap)!=0):
            max_node = self.maxNode()
            self.deleteHeapNode(index = 0)
            sorted_array.append(max_node)
        return sorted_array
            
class BinHeap:
    def __init__(self):
        self.heapList = [0]
        self.currentSize = 0


    def percUp(self,i):
        while i // 2 > 0:
          if self.heapList[i][-1] > self.heapList[i // 2][-1]:

             self.heapList[i // 2], self.heapList[i] = self.heapList[i], self.heapList[i // 2]

          i = i // 2

    def insert(self,heap_tuple):
      
      self.heapList.append(heap_tuple)
      self.currentSize = self.currentSize + 1
          
      self.percUp(self.currentSize)

    def percDown(self,i):
      while (i * 2) <= self.currentSize:
          mc = self.maxChild(i)
          if self.heapList[i][-1] < self.heapList[mc][-1]:

              self.heapList[i],self.heapList[mc] = self.heapList[mc],self.heapList[i]
          i = mc

    def maxChild(self,i):
      if i * 2 + 1 > self.currentSize:
          return i * 2
      else:
          if self.heapList[i*2][-1] > self.heapList[i*2+1][-1]:
              return i * 2
          else:
              return i * 2 + 1

    def delMax(self):
      retval = self.heapList[1]
      self.heapList[1] = self.heapList[self.currentSize]
      self.currentSize = self.currentSize - 1

      self.heapList.pop()
      self.percDown(1)
      return retval

    def delIndex(self, i):
      #print(i)
      self.heapList[i] = self.heapList[self.currentSize]
      self.currentSize = self.currentSize - 1
      
      self.heapList.pop()

      if(i//2 > 0):

          if(self.heapList[i][-1] > self.heapList[i // 2][-1]):
              self.percUp(i)
      else:          
          self.percDown(i)


    
    def heapSort(self):
        sorted_array = []
        while(len(self.heapList) > 1):
            max_node = self.delMax()
            sorted_array.append(max_node)
        return sorted_array


