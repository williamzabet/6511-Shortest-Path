#!/usr/bin/env python
# coding: utf-8

# In[234]:


import pandas as pd
import numpy as np
import numba as nb
import collections
from collections import defaultdict
import sys
import heapq
from math import*


# In[235]:


# initializing vertices dataframe from v.txt
vertices = pd.DataFrame([])
vtext = open("v.txt", "r")
for v in vtext:
    if not v.startswith("#"):
        v = v.split(",")
        vertices = vertices.append(pd.DataFrame({"Vertex_ID" : int(v[0]), "Square_ID" : int(v[1])}, 
                                          index=[0]), ignore_index=True)
vtext.close()

# initializing edges dataframe from e.txt
edges = pd.DataFrame([])
etext = open("e.txt", "r")
for e in etext:
    if not e.startswith("#"):
        e = e.split(",")
        edges = edges.append(pd.DataFrame({"From" : int(e[0]), "To" : int(e[1]), "Distance" : int(e[2])}, 
                                          index=[0]), ignore_index=True)
etext.close()


# In[236]:


# initial squares dictionary
init_squaresDict = {}

# adding {index:square_id}
for index, row in vertices.iterrows():
    init_squaresDict[index] = row["Square_ID"]

# another squares dictionary so I can append values to the keys (the squares)
squaresDict = {} 

# if the key already exists, the vertex value will get appended to the key
for key, value in init_squaresDict.items(): 
    if value not in squaresDict: 
        squaresDict[value] = [key] 
    else: 
        squaresDict[value].append(key)

# sorting the dictionary by square_ID
squaresDict = dict(collections.OrderedDict(sorted(squaresDict.items())))

# inverting the keys to values so I can have a key = vertex , value = square dictionary
squaresDict_inv = {row["Vertex_ID"] : row["Square_ID"] for index, row in vertices.iterrows()}

# initial edges dictionary
init_edgesDict = edges.set_index(edges.index).T.to_dict('list')
edgesDict = {}

# if the key already exists, the values will get appended (the edge with the weight)
for i in init_edgesDict.values():
    if i[0] not in edgesDict:
        edgesDict[i[0]] = {i[1]:i[2]}
    else:
        edgesDict[i[0]][i[1]] = i[2]
# additionally adding inverse edge order so the length of the items will be 100
for i in init_edgesDict.values():
    if i[1] not in edgesDict:
        edgesDict[i[1]] = {i[0]:i[2]}
    else:
        edgesDict[i[1]][i[0]] = i[2]
#sorting the dictionary by key (vertex: {edge: weight..})
edgesDict = dict(collections.OrderedDict(sorted(edgesDict.items())))


# In[285]:


# dijkstra alg function
def dijkstra(graph, start, end, output):
    #distances are initially set to inf for each vertex
    distances = {vertex: float("infinity") for vertex in graph}
    # the distance of the start vertex is set to 0
    distances[start] = 0

    #dictionary of total distance from path
    shortestPath = {start:0}
    # list of path nodes
    path = [start]
    # list of path nodes
    pq = [(0, start)]
    
    # while the length is greater than 0:..
    while len(pq) > 0:
        
        curr_distance, curr_vertex = heapq.heappop(pq)
        # if curr_distance is less than the distance of the path to the current node, continue
        if curr_distance > distances[curr_vertex]:
            continue
        # for the neighbors and weights for each node
        for neighbor, weight in graph[curr_vertex].items():
            # the distance is the current distance plus the weight of the current neighbor selected
            distance = curr_distance + weight
        
            # if the distance is less than the current distance of the neighbor then the distance gets updated
            if distance < distances[neighbor]:
                distances[neighbor] = distance
                heapq.heappush(pq, (distance, neighbor))
        # the node gets appended to the path
        if curr_vertex != start:
            path.append(curr_vertex)
        # once the current node selected is the end node, the loop breaks
        if curr_vertex == end:
            break
    # if the node is in the path, the shortestPath[node] = the distance
    for node, distance in distances.items():
        if node in path:
            shortestPath[node] = [distance]
    
    # dictionary is sorted
    shortestPath = {k: v for k, v in sorted(shortestPath.items(), key=lambda item: item[1])}
    
    if output == "yes":
        print("Path:", shortestPath)
        print()
        print("Total Nodes Visited (Cost):", len(distances.items()))
        print()
        print("Shortest Distance:", shortestPath[end])
        
    if output == "no":
        return shortestPath[end][0]
        


# In[286]:


dijkstra(edgesDict, 0, 99, "yes")
# as we can see all the nodes were visited and got the shortst distance of 6157


# In[279]:


# A* alg function
def aStar(graph, start_v, end_v): # if end_V is single digit, please put a 0 in fron! (i.e 9 -> 09)
    # initializing a dictionary of inf values
    aStarDict = {"Distance_From_Start": {vertex: float("infinity") for vertex in graph},
                 "Heuristic_Distance": {vertex: float("infinity") for vertex in graph},
                "F": {vertex: float("infinity") for vertex in graph},
                "Previous_Vertex": {vertex: float("infinity") for vertex in graph}}
    # start and end nodes
    start = start_v
    end = end_v
    
    # calculates the manhattan distance
    def man(r, c, i, j):
        return abs(r-i) + abs(c-j)
    
    #splits the node in the respective square into two digits so it can be calculated for heuristic distance (manhattan distance)
    init_end = [int(i) for i in str(squaresDict_inv[end])]
    
    # this loop gets the manhattan distance from each node based off what square the end node is in
    count = 0
    for x in range(10):
        for y in range(10):
            aStarDict["Heuristic_Distance"][count] = man(x,y, init_end[0], init_end[1])
            count+=1
            
    
    # closed list = the list of nodes already solved for 0 is in place for the start node
    closedList = [0]
    # open list = the list of nodes currently visiting and being able to extract
    openList = {}
    # the start distance is 0 for the start node
    aStarDict["Distance_From_Start"][start] = 0


    # once the start reaches the end the function stops
    while start != end:
        
        # initializing a variable for the next node to be assessed
        nextV = [float("infinity"), float("infinity")]
    
        # getting the neighbor and weights in the graph
        for neighbor, weight in graph[start].items():
            
            # if the neighbor isnt in the closed list
            if neighbor not in closedList:
                # the weight plus the cost so far = G
                g = weight + aStarDict["Distance_From_Start"][start]
                # the heuristic distance = H
                h = aStarDict["Heuristic_Distance"][neighbor]
                # F = G+H
                f = g+h
                
                # if G < current distance recorded for the node then it gets updated
                if g < aStarDict["Distance_From_Start"][neighbor]:
                    aStarDict["Distance_From_Start"][neighbor] = g
                
                # if F < current F recorded for node then it gets updated and so does the previous vertex (shortest path)
                if f < aStarDict["F"][neighbor]:
                    aStarDict["F"][neighbor] = f 
                    aStarDict["Previous_Vertex"][neighbor] = start
                
                #Openlist gets updated w/ values as well
                openList[neighbor] = aStarDict["F"][neighbor]
                
                #finds the minimum F value in the openlist so that it can be the next node to visit
                if aStarDict["F"][neighbor] < nextV[1]:
                    nextV = [neighbor, aStarDict["F"][neighbor]]
        # sorts the openlist
        openList = {k: v for k, v in sorted(openList.items(), key=lambda item: item[1])}
        # the next node gets added to the closed list
        closedList.append(nextV[0])
        # it becomes the start so the loop can run again.
        start = nextV[0]
        # then it gets popped from the list.
        openList.pop(nextV[0])
    
    
    
    print("Nodes Visited:", closedList)
    print()
    print("Total Nodes Visited (Cost):", len(closedList))
    print()
    print("Shortest Distance:", aStarDict["Distance_From_Start"][end], "From", aStarDict["Previous_Vertex"][end],"to", start)
    print()


# In[287]:


# cost was significantly less as only 61 nodes were visited as compared to 100
aStar(edgesDict, 0, 99)

