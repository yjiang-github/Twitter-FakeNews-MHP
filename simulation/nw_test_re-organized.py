# -*- coding: utf-8 -*-
"""
Created on Tue Dec  7 00:23:07 2021

@author: yjian
"""


"""
--- this file is for testing the package for generating a social network ---

"""

#%%
from itertools import combinations, groupby
import networkx as nx
from networkx.readwrite import json_graph
import random
import matplotlib.pyplot as plt 
import numpy as np
from tqdm import tqdm

#%%
""" assumption """

""" 
assume that we have 8000 users in the social network, we want each user to have 700 friends in average,

such that the expected number of edges for each node is 700, 

and the number of edges follows exponential distribution with scale = 700

"""

#%%
""" define function """


def gnp_random_connected_graph(n, r):
    """
    Generates a random undirected graph, similarly to an Erdős-Rényi 
    graph, but enforcing that the resulting graph is conneted
    
    """
    edges = combinations(range(n), 2)
    G = nx.Graph()
    G.add_nodes_from(range(n))
    
    for _, node_edges in groupby(edges, key=lambda x: x[0]):
        
        num_rel = int(np.random.exponential(scale=r))
        node_edges = list(node_edges)
        
        if num_rel == 0:
            # guarantee that each node has at least one edge connected to the network
            random_edge = random.choice(node_edges)
            G.add_edge(*random_edge)
        else:
            random_edges = random.choices(node_edges,k=num_rel)
            for random_edge in random_edges:
                G.add_edge(*random_edge)
    
    return G


#%%
""" generate network """


nodes = 8000
seed = random.randint(1,10)
num_relationship = 700
G = gnp_random_connected_graph(nodes, num_relationship)

#%%
""" plot """


plt.figure(figsize=(10,6))

nx.draw(G, node_color='lightblue', 
        with_labels=True, 
        node_size=500)


#%%

dict_nw = json_graph.adjacency_data(G)

#%%
""" save adjacency into dict """

dict_adjacency = {}

for i in tqdm(range(0,len(dict_nw['adjacency']))):
    dict_adjacency[i] = []
    for adj in dict_nw['adjacency'][i]:
        dict_adjacency[i].append(adj['id'])
