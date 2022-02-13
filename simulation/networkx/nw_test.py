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
import json
import numpy as np
from tqdm import tqdm
import os

#%%
""" assumption """

""" 
assume that we have 10000 users in the social network, we want each user to have 700 friends in average,

then for each user, the possible relationship is 9999, which is approximately 10000,

such that the average probability of setting edge for each node is 700/10000 = 0.07

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
    if r <= 0:
        return G
    if r >= n:
        return nx.complete_graph(n, create_using=G)
    for _, node_edges in groupby(edges, key=lambda x: x[0]):
        p = np.random.exponential(scale=r)/n
        node_edges = list(node_edges)
        random_edge = random.choice(node_edges)
        G.add_edge(*random_edge)
        for e in node_edges:
            if random.random() < p:
                G.add_edge(*e)
    return G

#%%
""" generate network """


nodes = 8000
seed = random.randint(1,10)
num_relationship = 700
G = gnp_random_connected_graph(nodes, num_relationship)


#%%

dict_nw = json_graph.adjacency_data(G)

#%%
""" save adjacency into dict """

dict_adjacency = {}

for i in tqdm(range(0,len(dict_nw['adjacency']))):
    dict_adjacency[i] = {}
    for adj in dict_nw['adjacency'][i]:
        dict_adjacency[i][adj['id']] = {}


#%%
""" save dict_adjacency in json """

path = r'C:\Users\yjian\OneDrive\Documents\research files\dissertation\Twitter Fake News\simulation\network generator'
with open(os.path.join(path, 'dict_adjacency.json'), 'w+', encoding="utf-8") as outfile:
    json.dump(dict_adjacency, outfile, ensure_ascii=False) 

""" complete info of  dict_adjacency """
with open(os.path.join(path, 'dict_nw.json'), 'w+', encoding="utf-8") as outfile:
    json.dump(dict_nw, outfile, ensure_ascii=False) 

#%%
""" plot """


plt.figure(figsize=(10,6))

nx.draw(G, node_color='lightblue', 
        with_labels=True, 
        node_size=500)
