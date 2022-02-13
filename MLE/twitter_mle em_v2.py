# -*- coding: utf-8 -*-
"""
Created on Sun Mar 21 10:00:53 2021

@author: yjian
"""


"""
--- this file is for performing MLE using EM algorithm
--- 2nd version

"""


#%%
import numpy as np
import pandas as pd
from tqdm import tqdm
import random
import math
import json
import os

#%%

createVar = locals()

#%%

""" import simulation data """
path = r'C:\Users\yjian\OneDrive\Documents\research files\dissertation\simulation'

with open(os.path.join(path, 'dict_events.json'), 'r',  encoding="utf-8") as f:
    dict_events = json.load(f)
    
#%%
""" dict -> list """

list_time = []

for gen in dict_events.keys():
    for num in dict_events[gen].keys():
            list_time.append((num,\
                            dict_events[gen][num]['time'],\
                              dict_events[gen][num]['stance'],\
                                  dict_events[gen][num]['type'],\
                                      dict_events[gen][num]['influenced by']))
                
""" ascending order """
list_time.sort(key=lambda x: x[1])


#%%
""" parameter initialization """

list_stance = ['s','d']
list_type = ['o','ret','q']

total_time = list_time[-1][1]

# initial mu, gamma, omega
for k in list_stance:
    # list -> list_stance, list_stance_o 
    createVar['list_time_'+k] = [event for event in list_time if event[2] == k]
    createVar['list_time_'+k+'_o'] = [event for event in createVar['list_time_'+k] if event[3] == 'o']
    createVar['initial_mu_'+k] = len(createVar['list_time_'+k+'_o'])/total_time
    createVar['initial_omega_'+k] = 1
    for m in list_stance:
        createVar['initial_gamma_'+m+k] = 0.5

# initial delta
for r in list_type:
    createVar['initial_delta_'+r] = 0.5

epsilon = 1e-5
num_iteration = 5000



#%%

#%%
""" main function """

for num in tqdm(range(num_iteration)):
    
    if num == 0:
        # set initial values
        # create arrays
        for k in list_stance:
            # array_h, array_pjj, array_pjl
            createVar['array_h_'+k], createVar['array_pjj_'+k], createVar['array_pjl_'+k] = \
                np.zeros((len(createVar['list_time_'+k]),len(createVar['list_time_'+k])-1)),\
                    np.zeros(len(createVar['list_time_'+k])),\
                        np.zeros((len(createVar['list_time_'+k]),len(createVar['list_time_'+k])-1))
            # mu, omega
            createVar['curr_mu_'+k] = createVar['initial_mu_'+k]
            createVar['curr_omega_'+k] = createVar['initial_omega_'+k]
            # gamma
            for m in list_stance:
                createVar['curr_gamma_'+m+k] = createVar['initial_gamma_'+m+k]
        # delta
        for r in list_type:
            createVar['curr_delta_'+r] = createVar['initial_delta_'+r]
        prior_Q = 1
        
    else:
        




