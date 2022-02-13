# -*- coding: utf-8 -*-
"""
Created on Thu Feb 10 23:47:56 2022

@author: yjian
"""




"""
--- this file is for performing MLE using EM algorithm
--- for fake news simulation with replies


--- PARTICULARLY, for version 5, only the first 4000 events were used for estimation    
    (all 4000 events are descendants of the initial event)

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
path = r'C:\Users\yjian\OneDrive\Documents\research files\dissertation\Twitter Fake News\simulation'

with open(os.path.join(path,'datasets','Scenario4','dict_events.json'), 'r',  encoding="utf-8") as f:
    dict_events = json.load(f)

with open(os.path.join(path, 'datasets','Scenario4','dict_params.json'), 'r',  encoding="utf-8") as f:
    dict_params = json.load(f)

with open(os.path.join(path, 'datasets','Scenario4','dict_adjacency.json'), 'r',  encoding="utf-8") as f:
    dict_adjacency = json.load(f)

#%%
beta_d = dict_params['beta_d']['true value']
beta_i = dict_params['beta_i']['true value']

#%%
""" dict -> list """

list_time = []

for num in dict_events.keys():
    list_time.append((num,\
        dict_events[num]['time'],\
          dict_events[num]['stance'],\
              dict_events[num]['type'],\
                  dict_events[num]['influenced by'],\
                     [beta_d if user_id in dict_adjacency[dict_events[num]['user_id']] else beta_i \
                      for user_id in dict_events[num]['influenced by'].split(',') if dict_events[num]['influenced by']!= '/']))
                
""" ascending order """
list_time = sorted(list_time, key=lambda x: x[1])[:4001]

#%%

""" parameter initialization """

list_stance = ['s','d']
list_type = ['o','ret','quo','rply']

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

array_events = np.array([event[1] for event in list_time])

# create array of time_diff between event_j and event_l
array_timediff = np.zeros((len(array_events),len(array_events)-1))
for j in range(len(array_events)):
    if j != 0:
        for l in range(0,j):
            array_timediff[j,l] = (array_events[j]-array_events[l])


# create array of stance in two dims and one dim
for stance in list_stance:
    createVar['array_stance_'+stance] = np.zeros((len(array_events),len(array_events)-1))
    createVar['array_stance_onedim_'+stance] = np.zeros(len(array_events))
    for j in range(len(array_events)):
        if list_time[j][2] == stance:
            createVar['array_stance_onedim_'+stance][j] = 1
        if j != 0:
            for l in range(0,j):
                if list_time[l][2] == stance:
                    createVar['array_stance_'+stance][j,l] = 1

# create array of event type, calculate number of event types
for type in list_type:
    createVar['array_type_'+type] = np.zeros((len(array_events),len(array_events)-1))
    createVar['n_'+type] = len([event[3] for event in list_time if event[3] == type])
    for j in range(len(array_events)):
        if j != 0:
            for l in range(0,j):
                if list_time[l][3] == type:
                    createVar['array_type_'+type][j,l] = 1

#%%
# create array of tweet relationships, calculate number of events with both stances
array_n = np.zeros((len(array_events),len(array_events)-1))
#for m in list_stance:
#    for k in list_stance:
#        createVar['n_'+m[0]+k[0]] = 0
for k in list_stance:
    createVar['n_'+k] = 0
    for m in list_stance:
        createVar['n_'+m+k] = 0

#%%
for j in tqdm(range(len(array_events))):
    if j != 0:
        createVar['n_'+list_time[j][2]] += 1
        if list_time[j][4] != '/':
            for l in range(0,j):
            # check if current tweet is triggered by any prior tweet(s)
                if list_time[l][0] in list_time[j][4].split(','):
                    index = list_time[j][4].split(',').index(list_time[l][0])
                    array_n[j,l] = list_time[j][5][index]
                    #createVar['n_'+list_time[l][2]+list_time[j][2]] += 1
                    
                
                    
                #  Attention!!: here we generated data such that the current event could only be
                # influenced by 1 prior event, such that we can break the loop right after finding the
                # relationship, but we have to run the whole loop for the real data since a real tweet
                # could be triggered by multiple prior events

#%%
""" --- main function --- """
for num in tqdm(range(num_iteration)):
    
    if num == 0:
        # set initial values
        # create arrays
        for k in list_stance:
            # array_h, array_pjj, array_pjl
            createVar['array_h_'+k], createVar['array_pjj_'+k], createVar['array_pjl_'+k] = \
                np.zeros((len(array_events),len(array_events)-1)),\
                    np.zeros(len(array_events)),\
                        np.zeros((len(array_events),len(array_events)-1))
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
        
        
        # create arrays
        for k in list_stance:
            ## array_h
            createVar['array_h_'+k] = np.zeros((len(array_events),len(array_events)-1))
            ## array_pjj array_pjl
            createVar['array_pjj_'+k], createVar['array_pjl_'+k] = \
                np.zeros(len(array_events)), np.zeros((len(array_events),len(array_events)-1))
                
    # create array_delta 
    array_delta = np.zeros((len(array_events),len(array_events)-1))
    for l in range(len(array_events)-1):
        array_delta[:,l] = \
            np.append(np.zeros(l+1),np.array([createVar['curr_delta_'+list_time[l][3]]]*(len(array_events)-l-1)))
    # create array_gamma
    for k in list_stance:
        createVar['array_gamma_'+k] = np.zeros((len(array_events),len(array_events)-1))
        for j in range(len(array_events)):
            if j != 0 and list_time[j][2] == k:
                for l in range(0,j):
                    createVar['array_gamma_'+k][j,l] = createVar['curr_gamma_'+list_time[l][2]+list_time[j][2]]
    
    # create&calculate array_decay (h_k(tj-tl)) 
    for k in list_stance:
        # delta_r*gamma_mk*omega_k*exp(-omega_k*(time-diff))
        createVar['array_h_'+k] = \
            array_delta*createVar['array_gamma_'+k]*array_n*createVar['curr_omega_'+k]*\
                math.e**(-createVar['curr_omega_'+k]*array_timediff)
    
    # calculate array_logh, array_pjj and array_pjl
    for k in list_stance:
        ## calculate array_logh for each k
        createVar['array_logh_'+k] = np.log(createVar['array_h_'+k])
        createVar['array_logh_'+k][np.isinf(createVar['array_logh_'+k])]=0
        ## calculate array_pjj
        createVar['array_pjj_'+k][0] = 1 #curr_mu_k/curr_mu_k
        createVar['array_pjj_'+k][1:] = \
            createVar['curr_mu_'+k]/(createVar['curr_mu_'+k]+createVar['array_h_'+k][1:].sum(axis=1))
        ## calculate array_pjl
        createVar['array_pjl_'+k][0] = np.float64(0) # l should < j-1
        createVar['array_pjl_'+k][1:] = \
            (createVar['array_h_'+k][1:].T/(createVar['curr_mu_'+k]+createVar['array_h_'+k][1:].sum(axis=1))).T
            
    # update parameters
    ## clear delta_r for the aggregating in the following procedures
    for r in list_type:
        createVar['curr_delta_'+r] = 0
        #createVar['array_delta_'+r[0]] = array_delta.copy()
        #createVar['array_delta_'+r[0]][createVar['array_delta_'+r[0]]!=createVar['curr_delta_'+r[0]]] = 0
    for k in list_stance:
        ## mu_k
        createVar['curr_mu_'+k] = (createVar['array_pjj_'+k]*createVar['array_stance_onedim_'+k]).sum()/total_time
        ## gamma_m,k
        for m in list_stance:
            createVar['curr_gamma_'+m+k] = \
                (createVar['array_stance_'+m]*createVar['array_pjl_'+k]).sum()/\
                    (createVar['n_'+m])
        ## omega_k
        createVar['curr_omega_'+k] = \
            createVar['array_pjl_'+k].sum()/(createVar['array_pjl_'+k]*array_timediff).sum()
        ## delta_r: aggregate delta_r under each k
        for r in list_type:
            createVar['curr_delta_'+r] += (createVar['array_type_'+r]*createVar['array_pjl_'+k]).sum()
    ## finalize delta_r
    for r in list_type:
        createVar['curr_delta_'+r] = createVar['curr_delta_'+r]/createVar['n_'+r] #len(list_stance)*len(array_events)
        
    
    # EM calculation
    Q = 0
    for k in list_stance:
        Q = Q + (createVar['array_pjj_'+k].sum())*math.log(createVar['curr_mu_'+k]) \
            - createVar['curr_mu_'+k]*total_time \
            + (createVar['array_pjl_'+k]*createVar['array_logh_'+k]).sum()
    
    
    if abs(Q - prior_Q) > epsilon:
        prior_Q = Q
    else:
        break
    
#%%

for param in dict_params.keys():
    if 'beta' not in param:
        dict_params[param]['estimated'] = createVar['curr_'+param]
