# -*- coding: utf-8 -*-
"""
Created on Tue Aug 10 20:44:21 2021

@author: yjian
"""




"""

--- this file is for simulation for dissertation topic ---
--- simulation on fake news spreading on twitter ---
--- LOOP ---

"""

#%%
from __future__ import division
from __future__ import print_function

import matplotlib
import csv
import json
import os
import datetime
from datetime import datetime
from matplotlib.dates import date2num, num2date
from tqdm import tqdm
import pandas as pd
import numpy as np
import math
import random
import sys
path_sys = r'C:\Users\yjian\OneDrive\Documents\research files\dissertation\Twitter Fake News\simulation'
sys.path.append(path_sys)

import Hawkes as hk

#%%
createVar = locals()

#%%
"""
--- twitter data simulation ---
--- case: original tweets with retweets and quotes;
            3 stances: supporting, denying and questioning ---
            

--- assumption: 
    an original tweet has 0.5 chance of triggering a new tweet;
    a retweet has 0.5 chance of triggering a new tweet; 
    a quote has 0.8 chance
    of triggering a new tweet


--- assumtpion: 
    a supporting tweet has 0.6 chance of generating a supporting tweet,
    0.2 chance of generating a denying tweet, and 0.2 chance of generating a questioning tweet;
    
    a denying tweet has 0.2 chance of generating a supporting tweet,
    0.5 chance of generating a denying tweet, and 0.3 chance of generating a questioning tweet;

    a quesioning tweet has 0.4 chance of generating a supporing tweet,
    0.4 chance of generating a denying tweet, and 0.2 chance of generating a questioning tweet.
    
    a commenting tweet has 0.2 chance of generating a supporting tweet,
    0.3 chance of generating a denying tweet, 0.1 chance of generating a questioning tweet,
    and 0.4 chance of generating a commenting tweet
        

"""
#%%
""" --- simulation --- """

"""
tweet type: delta
tweet type1: original tweet
tweet type2: retweet
tweet type3: quote

stance: gamma
stance1: supporting
stance2: denying
stance3: questioning
stance4: commenting


"""
list_stance = ['s', 'd', 'q']
list_type = ['o','ret','quo']

delta_o = 0.5
delta_ret = 0.5
delta_quo = 0.8


gamma_ss = 0.6
gamma_sd = 0.2
gamma_sq = 0.2
gamma_ds = 0.2
gamma_dd = 0.5
gamma_dq = 0.3
gamma_qs = 0.4
gamma_qd = 0.4
gamma_qq = 0.2


list_stance = ['s', 'd', 'q', 'c']
list_type = ['o','ret','quo']

delta_o = 0.5
delta_ret = 0.5
delta_quo = 0.8


gamma_ss = 0.6
gamma_sd = 0.2
gamma_sq = 0.1
gamma_sc = 0.1
gamma_ds = 0.1
gamma_dd = 0.4
gamma_dq = 0.2
gamma_dc = 0.3
gamma_qs = 0.3
gamma_qd = 0.4
gamma_qq = 0.2
gamma_qc = 0.1
gamma_cs = 0.2
gamma_cd = 0.3
gamma_cq = 0.1
gamma_cc = 0.4

""" --- define function: assign tweet type and tweet stance --- """

# assign tweet_type for the new-generated tweet
def assign_tweet_type():
    
    if random.uniform(0,1) > 0.5:
        tweet_type = 'ret'
    else:
        tweet_type = 'quo'
    return tweet_type

#%%

"""
--- simulation in a loop ---

"""
mu_s, mu_d, mu_q, mu_c = 0.02, 0.01, 0.01, 0.005
omega_s, omega_d, omega_q, omega_c = 20, 40, 25, 60

total_time = 100000


num_simulation = 25

max_gen = 100 # initial expected num of generations

""" main function """

for time in tqdm(range(num_simulation)):

    count = 0 # tweet label

    dict_events = {}
    dict_events_prior = {}
        
    for num_gen in range(max_gen):
        # break if no events on last generation
        if dict_events != {} and dict_events[num_gen-1] == {}:
            break
        else: dict_events[num_gen] = {}
        if num_gen == 0: # immigrants
            # generate immigrants following Poi(lambda*total_time) for each stance
            list_time = []
            for stance in list_stance:
                mu = createVar['mu_'+stance[0]]
                num_imm = np.random.poisson(lam = mu*total_time)
                # arrival times of immigrants follow Uni(0,total_time)
                list_time_temp = [(random.uniform(0, total_time),stance) for i in range(num_imm)]  
                list_time += list_time_temp     
            
            # ascending order
            list_time.sort(key=lambda x: x[0])  
            # save tweets
            for newtweet in list_time:
                dict_events[num_gen][str(count)] = {'time': newtweet[0],\
                                             'stance': newtweet[1],\
                                             'type':'o',\
                                             'influenced by':'/'}
                count += 1
                
        else: # descendants
    
            # get event times from last generation
            dict_events_prior = {}
    
            for key in dict_events[num_gen-1].keys():
                dict_events_prior[key] = dict_events[num_gen-1][key]
    
                    
            # generate cluster of descendants for each event from last generation
            for event in dict_events_prior.keys():
                # generate number of descendants for each event from last generation following Poi(gamma*delta)
                delta = createVar['delta_'+dict_events_prior[event]['type']]
                
                gamma_ps = createVar['gamma_'+dict_events_prior[event]['stance']+'s']
                gamma_pd = createVar['gamma_'+dict_events_prior[event]['stance']+'d']
                gamma_pq = createVar['gamma_'+dict_events_prior[event]['stance']+'q']
                gamma_pc = createVar['gamma_'+dict_events_prior[event]['stance']+'c']
                
                num_descendants_s = np.random.poisson(lam = delta*gamma_ps)
                num_descendants_d = np.random.poisson(lam = delta*gamma_pd)
                num_descendants_q = np.random.poisson(lam = delta*gamma_pq)
                num_descendants_c = np.random.poisson(lam = delta*gamma_pc)
    
                #For each stance, generate event time and save into dict_events:
                list_time = []
                for stance in list_stance:
                    # generate arrival times of descendant  following exp(1/omega)
                    omega = createVar['omega_'+stance]
                    num_descendants = createVar['num_descendants_'+stance]
     
                    list_time_temp = [(dict_events_prior[event]['time']+np.random.exponential(1/omega, size = 1)[0],stance)  for i in range(num_descendants)]
                    list_time += list_time_temp
                    
                # store descendants of each event if exist
                if list_time != []:
                    # ascending order
                    list_time.sort(key=lambda x: x[0])       
                    for newtweet in list_time:
                        # less than total simulation time
                        if newtweet[0] < total_time:
                            # generate tweet type & stance
                            tweet_type = assign_tweet_type()
                            dict_events[num_gen][str(count)] = {'time': newtweet[0],\
                                                                  'stance': newtweet[1],\
                                                                  'type':tweet_type,\
                                                                  'influenced by':event}
                            #if dict_events_prior[event]['stance'][0]== 'd' and newtweet[1][0] == 's':
                            #    print(count)
                            count += 1
                            
    """--- export dict_events data ---"""
    with open(os.path.join(path_sys, 'datasets','dict_events_'+str(time)+'.json'), 'w+', encoding="utf-8") as outfile:
        json.dump(dict_events, outfile, ensure_ascii=False) 

#%%
"""
--- export dict_params ---

"""

# export dict_params
dict_params = {}
for r in list_type:
    dict_params['delta_'+r] = {'true value': createVar['delta_'+r], 'estimated':''}
for stance in list_stance:
    for new_stance in list_stance:
        dict_params['gamma_'+stance+new_stance] = {'true value': createVar['gamma_'+stance+new_stance], 'estimated':''}
    dict_params['mu_'+stance] = {'true value': createVar['mu_'+stance], 'estimated':''}
    dict_params['omega_'+stance] = {'true value': createVar['omega_'+stance], 'estimated':''}
    
    
with open(os.path.join(path_sys, 'dict_params.json'), 'w+', encoding="utf-8") as outfile:
    json.dump(dict_params, outfile, ensure_ascii=False) 


