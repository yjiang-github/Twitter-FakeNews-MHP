# -*- coding: utf-8 -*-
"""
Created on Tue Feb  8 16:27:44 2022

@author: yjian
"""


"""
--- this file is the version5 of simulation in scenario4 using approach 1 from Prof. Porter ---
--- specifically, this approach will store all prior values into numpy such that each time the process
    will only calculate the values of current event for all users ---

"""

#%%
from __future__ import division
from __future__ import print_function

import matplotlib
import csv
import json
import os
#from matplotlib.dates import date2num, num2date
from tqdm import tqdm
import timeit
import pandas as pd
import numpy as np
import random
import math
import time
import sys
path_sys = r'C:\Users\yjian\OneDrive\Documents\research files\dissertation\Twitter Fake News\simulation'
sys.path.append(path_sys)
#import Hawkes as hk
#from multiprocessing import Pool, cpu_count
#from twitter_filteringfunction import*
#from config import*
#%%
createVar = locals()

#%%
""" import user network """
path_sim = r'C:\Users\yjian\OneDrive\Documents\research files\dissertation\Twitter Fake News\simulation'
with open(os.path.join(path_sim,'network generator','dict_adjacency.json'), 'r',  encoding="utf-8") as f:
    dict_adjacency = json.load(f)

for user_id in dict_adjacency.keys():
    dict_adjacency[user_id] = list(dict_adjacency[user_id].keys())

#%%
# temporarily set num_users to be 12
num_users = len(dict_adjacency)


#%%

"""
--- twitter data simulation ---
--- Scenario4: original tweets with retweets quotes and replies;
            2 stances: supporting, and denying ---    
            
--- this time we add n_l into the model, which is the relationships between users ---        

--- assumption: 
    an original tweet has 0.5 chance of triggering a new tweet;
    a retweet has 0.5 chance of triggering a new tweet; 
    a quote has 0.8 chance of triggering a new tweet;
    a reply has 0.1 chance of triggering a new tweet.


--- assumtpion: 
    a supporting tweet has 0.8 chance of generating a supporting tweet,
    0.2 chance of generating a denying tweet;
    
    a denying tweet has 0.2 chance of generating a supporting tweet,
    0.8 chance of generating a denying tweet;
    
--- assumption:
    n_l represents the cumulative relationship scores between the user of the current event and users of all prior events
    for each user pair:
        a socre of 1.0 represents that the current user directly follows the prior user;
        a score of 0.5 represents that the current user and the prior user follows the same user 
            (which is occurred in replies).

--- assumption:
    only original tweets and quotes can start a conversation,
    replies and retweets will not start a new conversation;
    retweets and replies to the retweets will directly count to the conversation, not retweets

"""

#%%
""" --- parameters --- """

"""
tweet type: delta
tweet type1: original tweet
tweet type2: retweet
tweet type3: quote
tweet type: reply

stance: gamma
stance1: supporting
stance2: denying



"""
list_stance = ['s', 'd']
list_type = ['o','ret','quo', 'rply']

delta_o = 0.5
delta_ret = 0.5
delta_quo = 0.8
delta_rply = 0.1

gamma_ss = 0.8
gamma_sd = 0.2
gamma_ds = 0.2
gamma_dd = 0.8

beta_d = 1.0
beta_i = 0.1

mu_s, mu_d = 0.015, 0.04
omega_s, omega_d = 0.5, 1

# assign tweet_type for the new-generated tweet
def assign_tweet_type():
    
    num = random.uniform(0,1)
    if num < 0.5:
        tweet_type = 'ret'
    elif 0.5 <= num < 0.9:
        tweet_type = 'quo'
    else:
        tweet_type = 'rply'
        
    return tweet_type

#%%
""" --- simulation --- """
""" first step: generate immigrants """



total_time = 7500

event_id = 0 # tweet id/label

dict_events = {} # save events
dict_userevents = {} # save events for each user
dict_convos = {} # save events as groups and conversations

for user_id in dict_adjacency.keys():
    dict_userevents[user_id] = {}

""" generate immigrants following Poi(lambda*total_time) for each stance  """


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
    # auto-assign immigrants to users
    user_id = random.randint(0,len(dict_adjacency)-1)
    # save event info into dict_events and dict_userevents
    dict_events[str(event_id)] = {'event_id': str(event_id),\
                                  'time': newtweet[0],\
                                  'stance': newtweet[1],\
                                  'user_id': str(user_id),\
                                  'type':'o',\
                                  'gamma_xs': createVar['gamma_'+newtweet[1]+'s'],\
                                  'gamma_xd': createVar['gamma_'+newtweet[1]+'d'],\
                                  'delta': createVar['delta_o'],\
                                  'omega_s': createVar['omega_s'],\
                                  'omega_d': createVar['omega_d'],\
                                  'convo_id': str(event_id),\
                                  'trgevent_id': '/',\
                                  'influenced by':'/'}
    dict_convos[str(event_id)] = {'event_info':{'event_id': str(event_id),\
                                                'time': newtweet[0],\
                                                'stance': newtweet[1],\
                                                'user_id': str(user_id),\
                                                'type':'o',\
                                                'gamma_xs': createVar['gamma_'+newtweet[1]+'s'],\
                                                'gamma_xd': createVar['gamma_'+newtweet[1]+'d'],\
                                                'delta': createVar['delta_o'],\
                                                'omega_s': createVar['omega_s'],\
                                                'omega_d': createVar['omega_d'],\
                                                'convo_id': str(event_id),\
                                                'trgevent_id': '/',\
                                                'influenced by':'/'},\
                                  'ret':{},\
                                  'quo':{},\
                                  'rply':{}}
    
    event_id += 1


max_event = 4000


df_events = pd.DataFrame(dict_events).T.sort_values(by='time').reset_index(drop=True)
# create a df for saving all accepted events and rejected events
df_eventstemp = pd.DataFrame(df_events, columns = ['event_id','time','type','convo_id','trgevent_id'])


#%%
""" create dictionaries for storing boolean matrix of priorinfluential events & weight matrix for all users """
np_bool = np.empty([num_users,0])
np_matrix1, np_matrix2, np_matrix3 = np.empty([num_users,0]), np.empty([num_users,0]), np.empty([num_users,0])


#%%
# def find_events(uid):
#     df_dummy = (df_curr.user_id.isin(dict_adjacency[uid])) & (df_curr.type!='rply')
#     return (df_dummy|(df_curr.convo_id.isin(df_curr[df_dummy].event_id)&(df_curr.type=='rply')))

# v2: faster
def filtering(uid):
    np_curr = df_curr[['delta','gamma_xs','gamma_xd','omega_s','omega_d']].to_numpy()
    df_dummy = (df_curr.user_id.isin(dict_adjacency[uid])) & (df_curr.type!='rply')
    l_prior = [] if np_bool[int(uid)].size == 0 else df_prior[np_bool[int(uid),]].event_id
    df_dummy = (df_dummy|(df_curr.convo_id.isin(l_prior)&(df_curr.type=='rply')))
    # 1: beta*delta*(\sum gamma*omega) 2: gamma_xs*omega_s 3: gamma_xd*omega_d
    return np.array(df_dummy,dtype=int)*df_curr.user_id.isin(dict_adjacency[uid]).replace({False:beta_i,True:beta_d}).to_numpy()*np_curr[:,0],\
        np_curr[:,1]*np_curr[:,3],np_curr[:,2]*np_curr[:,4], df_dummy

#%%
for i in tqdm(range(max_event)):
    
    df_eventstemp = df_eventstemp.sort_values(by='time').reset_index(drop=True)
    df_events = df_events.sort_values(by='time').reset_index(drop=True)
    
    curr_time = df_eventstemp.iloc[i].time
    df_prior = df_events[df_events.time < curr_time] # not include the current one
    
    # determine the convo_id and trgevent_id
    ## curr event is not a rejected event
    if df_eventstemp.iloc[i].type != 'rejected':
        trgevent_id = df_eventstemp.iloc[i].event_id
        convo_id = df_eventstemp.iloc[i].convo_id
        
        ## if current event is not a rejected event, find this event in df_events
        ## and then process the info of curr event for all users
        df_curr = df_events[df_events.event_id == trgevent_id]

        #start = time.time()
        np_matrix = np.array([filtering(str(j)) for j in range(num_users)])
        np_matrix1 = np.append(np_matrix1,np_matrix[:,0,:],axis=1)
        np_matrix2 = np.append(np_matrix2,np_matrix[:,1,:],axis=1)
        np_matrix3 = np.append(np_matrix3,np_matrix[:,2,:],axis=1)   
        np_bool = np.append(np_bool,np_matrix[:,3,:],axis=1)  
        #print('processing time:', time.time()-start, 'seconds')
        
        # ### update dict_bool
        # dict_bool = {str(j): np.append(dict_bool.get(str(j)),l_bool[j]) for j in range(num_users)}
        # ### update dict_matrix
        # dict_matrix = {str(j):[np.append(arr,l_matrix[j][k]) for k,arr in enumerate(dict_matrix.get(str(j)))] for j in range(num_users)}
        
    # rejected event
    else:
        trgevent_id = df_eventstemp.iloc[i].trgevent_id
        ## if triggering event is quo or o, then the convo_id is the id of the reiggering event itself
        ## if triggering event is ret or rply, then convo_id can also be found through trgevent_id using the same way
        convo_id = dict_events[trgevent_id]['convo_id']
        
        ## if current event is a rejected event, then no event will be processed, and we can directly implement existed weights
    
    # append curr event into df_prior
    df_prior = df_prior.append(df_curr)
    np_time = np.array(curr_time - df_prior.time)
    
    # cumulative intensity at the current time
    lambda_star = (np_matrix1*(np_matrix2*np.exp((-np.array(df_prior.omega_s)*np_time).astype(float))+\
                                  np_matrix3*np.exp((-np.array(df_prior.omega_d)*np_time).astype(float)))).sum()
    
    # generate new event time
    curr_time += -math.log(random.uniform(0,1))/lambda_star if lambda_star != 0.0 else float('inf')
    
    # rejection test
    if curr_time <= total_time:
        # calculate the cumulative intensity at the new event time
        np_time = np.array(curr_time - df_prior.time)
        lambda_Tarray = np.vstack((np_matrix1*np_matrix2*np.exp((-np.array(df_prior.omega_s)*np_time).astype(float)),\
                                   np_matrix1*np_matrix3*np.exp((-np.array(df_prior.omega_d)*np_time).astype(float))))
        
        lambda_T = lambda_Tarray.sum()
        ## check if it's out of intensity bound
        if random.uniform(0,1) <= (lambda_T/lambda_star):
            # now determine who generate the new event, and what type, what stance it is
            ## generate the user index based on the cumulative distribution of all non-zero probabilities
            np_candidates = np.nonzero(lambda_Tarray.sum(axis = 1))
            l_prob = list(lambda_Tarray[np_candidates]/lambda_Tarray[np_candidates].sum())
            index = np.random.choice(np_candidates[0].tolist(),1,l_prob)[0]
            user_id, curr_stance = str(index%num_users), list_stance[index//num_users]
            curr_type = assign_tweet_type()
            
            # convo_id: id the new event is a quo, then convo_id should be the new event id
            # bus still have to save it under the original convo
            if curr_type == 'quo':
                pre_convo_id, convo_id = convo_id, str(event_id)
            # influential events string
            str_ieventid = ','.join(df_prior[np_bool[int(user_id),]].event_id)
            # save events into dict_events, dict_convos, df_events, df_eventstemp
            ## df
            df_events = df_events.append({'event_id': str(event_id),\
                                          'time': curr_time,\
                                          'stance': curr_stance,\
                                          'user_id': str(user_id),\
                                          'type':curr_type,\
                                          'gamma_xs': createVar['gamma_'+curr_stance+'s'],\
                                          'gamma_xd': createVar['gamma_'+curr_stance+'d'],\
                                          'delta': createVar['delta_'+curr_type],\
                                          'omega_s': createVar['omega_s'],\
                                          'omega_d': createVar['omega_d'],\
                                          'convo_id': convo_id,\
                                          'trgevent_id': trgevent_id,\
                                          'influenced by': str_ieventid}, ignore_index=True)
            df_eventstemp = df_eventstemp.append({'event_id': str(event_id),\
                                                  'time': curr_time,\
                                                  'type': curr_type,\
                                                  'convo_id': convo_id,\
                                                  'trgevent_id': trgevent_id}, ignore_index=True)
            ## dict
            dict_events[str(event_id)] = {'event_id': str(event_id),\
                                          'time': curr_time,\
                                          'stance': curr_stance,\
                                          'user_id': str(user_id),\
                                          'type':curr_type,\
                                          'gamma_xs': createVar['gamma_'+curr_stance+'s'],\
                                          'gamma_xd': createVar['gamma_'+curr_stance+'d'],\
                                          'delta': createVar['delta_'+curr_type],\
                                          'omega_s': createVar['omega_s'],\
                                          'omega_d': createVar['omega_d'],\
                                          'convo_id': convo_id,\
                                          'trgevent_id': trgevent_id,\
                                          'influenced by': str_ieventid}
            ### dict_convos
            ### if new event is quo, then have to save it again as a new conversation
            # if curr_type == 'quo':
            #     dict_convos[convo_id] = {'event_info':{'event_id': str(event_id),\
            #                                                     'time': curr_time,\
            #                                                     'stance': curr_stance,\
            #                                                     'user_id': user_id,\
            #                                                     'type': curr_type,\
            #                                                     'convo_id': convo_id,\
            #                                                     'triggered by': trgevent_id,\
            #                                                     'influenced by': str_ieventid},\
            #                                 'ret':{},\
            #                                 'quo':{},\
            #                                 'rply':{}}  
            #     dict_convos[pre_convo_id]['quo'][str(event_id)] = {'event_id': str(event_id),\
            #                                            'time': curr_time,\
            #                                            'stance': curr_stance,\
            #                                            'user_id': user_id,\
            #                                            'type': curr_type,\
            #                                            'convo_id': convo_id,\
            #                                            'triggered by': trgevent_id,\
            #                                            'influenced by': str_ieventid}
            # else:
            #     dict_convos[convo_id][curr_type][str(event_id)] = {'event_id': str(event_id),\
            #                                                            'time': curr_time,\
            #                                                            'stance': curr_stance,\
            #                                                            'user_id': user_id,\
            #                                                            'type': curr_type,\
            #                                                            'convo_id': convo_id,\
            #                                                            'triggered by': trgevent_id,\
            #                                                            'influenced by': str_ieventid}  
            
            event_id += 1
            
        ## if out of bound
        else:
            # save the event into df_eventstemp only
            df_eventstemp = df_eventstemp.append({'event_id': '/',\
                                                  'time': curr_time,\
                                                  'type': 'rejected',\
                                                  'convo_id': convo_id,\
                                                  'trgevent_id': trgevent_id},ignore_index = True)

    if i+1 >= len(df_eventstemp):
        break

#%%

""" save dataset """
# dict_events
with open(os.path.join(path_sys, 'datasets','Scenario4', 'dict_events.json'), 'w+', encoding="utf-8") as outfile:
    json.dump(dict_events, outfile, ensure_ascii=False) 



# dict_params
dict_params = {}
for r in list_type:
    dict_params['delta_'+r] = {'true value': createVar['delta_'+r], 'estimated':''}
for stance in list_stance:
    for new_stance in list_stance:
        dict_params['gamma_'+stance+new_stance] = {'true value': createVar['gamma_'+stance+new_stance], 'estimated':''}
    dict_params['mu_'+stance] = {'true value': createVar['mu_'+stance], 'estimated':''}
    dict_params['omega_'+stance] = {'true value': createVar['omega_'+stance], 'estimated':''}
dict_params['beta_d'] = {'true value': beta_d, 'estimated':''}
dict_params['beta_i'] = {'true value': beta_i, 'estimated':''}
    
with open(os.path.join(path_sys, 'datasets','Scenario4', 'dict_params.json'), 'w+', encoding="utf-8") as outfile:
    json.dump(dict_params, outfile, ensure_ascii=False) 




# dict_adjacency
with open(os.path.join(path_sys, 'datasets','Scenario4', 'dict_adjacency.json'), 'w+', encoding="utf-8") as outfile:
    json.dump(dict_adjacency, outfile, ensure_ascii=False) 



# # dict_convos
# with open(os.path.join(path_sys, 'datasets','Scenario4', 'dict_convos.json'), 'w+', encoding="utf-8") as outfile:
#     json.dump(dict_convos, outfile, ensure_ascii=False) 



# # dict_userevents
# with open(os.path.join(path_sys, 'datasets','Scenario4', 'dict_userevents.json'), 'w+', encoding="utf-8") as outfile:
#     json.dump(dict_userevents, outfile, ensure_ascii=False) 



