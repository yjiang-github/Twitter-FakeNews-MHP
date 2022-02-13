# -*- coding: utf-8 -*-
"""
Created on Thu Feb  3 13:44:54 2022

@author: yjian
"""



"""
--- this file is for testing the new simulation method (approach 1 from Prof. Porter) ---
--- using multiprocessing package ---

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
import timeit
import pandas as pd
import numpy as np
import math
import time
import sys
path_sys = r'C:\Users\yjian\OneDrive\Documents\research files\dissertation\Twitter Fake News\simulation'
sys.path.append(path_sys)
import Hawkes as hk
from multiprocessing import Pool, cpu_count
from twitter_filteringfunction import*
from config import*


    
#%%
createVar = locals()

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


max_event = 10


df_events = pd.DataFrame(dict_events).T.sort_values(by='time').reset_index(drop=True)
# create a df for saving all accepted events and rejected events
df_eventstemp = pd.DataFrame(df_events, columns = ['event_id','time','type','convo_id','trgevent_id'] )

#%%
l = [(1,1,1),(2,2,2)]
dict_temp = {'0':[0,0,0],'1':[1,1,1]}

for i,num in enumerate(l):
    dict_temp[str(i)] = [dict_temp.get(str(i),[0,0,0])[j]+num[j] for j in range(len(dict_temp.get(str(i),[0,0,0])))]




    
#%%
num_processes = 10
chunk_size = int(num_users/num_processes)
l_temp = [[str(j) for j in range(i,i+chunk_size)] for i in range(0,num_users,chunk_size)]

if __name__ == '__main__':
    q = Queue()
    start = time.time()
    for l_uid in l_temp:
        p = Process(target = filtering_multiusers, args = (df_curr,l_uid,q))
        p.Daemon = True
        p.start()
    for l_uid in l_temp:
        p.join()
    print('processing time:',time.time()-start,' seconds')
    l = q.get()
    
#%%
if __name__ == '__main__':

    for i in tqdm(range(max_event)):
        
        df_eventstemp = df_eventstemp.sort_values(by='time').reset_index(drop=True)
        df_events = df_events.sort_values(by='time').reset_index(drop=True)
        
        curr_time = df_eventstemp.iloc[i].time
        
        
        # determine the convo_id and trgevent_id
        ## curr event is not a rejected event
        if df_eventstemp.iloc[i].type != 'rejected':
            trgevent_id = df_eventstemp.iloc[i].event_id
            convo_id = df_eventstemp.iloc[i].convo_id
        # rejected event
        else:
            trgevent_id = df_eventstemp.iloc[i].trgevent_id
            ## if triggering event is quo or o, then the convo_id is the id of the reiggering event itself
            ## if triggering event is ret or rply, then convo_id can also be found through trgevent_id using the same way
            convo_id = dict_events[trgevent_id]['convo_id']
        
        
        df_curr = df_events[df_events.time <= curr_time]
        # first dim: 
            
        np_temp = main(df_curr,l_uid)
        np_time = np.array(curr_time - df_curr.time)
        
        # cumulative intensity at the current time
        lambda_star = (np_temp[:,0,:]*(np_temp[:,1,:]*np.exp((-np.array(df_curr.omega_s)*np_time).astype(float))+\
                                      np_temp[:,2,:]*np.exp((-np.array(df_curr.omega_d)*np_time).astype(float)))).sum()
        
        # generate new event time
        curr_time += -math.log(random.uniform(0,1))/lambda_star if lambda_star != 0.0 else float('inf')
        
        # rejection test
        if curr_time <= total_time:
            # calculate the cumulative intensity at the new event time
            np_time = np.array(curr_time - df_curr.time)
            lambda_Tarray = np.vstack((np_temp[:,0,:]*np_temp[:,1,:]*np.exp((-np.array(df_curr.omega_s)*np_time).astype(float)),\
                                       np_temp[:,0,:]*np_temp[:,2,:]*np.exp((-np.array(df_curr.omega_d)*np_time).astype(float))))
            
            lambda_T = lambda_Tarray.sum()
            ## check if it's out of intensity bound
            if random.uniform(0,1) <= (lambda_T/lambda_star):
                # now determine who generate the new event, and what type, what stance it is
                ## if there's more than one user who has the hightest intensity, then pick up the first one
                index = np.where(lambda_Tarray.sum(axis=1)==lambda_Tarray.sum(axis=1).max())[0][0]
                user_id, curr_stance = str(index%num_users), list_stance[index//num_users]
                curr_type = assign_tweet_type()
                
                # convo_id: id the new event is a quo, then convo_id should be the new event id
                # bus still have to save it under the original convo
                if curr_type == 'quo':
                    pre_convo_id, convo_id = convo_id, str(event_id)
                # influential events string
                str_ieventid = ','.join(df_curr[find_events(df_curr,user_id)].event_id)
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
                if curr_type == 'quo':
                    dict_convos[convo_id] = {'event_info':{'event_id': str(event_id),\
                                                                    'time': curr_time,\
                                                                    'stance': curr_stance,\
                                                                    'user_id': user_id,\
                                                                    'type': curr_type,\
                                                                    'convo_id': convo_id,\
                                                                    'triggered by': trgevent_id,\
                                                                    'influenced by': str_ieventid},\
                                                'ret':{},\
                                                'quo':{},\
                                                'rply':{}}  
                    dict_convos[pre_convo_id]['quo'][str(event_id)] = {'event_id': str(event_id),\
                                                           'time': curr_time,\
                                                           'stance': curr_stance,\
                                                           'user_id': user_id,\
                                                           'type': curr_type,\
                                                           'convo_id': convo_id,\
                                                           'triggered by': trgevent_id,\
                                                           'influenced by': str_ieventid}
                else:
                    dict_convos[convo_id][curr_type][str(event_id)] = {'event_id': str(event_id),\
                                                                           'time': curr_time,\
                                                                           'stance': curr_stance,\
                                                                           'user_id': user_id,\
                                                                           'type': curr_type,\
                                                                           'convo_id': convo_id,\
                                                                           'triggered by': trgevent_id,\
                                                                           'influenced by': str_ieventid}  
                
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

        
#%%


for i in tqdm(range(max_event)):
    
    df_eventstemp = df_eventstemp.sort_values(by='time').reset_index(drop=True)
    df_events = df_events.sort_values(by='time').reset_index(drop=True)
    
    curr_time = df_eventstemp.iloc[i].time
    
    
    # determine the convo_id and trgevent_id
    ## curr event is not a rejected event
    if df_eventstemp.iloc[i].type != 'rejected':
        trgevent_id = df_eventstemp.iloc[i].event_id
        convo_id = df_eventstemp.iloc[i].convo_id
    # rejected event
    else:
        trgevent_id = df_eventstemp.iloc[i].trgevent_id
        ## if triggering event is quo or o, then the convo_id is the id of the reiggering event itself
        ## if triggering event is ret or rply, then convo_id can also be found through trgevent_id using the same way
        convo_id = dict_events[trgevent_id]['convo_id']
    
    
    df_curr = df_events[df_events.time < curr_time]
    # first dim: 
        
    np_temp = np.array([filtering(df_curr,str(j)) for j in range(num_users)])
    np_time = np.array(curr_time - df_curr.time)
    
    # cumulative intensity at the current time
    lambda_star = (np_temp[:,0,:]*(np_temp[:,1,:]*np.exp((-np.array(df_curr.omega_s)*np_time).astype(float))+\
                                  np_temp[:,2,:]*np.exp((-np.array(df_curr.omega_d)*np_time).astype(float)))).sum()
    
    # generate new event time
    curr_time += -math.log(random.uniform(0,1))/lambda_star if lambda_star != 0.0 else float('inf')
    
    # rejection test
    if curr_time <= total_time:
        # calculate the cumulative intensity at the new event time
        np_time = np.array(curr_time - df_curr.time)
        lambda_Tarray = np.vstack((np_temp[:,0,:]*np_temp[:,1,:]*np.exp((-np.array(df_curr.omega_s)*np_time).astype(float)),\
                                   np_temp[:,0,:]*np_temp[:,2,:]*np.exp((-np.array(df_curr.omega_d)*np_time).astype(float))))
        
        lambda_T = lambda_Tarray.sum()
        ## check if it's out of intensity bound
        if random.uniform(0,1) <= (lambda_T/lambda_star):
            # now determine who generate the new event, and what type, what stance it is
            ## if there's more than one user who has the hightest intensity, then pick up the first one
            index = np.where(lambda_Tarray.sum(axis=1)==lambda_Tarray.sum(axis=1).max())[0][0]
            user_id, curr_stance = str(index%num_users), list_stance[index//num_users]
            curr_type = assign_tweet_type()
            
            # convo_id: id the new event is a quo, then convo_id should be the new event id
            # bus still have to save it under the original convo
            if curr_type == 'quo':
                pre_convo_id, convo_id = convo_id, str(event_id)
            # influential events string
            str_ieventid = ','.join(df_curr[find_events(df_curr,user_id)].event_id)
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
            if curr_type == 'quo':
                dict_convos[convo_id] = {'event_info':{'event_id': str(event_id),\
                                                                'time': curr_time,\
                                                                'stance': curr_stance,\
                                                                'user_id': user_id,\
                                                                'type': curr_type,\
                                                                'convo_id': convo_id,\
                                                                'triggered by': trgevent_id,\
                                                                'influenced by': str_ieventid},\
                                            'ret':{},\
                                            'quo':{},\
                                            'rply':{}}  
                dict_convos[pre_convo_id]['quo'][str(event_id)] = {'event_id': str(event_id),\
                                                       'time': curr_time,\
                                                       'stance': curr_stance,\
                                                       'user_id': user_id,\
                                                       'type': curr_type,\
                                                       'convo_id': convo_id,\
                                                       'triggered by': trgevent_id,\
                                                       'influenced by': str_ieventid}
            else:
                dict_convos[convo_id][curr_type][str(event_id)] = {'event_id': str(event_id),\
                                                                       'time': curr_time,\
                                                                       'stance': curr_stance,\
                                                                       'user_id': user_id,\
                                                                       'type': curr_type,\
                                                                       'convo_id': convo_id,\
                                                                       'triggered by': trgevent_id,\
                                                                       'influenced by': str_ieventid}  
            
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

def find_events(uid):
    df_temp = (df_curr.user_id.isin(dict_adjacency[uid])) & (df_curr.type!='rply')
    return (df_temp|(df_curr.convo_id.isin(df_curr[df_temp].event_id)&(df_curr.type=='rply')))

# v2: faster
def filtering(uid):
    df_temp = find_events(uid).replace({False:0,True:1})
    np_curr = df_curr[['delta','gamma_xs','gamma_xd','omega_s','omega_d']].to_numpy()
    
    # 1: beta*delta*(\sum gamma*omega) 2: gamma_xs*omega_s 3: gamma_xd*omega_d
    return df_temp.to_numpy()*df_curr.user_id.isin(dict_adjacency[uid]).replace({False:beta_i,True:beta_d}).to_numpy()*np_curr[:,0],\
        np_curr[:,1]*np_curr[:,3],np_curr[:,2]*np_curr[:,4]