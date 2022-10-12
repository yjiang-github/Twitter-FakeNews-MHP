# -*- coding: utf-8 -*-
"""
Created on Tue Feb  8 16:27:44 2022

@author: yjian
"""


"""
--- this file is the version6 of simulation in scenario4 using the overall internsity---
--- specifically, this approach will store all prior values into numpy such that each time the process
    will only calculate the values of current event for all users ---

"""

#%%
from __future__ import division
from __future__ import print_function

import matplotlib.pyplot as plt
import seaborn as sns
import json
import os
from tqdm import tqdm
import pandas as pd
import numpy as np
import random
from scipy.stats import truncexpon
import math
from datetime import datetime


#%%
createVar = locals()

#%%
""" import user network """
path = ''
with open(os.path.join(path, 'dict_adjacency.json'), 'r',  encoding="utf-8") as f:
    dict_adjacency = json.load(f)

dict_adjpairs = {}

for i in tqdm(range(len(dict_adjacency))):
    for j in range(len(dict_adjacency)):
        # i has a friend j, such that i can see j's tweet: j's tweets (except rplies) can influence i
        dict_adjpairs[str(i)+'_'+str(j)] = True if str(j) in dict_adjacency[str(i)] else False

# temporarily set num_users to be 12
num_users = len(dict_adjacency)

#%%

"""
--- twitter data simulation ---
--- Scenario4: original tweets with retweets quotes and replies;
            2 stances: supporting, and denying ---    
            
--- this time we add n_l into the model, which is the relationships between users ---        

--- assumtpion: 
    a supporting tweet has 0.9 chance of generating a supporting tweet,
    0.1 chance of generating a denying tweet;
    
    a denying tweet has 0.5 chance of generating a supporting tweet,
    0.5 chance of generating a denying tweet;
    
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
list_destype = ['ret','quo','rply']

dict_stance = {stance:list_stance.index(stance) for stance in list_stance}

delta = 0.0001

delta_o = 25*delta
delta_ret = 0.45*delta
delta_quo = 0.045*delta
delta_rply = 0.15*delta

gamma_ss = 0.9
gamma_sd = 0.1
gamma_ds = 0.5
gamma_dd = 0.5

beta_d = 0.95
beta_i = 0.05

mu_s, mu_d = 0.15, 0.015
omega_s, omega_d = 3.5, 1.75

# probability of generating different types of event: sum to 1
num_destypes = 3
p_ret = 0.78
p_quo = 0.02
p_rply = 0.2

# assign tweet_type for the new-generated tweet
def assign_tweet_type():
    
    num = random.uniform(0,1)
    if num < p_ret:
        tweet_type = 'ret'
    elif p_ret <= num < p_ret + p_quo:
        tweet_type = 'quo'
    else:
        tweet_type = 'rply'
        
    return tweet_type


total_time = 6000

event_id = 0 # tweet id/label

dict_events = {} # save events
dict_convos = {} # save events as groups and conversations
#dict_userevents = {} # save events for each user


# for user_id in dict_adjacency.keys():
#     dict_userevents[user_id] = {}

#%%
""" --- simulation --- """
""" first step: generate immigrants """


""" generate immigrants - 
approach: following truncated expo(1/lambda) for each stance 
Use scipy.stats.truncexpon() function where bounds and scale param should be set

"""
lower, upper, scale = 0, total_time, total_time/6

dist = truncexpon(b=(upper-lower)/scale, loc=lower, scale=scale)
#l = dist.rvs(size = 100, random_state = 2022)

list_time = []

for stance in list_stance:
    mu = createVar['mu_'+stance[0]]
    num_imm = np.random.poisson(lam = mu*total_time)
    """ arrival time of immigrants should be expo(mean, num_events)"""
    list_time += [(time, stance) for time in dist.rvs(size = num_imm)]


#%%

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

    
    event_id += 1

# max_event = 20
max_event = 10000 


df_events = pd.DataFrame(dict_events).T.sort_values(by='time').reset_index(drop=True)
# create a df for saving all accepted events and rejected events
df_eventstemp = pd.DataFrame(df_events, columns = ['event_id','time','type','convo_id','trgevent_id'])


""" create numpy array for storing boolean matrix of prior influential events & weight matrix for all users """
""" create event type array for all combinations of users and stance """
np_bool = np.empty([num_users,len(df_events)+max_event], dtype = bool)

np_matrix1, np_matrix2, np_matrix3 = np.empty([num_users,len(df_events)+max_event]), \
    np.empty([num_users,len(df_events)+max_event]), np.empty([num_users,len(df_events)+max_event])

""" status array: describes what combinations of event type, user and stance are possible for the new events
## row: stance1*(user0, user1, user2 ... usern), stance2*(user0, user1, user2 ... usern)
## column: ret, quo, rply
"""
np_eventtype = np.stack([np.full((num_users*len(list_stance),), p_ret),np.full((num_users*len(list_stance),), p_quo),np.full((num_users*len(list_stance),), p_rply)], axis=1)

# remove the possibility that the users who generate original events will generate new events(except rply) in the same stance
for index in tqdm(df_events.index):
    user_id = df_events.iloc[index]['user_id']
    index_stance = list_stance.index(df_events.iloc[index]['stance'])
    index_comb = num_users*index_stance + int(user_id)
    np_eventtype[index_comb][0], np_eventtype[index_comb][1] = 0, 0

""" stance vector: whether the event is in stance k 
    default val is True
"""
np_stance_s = np.full([len(df_events)+max_event,], True)
np_stance_d = np.full([len(df_events)+max_event,], True)


#%%


def check_beta(dummy):
    return beta_d if dummy else beta_i

def filtering(uid):
    
    np_curr = df_curr[['delta','gamma_xs','gamma_xd','omega_s','omega_d']].to_numpy()
    
    # if curr event is generated by a user that directly connects to uid & curr event is not a rply
    df_dummy = (dict_adjpairs[uid+'_'+df_curr.user_id.iloc[0]]) & (df_curr.type!='rply')
    
    l_prior = [] if index == 0 else df_prior[np_bool[int(uid),:index]].event_id
    
    ## also, consider curr event as one rply from prior influential tweets
    df_dummy = (df_dummy|(df_curr.convo_id.isin(l_prior)&(df_curr.type=='rply')))
    
    # 1: beta*delta*(\sum gamma*omega) 2: gamma_xs*omega_s 3: gamma_xd*omega_d
    return np.array(df_dummy,dtype=int)*check_beta(dict_adjpairs[uid+'_'+df_curr.user_id.iloc[0]])*np_curr[:,0],\
        np_curr[:,1]*np_curr[:,3],np_curr[:,2]*np_curr[:,4], df_dummy



#%%
""" --- simulation --- """
""" second step: generate descendants """


index, i = 0, 0

progress_time, start_time = 0, datetime.now()

print('program starts, current time is {}'.format(start_time))


#for i in tqdm(range(max_event)):
while progress_time <= total_time:    # df_events.shape[0] <= len(list_time) + max_event

    df_eventstemp = df_eventstemp.sort_values(by='time').reset_index(drop=True)
    df_events = df_events.sort_values(by='time').reset_index(drop=True)
    
    curr_time, progress_time = df_eventstemp.iloc[i].time, df_eventstemp.iloc[i].time
    print('current event number is: {0:}, current simulation time:{1:}, current progress: {2:.0%}, time duration: {time}'.format(
            df_events.shape[0] - len(list_time), progress_time, progress_time/total_time, 
            time=datetime.now() - start_time))
    
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
        np_matrix1[:,index] = np_matrix[:,0,:].reshape([num_users,])
        np_matrix2[:,index] = np_matrix[:,1,:].reshape([num_users,])
        np_matrix3[:,index] = np_matrix[:,2,:].reshape([num_users,])
        np_bool[:,index] = np_matrix[:,3,:].reshape([num_users,])
        
        np_stance_s[index] = df_curr.iloc[0]['stance'] == 's'
        np_stance_d[index] = df_curr.iloc[0]['stance'] == 'd'
        
        
        index += 1

    ## rejected event
    else:
        trgevent_id = df_eventstemp.iloc[i].trgevent_id
        ## if triggering event is quo or o, then the convo_id is the id of the reiggering event itself
        ## if triggering event is ret or rply, then convo_id can also be found through trgevent_id using the same way
        convo_id = dict_events[trgevent_id]['convo_id']
        
        ## if current event is a rejected event, then no event will be processed, and we can directly implement existed weights
    
    # append curr event into df_prior
    if len(df_prior) > 0 and not df_curr.equals(df_prior.iloc[-1].to_frame().T):
        df_prior = df_prior.append(df_curr)
        
    np_time = np.array(curr_time - df_prior.time)
    
    # cumulative intensity at the current time
    # beta*delta* [(gamma_xs*omega_s*kernel*sum of prob of event types)+(gamma_xd*omega_d*kernel*prob of event types)]
    lambda_star = (np_matrix1[:,:index]*
                       (np_matrix2[:,:index]*np.exp((-np.array(df_prior.omega_s)*np_time).astype(float))*\
                            # ret (same stance as the prior event)
                            (np_stance_s[:index]*np_eventtype[:num_users,0].reshape(-1,1)+\
                             # quo & rply (do not need to be in the same stance)
                             np_eventtype[:num_users,1:].sum(axis=1).reshape(-1,1)) + \
                                                 
                        np_matrix3[:,:index]*np.exp((-np.array(df_prior.omega_d)*np_time).astype(float))*\
                            # ret (same stance as the prior event)
                            (np_stance_d[:index]*np_eventtype[num_users:,0].reshape(-1,1)+\
                             # quo & rply (do not need to be in the same stance)
                             np_eventtype[num_users:,1:].sum(axis=1).reshape(-1,1)))).sum()
                                         
                                         
    
    # generate new event time
    curr_time += -math.log(random.uniform(0,1))/lambda_star if lambda_star != 0.0 else float('inf')
    
    # rejection test
    if curr_time <= total_time:
        # calculate the cumulative intensity at the new event time
        np_time = np.array(curr_time - df_prior.time)
        ## calculate each combination of user, stance, and event type seperately
        np_matrix_s = np_matrix1[:,:index]*np_matrix2[:,:index]*np.exp((-np.array(df_prior.omega_s)*np_time).astype(float))
        np_matrix_d = np_matrix1[:,:index]*np_matrix3[:,:index]*np.exp((-np.array(df_prior.omega_d)*np_time).astype(float))
        ### stance_s*ret, stance_s*quo, stance_s*rply
        
        ### stance_d*ret, stance_d*quo, stance_d*rply        
        lambda_Tarray = np.vstack((
                            np.stack([(np_matrix_s*np_stance_s[:index]*np_eventtype[:num_users,0].reshape(-1,1)).sum(axis=1),
                                      (np_matrix_s*np_eventtype[:num_users,1].reshape(-1,1)).sum(axis=1),
                                      (np_matrix_s*np_eventtype[:num_users,2].reshape(-1,1)).sum(axis=1)], axis = 1),
                            np.stack([(np_matrix_d*np_stance_d[:index]*np_eventtype[num_users:,0].reshape(-1,1)).sum(axis=1),
                                      (np_matrix_d*np_eventtype[num_users:,1].reshape(-1,1)).sum(axis=1),
                                      (np_matrix_d*np_eventtype[num_users:,2].reshape(-1,1)).sum(axis=1)],axis = 1)))
        
        lambda_T = lambda_Tarray.sum()
        ## check if it's out of intensity bound
        print('lambda_T/lambda_star ==',lambda_T/lambda_star)
        if random.uniform(0,1) <= (lambda_T/lambda_star):
            
            # if new event is accepted, then randomly generate the new event with user, stance and event type
            # based on lambda_Tarray
            ### total combs: # of stance * # of users * # of event types
            lambda_Tcombs = lambda_Tarray.reshape(-1,)/lambda_T
            index_user = np.random.choice(np.arange(len(lambda_Tcombs)), 1, p = lambda_Tcombs)[0]
            
            # match the user_id, eventtype and stance
            index_stance = index_user//(num_users*num_destypes)
            curr_stance = list_stance[index_stance]
            index_usertype = index_user%(num_users*num_destypes)
            user_id = index_usertype//num_destypes
            index_type = index_user - (index_stance*(num_users*num_destypes)+user_id*num_destypes)
            curr_type = list_destype[index_type]
            
            # if new event is a ret/quo, then the prob of generating a ret/quo in the same stance becomes 0
            if curr_type in ['ret','quo']:
                index_comb = num_users*index_stance + int(user_id)
                np_eventtype[index_comb][0], np_eventtype[index_comb][1] = 0, 0
            
            
            # convo_id: id the new event is a quo, then convo_id should be the new event id
            # bus still have to save it under the original convo
            if curr_type == 'quo':
                pre_convo_id, convo_id = convo_id, str(event_id)
            # influential events string
            str_ieventid = ','.join(df_prior[np_bool[int(user_id),:index]].event_id)
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
    
    i += 1
    
#%%

""" save dataset """
# dict_events
with open(os.path.join(path, 'dict_events.json'), 'w+', encoding="utf-8") as outfile:
    json.dump(dict_events, outfile, ensure_ascii=False) 


# dict_params
dict_params = {}
for r in list_type:
    dict_params['delta_'+r] = {'true value': createVar['delta_'+r]}
    if r != 'o':
        dict_params['p_'+r] = {'true value': createVar['p_'+r]}
for stance in list_stance:
    for new_stance in list_stance:
        dict_params['gamma_'+stance+new_stance] = {'true value': createVar['gamma_'+stance+new_stance]}
    dict_params['mu_'+stance] = {'true value': createVar['mu_'+stance]}
    dict_params['omega_'+stance] = {'true value': createVar['omega_'+stance]}
dict_params['beta_d'] = {'true value': beta_d}
dict_params['beta_i'] = {'true value': beta_i}
    
with open(os.path.join(path, 'dict_params.json'), 'w+', encoding="utf-8") as outfile:
    json.dump(dict_params, outfile, ensure_ascii=False) 



