# -*- coding: utf-8 -*-
"""
Created on Tue Feb  8 01:01:09 2022

@author: yjian
"""


"""
--- this file is for setting up a configuration file that can be imported into 
    multiple modules as global variables ---

"""
#%%
import os
import json
#%%
""" import user network """
path_sim = r'C:\Users\yjian\OneDrive\Documents\research files\dissertation\Twitter Fake News\simulation'
with open(os.path.join(path_sim,'network generator','dict_adjacency.json'), 'r',  encoding="utf-8") as f:
    dict_adjacency = json.load(f)

for user_id in dict_adjacency.keys():
    dict_adjacency[user_id] = list(dict_adjacency[user_id].keys())

#%%
# temporarily set num_users to be 12
num_users = 12#len(dict_adjacency)
# list of connected users for each user, list of user_id
l_uid = [str(i) for i in range(num_users)]

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




