# -*- coding: utf-8 -*-
"""
Created on Mon Feb  7 20:01:55 2022

@author: yjian
"""


"""
--- this file is for seperating the functions as a simgle file ---
--- then import the function in order to run the pool.map function ---

"""
#%%
from multiprocessing import Process, Pool, Queue, cpu_count
from itertools import repeat
import random
import numpy as np
import pandas as pd
import sys
path_sys = r'C:\Users\yjian\OneDrive\Documents\research files\dissertation\Twitter Fake News\simulation'
sys.path.append(path_sys)
from config import *

#%%


""" --- define function: assign tweet type and tweet stance --- """

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

""" --- define function: return bool values for influential prior tweets """
def find_events(df_curr,uid):
    df_temp = (df_curr.user_id.isin(dict_adjacency[str(uid)])) & (df_curr.type!='rply')
    return (df_temp|(df_curr.convo_id.isin(df_curr[df_temp].event_id)&(df_curr.type=='rply')))

""" --- define function: calculate weight matrices --- """
# v2: faster
def filtering(df_curr,uid):
    
    df_temp = (find_events(df_curr,uid)).replace({False:0,True:1})
    np_curr = df_curr[['delta','gamma_xs','gamma_xd','omega_s','omega_d']].to_numpy()
    
    # 1: beta*delta*(\sum gamma*omega) 2: gamma_xs*omega_s 3: gamma_xd*omega_d
    return df_temp.to_numpy()*df_curr.user_id.isin(dict_adjacency[uid]).replace({False:beta_i,True:beta_d}).to_numpy()*np_curr[:,0],\
        np_curr[:,1]*np_curr[:,3],np_curr[:,2]*np_curr[:,4]

def filtering_multiusers(df_curr,l_uid,q):
    q.put([filtering(df_curr,uid) for uid in l_uid])

""" define function: multiprocessing """
def main(df_curr,l_uid):
    
    pool = Pool(processes = cpu_count())
    result = pool.starmap_async(filtering, zip(repeat(df_curr),l_uid))
    pool.close()
    pool.join()
    return np.array(result)
