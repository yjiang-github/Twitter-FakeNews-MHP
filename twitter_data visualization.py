# -*- coding: utf-8 -*-
"""
Created on Tue Jan  5 14:58:10 2021

@author: yjian
"""


"""
--- fake news data visualization ---

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
import math
import sys
path_sys = r'C:\Users\yjian\OneDrive\Documents\research files\dissertation\datasets'
sys.path.append(path_sys)




import copy,subprocess,itertools,pickle,warnings,gc,numbers
import numpy as np
import scipy as sp
from matplotlib import pyplot as plt

import Hawkes as hk


#%%
""" import the dataset """
""" dataset 1: Obama has bee arrested for espionage """
""" subdataset: conservativebeaver """

path = r"C:\Users\yjian\OneDrive\Documents\research files\dissertation\datasets\data collection\obama arrested for espionage" 

with open(os.path.join(path, 'conservativebeaver.json'), 'r', encoding="utf-8") as file:
    for line in file.readlines():
        dict_tweets = json.loads(line)

#%%
""" import the dataset """
""" dataset 2: covid vaccine female sterilization """
""" all subdatasets: dict_keyword, dict_url01 & dict_url02 """

path = r'C:\Users\yjian\OneDrive\Documents\research files\dissertation\datasets\data collection\covid vaccine female sterilization'

filenames = os.listdir(path)

dict_tweets = {}

for filename in filenames:
    if filename.endswith('.json'):
        with open(os.path.join(path, filename), 'r', encoding="utf-8") as file:
            for line in file.readlines():
                dict_temp = json.loads(line)
        
        for tweet_id in dict_temp.keys():
            if tweet_id not in dict_tweets.keys():
                dict_tweets[tweet_id] = dict_temp[tweet_id]

#%%
        
""" extract key info and convert time """

dict_data = {}

for tweet_id in tqdm(dict_tweets.keys()):
    
    time = datetime.strftime(datetime.strptime(dict_tweets[tweet_id]['created_at'],'%a %b %d %H:%M:%S +0000 %Y'), '%Y-%m-%d %H:%M:%S')
    dict_data[tweet_id] = {}
    dict_data[tweet_id]['text'] = dict_tweets[tweet_id]['full_text']
    dict_data[tweet_id]['time'] = time
    dict_data[tweet_id]['time_num'] = date2num(datetime.strptime(dict_tweets[tweet_id]['created_at'],'%a %b %d %H:%M:%S +0000 %Y'))
    # if quote
    if 'is_quote_status' in dict_tweets[tweet_id].keys() and dict_tweets[tweet_id]['is_quote_status'] == True:
        dict_data[tweet_id]['is_quote'] = 1
        dict_data[tweet_id]['is_retweet'] = 0
        dict_data[tweet_id]['is_reply'] = 0
        
    else:
        dict_data[tweet_id]['is_quote'] = 0
        # if retweet
        if 'retweeted_status' in dict_tweets[tweet_id].keys() and type(dict_tweets[tweet_id]['retweeted_status']) is dict:
            dict_data[tweet_id]['is_retweet'] = 1
            dict_data[tweet_id]['is_reply'] = 0
            
        else:
            dict_data[tweet_id]['is_retweet'] = 0
            # if reply
            if 'in_reply_to_status_id' in dict_tweets[tweet_id].keys() and type(dict_tweets[tweet_id]['in_reply_to_status_id']) is int:
                dict_data[tweet_id]['is_reply'] = 1
                
            else:
                dict_data[tweet_id]['is_reply'] = 0
                
    
#%%
    
df_data = pd.DataFrame.from_dict(dict_data).T.sort_values(by = 'time',axis = 0,ascending = True)
df_data.time_num = df_data.time_num.astype(float)


#%%

list_standardtime = []

# the time_num of the first tweet is 737758.7503009259
initial_num = int(df_data.iloc[0]['time_num'])

for tweet_id in tqdm(df_data.index.values):
    list_standardtime.append(df_data.loc[tweet_id]['time_num'] - initial_num)
    
df_data['time_num_hp'] = list_standardtime

#%%
""" fitting the Hawkes model and estimate the parameters """

itv = [0,1000]
T = df_data.time_num_hp
h2 = hk.estimator().set_kernel('exp').set_baseline('const')
h2.fit(T,itv)
print(h2.para)
print(h2.L)

alpha = h2.para['alpha']
delta = h2.para['beta']

#%%

""" calculate intensity values for each tweet"""

intensity = 0
list_intensity = []

for tweet_id in tqdm(df_data.index.values):
    
    intensity = 0
    # calculate intensity for current tweet
    intensity += alpha
    
    # calculate intensity for all past tweets
    ## get index
    index = list(df_data.index.values).index(tweet_id)
    list_pasttweet = list(df_data.index.values[0:index])
    list_pasttweet.sort(reverse = True)
    
    for past_id in list_pasttweet:
        ### calculate decay intensity for the past tweet
        timediff = dict_data[tweet_id]['time_num'] - dict_data[past_id]['time_num']
        decay = math.e ** (-(timediff) * delta)
        decay = decay * delta * alpha
        
        if decay < 1e-5:
            break
        else:
            intensity += decay
    list_intensity.append(intensity)
        
#%%
""" save intensity into df_data """
df_data['intensity'] = list_intensity

#%%
""" plot """

plt.style.use('ggplot')
fig = plt.figure(dpi = 80, figsize = (30, 10))
fig.suptitle('Intensity: Head of Pfizer Research_Covid Vaccine is Female Sterilization'+str(), fontsize=30)
plt.plot(num2date(df_data.time_num), df_data.intensity,color='k',label='Intensity')
plt.tick_params(labelsize=20,rotation=45)
#plt.xticks(index, y_test.index.values,fontsize = 16,rotation=45)
plt.xlabel('Datetime', fontsize=25)
plt.ylabel('Intensity', fontsize=25)
plt.legend()
plt.show()


#%%
""" find the earliest tweet that contains 'fact check' or 'fake' """

df_factcheck = pd.DataFrame()

for tweet_id in df_data.index.values:
    
    text = df_data.loc[tweet_id]['text'].lower()
    if 'fact check' in text or 'fake' in text:
        print(df_data.loc[tweet_id]['text'])
        print('tweet_id:' +str(tweet_id))
        print(df_data.loc[tweet_id]['time'])
        
        df_factcheck = pd.concat([df_factcheck,df_data.loc[tweet_id].to_frame().T],axis=0)
        
#%%
""" save these tweets """

path_meeting = r'C:\Users\yjian\OneDrive\Documents\research files\Meeting notes\20210106'
df_factcheck.to_csv(os.path.join(path_meeting,'Head of Pfizer Research_Covid Vaccine is Female Sterilization_fact check.csv'),index=True,quoting=1)



