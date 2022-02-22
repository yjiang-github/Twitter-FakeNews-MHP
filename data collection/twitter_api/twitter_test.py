# -*- coding: utf-8 -*-
"""
Created on Mon Nov 29 14:17:26 2021

@author: yjian
"""


"""
This file is for testing the New Twitter API key and token, which is accessable to free full archive

"""
#%%
"""
import packages
"""
import json
import os
import tweepy
import re
import time
from datetime import datetime, timedelta
from apscheduler.schedulers.background import BackgroundScheduler
import pandas as pd
from pytz import timezone
import logging
import OpenSSL
import datetime



#%%
createVar = locals()

#%%
"""
--- collecting tweets associated with fake news of "obama was arrested for espionage ---
"""

#%%
"""
--- key & token ---

"""

consumer_key_1 = ''  
consumer_secret_1 = ''  
access_token_key_1 = ''  
access_token_secret_1 = ''  
bearer_token = ''


"""
--- API ---
​
"""
auth_1 = tweepy.AppAuthHandler(consumer_key_1, consumer_secret_1)
api_1 = tweepy.API(auth_1)



#%%

keyword01 = 'obama arrest espionage'
url01 = 'url:'+'www.conservativebeaver.com/2020/11/28/former-president-barack-obama-arrested-for-espionage'

#%%
dict_keyword01 = {}
dict_url01 = {}

#%%


max_num = 100000

i = 0 # 0/1


# initial api
API = api_1

# Start and end times must be in UTC
start_time = datetime.datetime(2020, 11, 27, 0, 0, 0, 0, datetime.timezone.utc)
end_time = datetime.datetime(2021, 3, 27, 0, 0, 0, 0, datetime.timezone.utc)




#%%
""" search tweets by query (keywork or url) """

tweets = tweepy.Cursor(API.search_full_archive, label='fakenews', query=url01, fromDate=202011270000, toDate=202102270000, maxResults=100).items()


#%%


for tweet in tweets:
    if tweet.id not in dict_keyword01.keys():
        dict_url01[tweet.id] = tweet._json

#%%

""" save json """

path = r'C:\Users\yjian\OneDrive\Documents\research files\dissertation\Twitter Fake News\datasets\data collection\obama arrested for espionage'
with open(os.path.join(path, 'tweets_keyword01.json'), 'w+', encoding="utf-8") as outfile:
    json.dump(dict_keyword01, outfile, ensure_ascii=False) 



with open(os.path.join(path, 'tweets_url01.json'), 'w+', encoding="utf-8") as outfile:
    json.dump(dict_url01, outfile, ensure_ascii=False) 
        








