# -*- coding: utf-8 -*-
"""
Created on Sun Dec  6 13:48:16 2020

@author: yjian
"""


"""
--- this file is for collecting fake news/rumours tweets from twitter --- 
--- fake news from fake news website ---

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

#%%
createVar = locals()

#%%
""" I tweeted a blog link with fake news of 'Former President Barack Obama arrested for ESPIONAGE' """
""" now collected it to see how the link works """

#%%
"""
import keys and secrets for API
basically api_1 is for streaming, and api_2 is for re-collecting tweets by using tweet_id
"""



"""
--- my first key and secret ---
​
"""
consumer_key_1 = 'TEwiZkRDZjQWNIN8iby0TxDqX'  
consumer_secret_1 = 'l5g2Ngz97QwoebkpZ0HS6P3NFpv7Vw3scDGFIwBJEYtYXP9u4D'  
access_token_key_1 = '4876272161-kAVjVVZh02fYKTDeBru1AEftljoD0nqu1Tkbe7e'  
access_token_secret_1 = '594QERevczZ2osedoAf66ckXU5hf7IGoF1JusRvTHqDME'  


"""
--- my second key and secret ---
​
"""
consumer_key_2 = '6slRPLpFeJqGRWZ1DZ5uWQ7TF'  
consumer_secret_2 = 'r1tSESMauPzT24cM5O9zMh1SjPxXvhEjp1pqxz7CT98fE466jU'  
access_token_key_2 = '4876272161-yNb6YwyGSBBmI359nf7A1EYqjdKQ6hbv2Ky2SKS'  
access_token_secret_2 = 'nurDZsufBpL126vrBMik3Bu3zNMiyWRa1uSA8OuyQvo5O' 


"""
--- API ---
​
"""
auth_1 = tweepy.OAuthHandler(consumer_key_1, consumer_secret_1)  
auth_1.set_access_token(access_token_key_1, access_token_secret_1)  

api_1 = tweepy.API(auth_1, wait_on_rate_limit=True, wait_on_rate_limit_notify=True)


auth_2 = tweepy.OAuthHandler(consumer_key_2, consumer_secret_2)  
auth_2.set_access_token(access_token_key_2, access_token_secret_2)  

api_2 = tweepy.API(auth_2, wait_on_rate_limit=True, wait_on_rate_limit_notify=True)


#%%

tweets = api_1.user_timeline()


#%%
"""
tweet title: Former President Barack Obama arrested for ESPIONAGE

tweet link: https://www.conservativebeaver.com/2020/11/28/former-president-barack-obama-arrested-for-espionage/

tweet short link: https://t.co/b4eTerNqAE https://t.co/ESA79wYdhL

"""

"""
fact check links:
    
    https://t.co/QhIVKYvHNE
    https://obamagatescandal.com/fact-check-former-president-barack-obama-was-not-arrested-for-espionage-its-a-made-up-story/
    
"""


#%%
dict_tweets = {}
dict_tweets_2 = {}

#%%

""" search tweets with keywords """

query_1 = 'Former President Barack Obama arrested for ESPIONAGE'
query_2 = 'breaking obama has been arrested'



query = query_2

SearchingResults = api_1.search(q=query,lang='en',tweet_mode='extended',count=1000,pages=10)


#%%

""" save tweets """
for tweet in SearchingResults:
    if tweet.id not in dict_tweets_2:
        dict_tweets_2[tweet.id] = tweet._json


#%%    
""" search tweets posted before the earlist tweet collected """

max_id = min(list(dict_tweets_2.keys()))-1

# delete the past searching results
del SearchingResults

SearchingResults = api_2.search(q=query,lang='en',tweet_mode='extended', max_id = max_id, count=1000,pages=10)

""" save tweets """
for tweet in SearchingResults:
    if tweet.id not in dict_tweets_2:
        dict_tweets_2[tweet.id] = tweet._json

#%%
""" save dataset """
path = r'C:\Users\yjian\OneDrive\Documents\research files\dissertation\datasets\data collection\obama arrested for espionage'
with open(os.path.join(path, 'conservativebeaver.json'), 'w+', encoding="utf-8") as outfile:
    json.dump(dict_tweets, outfile, ensure_ascii=False) 
with open(os.path.join(path, 'shanesmedley.json'), 'w+', encoding="utf-8") as outfile:
    json.dump(dict_tweets, outfile, ensure_ascii=False) 
    
#%%

# check twitter user
list_2 = list(dict_tweets_2.keys())
list_2.sort(reverse=False)









