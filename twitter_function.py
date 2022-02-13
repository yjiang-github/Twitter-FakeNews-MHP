# -*- coding: utf-8 -*-
"""
Created on Fri Jan 15 10:13:26 2021

@author: yjian
"""

"""
--- this file is for setting up the function for data collection of fake news ---

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
import pandas as pd

#%%
""" simplified function """

def data_collection(query,API,delete_tweets,max_num,direction,dict_query):
    
    global tweets
    global api_1
    global api_2
    
    if delete_tweets == 'delete':
        del tweets
    
    if direction == 'backward':
        i = 0
    elif direction == 'forward':
        i = 1

    while i <= max_num:
        try:
            if i == 0:
                tweets = API.search(q=query,lang='en',tweet_mode='extended',count=1000,pages=10)
                # save tweets into dict_tweets_url
                for tweet in tweets:
                    if tweet.id_str not in dict_query.keys():
                        dict_query[tweet.id_str] = tweet._json
                i += 1
                
            else:
                
                # check rate limit first to see if necessary to switch api
                if i % 10 == 0:
                   # check rate limit
                   dict_limit = API.rate_limit_status() 
               
                   if dict_limit['resources']['search']['/search/tweets']['remaining'] <= 11:
                       # change to another api
                       if API == api_1:
                           API = api_2
                       else:
                           API = api_1
                
                # check collection direction
                if direction == 'forward':
                    since_id = max([int(tweet_id) for tweet_id in dict_query.keys()])+1 # the latest tweet id
                    tweets = API.search(q=query,lang='en',tweet_mode='extended', since_id = since_id, count=1000,pages=10)
                    
                elif direction == 'backward':
                    max_id = min([int(tweet_id) for tweet_id in dict_query.keys()])-1 # the earliest tweet id
                    tweets = API.search(q=query,lang='en',tweet_mode='extended', max_id = max_id, count=1000,pages=10)
                
                for tweet in tweets:
                    if tweet.id_str not in dict_query.keys():
                        dict_query[tweet.id_str] = tweet._json
                
                i += 1
                
                if len(tweets) == 0:
                    break
                
                del tweets
                
        except Exception as err:
            print(err)

#%%

def get_replies(tweet_id,dict_query,dict_replies,count,API):
    
    global tweets
    global tweet
    
    name = dict_query[tweet_id]['user']['screen_name']
    
    # check rate limit first to see if necessary to switch api
    if count % 10 == 0:
       # check rate limit
       dict_limit = API.rate_limit_status() 
       
       if dict_limit['resources']['search']['/search/tweets']['remaining'] <= 11:
           # change to another api
           if API == api_1:
               API = api_2
           else:
               API = api_1
    
    # define since_id
    ## if tweet has replies in dict_replies
    if tweet_id in dict_replies.keys():
        # tweet id of the latest reply
        since_id = max([int(reply_id) for reply_id in dict_replies[tweet_id].keys()])+1
    else:
        since_id = tweet_id
        
    # get replies
    tweets = tweepy.Cursor(API.search,q='to:'+name,since_id=since_id,timeout=999999).items(10000)
    
    for tweet in tweets:
        if hasattr(tweet, 'in_reply_to_status_is_str'):
            if (tweet.in_reply_to_status_id_str==tweet_id):
                if tweet_id not in dict_replies.keys():
                    dict_replies[tweet_id] = {}
                dict_replies[tweet_id][tweet.id_str] = tweet._json


    count += 1


